import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import BudgetSampler, EvalDataset, UnifiedData, make_example
from eppm import ParallelExpertMGPT

torch.set_num_threads(1)


def small_config(output):
    config = yaml.safe_load((ROOT / "configs/yelp.yaml").read_text())
    config.update(num_items=67, num_workers=0, output_dir=str(output), samples_per_epoch=4,
                  train_batch_size=2, eval_batch_size=2, max_epochs=2,
                  check_val_every_n_epoch=1, model_sequence_length=60)
    return config


def examples():
    return [make_example([1, 2, 3], [1, 2, 3], 2, 4, 50, 1),
            make_example([5, 6, 7, 8], [4, 3, 2, 1], 3, 9, 50, 2)]


class ReproductionTests(unittest.TestCase):
    def test_primary_configuration_budget(self):
        for dataset in ("ijcai", "yelp", "retail"):
            config = yaml.safe_load((ROOT / "configs" / (dataset + ".yaml")).read_text())
            self.assertEqual(config["seed"], 2025)
            self.assertEqual(config["sampler_seed"], 2025)
            self.assertEqual(config["samples_per_epoch"], 307200)
            self.assertEqual(config["samples_per_epoch"] // config["train_batch_size"] // config.get("accumulate_grad_batches", 1), 600)
            self.assertEqual(config["transition_rank"], 8)
            self.assertEqual(config["gate_warmup_steps"], 3000)
            self.assertEqual(config["patience"], 10)
            result = subprocess.run([sys.executable, str(ROOT / "src/main.py"), "--dataset", dataset, "--dry-run"],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["seed"], 2025)

    def test_sampler_is_seed_and_epoch_controlled(self):
        sampler = BudgetSampler(list(range(100)), 128, 2025)
        first = list(sampler)
        self.assertEqual(first, list(BudgetSampler(list(range(100)), 128, 2025)))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))

    def test_chronological_split(self):
        with tempfile.TemporaryDirectory() as directory:
            config = small_config(directory)
            path = Path(directory) / "dataset.pkl"
            artifact = {"train": {1: [1, 2, 3, 4]}, "train_b": {1: [1, 2, 3, 4]},
                        "val": {1: [5]}, "val_b": {1: [1]},
                        "smap": dict.fromkeys(range(1, 68)), "bmap": dict.fromkeys(range(1, 5)), "umap": {1: 1}}
            with path.open("wb") as stream:
                pickle.dump(artifact, stream)
            config["canonical_dataset_path"] = str(path)
            data = UnifiedData(config)
            self.assertEqual(len(data.train_dataset), 2)
            self.assertEqual(int(data.train_dataset[1]["target"]), 3)
            self.assertEqual(int(data.val_dataset[0]["target"]), 4)
            self.assertEqual(int(data.test_dataset[0]["target"]), 5)
            self.assertEqual(int(data.test_dataset[0]["anchor_item"]), 4)

    def test_historical_evaluation_window(self):
        sequence = list(range(1, 61))
        ds = EvalDataset({1: sequence}, {1: [1] * 60}, {1: [61]}, 50)
        batch = ds[0]
        self.assertEqual(batch["prefix_ids"].tolist(), list(range(11, 61)))
        self.assertEqual(batch["pre_ids"].tolist(), list(range(10, 60)))
        self.assertEqual(int(batch["target"]), 61)

    def test_exact_parallel_expert_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(2025)
            config = small_config(directory)
            model = ParallelExpertMGPT(config, 67, 4, directory).eval()
            batch = next(iter(DataLoader(examples(), batch_size=2)))
            with torch.no_grad():
                actual = model._logits(batch)
                native = model.backbone(batch["prefix_ids"], batch["prefix_behaviors"])
                historical = model.backbone(batch["pre_ids"], batch["pre_behaviors"])
                candidate, _, _ = model.acpt_update(historical[-1], model.backbone.item_embedding(batch["anchor_item"]),
                                                   model.backbone.type_embedding(batch["anchor_behavior"]), batch["anchor_behavior"], True)
                expert = native[-1] + model.expert_mix * (candidate - native[-1])
                weights = model.backbone.item_embedding.weight
                expected = torch.maximum(torch.stack([h @ weights.t() for h in native], dim=1).max(1).values,
                                         expert @ weights.t())
                expected[:, 0] = torch.finfo(expected.dtype).min
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.shape, (2, 68))

    def test_warmup_gate_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            config = small_config(directory)
            model = ParallelExpertMGPT(config, 67, 4, directory)
            batch = next(iter(DataLoader(examples(), batch_size=2)))
            for step in (0, 3000):
                model.zero_grad(set_to_none=True)
                model.completed_updates.fill_(step)
                loss = model.loss_fn(model._logits(batch), batch["target"])
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                for parameter in model.acpt_update.gate.parameters():
                    self.assertEqual(parameter.grad is None, step == 0)

    def test_full_item_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            config = small_config(directory)
            model = ParallelExpertMGPT(config, 67, 4, directory)
            scores = torch.arange(68).float().unsqueeze(0)
            scores[:, 0] = torch.finfo(scores.dtype).min
            model._logits = lambda batch: scores
            model._outputs = []
            model._evaluation_step({"target": torch.tensor([67])})
            self.assertEqual(float(model._outputs[0]["recall20"][0]), 1.0)
            self.assertEqual(float(model._outputs[0]["ndcg10"][0]), 1.0)

    def test_full_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = small_config(directory)
            fixture = examples()
            loader = DataLoader(fixture, batch_size=2, sampler=BudgetSampler(fixture, 2, 2025))

            def trainer(epochs):
                checkpoint = ModelCheckpoint(dirpath=Path(directory) / "checkpoints", filename="best", monitor="Val:NDCG@20",
                                             mode="max", save_top_k=1, save_last=True, auto_insert_metric_name=False,
                                             save_on_train_epoch_end=False)
                return pl.Trainer(accelerator="cpu", devices=1, max_epochs=epochs, logger=False,
                                  callbacks=[checkpoint, EarlyStopping(monitor="Val:NDCG@20", mode="max", patience=10,
                                                                      check_on_train_epoch_end=False)],
                                  enable_progress_bar=False, num_sanity_val_steps=0)
            model = ParallelExpertMGPT(config, 67, 4, directory)
            first = trainer(1)
            first.fit(model, loader, loader)
            path = Path(directory) / "checkpoints/last.ckpt"
            state = torch.load(path, map_location="cpu")
            self.assertTrue(state["optimizer_states"])
            self.assertIn("loops", state)
            resumed = ParallelExpertMGPT(config, 67, 4, directory)
            second = trainer(2)
            second.fit(resumed, loader, loader, ckpt_path=str(path))
            self.assertEqual(second.global_step, 2)
            self.assertEqual(int(resumed.completed_updates), 2)
            self.assertEqual(sorted(p.name for p in path.parent.glob("*.ckpt")), ["best.ckpt", "last.ckpt"])


if __name__ == "__main__":
    unittest.main()
