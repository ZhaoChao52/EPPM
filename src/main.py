import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(dataset, output_dir=None):
    path = ROOT / "configs" / (dataset + ".yaml")
    config = yaml.safe_load(path.read_text())
    config["canonical_dataset_path"] = str(ROOT / config["canonical_dataset_path"])
    config["output_dir"] = str(output_dir.resolve() if output_dir else ROOT / config["output_dir"])
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("ijcai", "yelp", "retail"))
    parser.add_argument("--mode", choices=("train", "test", "smoke"), default="train")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.dataset, args.output_dir)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import pytorch_lightning as pl
    import torch
    from data import UnifiedData
    from eppm import ParallelExpertMGPT
    from trainer import build_trainer

    data_path = Path(config["canonical_dataset_path"])
    expected_hash = json.loads((ROOT / "data/manifest.json").read_text())[args.dataset]["sha256"]
    if not data_path.exists():
        raise FileNotFoundError("Install the canonical dataset using scripts/prepare_data.py: " + str(data_path))
    if digest_file(data_path) != expected_hash:
        raise ValueError("Dataset SHA256 does not match the primary experiment")
    output = Path(config["output_dir"])
    saved = output / "config.yaml"
    last = output / "checkpoints/last.ckpt"
    best = output / "checkpoints/best.ckpt"
    if args.mode == "train":
        if saved.exists() and yaml.safe_load(saved.read_text()) != config:
            raise ValueError("Existing configuration differs from the requested run")
        if args.resume and not last.exists():
            raise FileNotFoundError(last)
        if not args.resume and (last.exists() or best.exists() or saved.exists()):
            raise FileExistsError("Use --resume for an existing run, or select another --output-dir")
    pl.seed_everything(int(config["seed"]), workers=True)
    data = UnifiedData(config)
    assert data.num_items == config["num_items"]
    assert data.num_behaviors == config["num_behaviors"]
    batches = config["samples_per_epoch"] // config["train_batch_size"]
    assert batches // int(config.get("accumulate_grad_batches", 1)) == 600
    output.mkdir(parents=True, exist_ok=True)
    model = ParallelExpertMGPT(config, data.num_items, data.num_behaviors, output)
    print("[CONFIG]", json.dumps(config, sort_keys=True), flush=True)
    print("[PARAMS]", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    if args.mode == "smoke":
        batch = next(iter(data.train_loader(config)))
        logits = model._logits(batch)
        assert logits.shape == (batch["target"].shape[0], data.num_items + 1)
        loss = model.loss_fn(logits, batch["target"])
        loss.backward()
        assert torch.isfinite(loss)
        print("[SMOKE_PASS]", float(loss), flush=True)
        return
    if args.mode == "train":
        saved.write_text(yaml.safe_dump(config))
        (output / "data_stats.json").write_text(json.dumps(data.stats, indent=2))
        trainer = build_trainer(config)
        trainer.fit(model, data.train_loader(config), data.val_loader(config),
                    ckpt_path=str(last) if args.resume else None)
        checkpoint = best
    else:
        checkpoint = args.checkpoint.resolve() if args.checkpoint else best
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    selection = {"checkpoint": str(checkpoint), "sha256": digest_file(checkpoint),
                 "epoch": state.get("epoch"), "global_step": state.get("global_step")}
    del state
    (output / "test_checkpoint.json").write_text(json.dumps(selection, indent=2))
    trainer = build_trainer(config)
    trainer.test(model, data.test_loader(config), ckpt_path=str(checkpoint))


if __name__ == "__main__":
    main()
