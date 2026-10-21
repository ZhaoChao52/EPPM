import bisect
import hashlib
import json
import pickle

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def _pad_left(values, length):
    values = list(values)[-length:]
    return [0] * (length - len(values)) + values


def make_example(items, behaviors, anchor_pos, target, max_len, user):
    assert 0 <= anchor_pos < len(items)
    assert target > 0
    prefix_items = items[: anchor_pos + 1]
    prefix_behaviors = behaviors[: anchor_pos + 1]
    pre_items = items[:anchor_pos]
    pre_behaviors = behaviors[:anchor_pos]
    assert len(prefix_items) == len(prefix_behaviors)
    return {
        "user": torch.tensor(user, dtype=torch.long),
        "prefix_ids": torch.tensor(_pad_left(prefix_items, max_len), dtype=torch.long),
        "prefix_behaviors": torch.tensor(
            _pad_left(prefix_behaviors, max_len), dtype=torch.long
        ),
        "pre_ids": torch.tensor(_pad_left(pre_items, max_len), dtype=torch.long),
        "pre_behaviors": torch.tensor(
            _pad_left(pre_behaviors, max_len), dtype=torch.long
        ),
        "anchor_item": torch.tensor(items[anchor_pos], dtype=torch.long),
        "target": torch.tensor(target, dtype=torch.long),
        "anchor_behavior": torch.tensor(behaviors[anchor_pos], dtype=torch.long),
    }


class TrainDataset(Dataset):
    def __init__(self, sequences, behaviors, max_len):
        self.sequences = sequences
        self.behaviors = behaviors
        self.max_len = int(max_len)
        self.users = []
        self.offsets = [0]
        for user in sorted(sequences):
            count = max(0, len(sequences[user]) - 1)
            if count:
                self.users.append(user)
                self.offsets.append(self.offsets[-1] + count)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        user_index = bisect.bisect_right(self.offsets, index) - 1
        user = self.users[user_index]
        anchor_pos = index - self.offsets[user_index]
        items = self.sequences[user]
        behaviors = self.behaviors[user]
        start = max(0, anchor_pos - self.max_len + 1)
        return make_example(
            items[start : anchor_pos + 1],
            behaviors[start : anchor_pos + 1],
            anchor_pos - start,
            items[anchor_pos + 1],
            self.max_len,
            user,
        )


class EvalDataset(Dataset):
    def __init__(self, sequences, behaviors, targets, max_len):
        self.sequences = sequences
        self.behaviors = behaviors
        self.targets = targets
        self.max_len = int(max_len)
        self.users = [
            user
            for user in sorted(targets)
            if user in sequences and len(sequences[user]) > 0
        ]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        items = self.sequences[user]
        behaviors = self.behaviors[user]
        return make_example(
            items,
            behaviors,
            len(items) - 1,
            self.targets[user][0],
            self.max_len,
            user,
        )


class BudgetSampler(Sampler):
    def __init__(self, dataset, samples, seed):
        self.total = len(dataset)
        self.samples = int(samples)
        self.seed = int(seed)
        self.epoch = 0
        assert self.total > 0 and self.samples > 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def indices(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randint(self.total, (self.samples,), generator=generator)

    def __iter__(self):
        indices = self.indices()
        digest = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
        print(f"[SAMPLER_STREAM] epoch={self.epoch} sha256={digest}", flush=True)
        return iter(indices.tolist())

    def __len__(self):
        return self.samples

    def audit(self):
        indices = self.indices()
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "count": len(indices),
            "unique": int(indices.unique().numel()),
            "sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
            "first20": json.dumps(indices[:20].tolist()),
        }


class UnifiedData:
    def __init__(self, config):
        with open(config["canonical_dataset_path"], "rb") as handle:
            raw = pickle.load(handle)
        train = raw["train"]
        train_b = raw["train_b"]
        heldout = raw["val"]
        heldout_b = raw["val_b"]

        legal_train = {}
        legal_train_b = {}
        val_seq = {}
        val_seq_b = {}
        val_target = {}
        test_seq = {}
        test_seq_b = {}
        test_target = {}

        for user in sorted(train):
            items = train[user]
            behaviors = train_b[user]
            assert len(items) == len(behaviors)
            has_heldout = (
                user in heldout
                and user in heldout_b
                and len(heldout[user]) == 1
                and len(heldout_b[user]) == 1
            )
            train_items = items[:-1] if has_heldout else items
            train_behaviors = behaviors[:-1] if has_heldout else behaviors
            if len(train_items) >= 2:
                legal_train[user] = train_items
                legal_train_b[user] = train_behaviors
            if has_heldout and len(items) >= 2:
                val_seq[user] = items[:-1]
                val_seq_b[user] = behaviors[:-1]
                val_target[user] = [items[-1]]
                test_seq[user] = items
                test_seq_b[user] = behaviors
                test_target[user] = heldout[user]

        max_len = int(config["max_len"])
        self.train_dataset = TrainDataset(legal_train, legal_train_b, max_len)
        self.val_dataset = EvalDataset(val_seq, val_seq_b, val_target, max_len)
        self.test_dataset = EvalDataset(test_seq, test_seq_b, test_target, max_len)
        self.num_items = len(raw["smap"])
        self.num_behaviors = len(raw["bmap"])
        micro_batches_per_epoch = int(config["samples_per_epoch"]) // int(
            config["train_batch_size"]
        )
        accumulation = int(config.get("accumulate_grad_batches", 1))
        self.stats = {
            "split_mode": "strict_v2",
            "train_users": len(legal_train),
            "legal_train_examples": len(self.train_dataset),
            "sampled_examples_per_epoch": int(config["samples_per_epoch"]),
            "micro_batches_per_epoch": micro_batches_per_epoch,
            "gradient_accumulation": accumulation,
            "optimizer_steps_per_epoch": micro_batches_per_epoch // accumulation,
            "val_users": len(self.val_dataset),
            "test_users": len(self.test_dataset),
            "num_items": self.num_items,
            "num_behaviors": self.num_behaviors,
        }
        print("[DATA_STATS] " + json.dumps(self.stats, sort_keys=True), flush=True)

    def train_loader(self, config):
        sampler = BudgetSampler(
            self.train_dataset, config["samples_per_epoch"], config["sampler_seed"]
        )
        return DataLoader(
            self.train_dataset,
            batch_size=config["train_batch_size"],
            sampler=sampler,
            num_workers=config["num_workers"],
            pin_memory=True,
            persistent_workers=config["num_workers"] > 0,
        )

    def val_loader(self, config):
        return DataLoader(
            self.val_dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            persistent_workers=config["num_workers"] > 0,
        )

    def test_loader(self, config):
        return DataLoader(
            self.test_dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            persistent_workers=config["num_workers"] > 0,
        )
