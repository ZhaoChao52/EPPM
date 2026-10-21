
import argparse
import json
import os
import pickle
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.source.open("rb") as handle:
        source = pickle.load(handle)

    required = {"train", "val", "train_b", "val_b", "umap", "smap", "bmap"}
    missing = required.difference(source)
    if missing:
        raise KeyError("Missing source keys: %s" % sorted(missing))

    train = {}
    val = {}
    train_b = {}
    val_b = {}
    train_behavior_counts = Counter()
    val_behavior_counts = Counter()
    test_behavior_counts = Counter()
    full_lengths = []

    for user in sorted(source["train"]):
        full_items = list(source["train"][user])
        full_behaviors = list(source["train_b"][user])
        held_out_items = list(source["val"].get(user, []))
        held_out_behaviors = list(source["val_b"].get(user, []))
        if len(full_items) != len(full_behaviors):
            raise AssertionError("Item/behavior length mismatch for user %s" % user)
        if len(held_out_items) != len(held_out_behaviors):
            raise AssertionError("Held-out length mismatch for user %s" % user)
        if len(held_out_items) > 1:
            raise AssertionError("Expected at most one native held-out event")



        full_items.extend(held_out_items)
        full_behaviors.extend(held_out_behaviors)
        if len(full_items) < 3:
            continue




        train[user] = full_items[:-1]
        train_b[user] = full_behaviors[:-1]
        val[user] = [full_items[-1]]
        val_b[user] = [full_behaviors[-1]]
        full_lengths.append(len(full_items))
        train_behavior_counts.update(full_behaviors[:-2])
        val_behavior_counts.update([full_behaviors[-2]])
        test_behavior_counts.update([full_behaviors[-1]])

    canonical = {
        "train": train,
        "val": val,
        "train_b": train_b,
        "val_b": val_b,
        "val_num": len(val),
        "umap": source["umap"],
        "smap": source["smap"],
        "bmap": source["bmap"],
    }
    num_items = len(source["smap"])
    if any(item <= 0 or item > num_items for seq in train.values() for item in seq):
        raise AssertionError("Canonical item IDs are outside the dense item vocabulary")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "dataset.pkl"
    temporary_path = dataset_path.with_suffix(".pkl.tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(canonical, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(str(temporary_path), str(dataset_path))

    train_examples = sum(max(0, length - 3) for length in full_lengths)
    stats = {
        "protocol": "ACPT strict-v2 canonical IJCAI",
        "canonical_source": str(args.source.resolve()),
        "storage_semantics": {
            "train": "e1..e(T-1)",
            "val": "eT",
            "strict_v2_training_region": "e1..e(T-2)",
            "validation_target": "e(T-1)",
            "test_target": "eT",
        },
        "num_users_total": len(train),
        "num_items": num_items,
        "num_behaviors": len(source["bmap"]),
        "bmap": source["bmap"],
        "full_interactions": sum(full_lengths),
        "full_seq_len_min": min(full_lengths),
        "full_seq_len_max": max(full_lengths),
        "full_seq_len_mean": sum(full_lengths) / len(full_lengths),
        "train_users": len(train),
        "train_examples": train_examples,
        "val_users": len(val),
        "val_cases": len(val),
        "test_users": len(val),
        "test_cases": len(val),
        "train_behavior_counts": dict(sorted(train_behavior_counts.items())),
        "val_behavior_counts": dict(sorted(val_behavior_counts.items())),
        "test_behavior_counts": dict(sorted(test_behavior_counts.items())),
        "leakage_audit": {
            "status": "PASS by construction",
            "training_targets": "strictly inside e1..e(T-2)",
            "held_out_events_in_training": False,
        },
    }
    with (args.output_dir / "split_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
