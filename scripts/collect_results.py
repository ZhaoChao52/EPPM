import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("Recall@10", "NDCG@10", "Recall@20", "NDCG@20")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    reference = json.loads((ROOT / "results/table2_eppm.json").read_text())
    failed = False
    print("| Dataset | R@10 | N@10 | R@20 | N@20 | Check |")
    print("|---|---:|---:|---:|---:|---|")
    for dataset, expected in reference.items():
        path = ROOT / "outputs" / dataset / "seed2025/test_metrics.json"
        if path.exists():
            raw = json.loads(path.read_text())
            scores = {name: float(raw["Test:" + name]) for name in METRICS}
            status = "match" if all(abs(scores[name] - expected[name]) <= 0.0000051 for name in METRICS) else "different"
            cells = ["%.5f" % scores[name] for name in METRICS]
        else:
            status, cells = "missing", ["TBD"] * 4
        failed |= status != "match"
        print("| " + " | ".join([dataset] + cells + [status]) + " |")
    if args.check and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
