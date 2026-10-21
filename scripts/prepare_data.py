import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("ijcai", "yelp", "retail"))
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256()
    with args.source.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    metadata = json.loads((ROOT / "data/manifest.json").read_text())[args.dataset]
    if digest.hexdigest() != metadata["sha256"]:
        raise ValueError("Source does not match the canonical dataset used in Table 2")
    target = ROOT / "data/canonical" / args.dataset / "dataset.pkl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.source.resolve() != target.resolve():
        if target.exists():
            raise FileExistsError(target)
        shutil.copyfile(args.source, target)
    print(args.dataset, metadata["sha256"])


if __name__ == "__main__":
    main()
