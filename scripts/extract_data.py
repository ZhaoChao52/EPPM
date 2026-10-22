import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_dataset(dataset, archives, expected):
    target = ROOT / "data/canonical" / dataset / "dataset.pkl"
    if target.exists():
        if digest_file(target) != expected:
            raise ValueError("Existing dataset checksum mismatch: " + dataset)
        print(dataset + ": installed and verified")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(target.parent)) as temporary:
        archive = Path(temporary) / "dataset.zip"
        with archive.open("wb") as output:
            for part in archives["parts"]:
                source = ROOT / "data/archives" / part["file"]
                if source.stat().st_size != part["size_bytes"] or digest_file(source) != part["sha256"]:
                    raise ValueError("Archive part checksum mismatch: " + source.name)
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        if digest_file(archive) != archives["archive_sha256"]:
            raise ValueError("Archive checksum mismatch: " + dataset)
        extracted = Path(temporary) / "dataset.pkl"
        with zipfile.ZipFile(archive) as bundle:
            if bundle.namelist() != ["dataset.pkl"]:
                raise ValueError("Unexpected archive contents: " + dataset)
            with bundle.open("dataset.pkl") as source, extracted.open("wb") as output:
                shutil.copyfileobj(source, output)
        if digest_file(extracted) != expected:
            raise ValueError("Dataset checksum mismatch: " + dataset)
        extracted.replace(target)
    print(dataset + ": extracted and verified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", "ijcai", "yelp", "retail"), default="all")
    args = parser.parse_args()
    archives = json.loads((ROOT / "data/archive_manifest.json").read_text())
    datasets = json.loads((ROOT / "data/manifest.json").read_text())
    selected = ("ijcai", "yelp", "retail") if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        extract_dataset(dataset, archives[dataset], datasets[dataset]["sha256"])


if __name__ == "__main__":
    main()
