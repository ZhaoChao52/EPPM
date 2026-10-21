# Data

Training requires the frozen canonical dataset artifacts used in Table 2.
The local project contains these files in `data/canonical/`. They are also
packaged separately as `EPPM_datasets.zip`; the code archive excludes them.
Extract the data archive into the repository root. It contains
`data/canonical/{ijcai,yelp,retail}/dataset.pkl` and a copy of the manifest.
No model checkpoints or public download link are included.

The three files occupy approximately 314 MiB uncompressed. All three were
checked against SHA256 values computed directly from the server artifacts.
Before publishing the data archive, confirm the redistribution terms of each
source. Locally prepared data do not imply public redistribution permission.

## Format

Each trusted `dataset.pkl` is a Python pickle with these keys:

| Key | Meaning |
|---|---|
| `train` | User ID to observed item sequence ending at the penultimate event |
| `train_b` | Corresponding observed behavior sequence |
| `val` | Final held-out item event per eligible user |
| `val_b` | Corresponding held-out behavior, retained for bookkeeping only |
| `umap`, `smap`, `bmap` | Frozen user, item, and behavior mappings |

The loader derives training anchors from the region before the validation
target. The penultimate event is the validation target; the final event is the
test target. Held-out future behavior is never passed to the model.
Only unpickle trusted artifacts.

The original evaluation code separately truncates the anchor-inclusive prefix
and history-only view to `max_len`. Training clips the anchor-inclusive segment
before constructing its two views. This packaging retains that historical
behavior rather than silently changing the inputs used in the reported runs.

## Identity Checks

| Dataset | Items | Users in canonical artifact | SHA256 prefix |
|---|---:|---:|---|
| IJCAI | 874328 | 423423 | `c650c891` |
| Yelp | 22734 | 19800 | `81c571ca` |
| Retail | 99037 | 147894 | `806fee4b` |

Full checksums and aggregate statistics are in [manifest.json](manifest.json).
`scripts/prepare_data.py` verifies file bytes before installing an artifact.
Matching only the dataset name, vocabulary sizes, or split counts is not enough.

## Sources

IJCAI originates from the IJCAI-15 repeat-buyer interaction data distributed
through [Tianchi](https://tianchi.aliyun.com/dataset/42). Yelp uses the KHGT
timestamp-matrix release, with negative, neutral, positive and tip behaviors.
Retail retains its source-specific filtering and chronological conversion.
The precise canonical artifacts, including mappings and timestamp tie order,
are needed for the seed-controlled sampling stream.

The included `scripts/build_ijcai_canonical.py` converts the trusted native
chronological IJCAI cache into the storage convention above:

```bash
python scripts/build_ijcai_canonical.py --source /path/to/native/ijcai/dataset.pkl --output-dir /path/to/converted/ijcai
python scripts/prepare_data.py --dataset ijcai --source /path/to/converted/ijcai/dataset.pkl
```

This conversion does not reconstruct the original cache from raw transactions
and is not a replacement for the source release. A serialization or mapping
change can produce a different checksum even when aggregate counts agree.
The package intentionally provides no substitute raw-to-canonical procedure
for Yelp or Retail without the original conversion provenance.
