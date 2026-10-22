# Data

The frozen datasets used in Table 2 are included in [archives](archives).
Download the complete repository, then run from its root:

```bash
python scripts/extract_data.py
```

For a single dataset:

```bash
python scripts/extract_data.py --dataset yelp
```

The script joins archive parts in order, verifies their SHA256 checksums,
and extracts `data/canonical/{ijcai,yelp,retail}/dataset.pkl`. Existing files
are verified rather than overwritten. Archive parts must all be present;
individual `.part` files are not standalone ZIP files.

The three files occupy approximately 314 MiB uncompressed. Dataset SHA256
values are recorded in [manifest.json](manifest.json); archive checksums are
in [archive_manifest.json](archive_manifest.json). Dataset use and
redistribution remain subject to the source terms.

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

All splits first truncate the anchor-inclusive prefix to `max_len` and then
remove its final event to construct the history-only view. With `max_len=50`,
this yields at most 49 historical events and one separately supplied anchor.
The native prefix and post-action branch therefore share the same event window.

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

| Dataset | Upstream access | Contents |
|---|---|---|
| IJCAI | [Tianchi](https://tianchi.aliyun.com/dataset/42) | IJCAI-15 repeat-buyer source data |
| Yelp | [KHGT Yelp directory](https://github.com/akaxlh/KHGT/tree/master/Datasets/Yelp) | `trn_neg`, `trn_neutral`, `trn_pos`, and `trn_tip` timestamp matrices |
| Retail | [PBAT Google Drive](https://drive.google.com/file/d/1Ki8ibuTgkxnBITCe4gNLtrR0mG1vxevl/view?usp=sharing) | `retail.txt`, linked by the [PBAT authors](https://github.com/TiliaceaeSU/PBAT#dataset) |

These are third-party source releases, not drop-in replacements for the frozen
artifacts in `manifest.json`. They require the corresponding preprocessing.
Some providers may require sign-in. The MB-STR README also lists an
[upstream data share](https://github.com/yuanenming/mb-str#quick-start), but its
linked cloud endpoint could not be verified during this check.

Yelp uses the KHGT timestamp-matrix release, with negative, neutral, positive
and tip behaviors. Retail retains its source-specific filtering and
chronological conversion.
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
For Yelp and Retail, use the included frozen artifacts rather than attempting
to substitute a different upstream conversion.
