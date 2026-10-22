# Reproducibility

## Scope

This artifact targets only the EPPM row of Table 2 on IJCAI, Yelp, and Retail.
It contains the M-GPT Parallel-Expert implementation, the primary seed-2025
configurations, full-item evaluation, and reference test metrics. It does not
contain other Table 2 methods or claim to reproduce the transfer and ablation
tables.

## Configuration and Tests

From the repository root, with the dependencies in the README installed, run:

```bash
python src/main.py --dataset yelp --dry-run
python -m unittest discover -s tests -v
```

The first command prints configuration only. The second runs the CPU tests
using small synthetic fixtures.

Run `python scripts/extract_data.py` to install and verify the included frozen
datasets. Then run the training commands in the README on an
available CUDA GPU. The primary experiments used Linux. Windows can be used
for inspection and CPU tests, but identical performance on Windows is not
certified.

## Data and Prediction Boundary

The stored `train` sequence ends at the penultimate event, and stored `val`
contains the final event. The loader reserves the penultimate event for
validation and the final event for testing. Training targets precede both.
The model receives observed item and behavior information, not the future
target behavior. Evaluation ranks the full nonpadding item catalog.

Training, validation, and testing first truncate the anchor-inclusive prefix
to at most 50 events. The history-only view is the same window without its
final anchor, giving at most 49 historical events. Thus neither EPPM branch
accesses an event outside the native prefix window.

## Checkpoints

Checkpoints are selected by validation NDCG@20. The training entry point tests
`checkpoints/best.ckpt` after fitting, not the latest epoch. The resume command
uses `checkpoints/last.ckpt`, including optimizer and training-loop state.
IJCAI validates and saves every two epochs; Yelp and Retail do so every epoch.
Patience is measured in validation checks, not epochs.

Only the best and latest checkpoints are retained by the configured callback.
Checkpoint files are not included in the repository. An explicit
`--checkpoint` in test mode is the caller's selection and need not be the
validation-best model.

## Tests and Reference Results

The CPU suite checks sampling, chronological splits, shared-window truncation,
Parallel-Expert score construction, gate warmup gradients, padding exclusion,
configuration budgets, and checkpoint resumption. These checks do not replace
a complete retraining run on each dataset.

The recorded Table 2 values are reference measurements from the original
runs. The collector reads test outputs and reports missing results as `TBD`.
Hardware, software, and resumed data-loader state can affect numerical
agreement. Dataset checksums are recorded in `data/manifest.json`.

See `THIRD_PARTY_NOTICES.md` before public redistribution of code or data.
