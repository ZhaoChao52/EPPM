# Reproducibility

## Scope

This artifact targets only the EPPM row of Table 2 on IJCAI, Yelp, and Retail.
It contains the M-GPT Parallel-Expert implementation, the primary seed-2025
configurations, full-item evaluation, and reference test metrics. It does not
contain other Table 2 methods or claim to reproduce the transfer and ablation
tables.

## Run From VS Code

Open the `EPPM` folder and select the environment containing the dependencies
listed in the README. Open a terminal at the repository root and run:

```bash
python src/main.py --dataset yelp --dry-run
python -m unittest discover -s tests -v
```

The first command prints configuration only. The second uses small synthetic
fixtures on CPU; it does not start a paper-scale training run.

After verifying the datasets, run the training commands in the README on an
available CUDA GPU. The primary experiments used Linux. Windows can be used
for inspection and CPU tests, but identical performance on Windows is not
certified.

## Data and Prediction Boundary

The stored `train` sequence ends at the penultimate event, and stored `val`
contains the final event. The loader reserves the penultimate event for
validation and the final event for testing. Training targets precede both.
The model receives observed item and behavior information, not the future
target behavior. Evaluation ranks the full nonpadding item catalog.

The historical loader independently truncates the anchor-inclusive and
history-only evaluation views to 50 events. For long sequences, the latter
can therefore contain one earlier event absent from the former. Training
first clips the anchor-inclusive segment before constructing its two views.
The release preserves these original semantics rather than changing the
inputs used in the reported experiments.

## Checkpoints

Checkpoints are selected by validation NDCG@20. The training entry point tests
`checkpoints/best.ckpt` after fitting, not the latest epoch. The resume command
uses `checkpoints/last.ckpt`, including optimizer and training-loop state.
IJCAI validates and saves every two epochs; Yelp and Retail do so every epoch.
Patience is measured in validation checks, not epochs.

Only the best and latest checkpoints are retained by the configured callback.
Checkpoint files are not included in the prepared distribution. An explicit
`--checkpoint` in test mode is the caller's selection and need not be the
validation-best model.

## Verification Scope

The CPU suite checks sampling, chronological splits, historical truncation,
Parallel-Expert score construction, gate warmup gradients, padding exclusion,
configuration budgets, and checkpoint resumption. These checks do not replace
a complete retraining run on each dataset.

The recorded Table 2 values are reference measurements from the original
runs. The collector reads only new test outputs and reports missing results
as `TBD`. Hardware, software, and resumed data-loader state can affect numerical
agreement. No new paper-scale retraining was performed while packaging.

## Source Integrity

`results/source_manifest.json` records the retained module provenance.
`results/file_manifest.json` records SHA256 values for the distributed code
and documentation. Data checksums are recorded separately in
`data/manifest.json`. Python source files contain no comments or docstrings.

See `THIRD_PARTY_NOTICES.md` before public redistribution of code or data.
