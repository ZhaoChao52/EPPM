# Source Notices

EPPM uses an M-GPT backbone implementing interaction-level graph encoding,
multi-grained preference extraction, and native expert scoring. The retained
backbone is not presented as a new contribution of EPPM.

The provided module snapshots and primary configurations were used by the
reported experiments. File checksums are recorded in
`results/source_manifest.json`. The public entry point changes path handling
and adds explicit checkpoint resumption; the model and evaluator are retained.
Python comments and docstrings have been removed from the code-only release.

No upstream license file was found in the inspected M-GPT snapshot. This
repository does not assert a new permissive license over third-party work.
The applicable upstream permission and attribution terms must be confirmed
before public redistribution. This notice does not replace such permission.

Dataset terms are independent of code terms. No dataset redistribution rights
are asserted or granted by this repository.
