# Sibling staging deployment (development)

Date: 2026-10-17 (Asia/Shanghai)

The committed P6 code at `0f8cee6` was copied to the isolated server staging
directory:

`/home/wlwuser/P6-Roll_the_dice/staging-0f8cee6/`

The deployed files are `train.py`, `evaluator.py`, `rescore.py`,
`validate_sibling_manifest.py`, and `run_sibling_confirmation.py`. Their
SHA-256 values were checked against the local worktree. Remote `py_compile`,
manifest validation, and evaluator selftest passed.

The launcher was run in `--dry-run` mode for
`ntp_fixed_2hash/seed42`. It resolved the derived candidate data and graph
files and produced the expected 800,000-step command. No output run directory,
checkpoint, model weight, or heldout score was created.

At the verification time, gpu-gfkd GPUs 4 and 5 each showed about 22 GB free,
but `fuser` showed shared Python/torch/monitor processes holding both device
files. The free-memory reading therefore was not treated as permission to
start a job. The formal freeze manifest remains `blocked_pending_artifacts`.
