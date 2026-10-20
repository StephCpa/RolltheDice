# Circle graph-factorial launch record

Formal queue start: 2026-10-18 11:34:55 CST  
Host: `gpu-lzn` (`gfkd-R2280-V6-F201`)  
Devices: NVIDIA L40, CUDA devices 0 and 1  
Runtime: `/home/lzn/envs/dllm_cf_l40_py312/bin/python` (torch 2.4.0+cu121,
transformers 4.47.1)  
Code: `1274bde` under `code-v1274bde`

The queue started `G12` seeds 42 and 43 as the first two runs. Their logs
contain `run_meta.json` with the task, model revision, data directory, device,
and full argument vector. At the first progress check, both processes were
training normally (approximately 0.125–0.136 seconds per step); no held-out
evaluation has been run.

Two earlier attempts are retained as preflight failures: the first used a code
copy without `--model_revision`; the second used a runtime whose optional
`flash_attn` binary required unavailable `GLIBC_2.32`. Neither reached a
training step or produced a checkpoint. The formal outputs use the append-only
`...-20261018-v3` root.

