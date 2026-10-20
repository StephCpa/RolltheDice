# Circle graph-factorial run record

This is a frozen development run manifest created on 2026-10-18 after two L40
GPUs became available on `gpu-lzn`. It authorizes 12 Circle training runs:
four graph cells × three seeds, using `ntp_fixed`, batch 128, 40,000 steps.

The run is independent of the still-running Sibling Gate A and will not use
Sibling intermediate results to change its design. Outputs are append-only
under a new run root. The first launcher attempt (`...-20261018`) stopped
before training because the copied server code lacked the frozen
`--model_revision` argument; its logs are retained as an engineering
preflight record. The formal queue uses code snapshot `1274bde` under
`code-v1274bde` and root `...-20261018-v3`. The second preflight attempt
(`...-v2`) also stopped before training because its Python runtime loaded a
`flash_attn` binary requiring unavailable `GLIBC_2.32`; that log is retained.
The formal queue uses the compatible `dllm_cf_l40_py312` environment.
Checkpoint selection and held-out scoring remain separate actions; no held-out
result exists at freeze time.

