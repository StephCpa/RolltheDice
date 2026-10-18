# Circle graph-factorial run record

This is a frozen development run manifest created on 2026-10-18 after two L40
GPUs became available on `gpu-lzn`. It authorizes 12 Circle training runs:
four graph cells × three seeds, using `ntp_fixed`, batch 128, 40,000 steps.

The run is independent of the still-running Sibling Gate A and will not use
Sibling intermediate results to change its design. Outputs are append-only
under a new run root. Checkpoint selection and held-out scoring remain separate
actions; no held-out result exists at freeze time.

