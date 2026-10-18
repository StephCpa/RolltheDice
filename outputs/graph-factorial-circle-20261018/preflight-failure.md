# Preflight failure record

The first Circle queue invocation created two log files and exited before
training because the server copy of `train.py` did not accept
`--model_revision`. No checkpoint or model output was produced. The failed
attempt is retained under `runs/icml2027-graph-factorial-circle-20261018`.

The corrected code snapshot is `1274bde` under `code-v1274bde`; the formal queue
uses the separate append-only root
`runs/icml2027-graph-factorial-circle-20261018-v2`.

