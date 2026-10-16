# Sibling derived-5x500 no-heldout smoke record (development)

Date: 2026-10-17 (Asia/Shanghai)
Remote host: gpu-gfkd
Run: runs/icml2027-dev-20261017/sibling-smoke-derived/ntp_fixed/s42
Data: accepted-derived-5x500 base; graph: accepted-derived-5x500/graph.
Model: the same offline GPT-2 snapshot used by the Circle development batch,
revision gpt2-local-snapshot-607a30d7, model.safetensors SHA
248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707.

## Check

An 8-step, batch-4, seed-42 run completed on an NVIDIA RTX A6000 with explicit
--task sibling and explicit graph directory. Checkpoint/evaluation plumbing
completed at steps 4 and 8. The run metadata recorded the graph schema and both
graph SHA values. Remote artifact SHA values were:

- run_meta.json: 99cf47721a49a271f15423837f11f196bd804c13f554d2eae9a3c982d75c8011
- throughput.json: 90406c45a4018ebeafc894d4bbb81775141e44ee5896f65dd82347dc776ca
- scores.jsonl: 85a656b208cefd8f561f2f3251aa2d432b2062b0614aa591ad27db728788ddaa

## Boundary

The smoke produced zero validity/creativity at steps 4 and 8, as expected for an
undertrained 8-step run. Those numbers are engineering diagnostics only and
cannot support a Sibling scientific claim. The formal freeze remains separate;
the ready manifest records this smoke as an engineering gate, not as result evidence.

## Remote launch preflight

The accepted candidate and ready freeze manifest were copied to a separate remote
data/config path. The manifest-driven launcher dry-run for ntp_fixed_2hash/seed42
passed with the same file hashes, explicit sibling task, graph directory, model
revision, 800,000-step protocol, and no output run created.
