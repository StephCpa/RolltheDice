# Sibling derived-5x500 acceptance and fresh heldout freeze record (development)

Date: 2026-10-17 (Asia/Shanghai)
Decision source: the project owner explicitly accepted derived-5x500 in this session and requested one fresh heldout set after freeze.
Original candidate: candidate-public-5x500-20261017/; accepted read-only copy: accepted-derived-5x500-20261017/.
The old candidate_manifest.json was not overwritten; this record uses a separate copy and freeze manifest.

## Frozen inputs

- Data candidate: sibling.derived-5x500.20261017, 5 A entities, 500 B entities per A, 10-character hashes.
- Graph rule: derived from the upstream notebook contiguous-block rule; graph_is_official=false.
- Base/2hash/graph file SHA values are preserved in the candidate manifest and checked in the accepted copy.
- Fresh heldout: fresh-heldout-20261017.json, 2048 new hashes, generation seed 2026101701.
- Fresh heldout SHA256: edcdd3baa2dbe0067f0cb3a24cd60a7a7c84f3203bd640d625111ad42b95686a.
- Before generation, 103000 hashes in the accepted 2hash train/valid/test union were excluded; base and variant share this heldout file.

## Gate status

The data contract, derived graph identity, 2hash disjointness, and heldout-generation gates are recorded as passed. The no-heldout model/training throughput smoke is still pending, so the new freeze manifest remains blocked_pending_artifacts with throughput_smoke recorded as the next gate; no Sibling formal training has started. This record remains development input, not a scientific result.

