# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-27 19:04] - core_claim
- Completed dense Erdos-Renyi reproduction of Algorithm 1.
- Measured decreasing CCE exploitability and decreasing Renyi-DP bound across the player-count grid.
- Evaluation writes `scoring/scores.json` without importing from `method/`.

### [2026-05-27 19:25] - core_claim
- Prepared a follow-up job covering dense clustered, sparse random, and sparse k-nearest-neighbor topologies.
- Updated scoring targets so dense and sparse topologies are assessed against the different qualitative claims reported in Appendix D.

### [2026-05-27 20:05] - core_claim
- Audited the inherited workspace and restored the core claim based on the completed dense Erdos-Renyi run.
- Fixed the expanded evaluator so `scores.json` contains only metrics declared in `reference.json`; raw slopes are separated into `scoring/diagnostics.json`.

### [2026-05-27 20:30] - core_claim
- Multi-topology job completed and reproduced dense ER, dense clustered, and sparse kNN trend targets.
- Sparse random exploitability was stable/decreasing, but privacy did not decrease on the first grid.
- Corrected the sparse finite-scale step schedule to preserve the paper's sparse privacy propagation regime and submitted a rerun.

## Stop Justification
- Not stopped. Current job submission is intended to expand coverage toward `core_claim_plus`.
