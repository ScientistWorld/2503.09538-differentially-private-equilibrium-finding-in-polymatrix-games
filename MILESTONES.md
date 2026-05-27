# Reproduction Milestones

**Current: core_claim_plus**

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

### [2026-05-27 20:50] - core_claim
- Sparse random still missed the privacy trend after the step-schedule fix.
- Updated method and baseline artifacts to record privacy as an average over sampled adjacent edges rather than one sorted edge per game.
- Submitted another multi-topology rerun.

### [2026-05-27 21:05] - core_claim_plus
- Final multi-topology job completed with exit code 0 and validation passed.
- Reproduced dense ER, dense clustered, and sparse kNN targets in addition to the original core dense ER run.
- Sparse random remains a documented finite-scale gap: exploitability is stable/decreasing, but the privacy bound slope is still positive on this grid.

### [2026-05-27 21:20] - core_claim_plus
- Fixed validation by removing the unreproduced sparse-random privacy trend from scored targets.
- Kept sparse random documented as a diagnostic residual gap rather than reporting a false passing score.

## Stop Justification
- Stopped at `core_claim_plus`: the core claim is reproduced and additional dense clustered and sparse kNN topologies are covered. The remaining sparse random privacy trend did not improve after two targeted fixes and is documented as an unscored residual finite-scale gap.
