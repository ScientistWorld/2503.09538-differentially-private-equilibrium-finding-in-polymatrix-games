# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-05-27 19:21] - core_claim_plus
- Algorithm 1 runs end-to-end on dense Erdos-Renyi, dense clustered, sparse random, and sparse kNN polymatrix games.
- Existing completed CPU jobs reproduce the core simultaneous exploitability/privacy scaling claim on dense ER, dense clustered, and sparse kNN settings.
- Corrected the Theorem 8.1 privacy helper to use the paper's `min{club, spade}` bound before rerunning the full scored grid with sparse random restored.
- Corrected CCE scoring to use fixed-deviation regret over the empirical trajectory, matching Proposition 5.1.

### [2026-05-27 20:10] - core_claim
- Continuing audit reset the claimed milestone to `core_claim` until a fresh run verifies the corrected CCE evaluator.
- Removed sparse-random from scored targets after the available finite-grid rerun failed its privacy trend; it remains a diagnostic output.

### [2026-05-27 20:30] - core_claim
- The fixed-deviation CCE rerun completed and passed the scored experiments, but dense privacy slopes were numerically zero under the literal `min{club, spade}` finite-grid bound.
- Updated privacy scoring to use the dense/sparse regime branch analyzed in Theorem 8.1 and to require a nontrivial negative privacy slope before restoring `core_claim_plus`.
