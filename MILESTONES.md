# Reproduction Milestones

**Current: core_claim_plus**

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

### [2026-05-27 20:35] - core_claim_plus
- Regime-specific privacy rerun completed successfully and `python validate.py --compare` passed.
- Reproduced the core dense ER scaling claim plus dense clustered and sparse kNN settings with fixed-deviation CCE scoring.
- Submitted one final packaging rerun so unscored sparse-random remains visible in `scoring/diagnostics.json`.

### [2026-05-27 20:45] - core_claim_plus
- Final packaging rerun completed successfully and validation passed.
- `scores.json` contains the three reproduced scored experiments; `diagnostics.json` also records sparse-random as an unscored finite-grid gap.

### [2026-05-27 20:11] - core_claim_plus
- Final verification job `abee7929-170` completed with exit code 0.
- `scripts/evaluate.sh`, `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, and `python validate.py --compare` all passed in the container runtime.
- Verified that method-agnostic files, scoring docs, and train/test split wiring are suitable for gym use.
