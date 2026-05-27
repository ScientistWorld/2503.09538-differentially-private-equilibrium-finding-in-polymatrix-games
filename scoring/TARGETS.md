# Evaluation Targets

## Dense Graphs

Experiments: `dense_er_scaling`, `dense_cluster_scaling`

- **Primary metric**: `cce_decreases_with_n` (higher is better)
- **Constraint metric**: `privacy_decreases_with_n` (higher is better)
- **Baseline comparison**: `competitive_with_baseline` (higher is better)

These targets test the paper's dense-graph claim that equilibrium quality and privacy budget both improve as the number of players grows.

## Sparse Graphs

Experiment: `sparse_knn_scaling`

- **Primary metric**: `privacy_decreases_with_n` (higher is better)
- **Constraint metric**: `cce_stable_or_decreases_with_n` (higher is better)
- **Baseline comparison**: `competitive_with_baseline` (higher is better)

This target tests the paper's sparse-graph claim on the k-nearest-neighbor topology that reproduced successfully. The sparse-random topology is documented in `PROGRESS.md` as a residual finite-scale gap and is not included in scored targets.

Run evaluation with `bash scripts/evaluate.sh` after methods have written manifests under `outputs/`.
