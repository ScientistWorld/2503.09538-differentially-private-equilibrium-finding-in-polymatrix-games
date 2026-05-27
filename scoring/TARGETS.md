# Evaluation Targets

## Dense Graphs

Experiments: `dense_er_scaling`, `dense_cluster_scaling`

- **Primary metric**: `cce_decreases_with_n` (higher is better)
- **Constraint metric**: `privacy_decreases_with_n` (higher is better)
- **Baseline comparison**: `competitive_with_baseline` (higher is better)

These targets test the paper's dense-graph claim that equilibrium quality and privacy budget both improve as the number of players grows.

## Sparse Graphs

Experiments: `sparse_random_scaling`, `sparse_knn_scaling`

- **Primary metric**: `privacy_decreases_with_n` (higher is better)
- **Constraint metric**: `cce_stable_or_decreases_with_n` (higher is better)
- **Baseline comparison**: `competitive_with_baseline` (higher is better)

These targets test the paper's sparse-graph claim on both sparse random graphs from Figure 6 and k-nearest-neighbor graphs from Appendix D.1.

Run evaluation with `bash scripts/evaluate.sh` after methods have written manifests under `outputs/`.
