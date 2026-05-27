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

These targets test the paper's sparse-graph claim that privacy improves with player count while exploitability does not meaningfully degrade.

Run evaluation with `bash scripts/evaluate.sh` after methods have written manifests under `outputs/`.
