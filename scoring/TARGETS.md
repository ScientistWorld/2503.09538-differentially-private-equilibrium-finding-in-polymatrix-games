# Evaluation Targets

## Primary: dense_er_scaling
Tests whether equilibrium quality improves as random dense games grow in player count.
- **Metric**: `cce_decreases_with_n` (higher is better)

## Constraint: dense_er_scaling
Tests whether the privacy budget improves rather than being sacrificed for accuracy.
- **Metric**: `privacy_decreases_with_n` (higher is better)

## Baseline Comparison: dense_er_scaling
Tests whether equilibrium quality is competitive with the adapted private distributed-optimization baseline under the same generated games and random seeds.
- **Metric**: `competitive_with_baseline` (higher is better)

Run evaluation with `bash scripts/evaluate.sh` after one or more methods have written manifests under `outputs/`.
