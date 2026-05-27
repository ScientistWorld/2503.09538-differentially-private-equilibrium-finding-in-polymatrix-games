# Evaluation Targets

## Primary: dense_er_scaling
Tests whether equilibrium quality improves as random dense games grow in player count.
- **Metric**: `cce_loglog_slope` (lower is better; negative means exploitability decreases with size)

## Constraint: dense_er_scaling
Tests whether the privacy budget improves rather than being sacrificed for accuracy.
- **Metric**: `privacy_loglog_slope` (lower is better; negative means the privacy bound decreases with size)

## Baseline Comparison: dense_er_scaling
Tests whether equilibrium quality is competitive with the adapted private distributed-optimization baseline under the same generated games and random seeds.
- **Metric**: `mean_exploitability_vs_baseline` (lower is better)

Run evaluation with `bash scripts/evaluate.sh` after one or more methods have written manifests under `outputs/`.
