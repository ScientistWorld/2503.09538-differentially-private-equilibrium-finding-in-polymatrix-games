# Evaluation Targets

Every experiment is scored with the same four metrics (definitions in `scoring/reference.json` and `briefing/evaluation.md`). Lower is better for the three exploitability metrics; a failed experiment scores 10 on them and its compliance fraction.

## All experiments (`dense_er_scaling`, `dense_cluster_scaling`, `sparse_knn_scaling`)

- **Primary metric**: `comparator_exploitability_ratio` -- exploitability relative to the fixed private comparator at the same privacy budget. Paper (read from its figures, larger N): 0.84 dense ER, 0.83 dense clustered, about 0.8 nearest-neighbour graphs.
- **Constraint metric**: `privacy_protocol_compliance` -- must be 1.0. Any failing or missing instance fails the experiment, so exploitability cannot be bought with more rounds, less or selected noise, or stronger payoff dependence of the messages.
- **Secondary metrics**: `relative_exploitability` (exploitability / uniform play's), `relative_final_nash_gap`.

## Reference values (hidden test slice, odd seeds; visible slice in brackets)

| Experiment | Method | comparator ratio | relative exploitability | relative final Nash gap | compliance |
|---|---|---:|---:|---:|---:|
| dense_er_scaling | our_method (Algorithm 1) | 0.987 [0.988] | 0.979 [0.958] | 1.249 [1.228] | 1.0 |
| | structure_free_baseline | 1.000 | 0.992 [0.970] | 1.265 [1.244] | 1.0 |
| | no_payoff_baseline | 1.102 [1.119] | 1.081 [1.068] | 1.259 [1.256] | 1.0 |
| dense_cluster_scaling | our_method | 0.986 [0.987] | 0.996 [0.984] | 1.250 [1.250] | 1.0 |
| | structure_free_baseline | 1.000 | 1.010 [0.997] | 1.270 [1.267] | 1.0 |
| | no_payoff_baseline | 1.082 [1.087] | 1.082 [1.072] | 1.240 [1.255] | 1.0 |
| sparse_knn_scaling | our_method | 0.715 [0.710] | 0.910 [0.903] | 1.068 [1.065] | 1.0 |
| | structure_free_baseline | 1.000 | 1.275 [1.272] | 1.482 [1.479] | 1.0 |
| | no_payoff_baseline | 0.904 [0.902] | 1.152 [1.146] | 1.273 [1.267] | 1.0 |

The paper's direction (lower exploitability than the budget-matched prior-work-style comparator) holds in all three experiments, but in the dense families only by about 1.3% at this scale (see `HARDENING.md`). None of the exploitability metrics is near its bound: a protocol-compliant variant of the reference with stronger shrinkage towards uniform reaches comparator ratios of about 0.865 / 0.862 / 0.611.
