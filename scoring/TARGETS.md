# Evaluation Targets

Every experiment is scored with the same four metrics (definitions in `scoring/reference.json` and `briefing/evaluation.md`). Lower is better for the three exploitability metrics; a failed experiment scores 10 on them and its compliance fraction.

## All experiments (`dense_er_scaling`, `dense_cluster_scaling`, `sparse_knn_scaling`)

- **Primary metric**: `comparator_exploitability_ratio` -- exploitability relative to the fixed private comparator at the same privacy budget. Paper (read from its figures, larger N): 0.84 dense ER, 0.83 dense clustered, about 0.8 nearest-neighbour graphs.
- **Constraint metric**: `privacy_protocol_compliance` -- must be 1.0. Any failing instance fails the experiment. Exploitability therefore cannot be bought with stronger payoff dependence of the messages, or with players using information beyond their own view and received messages (locality audit). The evaluator fixes the rounds and the noise.
- **Secondary metrics**: `relative_exploitability` (exploitability / uniform play's), `relative_final_nash_gap`.

## Reference values (hidden test slice, 8 hidden instances per size; visible train slice in brackets)

| Experiment | Method | comparator ratio | relative exploitability | relative final Nash gap | compliance |
|---|---|---:|---:|---:|---:|
| dense_er_scaling | our_method (Algorithm 1) | 0.989 [0.988] | 0.969 [0.958] | 1.213 [1.228] | 1.0 |
| | structure_free_baseline | 1.000 | 0.980 [0.970] | 1.226 [1.244] | 1.0 |
| | no_payoff_baseline | 1.113 [1.119] | 1.076 [1.068] | 1.244 [1.256] | 1.0 |
| dense_cluster_scaling | our_method | 0.987 [0.987] | 0.993 [0.984] | 1.241 [1.250] | 1.0 |
| | structure_free_baseline | 1.000 | 1.006 [0.997] | 1.257 [1.267] | 1.0 |
| | no_payoff_baseline | 1.083 [1.087] | 1.079 [1.072] | 1.256 [1.255] | 1.0 |
| sparse_knn_scaling | our_method | 0.711 [0.710] | 0.921 [0.903] | 1.070 [1.065] | 1.0 |
| | structure_free_baseline | 1.000 | 1.296 [1.272] | 1.490 [1.479] | 1.0 |
| | no_payoff_baseline | 0.901 [0.902] | 1.166 [1.146] | 1.280 [1.267] | 1.0 |

The paper's direction (lower exploitability than the budget-matched prior-work-style comparator) holds in all three experiments. In the dense families it holds only by about 1.1-1.3% at this scale, though well outside the test's noise: the paired 95% interval of the comparator-minus-reference ratio is [0.010, 0.013] (dense ER) and [0.012, 0.014] (dense clustered); see `HARDENING.md`. None of the exploitability metrics is near its bound. A protocol-compliant variant of the reference with stronger shrinkage towards uniform reaches comparator ratios of about 0.868 / 0.866 / 0.606 on the hidden test.
