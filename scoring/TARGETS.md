# Evaluation Targets

Every experiment is scored with the same four metrics (see `scoring/reference.json` and `briefing/evaluation.md`). Lower is better for the three exploitability metrics; a failed experiment scores 10 on them.

## All experiments (`dense_er_scaling`, `dense_cluster_scaling`, `sparse_knn_scaling`)

- **Primary metric**: `comparator_exploitability_ratio` -- exploitability relative to the fixed comparator at the same privacy budget (paper: 0.84 dense ER, 0.83 dense clustered, about 0.8 nearest-neighbour graphs, read from its figures at larger N).
- **Constraint metric**: `privacy_protocol_compliance` -- must be 1.0; any failing or missing instance fails the experiment.
- **Secondary metrics**: `relative_exploitability` (vs uniform play), `relative_final_nash_gap`.

Reference values (Algorithm 1 on this benchmark) are filled in below from `scoring/scores_test.json`.
