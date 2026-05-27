# Design Notes

## Checks Performed

- Compared `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`: train invokes `eval/train/` and writes `scoring/scores_train.json`; test invokes `eval/test/`, writes `scoring/scores_test.json`, and keeps the existing method-key match against train scores.
- Checked for cross-imports between `eval/train/` and `eval/test/`; none were present or added.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` do not exist.
- Ran `bash -n` on both train/test evaluation scripts.
- Reviewed the train/test evaluator logic for whether scores depend on saved method artifacts rather than importing `method/`; both slices read manifest, game, and trajectory artifacts.

## Changes Made

- Fixed a shortcut in both train and test evaluators: `privacy_budget_mean` and `privacy_decreases_with_n` now recompute the Renyi-DP bound from the saved game plus manifest hyperparameters using the benchmark's deterministic privacy-edge sampling. The score no longer trusts the method-authored `privacy_bound` manifest field.
- Added trajectory contract checks before scoring: trajectories must be finite arrays with shape `(*, n_players, n_actions)`.
- Added `strategy_simplex_violation_mean` to both train and test outputs. This penalizes invalid strategy trajectories that might otherwise game exploitability metrics by leaving the probability simplex.
- Updated `scoring/reference.json` so every nonzero metric has `higher_is_better`, `coefficient`, and `description`.
- Changed lower-is-better metric coefficients in `scoring/reference.json` to negative values to match the intended reward/penalty semantics.

## Runtime Concerns For Next Iteration

- The runner should verify that recomputed privacy values match the existing reproduction closely enough. The deterministic edge sampling mirrors the current method and baseline code, but this pass did not execute the evaluators.
- The privacy recomputation assumes output artifact names contain `_seedN`; otherwise it falls back to a hash-derived seed. If a future method changes artifact naming, scores remain deterministic but may not use the original privacy-edge sample.
- `strategy_simplex_violation_mean` is expected to be near zero for valid reproduced trajectories. The runner should check for saturation at exactly zero; this is acceptable as a constraint metric, but it should not dominate benefit metrics.
