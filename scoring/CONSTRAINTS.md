# Scientist Constraints

## Problem Instances
Solve every public instance of `data/polymatrix.benchmark_instances()` (3 graph families x N in {64, 128, 256, 512} x seeds 0-11) on the games produced by `data/polymatrix.benchmark_game`. The evaluators regenerate the games themselves; outputs computed on any other game are scored against the benchmark game anyway. A missing instance fails its whole experiment.

## Privacy Operating Point (fixed)
Use exactly T rounds, the per-instance noise stream `data/polymatrix.broadcast_noise(topology, N, seed)` (key 0) and the step scale eta of `data/polymatrix.benchmark_hyperparameters`. Each round, every player broadcasts the profile it is playing plus that noise; the next profile must satisfy the per-round payoff-dependence bound described in `briefing/evaluation.md`. Do not select among noise streams, reshape the noise, add rounds, or report a played profile different from the one whose noisy version was broadcast.

## Information Use
A player's update may use only its own utility matrices, the public graph, and the noisy messages it has received (current and past). Using other players' noiseless strategies, their utilities, or the known future noise is a privacy violation even when the transcript passes the checks; such submissions are disqualified by review.

## Output Contract
One `manifest.json` per method directory under `outputs/` and one `.npz` per instance with `trajectory` and `broadcasts` arrays of shape (T, N, 5), as written by `data/submission.py`. Scores are produced only by `scripts/evaluate*.sh`; numbers written by the method are ignored.

## Compliance Checklist
- [ ] Every benchmark instance is present, with exactly T rounds.
- [ ] Broadcasts are the played profile plus the fixed instance noise.
- [ ] Updates stay within the per-round payoff-dependence bound (the evaluator reports violations).
- [ ] Updates use only own utilities, the public graph and received noisy messages.
- [ ] Scores come from `bash scripts/evaluate.sh` (visible) and the hidden `scripts/evaluate_test.sh`.
