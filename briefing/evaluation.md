# Evaluation

## What a submission provides

For every instance (topology, N, seed) a `.npz` file with two float arrays of shape (T, N, 5):

- `trajectory[t]` -- the profile played after round t (t = 0..T-1);
- `broadcasts[t]` -- the messages sent in round t: the profile played before round t (uniform for t = 0, `trajectory[t-1]` afterwards) plus `broadcast_noise(topology, N, seed)[t]`.

A `manifest.json` per method directory under `outputs/` lists the runs (`method`, `topology`, `n_players`, `seed`, `trajectory_path`); `data/submission.py` writes this format. The manifest's `method` field names the scored entry (`our_method` is the entry ranked against the reference). Any other manifest field (claimed step sizes, privacy numbers, game files, ...) is ignored.

## Protocol checks (privacy constraint)

The evaluator regenerates each game, its operating point (T, eta, sigma) and its noise stream itself, and accepts a transcript only if:

1. both arrays have exactly shape (T, N, 5) and are finite;
2. every played strategy is a probability vector (tolerance 1e-6);
3. every broadcast equals the previously played profile plus the instance noise (tolerance 1e-8);
4. every update has bounded payoff dependence: with P = the player's own broadcast projected onto the simplex, u = uniform, and g = the player's payoff gradient against all players' projected broadcasts (centred over actions), the new strategy must lie within eta * ||g|| of the segment from P to (P + eta * u) / (1 + eta).

Check 4 bounds, per round, how much a player's next message can depend on its private payoffs beyond public information; with the fixed noise this fixes the privacy budget of every compliant submission. Any failing or missing instance fails the whole experiment (graph family) for that method.

## Metrics (per graph family; lower is better unless noted)

- **comparator_exploitability_ratio** (primary): mean over the four player counts of the submission's summed CCE exploitability divided by the summed CCE exploitability of a fixed private comparator on the same instances and noise. The comparator plays noisy projected-gradient dynamics whose step size is calibrated so that a privacy accounting that ignores the interaction graph meets the same per-instance budget; its values are fixed in `eval/`.
- **relative_exploitability**: mean over instances of CCE exploitability divided by the exploitability of the uniform profile on the same game (1 = no better than uniform play).
- **relative_final_nash_gap**: mean over instances of the last profile's Nash gap divided by the uniform profile's Nash gap.
- **privacy_protocol_compliance** (higher is better): fraction of required instances passing the protocol checks.

A failed experiment scores 10 on the three exploitability metrics (valid values are clipped to 10) and its compliance fraction, so a failing submission ranks below every valid one.

## Ranking

The rank of a submission is the sum over experiments of `weight x sum_metric coefficient x (value - reference value)` with the weights and coefficients in `scoring/reference.json` (dense ER 0.4, dense clustered 0.3, sparse nearest-neighbour 0.3).

## Protocol

- `bash scripts/evaluate.sh` scores the visible slice (even seeds) into `scoring/scores.json`; `scripts/evaluate_train.sh` writes the same slice to `scoring/scores_train.json`.
- The hidden test (`scripts/evaluate_test.sh`, no arguments) scores the odd seeds in a fresh container with its own copy of the games, noise and comparator values. It never reruns a method.
- Per-player-count numbers, failure reasons, and the operating point's analytic privacy budgets are written to `scoring/diagnostics*.json`.
