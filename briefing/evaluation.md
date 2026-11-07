# Evaluation

## What a submission provides

A **player policy**: a Python module, importable from the workspace root, with a factory `make_player(view)` that returns an object with a `step(t, own_broadcast, received)` method. It is registered by one `policy.json` in its own directory under `outputs/`:

```json
{"method": "our_method", "module": "method.my_policy", "factory": "make_player", "options": {}}
```

`data/submission.write_policy` writes this file. The manifest's `method` field names the scored entry; `our_method` is the entry ranked against the reference. `options` (any JSON object) is passed to every player. Any other field (claimed step sizes, privacy numbers, scores, ...) is ignored. Nothing is trained, and nothing else is read from a submission: the evaluators run the protocol themselves.

## The protocol the evaluators run

For every instance the evaluator builds each player once with its own local view, then runs T rounds. Before round 0 every player plays the uniform strategy. In round t (t = 0..T-1):

- each player's message is the strategy it is currently playing plus Gaussian noise of standard deviation sigma, drawn by the evaluator;
- `step(t, own_broadcast, received)` is called for every player with its own message of this round (shape (A,)) and its neighbours' messages (shape (d, A), in `view["neighbors"]` order). It returns the strategy the player plays next.

The view of player i (see `data/protocol.py`) contains `player`, `n_players`, `n_actions`, the public graph `edges`, its `neighbors`, its own payoff matrices `utilities` (shape (d, A, A); its payoff is the average over neighbours of pi_i^T U_ij pi_j), the operating point `rounds`, `eta`, `sigma`, a `seed` for any randomness, and `options`. A player never sees another player's payoffs, anybody's noiseless strategy, or noise of future rounds.

## Protocol checks (privacy constraint)

The evaluator regenerates each game, its operating point (T, eta, sigma) and its noise stream, recomputes every message from the returned strategies, and accepts an instance only if:

1. every call returned a finite vector of length A, for exactly T rounds;
2. every played strategy is a probability vector (tolerance 1e-6);
3. every update has bounded payoff dependence. Let P be the player's own projected message (projected onto the simplex), u the uniform strategy, and g the player's payoff gradient against its neighbours' projected messages, centred over actions. The new strategy must lie within eta * ||g|| of the segment from P to (P + eta * u) / (1 + eta);
4. **locality audit**: in every instance, randomly drawn players (8 per instance, drawn by the evaluator) are re-executed in a separate fresh process from their own view and the messages they received. That process holds nothing else, and evaluator modules cannot be imported there. Their strategies must be reproduced (tolerance 1e-7).

Check 3 bounds, per round, how much a player's next message can depend on its private payoffs beyond public information. With the noise fixed by the evaluator, this fixes the privacy budget of every compliant submission. Check 4 requires each player to use only its own information. A `step` must be a deterministic function of the view and the messages received so far (use `view["seed"]` for randomness). Players must not share state or read anything else (other players' data, evaluator internals or files).

A policy exception, a missing round, or any failed check fails the instance. Any failed instance fails the whole experiment (graph family) for that method.

## Metrics (per graph family; lower is better unless noted)

- **comparator_exploitability_ratio** (primary): the mean over the four player counts of the submission's summed CCE exploitability divided by the summed CCE exploitability of a fixed private comparator on the same instances and noise. The comparator plays noisy projected-gradient dynamics. Its step size is calibrated so that a privacy accounting that ignores the interaction graph meets the same per-instance budget. Its values are fixed in `eval/`.
- **relative_exploitability**: the mean over instances of CCE exploitability divided by the exploitability of the uniform profile on the same game (1 = no better than uniform play).
- **relative_final_nash_gap**: the mean over instances of the last profile's Nash gap divided by the uniform profile's Nash gap.
- **privacy_protocol_compliance** (higher is better): the fraction of required instances passing all protocol checks.

CCE exploitability of a run: the played profiles of the T rounds define an empirical distribution. Its exploitability is the average over players of the gain from switching to the best single fixed strategy against that distribution.

A failed experiment scores 10 on the three exploitability metrics (valid values are clipped to 10) plus its compliance fraction, so a failing submission ranks below every valid one.

## Ranking

A submission's rank is the sum over experiments of `weight x sum_metric coefficient x (value - reference value)`, using the weights and coefficients in `scoring/reference.json` (dense ER 0.4, dense clustered 0.3, sparse nearest-neighbour 0.3).

## Protocol

- `bash scripts/evaluate.sh` scores the visible train slice into `scoring/scores.json`; `scripts/evaluate_train.sh` writes the same slice to `scoring/scores_train.json`. The train slice is the public development instances with even seeds 0-10: games from `data/polymatrix.benchmark_game`, noise `data/polymatrix.broadcast_noise(..., key=0)`.
- The hidden test (`scripts/evaluate_test.sh`, no arguments) runs in a fresh container on 8 instances per (graph family, N). They are drawn from the same generator and noise model, but their seeds and noise exist only in `eval/test/`. Every policy runs in a child process; the scoring process never executes submitted code.
- `data/protocol.py` runs the same protocol on any public instance for development.
- `scoring/diagnostics*.json` holds per-player-count and per-instance numbers, failure reasons, locality-audit results and the operating point's analytic privacy budgets.
