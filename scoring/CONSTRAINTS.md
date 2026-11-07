# Scientist Constraints

## What You Submit
A player policy: a module importable from the workspace root with a factory `make_player(view)` returning an object with `step(t, own_broadcast, received)`. It is registered by `outputs/<dir>/policy.json` (`data/submission.write_policy`; interface in `data/protocol.py` and `briefing/evaluation.md`). The evaluators run the protocol with it on their own instances; scores are produced only by `scripts/evaluate*.sh`. Numbers written by the method, or extra manifest fields, are ignored.

## Privacy Operating Point (fixed)
The evaluator fixes the rounds T, the noise level and the step scale eta, and it draws the broadcast noise. Each round, the next strategy must satisfy the per-round payoff-dependence bound described in `briefing/evaluation.md`. You cannot add rounds, change or select noise, or report a played strategy other than the one that was broadcast.

## Information Use
A player's step may use only its own view (own payoff matrices, the public graph, the operating point, its seed and the policy options) and the noisy messages it has received in the current and past rounds. The following are privacy violations:
- sharing state between players (module-level registries, files, class attributes);
- using other players' payoffs or noiseless strategies;
- using future noise;
- reading evaluator code, data or process internals (e.g. the call stack, `__main__`, files under `eval/`).

The evaluator re-executes randomly chosen players in isolation and fails the instance if their strategies are not reproduced. Code review disqualifies violations the re-execution cannot see.

## Determinism
A step must be a deterministic function of the view and the messages received so far. Use `view["seed"]` for any randomness.

## Compliance Checklist
- [ ] `step` returns a probability vector of length A, without raising, on every instance.
- [ ] Updates stay within the per-round payoff-dependence bound (the evaluator reports violations).
- [ ] Each player uses only its own view and received messages; no state is shared between players.
- [ ] Deterministic given the view and the messages (seeded with `view["seed"]`).
- [ ] Scores come from `bash scripts/evaluate.sh` (visible) and the hidden `scripts/evaluate_test.sh`.
