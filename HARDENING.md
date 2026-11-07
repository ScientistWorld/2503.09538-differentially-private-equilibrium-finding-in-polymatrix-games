# Hardening Report (final iteration)

Rank gain = hidden-test gain over the reference, `sum_exp weight x sum_metric coefficient x (value - reference)`, with the weights and coefficients of `scoring/reference.json`.

- **Before** = the redteam's audit of the previous gym (`REDTEAM.md`, commit `76a0754`).
- **After** = `python -m redteam.hardening_v3.harness`. Each case registers only its own policy (`outputs/candidate/policy.json`, method `our_method`). It is then scored by the unmodified, no-argument `bash scripts/evaluate_test.sh` and `bash scripts/evaluate_train.sh` in fresh processes. Per-case scores, diagnostics and `result.json` are in `redteam/hardening_v3/cases/`; the summary is `redteam/hardening_v3/summary.json`.
- Every case sees the same hidden instances and noise, so the "SE" column is a paired bootstrap over instances (resampled within each player count, 2000 draws).

## What was wrong

The redteam found one major shortcut. The test games and their broadcast noise could be regenerated from the public `data/` generator, and the evaluator only saw transcripts. A method could therefore centralise the computation:
- read every player's payoffs, the noiseless strategies and the future noise;
- run a joint Nash-gap optimiser;
- clip each update into the per-round step allowance.

It passed every check with compliance 1.0 and scored **+0.543**, more than twice the best honest variant (+0.226). Its oracle-selection variant used the same mechanism. Transcript checks alone cannot establish that a player used only its own information.

## What changed

1. **A submission is a per-player policy, not a transcript.** It is registered by `outputs/<dir>/policy.json`, naming a module with `make_player(view)` and a `step(t, own_broadcast, received)` method (`data/protocol.py`, `briefing/evaluation.md`).
   - A player's view is its own payoff matrices, the public graph, the operating point, a seed and the policy options.
   - Each round it receives only its own noisy message and its neighbours' noisy messages.
   - Algorithm 1 and both baselines were rewritten as policies. They reproduce the previous vectorised dynamics to 2e-16 (smoke test in `scripts/run.sh`).
2. **Hidden test instances and noise.** The test games, their noise, the per-player seeds and the audit draw come from salts that exist only in `eval/test/evaluate.py`. The test now uses 8 instances per (family, N), 96 in total (up from 72).
3. **The evaluator runs the protocol itself.**
   - Each policy runs in a child process.
   - The scoring process never executes submitted code. It regenerates every game and noise stream, recomputes every broadcast from the returned strategies, and applies the existing checks: exact rounds, simplex, and the per-round step allowance.
4. **Locality audit.** In every instance, 8 randomly drawn players are re-executed from their own view and received messages, which the scoring process computes itself. Their strategies must be reproduced to 1e-7.
   - The re-execution runs in a fresh process, `eval/<slice>/replay.py`. That module holds no generator, noise or secret, and it blocks imports of the evaluator package before loading the policy.
   - The protocol run's files are deleted before the replay starts.
   - A non-reproduced player fails its instance, and a failed instance fails the experiment.
5. **Unchanged:**
   - metrics, coefficients and weights;
   - the comparator's definition and per-instance budget calibration (its values for the new test instances are in `eval/test/comparator_test.json`);
   - the operating point and the failure semantics;
   - the train slice: the public even seeds with key-0 noise. The reference's train scores through the new interface equal the previous transcript evaluator's to within 3.3e-16.

## Does a better result score higher?

Yes. Every honest variant keeps its place in the order, with gaps of 30-80x the paired SE.

| Variant | Before (old test) | After: hidden test (paired SE) | After: train |
|---|---:|---:|---:|
| better_shrink_075 (anchor 0.75*P + 0.25*u, plain step) | +0.2260 | **+0.2233** (0.0027) | +0.2201 |
| better_max_regularization (regularizer weight 1) | +0.1372 | **+0.1303** (0.0040) | +0.1265 |
| reference (Algorithm 1) | 0 | 0 | 0 |
| comparator submitted as method | -0.1886 | **-0.1895** (0.0018) | -0.1892 |
| payoff-blind protocol baseline | -0.2167 | **-0.2266** (0.0061) | -0.2299 |
| weaker_half_step | -0.286 (previous hardening) | **-0.2899** (0.0025) | -0.2887 |
| weaker_no_gradient | -0.6125 | **-0.6248** (0.0064) | -0.6215 |

- **Comparator ratios (ER / clustered / kNN).** The best variant reaches 0.868 / 0.866 / 0.606, against 0.989 / 0.987 / 0.711 for the reference.
- **Train and test agree.** Every honest case's train and test gains agree to within 0.01.
- **Seed noise.** The methods have no randomness besides the evaluator's fixed noise. Reruns are bit-identical: the locality audit re-executed 768 reference players per test run with a maximum deviation of exactly 0.0. The remaining noise is instance sampling, which is what the paired SE measures.
- **The paper's effect against noise.** The paired 95% interval of (comparator - reference) primary ratio is:
  - dense ER: [0.0099, 0.0125];
  - dense clustered: [0.0115, 0.0138];
  - sparse kNN: [0.284, 0.294].

  Even the small dense effect is about 16x its SE.

## Major shortcuts: fix and score before/after

| Shortcut (case) | Before | Fix | After: hidden test | After: train |
|---|---:|---|---:|---:|
| Global private optimizer; oracle selection (`shortcut_public_instance_oracle`: the redteam's optimizer, ported: identifies the instance from the public generator, rebuilds payoffs + noise) | **+0.5434** | Hidden instances and noise; per-player views | **0.0000**: no hidden instance is public, so it falls back to Algorithm 1 | +0.5286: works on the public train slice, gains nothing on test |
| Players pool payoffs and noiseless strategies in a module registry (`shortcut_global_registry`) | not possible before (no players) | Locality audit | **-16.82**: fails the audit in every family (its transcripts alone would have scored +0.245) | -16.82 |
| Read the runner's full game and future noise from the call stack (`shortcut_frame_oracle`) | n/a | Locality audit | **-16.82**: fails the audit (its transcripts alone would have scored +0.534) | -16.82 |
| Regenerate the hidden instance through the evaluator module it runs in (`shortcut_hidden_evaluator_oracle`) | **+0.5341** in this iteration's first run, when the replay ran inside the evaluator module (`summary_run1_before_replay_isolation.json`) | Isolated replay module without generator, imports of `eval` blocked | {{HIDDEN}} | {{HIDDEN_TRAIN}} |
| Monkeypatch the evaluator's checks/metrics and write forged score files (`shortcut_scorer_tamper`, payoff-blind dynamics) | n/a | Scoring process never runs submitted code and writes the scores itself | -0.2266: exactly the payoff-blind baseline | -0.2299 |
| Patch the runner to deliver noiseless messages (`shortcut_runner_noise_patch`) | -16.81 (remove noise) | Scorer recomputes broadcasts with its own noise; step allowance | **-16.81**: fails the step allowance | -16.81 |
| Forged manifest claims: perfect scores, eta = 0, 1 round, huge sigma (`shortcut_forged_policy_claims`, payoff-blind dynamics) | 0 (tie) | Every field except module/factory/options is ignored | -0.2266: exactly its dynamics | -0.2299 |
| Move the comparison baseline; replace the game artifact | 0 (tie) | Comparator values fixed in `eval/`; games regenerated by the evaluator | n/a: nothing of either is read from a submission | |
| Reduce the test size (`shortcut_crash_small_games`: raises on N=64) | -16.06 | Evaluator enumerates instances; missing = failed | -16.07 | -16.07 |
| NaN / uniform outputs (`shortcut_nonfinite`, `shortcut_uniform`) | -16.81 | Per-round finiteness, simplex and allowance checks | -16.82 / -16.82 | -16.82 |

## Shortcuts left open

| Left shortcut | Hidden-test gain | Why it is left |
|---|---:|---|
| `left_shortcut_cache_evasion`: computes the oracle during the protocol run by reaching into the evaluator process, and writes its strategies to a disk cache that the isolated replay reads back | {{CACHE}} | It needs code written to defeat the audit: (a) access to evaluator internals (its module, call stack, or files under `eval/`) during the protocol run, and (b) a cross-process disk cache keyed to the replayed player. Closing it needs OS-level isolation: no shared writable filesystem between the run and the replay, and no access to `eval/` at test time. `scoring/CONSTRAINTS.md` forbids both steps; the separate grader must reject any submission that touches `eval/`, `__main__`, the call stack (`sys._getframe`, `inspect`, `gc`), or shares state through files. |
| Loading `eval/test/evaluate.py` by file path inside the replay (`importlib.util.spec_from_file_location` bypasses the import block) | not probed; same class as above | Same reasons. The scientist never sees `eval/test/`, so this means reading undisclosed evaluator code at test time. |
| Information shared among the 8 audited players of one instance inside the replay process | negligible | They are 8 random players out of 64-512, and the protocol run needs all players' information. |
| Anything on the public train slice | e.g. +0.5286 for the public-instance oracle | The train games and noise are public by design. The hidden test does not reward train-tuned shortcuts (that oracle scores 0 on test). |

The audit is a spot check: 8 players per instance, 768 per test run. A policy that broke locality for a random fraction f of players would escape with probability (1-f)^768 and gain at most about f x 0.5.

## Reference score before and after

- **Before** (old hidden test, 6 public odd seeds per size):
  - comparator ratio 0.9869 / 0.9862 / 0.7145;
  - relative exploitability 0.979 / 0.996 / 0.910;
  - relative final Nash gap 1.249 / 1.250 / 1.068.
- **After** (8 hidden instances per size):
  - comparator ratio 0.9889 / 0.9873 / 0.7108;
  - relative exploitability 0.969 / 0.993 / 0.921;
  - relative final Nash gap 1.213 / 1.241 / 1.070;
  - compliance 1.0 everywhere, with all 768 audited players reproduced exactly.
- **Train** (unchanged): 0.988 / 0.987 / 0.710.
- **Source of the score files.** `scoring/scores_{train,test}.json` and `scoring/scores.json` were regenerated by `scripts/reproduce.sh` from the final code. They contain only the reference package: `our_method` and the two baselines.

## Headroom check

- **Reference at a bound:** only `privacy_protocol_compliance` (1.0, a constraint).
- **Reference far from bounds** (ER / clustered / kNN):
  - primary 0.989 / 0.987 / 0.711 (bound 0);
  - relative exploitability 0.969 / 0.993 / 0.921 (bound 0);
  - final-gap 1.21 / 1.24 / 1.07 (bound 0).
- **Demonstrated room:** a compliant local variant scores +0.223, and the forbidden centralised optimiser shows how much exploitability information alone could buy (its transcripts reach 0.758 / 0.757 / 0.341).
- **Baselines:** the payoff-blind baseline scores clearly below the reference (-0.227), as does the budget-matched comparator (-0.189).
- **No experiment added.** `sparse_knn_scaling` is already where the paper's method falls short: the paper reports no exploitability gain on sparse graphs, and here Algorithm 1 sits at 0.92 of uniform play. In the dense families its exploitability is close to uniform play's (0.97-0.99).

## Paper direction and effect size

- **Ours vs the prior-work baseline at equal privacy (primary).** Reproduced in the paper's direction in all three families on both slices, and outside the test's noise.
  - Sparse nearest-neighbour graphs: a large gap (0.711).
  - Dense graphs: a small gap (1.1-1.3%). The paper's dense accounting term is vacuous at N <= 512, so the budget-matched baseline gets the same step size, and only the degree-adaptive regularizer separates the two.
  - The paper's ~0.84 dense ratio, at N up to 4096 and with unpublished baseline constants, is not reproduced in magnitude.
- **Exploitability falls with N in dense graphs.** It falls in absolute terms, but the comparator falls equally fast (see `scoring/diagnostics_test.json`, `cce_slope_vs_n`).

## Weights and coefficients

Unchanged: 0.4 / 0.3 / 0.3 (dense ER is the paper's main figure); -1 / -0.5 / -0.25 on the three exploitability metrics, +1 on compliance. Nothing found in this iteration argued for a different weighting.

## Not fixed, stated plainly

- The left shortcuts above. A determined submission that reaches into the evaluator process and fakes the replay through the filesystem would pay {{CACHE}} on the hidden test. Only code review, or OS-level sandboxing of the policy processes, can stop it.
- The dense-graph gap to the prior-work baseline is small (~1.2%), and the paper's magnitude is not reproduced. Our baseline is a principled budget-matched stand-in for the paper's unpublished Huang et al. adaptation.
- The step allowance is a structural privacy certificate (bounded payoff-dependent movement per round at fixed noise), not a proof of a specific Renyi epsilon for arbitrary update rules.
- Running a policy through the hidden test takes about a minute per policy on one CPU. The hidden test now executes submitted code. This is allowed because the method needs no training, but it is what makes the left shortcut class possible at all.
- `redteam/audit_v2*`, `redteam/audit.py` and `redteam/hardening/` target the previous transcript interface and no longer run; their evidence is kept.
