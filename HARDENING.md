# Hardening Report

All rank gains are hidden-test gains over the reference, `sum_exp weight x sum_metric coefficient x (value - reference)`, computed with the `scoring/reference.json` that each evaluator shipped with. "Before" = the redteam's audit of the previous gym (`REDTEAM.md`, `redteam/results.json`); "after" = `python -m redteam.hardening.harness`, which builds each candidate from the public `data/` API (plus method/baseline code) and scores it with the unmodified, no-argument `bash scripts/evaluate_test.sh` in a fresh process (`redteam/hardening/cases/<case>/`, summary in `redteam/hardening/summary.json`). Before/after numbers are on different rank scales (the metric set changed), so compare signs and orderings, not magnitudes.

## What was wrong

The previous test trusted the method for everything on the other side of the comparison: the game files, the comparator outputs, the instance grid, and the step size `eta`, Renyi order `alpha` and step count feeding an analytic privacy formula whose value (about 6400) dominated the rank. Seven shortcuts beat the reference, by up to +7682. Privacy was a number the method could set rather than a constraint it had to meet. The comparator (`noisy_omd`) was Algorithm 1 with a different regularizer and tied the reference (+0.004, inside seed spread). Only 2 test seeds were used per size.

## What changed

1. **Everything on the other side is fixed by `eval/`.** `eval/train/evaluate.py` and `eval/test/evaluate.py` are self-contained copies (numpy only, no imports from `data/`, `method/` or each other). Each one regenerates the games, the operating point (T, eta, sigma) and a fixed per-instance broadcast-noise stream itself. Comparator values for its split are stored next to it (`comparator_{train,test}.json`, with game and noise fingerprints checked at score time). Only the instance identity and the `trajectory`/`broadcasts` arrays are read from a submission; every claimed number (`eta`, `alpha`, `privacy_bound`, `game_path`, ...) is ignored.
2. **Privacy is a verified constraint at a fixed operating point**, as in the paper's experiments ("both run under the same differential privacy budget"). A transcript must have exactly T rounds, valid probability vectors, and broadcasts equal to the played profile plus the fixed noise. Each update must lie within `eta x ||centred public payoff gradient||` of the segment from the player's projected noisy broadcast P to `(P + eta*u)/(1+eta)`. This bounds the payoff-dependent part of every message to O(eta) per round; the reference satisfies it by construction (non-expansive projection, regularizer tau_i <= 0.75 < 1 on this grid). Common noise also removes noise cherry-picking and noise shaping.
3. **Failure is failure.** A missing, duplicate, unreadable, non-finite, off-simplex or protocol-violating instance fails its whole experiment. Failure scores 10 on every exploitability metric (valid values are clipped to 10) plus the compliance fraction, so a failing submission ranks far below every valid one. Failing even the lowest-weight experiment on one instance costs at least -4.8. Driving every exploitability metric of every experiment to its bound of 0 would gain only +1.69 in total, so no failure can be compensated.
4. **Scale-free metrics with headroom.** Primary: `comparator_exploitability_ratio` (exploitability / fixed comparator's at the same budget, the paper's headline comparison). Secondary: `relative_exploitability` (/ uniform play) and `relative_final_nash_gap`. Constraint: `privacy_protocol_compliance`. The old 0/1 and trend scores (constant across methods, or saturated at about 1.00) moved to diagnostics.
5. **Comparator = the paper's baseline mechanism.** A prior-work-style private learner (Huang et al. adaptation) cannot exploit graph structure in its privacy accounting. `structure_free_baseline` uses the same noise and rounds with plain projected-gradient steps, calibrated so that a structure-free accounting (all N players counted in the locality term, i.e. 2A) equals the per-instance budget of the paper's accounting min{dense term, locality term}: `gamma = eta*sqrt(min(club, spade_max)/(2A))`.
6. **Bigger test, fixed randomness.** 6 test seeds per (family, N), up from 2, i.e. 72 test instances. The noise is common to all methods, so comparisons are paired.

## Does a better result now score higher?

| Variant | Before (old test) | After (hidden test) |
|---|---:|---:|
| Reference | 0 | 0 |
| Reference, alternative noise streams 1/2/3 (seed noise) | +0.004 / -0.008 (algorithm-RNG reruns) | +0.037 / +0.011 / +0.012 |
| better_shrink_075 (compliant: anchor 0.75*P + 0.25*u, plain gradient step) | -- | **+0.226** |
| better_max_shrink (compliant: regularizer weight 1 for every player) | -- | **+0.137** |
| better_denoised_gradient (compliant: running-mean gradient direction) | -- | -0.130 (a worse idea, scored lower) |
| weaker_half_step | -- | **-0.286** |
| weaker_no_gradient | -0.496 | **-0.612** |
| comparator submitted as method | +0.004 (tie) | **-0.189** |
| payoff-blind protocol baseline | -0.111 (constant uniform) | -0.217 |
| redteam better_fixed_transcript_readout (separate output state) | +0.075 | fails: -16.81 |
| redteam better_persistent_anchor | +0.109 | fails: -16.81 |
| redteam better_final_average | -0.002 | fails: -16.81 (T+1 rounds) |

Genuine compliant improvements now score above the reference, by 4-6x the largest noise-stream shift. Weaker variants and both baselines score below it. The ordering follows the exploitability each variant actually reaches: comparator ratios 0.865 / 0.862 / 0.611 for the best variant, against 0.987 / 0.986 / 0.715 for the reference. The two redteam "better" variants now fail on purpose:
- The separate readout reports played strategies that are not the ones whose noisy versions were broadcast. Allowing that would also let a central solver's output be reported next to a valid transcript.
- The persistent anchor starts its step from the private noiseless strategy, which the paper itself rules out for privacy (Section 5: the step must start from the noisy broadcast so that adjacent games' observations do not drift apart).

**Effect vs seed noise (paired).** The reference and the comparator were rerun on three further noise streams. Each pair shares a stream; the evaluator was run with `--noise-key`, and every run is compared with the fixed key-0 comparator values.

| Noise stream | Rank(reference) - rank(comparator) | Reference / comparator: dense ER | dense cluster | sparse kNN |
|---|---:|---:|---:|---:|
| 0 (benchmark) | 0.1886 | 0.9869 | 0.9862 | 0.7145 |
| 1 | 0.1879 | 0.9885 | 0.9869 | 0.7109 |
| 2 | 0.1882 | 0.9873 | 0.9863 | 0.7161 |
| 3 | 0.1850 | 0.9880 | 0.9859 | 0.7126 |

The paper's effect (ours vs the prior-work baseline at equal privacy) is 0.185-0.189 in rank, against a largest unpaired shift of the reference's own score of 0.037 across streams. Even the small dense-graph effect (1.2-1.4%) varies by only about 0.0016 across streams, roughly 8x below its size. The benchmark fixes the noise stream, so all hidden-test comparisons are paired in exactly this way.

## Major shortcuts: fix and score before/after

| Shortcut | Before | Fix | After |
|---|---:|---|---:|
| Claim eta = 0 | +3840.92 | eta fixed by eval; manifest numbers ignored | 0 (identical to reference) |
| Claim alpha = -2 | +7681.86 | alpha fixed by eval | 0 |
| Fabricated privacy_bound | 0 (tie) | ignored | 0 |
| Replace the game file with zero utilities | +0.481 | eval regenerates and fingerprints the games | 0 |
| Degrade the comparator outputs on disk | +0.286 | comparator values fixed in `eval/test/comparator_test.json` | 0 (the fake baseline only hurts itself) |
| Drop the n=64 games | +1042.42 | required instance grid fixed by eval; missing instance fails | -16.06 |
| Public games, no noise | +0.147 | broadcasts must equal played + fixed noise | -16.81 |
| All-zero strategies (+ eta = 0) | -0.019 (+3840.90) | simplex gate, failure values | -16.81 (-16.81) |
| Constant uniform, one round, no broadcasts | -0.111 | exact rounds and both arrays required | -16.81 |
| NaN strategies / empty trajectory | evaluator crash, no score | scored as failed instead of crashing | -16.81 / -16.81 |
| New: noiseless dynamics + fabricated noise afterwards | -- | step allowance from the public anchor | -16.81 |
| New: 3x step size | -- | step allowance | -16.81 |
| New: best of 8 noise streams per instance | -- | fixed noise stream | -16.76 |
| New: T-1 rounds | -- | exact rounds | -16.81 |
| New: gradient from neighbours' true strategies (unscaled) | -- | step allowance (incidentally) | -16.81 |

The "0" rows are ties because the forged field no longer does anything: the submission is the reference itself.

## Shortcuts left open

Neither shortcut below can be seen in a transcript: each is a valid execution of the protocol, and only a code review can reject it. Both were probed through the hidden test; neither pays.

| Left shortcut | What it does | Hidden-test rank gain | Why it is left |
|---|---|---:|---|
| `shortcut_known_noise_gradient_rescaled` | Uses neighbours' true (pre-noise) strategies, i.e. information leaked by knowing the public noise stream, for the gradient direction, rescaled to stay within the step allowance. | **-0.204** (worse than the reference: ratios 1.071 / 1.062 / 0.898) | Information misuse inside the allowance is invisible in the transcript. It does not pay here: the noisy broadcasts predict where neighbours will actually be next round better than their true strategies do. |
| `shortcut_select_best_variant` | Per instance, keeps the best of five valid runs by exploitability computed with all utilities (a centralised selection). | +0.226, identical to the best single compliant variant (`better_shrink_075`), which it selected on every instance | Selecting runs with global knowledge is a centralised computation. The simulation necessarily hands a method every player's utilities, so only review can rule it out. With these variants it gains nothing beyond the best honest variant. Selection over a much larger pool of compliant runs could gain a little more; `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md` forbid it. |

Further members of the same class, not probed and closable only by review:
- choosing the next strategy using the known *future* noise of one's own broadcast;
- using other players' utilities to choose a step direction within the allowance;
- tuning per-instance hyperparameters on the public instances, whose seeds include the test seeds. The 12 public seeds include the 6 test seeds because a method must produce outputs for them, and the test never reruns a method.

The step allowance caps how much any of these can move a strategy per round to the same O(eta) budget an honest update has. The two probed versions confirm the effect is nil or negative here.

## Reference before and after

- **Before:** test CCE 0.0659 / 0.0634 / 0.1268 (ER / cluster / kNN, 2 seeds). Ratio to `noisy_omd` 0.988 / 1.020 / 1.013. Privacy "budget" about 6400 (method-claimed formula).
- **After (hidden test, 6 seeds, fixed noise):** comparator ratio 0.987 / 0.986 / 0.715; relative exploitability 0.979 / 0.996 / 0.910; relative final Nash gap 1.249 / 1.250 / 1.068; compliance 1.0 everywhere.
- **Visible slice:** 0.988 / 0.987 / 0.710; relative exploitability 0.958 / 0.984 / 0.903. Train and test agree to within about 0.02.
- The reference was regenerated from the final code by `scripts/reproduce.sh`. `scoring/scores_{train,test}.json` contain only the reference package: `our_method` and the two baselines.

## Paper direction and effect size

- **Ours vs prior-work baseline at equal privacy (primary).** Reproduced in the paper's direction in all three families: ratio < 1 on both slices.
  - Sparse nearest-neighbour graphs: the gap is large (0.71). The paper's locality accounting lets Algorithm 1 take steps 2.5-4x larger than a structure-free learner at the same budget.
  - Dense graphs: only about 1.3%, though with the same sign on train and test and in both dense families. Here the paper's dense accounting term (16A^3 (log N)^2 / hdeg^(4/9), about 14000) is far above the locality term (2A = 10) at N <= 512, so its privacy advantage is vacuous at this scale. The budget-matched baseline gets the same step, and only the degree-adaptive regularizer separates the two.
  - The paper's ~0.84 dense ratio (at N up to 4096 with unpublished baseline constants) is **not** reproduced in magnitude.
- **Exploitability falls with N in dense graphs (paper Fig. 5).** Reproduced in absolute terms: log-log slope -0.40 (ER) and -0.36 (cluster), roughly flat in kNN (-0.04, matching the paper's "unchanged" sparse result). Two caveats:
  - The comparator falls equally fast, so the paper's *rising* baseline is not reproduced.
  - Relative to uniform play, Algorithm 1 gets worse with N at this operating point (0.83 at N=64 to 1.10 at N=512, ER). This is a limitation of the method at this scale and is where the headroom is.
- **Privacy budget falls with N.** True for the operating point's analytic budget with the regime branch the earlier iterations used (slope -0.09 dense ER, -0.06 cluster, -4.7 kNN, in `scoring/diagnostics_*.json`). With the literal min{club, spade} bound the dense budget is flat (320) at this scale. The budget is a property of the fixed operating point, identical for every compliant method, so it is a diagnostic, not a scored metric.

## Headroom check

- **Reference at a bound:** only `privacy_protocol_compliance` (1.0, a constraint).
- **Reference far from bounds:** primary 0.987 / 0.986 / 0.715 (bound 0); relative exploitability 0.979 / 0.996 / 0.910; final-gap 1.25 / 1.25 / 1.07.
- **Demonstrated room:** a compliant variant reaches 0.865 / 0.862 / 0.611 (+0.226).
- **Baselines:** the payoff-blind baseline scores clearly below the reference everywhere (1.10 / 1.08 / 0.90).
- **No experiment added:** `sparse_knn_scaling` already covers where the paper's method falls short (the paper: exploitability does not improve on sparse graphs), and the dense experiments expose the method's degradation relative to uniform play as N grows.

## Weights and coefficients

- Experiment weights unchanged: 0.4 / 0.3 / 0.3. Dense ER is the paper's main figure; the other two are appendix settings.
- Coefficients:
  - -1.0 on the primary comparator ratio (the paper's headline claim);
  - -0.5 on exploitability relative to uniform (the same quantity on an absolute scale);
  - -0.25 on the final Nash gap (the paper's guarantee is for the CCE, not the last iterate);
  - +1.0 on compliance.

## Not fixed / limitations, stated plainly

- The dense-graph gap to the prior-work baseline is small (~1.3%) at the feasible scale, and the paper's magnitude (0.84) is not reproduced. The exact baseline of the paper (constants of its Huang et al. adaptation) is unpublished; ours is a principled budget-matched stand-in, not a replica.
- The step allowance is a structural privacy certificate (bounded payoff-dependent movement per round at fixed noise), not a proof of a specific Renyi epsilon for arbitrary update rules. Information misuse that stays inside it cannot be seen in the transcript (see "Shortcuts left open").
- The allowance rejects some trivially private behaviours, such as jumping straight back to uniform play every round. That is the price of not letting a player choose large jumps based on its private payoffs. Uniform play is therefore only the normalizer, not a valid submission.
- `redteam/audit.py` / `redteam/methods.py` target the removed pre-hardening interface and no longer run; their evidence is kept. The hardening harness supersedes them.
