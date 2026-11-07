**Genuine improvements score higher in the matched controls, but the reference does not consistently beat its comparator.** A local output improvement gains **+0.075492** and removing payoff gradients loses **−0.495531**, both beyond the **0.012086** reference seed spread. The overall ranking is nevertheless unsuitable: seven shortcuts beat the reference, one shortcut ties it, and an invalid method can win by thousands of points.

All scores below are **test rank gains over the reference, whose gain is 0**. I used the specified sum of experiment weight × signed metric coefficient × difference from `scoring/scores_test.json`, across all 24 scored metric instances. Experiment weights are 0.4 ER, 0.3 clustered, and 0.3 kNN. CCE, Nash gap, and privacy columns are experiment-weighted means for readability; the rank also includes trends, baseline ratios, competitiveness, and simplex violations. Full scores and every rank contribution are saved per run and in [results.json](redteam/results.json).

The scientific controls use the same generated games, sizes, noise scales, communication counts, and comparator. The primary improvement keeps the original communication protocol and broadcasts exactly unchanged. Each player maintains an additional local output state, updated from its own utility matrices and already received noisy neighbor messages; this state never feeds back into communications. It improves equilibrium quality without changing anything seen by the channel observer in the stated privacy setting.

| Variant (each link runs the method and hidden test) | Test CCE ↓ | Test final Nash gap ↓ | Test privacy bound ↓ | Rank gain |
|---|---:|---:|---:|---:|
| [Reference, seed 0](redteam/reference_seed0/run.sh) | 0.083415 | 0.103598 | 6401.558784 | 0.000000 |
| [Better: separate local output state](redteam/better_fixed_transcript_readout/run.sh) | 0.073229 | 0.072361 | 6401.558784 | **+0.075492** |
| [Same output change, learning rate 2](redteam/better_fixed_transcript_stronger_readout/run.sh) | 0.078181 | 0.083958 | 6401.558784 | +0.049400 |
| [Persistent strategy anchor](redteam/better_persistent_anchor/run.sh) | 0.067357 | 0.061854 | 6401.558784 | +0.109339 |
| [Append temporal average](redteam/better_final_average/run.sh) | 0.084821 | 0.090760 | 6401.558784 | −0.002382 |
| [Weaker: remove payoff gradients](redteam/weaker_no_gradient/run.sh) | 0.121908 | 0.147454 | 6401.558784 | **−0.495531** |
| [Comparator submitted as method](redteam/comparator_as_method/run.sh) | 0.082899 | 0.098591 | 6401.558784 | +0.003892 |

The persistent-anchor candidate improves quality, but changes the communication dynamics; its unchanged *reported* privacy bound alone does not prove the modified protocol's privacy guarantee. It is exploratory evidence, not the primary scientific control. Appending an average improves the final Nash gap but slightly worsens CCE and aggregate rank; this change is within seed noise and is not a confirmed improvement. The `better_*` folder names record the hypotheses tested.

Three reference reruns vary the algorithm RNG while holding the test games and comparator fixed. The stream for game seed `s` is `2000 + s + 10000 × replicate`.

| Replicate | Test rank gain | Test CCE |
|---|---:|---:|
| [0](redteam/reference_seed0/scores_test.json) | 0.000000000 | 0.083414605 |
| [1](redteam/reference_seed1/scores_test.json) | +0.003886488 | 0.082328722 |
| [2](redteam/reference_seed2/scores_test.json) | −0.008199706 | 0.085058981 |

The max-minus-min rank spread is **0.012086194101**; population standard deviation is 0.005037813161. A gain below that observed spread is noise for this audit. This measures algorithm randomness on the fixed test games, not uncertainty across newly sampled game populations, and three seeds do not establish statistical significance. Replicate 0 reproduces every stored reference test metric exactly.

The reference/comparator CCE ratios are **0.988163 ER**, **1.020007 clustered**, and **1.013368 kNN**. Thus the reference is slightly better on ER and worse on the other two families. Submitting the comparator itself scores +0.003892, within reference seed spread: the gym cannot reliably distinguish these honest alternatives. The documented “within 10%” competitiveness target passes on all three families; strict comparator superiority is not reproduced. No paper was read, so this check establishes the direction supported by the supplied briefing and comparator, not any unprovided paper-specific numerical margin.

The reference can be beaten by a genuine method; it is not pinned at the overall score ceiling. Every exactly saturated favorable metric instance is listed here, together with the simplex metric already at its numerical floor:

| Experiment | `competitive_with_baseline` (upper bound 1) | `strategy_simplex_violation_mean` (lower bound 0) |
|---|---:|---:|
| Dense ER | **1, saturated** | 4.788785e−17, numerical floor |
| Dense clustered | **1, saturated** | 4.903304e−17, numerical floor |
| Sparse kNN | **1, saturated** | 4.531965e−17, numerical floor |

There are no other saturated reference metric instances. Both trend scores have range [0, 2]; reference CCE trends are 1.000513–1.003631 and privacy trends 1.000520–1.048792. CCE (0.063354–0.126841), final Nash gap (0.083473–0.149045), baseline ratio (0.988163–1.020007), and valid privacy bounds (5482.412350–6937.685412) all have room toward their favorable lower bound of zero. Invalid inputs defeat that last bound in the implementation. The simple constant-uniform shortcut scores **−0.111140**, more than nine seed spreads below reference, with CCE 0.090253. The task therefore distinguishes this simple baseline, although its designated comparator essentially ties the reference.

Each shortcut below has its own `method.py`, `run.sh`, and `result.json` under `redteam/<name>/`; successful evaluations also retain `scores_test.json` and diagnostics. Links in the first column run the shortcut; the rank links open its evidence. A tie counts as a paying shortcut as requested.

| Shortcut | What it does | Test rank gain vs reference | Works or fails |
|---|---|---:|---|
| [Zero claimed learning rate](redteam/shortcut_eta_zero/run.sh) | Keep reference trajectories; change manifest `eta` to 0. Recomputed privacy becomes 0. | [+3840.920138](redteam/shortcut_eta_zero/scores_test.json) | **Works** |
| [Negative claimed Rényi order](redteam/shortcut_negative_alpha/run.sh) | Keep reference trajectories; set `alpha=-2`. Accepted privacy becomes negative. | [+7681.855408](redteam/shortcut_negative_alpha/scores_test.json) | **Works** |
| [Fabricated raw bound](redteam/shortcut_reported_bound/run.sh) | Set manifest `privacy_bound=-1e30`; the evaluator ignores that field and recomputes it. | [0.000000, exact tie](redteam/shortcut_reported_bound/scores_test.json) | **Works as tie**; forgery adds no gain |
| [Replace fixed game artifact](redteam/shortcut_zero_game/run.sh) | Zero the proposed game's utilities while preserving graph and reference strategies; leave comparator games intact. Evaluated CCE and Nash gap become 0. | [+0.480634](redteam/shortcut_zero_game/scores_test.json) | **Works** |
| [Degrade fixed comparator](redteam/shortcut_degrade_baseline/run.sh) | Leave proposed outputs unchanged; replace baseline outputs with deliberate poor play. Weighted baseline ratio falls from 1.005278 to 0.291285. | [+0.285597](redteam/shortcut_degrade_baseline/scores_test.json) | **Works** |
| [Drop small test games](redteam/shortcut_drop_small_games/run.sh) | Remove n=64 from both manifests. Evaluator accepts a smaller grid; kNN privacy mean falls from 5482.412350 to 0.535296. | [+1042.418260](redteam/shortcut_drop_small_games/scores_test.json) | **Works** |
| [Public games without privacy noise](redteam/shortcut_public_game_no_noise/run.sh) | Rebuild games from visible generator/seeds and run the same updates with `sigma=0`, reported honestly. Noise removal is not reflected in the privacy bound. | [+0.147247](redteam/shortcut_public_game_no_noise/scores_test.json) | **Works** |
| [Constant uniform output](redteam/shortcut_constant_uniform/run.sh) | Return one valid uniform profile without payoff optimization. | [−0.111140](redteam/shortcut_constant_uniform/scores_test.json) | Fails |
| [All-zero strategies](redteam/shortcut_zero_outputs/run.sh) | Return invalid probability vectors. Both equilibrium errors become 0; simplex violation is 1. | [−0.019366](redteam/shortcut_zero_outputs/scores_test.json) | Fails to pay; invalid outputs accepted |
| [All-zero strategies and eta](redteam/shortcut_zero_outputs_zero_eta/run.sh) | Combine invalid zero vectors with `eta=0`; evaluated equilibrium errors and privacy are 0. | [+3840.900772](redteam/shortcut_zero_outputs_zero_eta/scores_test.json) | **Works** despite failing the task |
| [NaN strategies](redteam/shortcut_nan_output/run.sh) | Return nonfinite values. | [No score; test exit 1](redteam/shortcut_nan_output/test_error.txt) | Fails; explicit finite-value rejection |
| [Empty trajectory](redteam/shortcut_empty_output/run.sh) | Return zero time steps. | [No score; test exit 1](redteam/shortcut_empty_output/test_error.txt) | Fails; aggregation raises an error |

The public-game shortcut reconstructs test *inputs* and computes noiseless outputs; it does not claim to recover an equilibrium-answer cache. There is no enforced hidden game identity: the test scores game paths supplied in method manifests. The zero-game shortcut demonstrates replacement of that supposedly fixed artifact, while the comparator shortcut demonstrates moving the other side of the comparison. Both work without a better proposed method.

The rank fails the task-validity check. It penalizes simplex violation 1 by only **0.5** after experiment weighting. Reducing the claimed privacy mean from 6401.558784 to zero contributes **+3840.935270** before the small trend adjustment. The invalid all-zero shortcut therefore outranks the reference by +3840.900772. Neither a validity gate nor comparable metric scales prevent a task failure from winning. The negative-order shortcut also shows an algebraically unbounded direction if increasingly negative orders are accepted; only `alpha=-2` was tested.

The evaluator does recompute exploitability and ignores a raw self-reported privacy bound, but its replacement privacy formula trusts `eta`, `alpha`, step count, and the submitted graph. It samples adjacency edges itself. It does not establish that the submitted noise or protocol produces the claimed privacy. Required game coverage, fixed game identity, and fixed comparator identity are also unenforced. These are scoring issues; no evaluator process, imports, or Python environment were tampered with.

For the next gym iteration, fix the test game/grid and comparator outside submitted artifacts; reject invalid strategies and invalid privacy parameters before ranking; and bind privacy accounting to the executed protocol and noise. Normalize metric scales and make task constraints gates rather than small additive penalties. These recommendations were not applied to this gym.

Execution used two completed CPU jobs, `0353ab9d-cdc` and `02fec3f7-5e7`, covering 21 variants/shortcuts. Each candidate preparation process exited before the unchanged no-argument `bash scripts/evaluate_test.sh` ran as a separate process. Only persisted files were consumed; no background processes or environment patches were used. Evaluator/script/shared-data hashes were checked on every run. The original outputs and scoring files were restored after each batch, and the temporary job script was restored at completion. Generated trajectories are ignored and rebuildable from the public generator; all small score evidence is committed. Audit progress is complete; the existing reproduction milestone was not changed by this scoring audit.

Run [redteam/summarize.py](redteam/summarize.py) to verify the saved JSON and regenerate [audit_summary.json](redteam/audit_summary.json) without rerunning computation. The full CPU audit is reproducible with `python3 -m redteam.audit batch`; remove saved per-case `result.json` files to request reruns instead of resuming completed cases.
