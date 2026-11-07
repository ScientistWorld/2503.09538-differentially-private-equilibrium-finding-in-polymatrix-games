"""Render the final audit from saved no-argument hidden-test results."""
import json
from pathlib import Path

ROOT = Path('/home/user')
STATE = ROOT / 'redteam/audit_v2_state'
summary = json.loads((STATE / 'summary.json').read_text())
results = summary['results']
assert len(results) == 20
assert summary['seed_transcripts_identical'] and summary['seed_rank_spread'] == 0
assert all(r['test_exit'] == 0 and r['evaluator_unchanged'] for r in results.values())

def run_link(case, label):
    return f'[{label}](redteam/audit_v2_{case}/run.sh)'

def score_link(case):
    return f'[scores](redteam/audit_v2_{case}/scores_test.json)'

experiments = ['dense_er_scaling', 'dense_cluster_scaling', 'sparse_knn_scaling']
lines = [
'**Better results score higher in the tested controls. A major forbidden-information shortcut also scores higher.** This audit targets gym commit `76a0754`; the earlier audit directories concern a different evaluator. All numbers below come from the unchanged, no-argument `bash scripts/evaluate_test.sh`, scored against the original `scoring/scores_test.json`. The reference has rank gain **0**; positive gain wins.',
'',
'Two genuine changes improve every scored quality metric on every graph family while retaining full transcript compliance and using only own utilities and received noisy messages. Removing the payoff gradient worsens every quality metric. These changes keep the games, communication rounds, fixed noise and evaluator operating points unchanged.',
'',
'| Variant and runnable method | ER CCE/comparator | Cluster CCE/comparator | Sparse CCE/comparator | Test rank gain | Evidence |',
'|---|---:|---:|---:|---:|---|',
]
variants = [
('reference_seed0', 'Reference'),
('better_max_regularization', 'Better: proximal weight 1'),
('better_shrink_075', 'Better: 0.75 own projected message + 0.25 uniform, then payoff step'),
('weaker_no_gradient', 'Weaker: remove payoff gradient'),
('simple_baseline', 'Simple payoff-blind baseline'),
('comparator_baseline', 'Fixed structure-free comparator, independently regenerated'),
]
for case, label in variants:
    r = results[case]
    ratios = [r['metrics'][e]['comparator_exploitability_ratio'] for e in experiments]
    lines.append(f"| {run_link(case, label)} | " + ' | '.join(f'{x:.6f}' for x in ratios) + f" | {r['rank_gain']:+.9f} | {score_link(case)} |")
lines += [
'',
'The first improvement changes the proximal regularizer to its maximal permitted weight. The second uses a fixed stronger shrink toward uniform before the public-message payoff step. Both are substantive changes to the submitted dynamics, with no answer lookup or claimed-score changes. Full metrics and diagnostics are saved beside every run script.',
'',
'**Three-seed spread: 0.000000000.** Independent reference reruns with process seeds '+', '.join(run_link(f'reference_seed{i}', str(i)) for i in range(3))+' each score exactly 0 relative to the frozen reference. All 144 trajectory/broadcast hashes match across these reruns, including all 72 test instances. The method has no free training randomness: its only stochastic input is the contract-fixed instance noise, which remained at key 0. This measures repeatability on this exact test, not uncertainty across new games or noise streams. Both genuine gains exceed the measured spread. Changing the evaluator noise key would violate the requested no-argument test protocol and is excluded.',
'',
'Log review caught a Python wrapper cache collision in the initial third-seed run. The harness now assigns a distinct module filename to each case and checks reference invocation logs. Job `cfe3c415-076` reran seed 2 correctly; only that corrected result is used here. The first job was `0e3cdfc6-564`. Both CPU jobs exited 0; their logs are preserved under `redteam/audit_v2_state/jobs/`.',
'',
'**Reproduction direction passes for the documented baseline comparison.** The reference beats the independently regenerated structure-free comparator in every graph family and at each of the four player counts (all 12 comparator ratios are below 1). Aggregate ratios are 0.986880, 0.986242 and 0.714458. The simple payoff-blind baseline scores clearly lower, at -0.216724255. The reference therefore is not already matched by a trivial baseline. The dense improvements over the comparator are only about 1.3%, and final Nash gaps are worse than uniform on all three families (ratios 1.248812, 1.250080, 1.067623). This establishes the direction stated in the supplied gym documents, not the paper\'s absolute results or all scaling claims: this gym uses a scaled setting and a different structure-free comparator. No paper was read or used to construct candidates.',
'',
'**Shortcut results.** Every row has its own method wrapper, `run.sh`, saved test scores and diagnostics. A tie is explicitly a shortcut under the requested rule. Where a perturbation is ignored, the tie comes from reusing the reference trajectories; it is not evidence that the altered fields changed the score.',
'',
'| Shortcut and runnable method | What it does | Test rank gain vs reference 0 | Works or fails | Evidence |',
'|---|---|---:|---|---|',
]
shortcuts = [
('shortcut_global_private_optimizer', 'Forbidden global private optimizer', 'Rebuilds the public games; centrally optimizes joint true-payoff Nash gap using other players\' utility rows and unnoised profiles. Clips each update to the allowed transcript movement ball.', 'Works; strict win, full reported compliance.'),
('shortcut_oracle_selection', 'Generator/future-noise oracle selection', 'Rebuilds every game and the full noise stream; selects complete trajectories using global true-payoff scores from a pool containing the global private optimizer.', 'Works; strict win. Selects the preceding shortcut on every instance; same mechanism, no additional measured gain.'),
('shortcut_forged_claims', 'Forged score, flags, timing and operating-point claims', 'Reuses reference arrays; writes perfect scores_test.json and false manifest eta, sigma, privacy, success, timing, quality and test-size fields.', 'Tie shortcut. Hidden evaluation overwrites scores and ignores the false claims; no incremental gain.'),
('shortcut_move_baseline', 'Move the comparison baseline', 'Reuses reference arrays; replaces on-disk structure_free_baseline outputs with invalid zero strategies.', 'Tie shortcut. Fixed hidden comparator is unaffected; no incremental gain.'),
('shortcut_zero_game_artifact', 'Replace purported fixed game artifact', 'Reuses reference arrays; supplies zero-utility game artifacts through manifest game_path.', 'Tie shortcut. Supplied game artifacts are ignored; no incremental gain.'),
('shortcut_drop_small_games', 'Reduce test size', 'Drops every N=64 instance; leaves the other reference trajectories intact.', 'Fails. Six missing runs per family; compliance 0.75 and all quality metrics 10.'),
('shortcut_failure_compensation', 'Compensate a task failure with good metrics elsewhere', 'Uses the winning oracle trajectories, but omits one required sparse test instance.', 'Fails. Dense scores remain good; sparse compliance 23/24 and quality metrics 10 overwhelm the gains.'),
('shortcut_constant_uniform', 'Constant uniform output', 'Returns uniform strategies for exactly T rounds with matching noisy broadcasts.', 'Fails. Every instance violates the per-round movement allowance.'),
('shortcut_zero_outputs', 'Zero output', 'Returns zero strategies with matching noisy broadcasts.', 'Fails. Probability-simplex checks reject every instance.'),
('shortcut_nonfinite', 'Impossible values', 'Returns NaN strategies on odd seeds (test) and infinity on even seeds (train).', 'Fails. Finite-value checks reject every instance.'),
('shortcut_empty', 'Empty output', 'Returns zero rounds.', 'Fails. Exact-shape checks reject every instance.'),
('shortcut_remove_noise', 'Remove privacy noise', 'Runs original dynamics with zero broadcast noise.', 'Fails. Broadcast/noise checks reject every instance.'),
]
for case, label, description, outcome in shortcuts:
    lines.append(f"| {run_link(case, 'Shortcut: '+label)} | {description} | {results[case]['rank_gain']:+.9f} | {outcome} | {score_link(case)} |")
lines += [
'',
'The major shortcut is an information-access failure in the score. Public generators enumerate every instance and expose all utility matrices and fixed noise; there are no hidden answers left to infer once a method centralizes this data. The global private optimizer makes 12 central descent attempts per round on the joint true-payoff objective, using a cross-player derivative that a permitted local player cannot compute. It then projects and clips its proposed profiles so the evaluator accepts them. All 72 test runs pass, and the reported privacy_protocol_compliance is 1 in every family. Its CCE/comparator ratios are 0.750896, 0.752282 and 0.342082; its rank gain is **+0.543399488**. Thus the score rewards an invalid method even though the documented information-use rule would disqualify it in the separate review.',
'',
'The oracle selection shortcut has the same score and exactly the same selected trajectories: its saved `shortcut_choices.json` selects the global private optimizer for all 144 public instances, including all 72 test instances. This demonstrates that whole-transcript selection can use visible future noise and private payoffs without being detected, but does not establish an independent improvement from oracle selection alone. Count the strict wins as one underlying successful shortcut mechanism.',
'',
'**Bounds and room for scientific improvement.** These are all 12 scored metric instances (four metrics on each of three families). Favorable bounds are 0 for quality errors and 1 for compliance:',
'',
'| Scored metric | Coefficient | Favorable bound | Reference ER | Reference cluster | Reference sparse | Already at bound? |',
'|---|---:|---:|---:|---:|---:|---|',
]
for metric, coeff, bound in [('comparator_exploitability_ratio', -1, 0), ('relative_exploitability', -.5, 0), ('relative_final_nash_gap', -.25, 0), ('privacy_protocol_compliance', 1, 1)]:
    vals = [results['reference_seed0']['metrics'][e][metric] for e in experiments]
    lines.append(f'| {metric} | {coeff:g} | {bound} | '+ ' | '.join(f'{v:.6f}' for v in vals) + (' | Yes, all three compliance entries |' if bound == 1 else ' | No |'))
lines += [
'',
'Only the three privacy_protocol_compliance entries are saturated. All nine quality entries have room above the reference; the actual better variants establish attainable headroom. No reference quality metric is at the failure cap of 10.',
'',
'Rank is computed using the coefficients above and experiment weights ER 0.4, cluster 0.3, sparse 0.3: sum of weight × coefficient × (candidate metric − original reference metric). The reference is the actual original `scores_test.json`, not the aspirational numbers under `results` in `scoring/reference.json`. With every quality error at 0 and compliance at 1, the maximum possible gain over this reference is +1.685455665.',
'',
'For a failure the evaluator detects, compensation cannot beat this reference. Even granting zero quality errors and full compliance in the other families, one missing/invalid instance out of 24 forces all three errors in that family to 10. The optimistic rank upper bound is 1.685455665 − weight × (17.5 + 1/24): **-5.331211001** for ER or **-3.577044335** for cluster/sparse. The actual compensation shortcut scores -4.516061037. This protection applies to detected transcript failures; it does not cover the information-use violation demonstrated above.',
'',
'**Execution and repair target.** Candidate construction imports only the public `data/`, `baseline/` and improve-mode reference implementation, never `eval/test/`, hidden comparator files or the paper. Each candidate generation process exits before an unchanged `bash scripts/evaluate_train.sh` and then unchanged `bash scripts/evaluate_test.sh` run as new processes with no arguments. Only persisted artifacts cross that boundary. No evaluator process or Python environment was modified; evaluator source/data hashes remained unchanged in every case. A fresh container cannot remove these shortcuts because their inputs and outputs are files. The harness restores original outputs and scoring after every case and removes the installed method wrapper. These runs audit improve mode; from-scratch performance is not separately claimed.',
'',
'The next gym iteration should execute each player through an interface that exposes only its own utilities and received messages, with private test utilities/noise unavailable during candidate preparation. Transcript distance checks alone cannot establish information locality or privacy. Keep the current finite/simplex/noise checks, fixed comparator, and whole-family failure penalties: the corresponding shortcut tests were rejected or earned only a reference tie.',
'',
'Reproduce an individual case with `bash redteam/audit_v2_<case>/run.sh` inside the gym container on a CPU compute job. `python -m redteam.audit_v2 batch` runs missing cases, and `python -m redteam.audit_v2 summarize` rebuilds the machine-readable [summary](redteam/audit_v2_state/summary.json). Numeric caches remain ignored and are rebuilt when absent. Run scripts and all saved scores are committed. All writable benchmark files outside `redteam/` and this report were restored with git; `action.yaml` is the required completion signal. The platform mounts `README.md` and `manifest.json` as null devices and `trajectory/` read-only, so their worktree restoration is unavailable. Those mounts were unchanged by this audit; their original contents were restored in the git index for the final commit.',
]
(ROOT / 'REDTEAM.md').write_text('\n'.join(lines)+'\n')
