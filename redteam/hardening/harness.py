"""Hardening audit: candidate methods and shortcuts through the hidden test.

Every candidate is produced with the public problem API only (data/, plus the
reference method and baseline code) -- nothing here imports eval/.  Each case
is then scored by the unmodified, no-argument ``bash scripts/evaluate_test.sh``
in a fresh process with only that case's manifests under outputs/.  The rank
gain is sum_exp weight * sum_metric coefficient * (case - reference) with the
weights and coefficients of scoring/reference.json, the reference being the
``reference`` case's ``our_method`` entry.

Noise-stream replicates of the reference are scored with
``python -m eval.test.evaluate --noise-key k`` (robustness study only).

Usage:  python -m redteam.hardening.harness [case ...]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from baseline.no_payoff import run_no_payoff
from baseline.structure_free import budget_matched_step, run_structure_free
from data.polymatrix import (
    N_ACTIONS,
    SIZES,
    TOPOLOGIES,
    benchmark_game,
    benchmark_hyperparameters,
    broadcast_noise,
    cce_exploitability,
    gradients,
    payoff_operator,
    project_simplex,
)
from data.submission import save_run, write_manifest
from method.dp_cce import degree_adaptive_regularizer, run_algorithm1

ROOT = Path("/home/user")
HERE = ROOT / "redteam" / "hardening"
RUNS = HERE / "_runs"
RESULTS = HERE / "cases"
# The hidden test reads only these instances; producing the other public seeds
# would not change any test score, so the audit skips them for speed.
AUDIT_SEEDS = (1, 3, 5, 7, 9, 11)
REPLICATE_KEYS = (1, 2, 3)

CASES = {
    # name: (kind, description)
    "reference": ("reference", "Algorithm 1 exactly as shipped (method/dp_cce.py)."),
    "better_shrink_075": ("better", "Anchor at 0.75*own projected broadcast + 0.25*uniform, plain gradient step, no degree regularizer."),
    "better_max_shrink": ("better", "Algorithm 1 update with the regularizer weight set to 1 for every player (maximal allowed shrink)."),
    "better_denoised_gradient": ("better", "Algorithm 1, gradient direction from the running mean of public projected broadcasts, rescaled to the current public-gradient norm."),
    "weaker_no_gradient": ("weaker", "Algorithm 1 with all payoff gradients removed."),
    "weaker_half_step": ("weaker", "Algorithm 1 with half the step size."),
    "comparator_as_method": ("baseline", "The structure-free budget-matched comparator submitted as the method."),
    "no_payoff_as_method": ("baseline", "The payoff-blind protocol baseline submitted as the method."),
    "old_better_separate_readout": ("redteam_better", "Redteam 'better_fixed_transcript_readout': reference broadcasts, separately evolved private output states reported as the strategies."),
    "old_better_persistent_anchor": ("redteam_better", "Redteam 'better_persistent_anchor': gradient step from the private noiseless strategy instead of the projected broadcast."),
    "old_better_final_average": ("redteam_better", "Redteam 'better_final_average': reference trajectory with the temporal average appended as an extra round."),
    "shortcut_eta_zero": ("shortcut", "Reference arrays; manifest claims eta = 0."),
    "shortcut_negative_alpha": ("shortcut", "Reference arrays; manifest claims Renyi order alpha = -2."),
    "shortcut_reported_bound": ("shortcut", "Reference arrays; manifest claims privacy_bound = -1e30."),
    "shortcut_zero_game": ("shortcut", "Reference arrays; manifest points game_path at all-zero-utility games."),
    "shortcut_degrade_baseline": ("shortcut", "Reference arrays as our_method plus deliberately terrible 'structure_free_baseline' outputs on disk."),
    "shortcut_drop_small_games": ("shortcut", "Reference arrays with every n=64 run removed."),
    "shortcut_public_game_no_noise": ("shortcut", "Algorithm 1 on the public games with the noise removed (broadcasts = strategies)."),
    "shortcut_constant_uniform": ("shortcut", "Redteam form: a single uniform round and no broadcasts."),
    "shortcut_zero_outputs": ("shortcut", "All-zero strategy vectors with the reference broadcasts."),
    "shortcut_zero_outputs_zero_eta": ("shortcut", "All-zero strategies plus a claimed eta = 0."),
    "shortcut_nan_output": ("shortcut", "Non-finite strategies."),
    "shortcut_empty_output": ("shortcut", "Zero rounds."),
    "shortcut_fabricated_noise": ("shortcut", "Noiseless Algorithm 1 dynamics; broadcasts fabricated afterwards as strategy + the instance noise."),
    "shortcut_large_step": ("shortcut", "Algorithm 1 with three times the operating-point step."),
    "shortcut_best_of_8_noise": ("shortcut", "Algorithm 1 under 8 noise streams per instance, keeping the lowest-exploitability run."),
    "shortcut_fewer_rounds": ("shortcut", "Reference arrays truncated to T-1 rounds."),
    "shortcut_known_noise_gradient": ("shortcut", "Algorithm 1 but gradients use neighbours' true (pre-noise) strategies -- a privacy violation the transcript cannot reveal."),
}


# ---------------------------------------------------------------------------
# Candidate dynamics (public API only)
# ---------------------------------------------------------------------------
def simulate(game, steps, eta, noise, *, tau=None, keep=1.0, step_scale=1.0, grad_mode="public",
             anchor_mode="public", denoise=False):
    n, a = game.n_players, game.n_actions
    uniform = np.full((n, a), 1.0 / a)
    op = payoff_operator(game)
    tau = np.zeros(n) if tau is None else tau
    s = uniform.copy()
    running = np.zeros((n, a))
    traj, bc = [], []
    for t in range(steps):
        b = s + noise[t]
        anchor_public = project_simplex(b)
        running += anchor_public
        if grad_mode == "public":
            g = gradients(game, anchor_public, op)
        elif grad_mode == "true":
            g = gradients(game, s, op)
        else:
            g = np.zeros((n, a))
        if denoise:
            gd = gradients(game, running / (t + 1), op)
            gd = gd - gd.mean(axis=1, keepdims=True)
            gp = g - g.mean(axis=1, keepdims=True)
            scale = np.minimum(1.0, np.linalg.norm(gp, axis=1) / np.maximum(np.linalg.norm(gd, axis=1), 1e-12))
            g = gd * scale[:, None]
        base = anchor_public if anchor_mode == "public" else s
        base = keep * base + (1.0 - keep) * uniform
        s = project_simplex((base - step_scale * eta * g) / (1.0 + step_scale * eta * tau[:, None]))
        traj.append(s.copy())
        bc.append(b.copy())
    return np.stack(traj), np.stack(bc)


def separate_readout(game, eta, broadcasts, tau):
    n, a = game.n_players, game.n_actions
    op = payoff_operator(game)
    shadow = np.full((n, a), 1.0 / a)
    out = []
    for message in broadcasts:
        g = gradients(game, project_simplex(message), op)
        shadow = project_simplex((shadow - eta * g) / (1.0 + eta * tau[:, None]))
        out.append(shadow.copy())
    return np.stack(out)


def build_instance(topology, n, seed):
    steps, eta, sigma = benchmark_hyperparameters(n, topology)
    game = benchmark_game(topology, n, seed)
    noise = broadcast_noise(topology, n, seed)
    tau = degree_adaptive_regularizer(game)
    uniform = np.full((n, N_ACTIONS), 1.0 / N_ACTIONS)
    ref_t, ref_b = run_algorithm1(game, steps, eta, noise)
    out = {"reference": (ref_t, ref_b, {})}
    out["better_shrink_075"] = (*simulate(game, steps, eta, noise, keep=0.75), {})
    out["better_max_shrink"] = (*simulate(game, steps, eta, noise, tau=np.ones(n)), {})
    out["better_denoised_gradient"] = (*simulate(game, steps, eta, noise, tau=tau, denoise=True), {})
    out["weaker_no_gradient"] = (*simulate(game, steps, eta, noise, tau=tau, grad_mode="none"), {})
    out["weaker_half_step"] = (*simulate(game, steps, eta, noise, tau=tau, step_scale=0.5), {})
    out["comparator_as_method"] = (*run_structure_free(game, steps, budget_matched_step(game, steps, eta), noise), {})
    out["no_payoff_as_method"] = (*run_no_payoff(n, steps, eta, noise), {})
    out["old_better_separate_readout"] = (separate_readout(game, eta, ref_b, tau), ref_b, {})
    out["old_better_persistent_anchor"] = (*simulate(game, steps, eta, noise, tau=tau, anchor_mode="private"), {})
    out["old_better_final_average"] = (np.concatenate([ref_t, ref_t.mean(axis=0, keepdims=True)]), ref_b, {})
    out["shortcut_eta_zero"] = (ref_t, ref_b, {"eta": 0.0})
    out["shortcut_negative_alpha"] = (ref_t, ref_b, {"alpha": -2.0})
    out["shortcut_reported_bound"] = (ref_t, ref_b, {"privacy_bound": -1e30})
    out["shortcut_zero_game"] = (ref_t, ref_b, {"zero_game": True})
    out["shortcut_degrade_baseline"] = (ref_t, ref_b, {"degrade_baseline": True})
    out["shortcut_drop_small_games"] = (ref_t, ref_b, {"skip": n == 64})
    out["shortcut_public_game_no_noise"] = (*simulate(game, steps, eta, np.zeros_like(noise), tau=tau), {})
    out["shortcut_constant_uniform"] = (uniform[None].copy(), None, {})
    out["shortcut_zero_outputs"] = (np.zeros_like(ref_t), ref_b, {})
    out["shortcut_zero_outputs_zero_eta"] = (np.zeros_like(ref_t), ref_b, {"eta": 0.0})
    out["shortcut_nan_output"] = (np.full_like(ref_t, np.nan), ref_b, {})
    out["shortcut_empty_output"] = (np.empty((0, n, N_ACTIONS)), np.empty((0, n, N_ACTIONS)), {})
    clean_t, _ = simulate(game, steps, eta, np.zeros_like(noise), tau=tau)
    out["shortcut_fabricated_noise"] = (clean_t, np.concatenate([uniform[None], clean_t[:-1]]) + noise, {})
    out["shortcut_large_step"] = (*simulate(game, steps, eta, noise, tau=tau, step_scale=3.0), {})
    best = None
    for key in range(8):
        cand_t, cand_b = run_algorithm1(game, steps, eta, broadcast_noise(topology, n, seed, key=key))
        value = cce_exploitability(game, cand_t)[0]
        if best is None or value < best[0]:
            best = (value, cand_t, cand_b, key)
    out["shortcut_best_of_8_noise"] = (best[1], best[2], {"selected_noise_key": best[3]})
    out["shortcut_fewer_rounds"] = (ref_t[:-1], ref_b[:-1], {})
    out["shortcut_known_noise_gradient"] = (*simulate(game, steps, eta, noise, tau=tau, grad_mode="true"), {})
    replicates = {}
    for key in REPLICATE_KEYS:
        replicates[f"reference_noise_key{key}"] = run_algorithm1(game, steps, eta, broadcast_noise(topology, n, seed, key=key))
    bad_baseline = None
    g_uniform = gradients(game, uniform)
    worst = np.eye(N_ACTIONS)[g_uniform.argmax(axis=1)]
    bad_baseline = (np.repeat(worst[None], steps, axis=0), np.concatenate([uniform[None], np.repeat(worst[None], steps - 1, axis=0)]) + noise)
    return game, out, replicates, bad_baseline


def save_case_run(case, topology, n, seed, traj, bcast, info, method="our_method"):
    folder = RUNS / case / topology
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{method}_{topology}_n{n}_seed{seed}.npz"
    arrays = {"trajectory": traj}
    if bcast is not None:
        arrays["broadcasts"] = bcast
    np.savez_compressed(path, **arrays)
    run = {"method": method, "topology": topology, "n_players": n, "seed": seed, "trajectory_path": str(path)}
    run.update(info)
    return run


def generate_all():
    runs = defaultdict(lambda: defaultdict(list))
    extra = defaultdict(lambda: defaultdict(list))
    for topology in TOPOLOGIES:
        for n in SIZES:
            for seed in AUDIT_SEEDS:
                start = time.time()
                game, out, replicates, bad = build_instance(topology, n, seed)
                for case, (traj, bcast, info) in out.items():
                    info = dict(info)
                    if info.pop("skip", False):
                        continue
                    if info.pop("zero_game", False):
                        zero_path = RUNS / case / topology / f"zero_game_n{n}_seed{seed}.npz"
                        zero_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(zero_path, n_players=n, n_actions=N_ACTIONS, edges=game.edges,
                                            utilities=np.zeros_like(game.utilities))
                        info["game_path"] = str(zero_path)
                    if info.pop("degrade_baseline", False):
                        extra[case][topology].append(save_case_run(case, topology, n, seed, bad[0], bad[1], {},
                                                                   method="structure_free_baseline"))
                    runs[case][topology].append(save_case_run(case, topology, n, seed, traj, bcast, info))
                for case, (traj, bcast) in replicates.items():
                    runs[case][topology].append(save_case_run(case, topology, n, seed, traj, bcast, {}))
                print(f"generated {topology} n={n} seed={seed} in {time.time() - start:.1f}s", flush=True)
    for case in runs:
        for topology in TOPOLOGIES:
            write_manifest(RUNS / case / topology, "our_method", runs[case][topology])
            if extra[case][topology]:
                write_manifest(RUNS / case / "baseline_side" / topology, "structure_free_baseline", extra[case][topology])


# ---------------------------------------------------------------------------
# Scoring through the hidden-test entry point
# ---------------------------------------------------------------------------
def install_case(case):
    shutil.rmtree(ROOT / "outputs", ignore_errors=True)
    for topology in TOPOLOGIES:
        src = RUNS / case / topology / "manifest.json"
        if src.exists():
            dst = ROOT / "outputs" / "candidate" / topology / "manifest.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        side = RUNS / case / "baseline_side" / topology / "manifest.json"
        if side.exists():
            dst = ROOT / "outputs" / "structure_free_baseline" / topology / "manifest.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(side, dst)


def rank_gain(scores, reference_scores, rubric):
    total, parts = 0.0, {}
    for exp, cfg in rubric["experiments"].items():
        actual = scores["experiments"][exp]["results"]["our_method"]
        target = reference_scores["experiments"][exp]["results"]["our_method"]
        parts[exp] = {m: cfg["weight"] * spec["coefficient"] * (actual[m] - target[m]) for m, spec in cfg["metrics"].items()}
        total += sum(parts[exp].values())
    return total, parts


def evaluator_hash():
    h = hashlib.sha256()
    for rel in ("eval/test/evaluate.py", "eval/test/comparator_test.json", "scripts/evaluate_test.sh"):
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()


def score_case(case):
    folder = RESULTS / case
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("scores_test.json", "diagnostics_test.json"):
        (ROOT / "scoring" / name).unlink(missing_ok=True)
    before = evaluator_hash()
    start = time.time()
    if case.startswith("reference_noise_key"):
        key = case.removeprefix("reference_noise_key")
        install_case(case)
        manifests = sorted(str(p) for p in (ROOT / "outputs").rglob("manifest.json"))
        cmd = [sys.executable, "-m", "eval.test.evaluate", "--manifest", *manifests, "--noise-key", key]
    else:
        install_case(case)
        cmd = ["bash", "scripts/evaluate_test.sh"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    result = {"case": case, "test_exit": proc.returncode, "elapsed_seconds": round(time.time() - start, 1),
              "evaluator_unchanged": evaluator_hash() == before}
    if case in CASES:
        result["kind"], result["description"] = CASES[case]
    else:
        result["kind"], result["description"] = "replicate", f"Reference under alternative noise stream {case[-1]} (seed-noise study)."
    (folder / "evaluate_test.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
    if proc.returncode == 0:
        shutil.copy2(ROOT / "scoring" / "scores_test.json", folder / "scores_test.json")
        shutil.copy2(ROOT / "scoring" / "diagnostics_test.json", folder / "diagnostics_test.json")
        scores = json.loads((folder / "scores_test.json").read_text())
        diag = json.loads((folder / "diagnostics_test.json").read_text())
        result["metrics"] = {exp: block["results"].get("our_method") for exp, block in scores["experiments"].items()}
        result["status"] = {exp: d.get("our_method", {}).get("status") for exp, d in diag["experiments"].items()}
        result["failure_reasons"] = {exp: d.get("our_method", {}).get("failure_reasons") for exp, d in diag["experiments"].items()}
        result["max_step_utilization"] = {exp: d.get("our_method", {}).get("max_step_utilization") for exp, d in diag["experiments"].items()}
    (folder / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: result.get(k) for k in ("case", "test_exit", "status", "elapsed_seconds")}), flush=True)
    return result


def main(selected):
    backup = HERE / "_backup"
    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir(parents=True)
    if (ROOT / "outputs").exists():
        shutil.move(str(ROOT / "outputs"), backup / "outputs")
    for name in ("scores_test.json", "diagnostics_test.json"):
        if (ROOT / "scoring" / name).exists():
            shutil.copy2(ROOT / "scoring" / name, backup / name)
    try:
        if not (RUNS / "ready.json").exists():
            shutil.rmtree(RUNS, ignore_errors=True)
            generate_all()
            (RUNS / "ready.json").write_text("{}")
        cases = selected or (list(CASES) + [f"reference_noise_key{k}" for k in REPLICATE_KEYS])
        results = {}
        for case in cases:
            results[case] = score_case(case)
        rubric = json.loads((ROOT / "scoring" / "reference.json").read_text())
        ref_path = RESULTS / "reference" / "scores_test.json"
        reference_scores = json.loads(ref_path.read_text())
        for case in sorted(p.name for p in RESULTS.iterdir() if (p / "result.json").exists()):
            result = json.loads((RESULTS / case / "result.json").read_text())
            if result.get("test_exit") == 0:
                scores = json.loads((RESULTS / case / "scores_test.json").read_text())
                result["rank_gain"], result["rank_components"] = rank_gain(scores, reference_scores, rubric)
                (RESULTS / case / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
            results[case] = result
        summary = {case: {"kind": r.get("kind"), "test_exit": r.get("test_exit"), "rank_gain": r.get("rank_gain"),
                          "status": r.get("status"),
                          "comparator_exploitability_ratio": {e: (m or {}).get("comparator_exploitability_ratio") for e, m in (r.get("metrics") or {}).items()},
                          "relative_exploitability": {e: (m or {}).get("relative_exploitability") for e, m in (r.get("metrics") or {}).items()}}
                   for case, r in sorted(results.items())}
        (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        for case, r in sorted(summary.items(), key=lambda kv: -(kv[1]["rank_gain"] if kv[1]["rank_gain"] is not None else -1e9)):
            print(f"{case:40s} exit={r['test_exit']} rank={r['rank_gain']}", flush=True)
    finally:
        shutil.rmtree(ROOT / "outputs", ignore_errors=True)
        if (backup / "outputs").exists():
            shutil.move(str(backup / "outputs"), ROOT / "outputs")
        for name in ("scores_test.json", "diagnostics_test.json"):
            if (backup / name).exists():
                shutil.copy2(backup / name, ROOT / "scoring" / name)


if __name__ == "__main__":
    main(sys.argv[1:])
