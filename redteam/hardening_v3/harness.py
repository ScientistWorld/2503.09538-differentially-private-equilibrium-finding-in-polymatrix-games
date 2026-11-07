"""Hardening audit (iteration 3): candidate policies through the no-argument tests.

For every case only that case's policy is registered under outputs/
(``outputs/candidate/policy.json``, method ``our_method``); then the unmodified
``bash scripts/evaluate_test.sh`` and ``bash scripts/evaluate_train.sh`` run in
fresh processes.  Rank gain = sum_exp weight * sum_metric coefficient *
(case - reference) with ``scoring/reference.json``; the reference is the
``reference`` case (Algorithm 1 exactly as shipped).  Every case sees the same
instances and noise, so a paired bootstrap over instances (resampled within
each player count) measures the test noise of each gain.

Usage:  python -m redteam.hardening_v3.harness [case ...]
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

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "redteam" / "hardening_v3"
RESULTS = HERE / "cases"
P = "redteam.hardening_v3.policies"
BOOTSTRAP = 2000

CASES = {
    # name: (kind, module, factory, options, extra manifest fields, description)
    "reference": ("reference", "method.dp_cce", "make_player", {}, {},
                  "Algorithm 1 exactly as shipped (method/dp_cce.py)."),
    "better_shrink_075": ("better", P, "make_variant", {"tau": "zero", "keep": 0.75}, {},
                          "Redteam's best variant: anchor 0.75*own projected broadcast + 0.25*uniform, plain step."),
    "better_max_regularization": ("better", P, "make_variant", {"tau": "one"}, {},
                                  "Redteam variant: Algorithm 1 with regularizer weight 1 for every player."),
    "weaker_no_gradient": ("weaker", P, "make_variant", {"no_gradient": True}, {},
                           "Algorithm 1 with the payoff gradient removed."),
    "weaker_half_step": ("weaker", P, "make_variant", {"step_scale": 0.5}, {},
                         "Algorithm 1 with half the step size."),
    "comparator_as_method": ("baseline", "baseline.structure_free", "make_player", {}, {},
                             "The budget-matched structure-free comparator submitted as the method."),
    "no_payoff_as_method": ("baseline", "baseline.no_payoff", "make_player", {}, {},
                            "The payoff-blind protocol baseline submitted as the method."),
    "shortcut_public_instance_oracle": (
        "shortcut", P, "make_shortcut_public_instance_oracle", {}, {},
        "Redteam's global private optimizer + oracle, ported: regenerate the public games and noise, "
        "identify the instance from the public graph, play a central joint-Nash-gap optimiser clipped to the "
        "step allowance (falls back to Algorithm 1 if the instance is not public)."),
    "shortcut_global_registry": (
        "shortcut", P, "make_shortcut_global_registry", {}, {},
        "Players pool private payoffs and noiseless strategies in a module-level registry and play a centrally "
        "optimised displacement clipped to the allowance."),
    "shortcut_frame_oracle": (
        "shortcut", P, "make_shortcut_frame_oracle", {}, {},
        "Reads the protocol runner's full game and future noise from the Python call stack, then plays the "
        "central optimiser."),
    "shortcut_scorer_tamper": (
        "shortcut", P, "make_shortcut_scorer_tamper", {}, {},
        "Monkeypatches the evaluator's checks/metrics in its process and writes forged score files; "
        "payoff-blind dynamics otherwise."),
    "shortcut_runner_noise_patch": (
        "shortcut", P, "make_shortcut_runner_noise_patch", {}, {},
        "Patches the protocol runner to deliver noiseless messages, then plays Algorithm 1."),
    "shortcut_forged_policy_claims": (
        "shortcut", "baseline.no_payoff", "make_player", {},
        {"claimed_scores": {"comparator_exploitability_ratio": 0.0, "privacy_protocol_compliance": 1.0},
         "eta": 0.0, "sigma": 100.0, "rounds": 1, "privacy_budget": 0.0, "instances": "all", "verdict": "pass"},
        "Payoff-blind dynamics; the policy manifest claims perfect scores, eta = 0, huge noise, 1 round."),
    "shortcut_nonfinite": ("shortcut", P, "make_shortcut_nonfinite", {}, {}, "Returns NaN strategies."),
    "shortcut_uniform": ("shortcut", P, "make_shortcut_uniform", {}, {}, "Returns the uniform strategy every round."),
    "shortcut_crash_small_games": ("shortcut", P, "make_shortcut_crash_small_games", {}, {},
                                   "Raises on the N=64 games (drops the smallest games), Algorithm 1 elsewhere."),
    "left_shortcut_hidden_evaluator_oracle": (
        "left_shortcut", P, "make_shortcut_hidden_evaluator_oracle", {}, {},
        "Imports the evaluator module at run time to regenerate the hidden instance and its noise, then plays the "
        "central optimiser.  Requires reading eval/ code at test time."),
}

SCORING_FILES = ("scores.json", "scores_train.json", "scores_test.json",
                 "diagnostics.json", "diagnostics_train.json", "diagnostics_test.json")


def install_case(case: str) -> None:
    kind, module, factory, options, extra, _ = CASES[case]
    shutil.rmtree(ROOT / "outputs", ignore_errors=True)
    folder = ROOT / "outputs" / "candidate"
    folder.mkdir(parents=True)
    payload = {"method": "our_method", "module": module, "factory": factory, "options": options}
    payload.update(extra)
    (folder / "policy.json").write_text(json.dumps(payload, indent=2))


def evaluator_hash() -> str:
    h = hashlib.sha256()
    for rel in ("eval/test/evaluate.py", "eval/test/comparator_test.json", "scripts/evaluate_test.sh",
                "eval/train/evaluate.py", "eval/train/comparator_train.json", "scripts/evaluate_train.sh"):
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()


def rank_gain(scores: dict, reference: dict, rubric: dict) -> float:
    total = 0.0
    for exp, cfg in rubric["experiments"].items():
        actual = scores["experiments"][exp]["results"]["our_method"]
        target = reference["experiments"][exp]["results"]["our_method"]
        total += sum(cfg["weight"] * spec["coefficient"] * (actual[m] - target[m]) for m, spec in cfg["metrics"].items())
    return float(total)


def metrics_gain(metrics: dict, reference: dict, rubric: dict) -> float:
    total = 0.0
    for exp, cfg in rubric["experiments"].items():
        total += sum(cfg["weight"] * spec["coefficient"] * (metrics[exp][m] - reference[exp][m])
                     for m, spec in cfg["metrics"].items())
    return float(total)


def _arrays(diag: dict, exp: str) -> dict:
    per = diag["experiments"][exp]["our_method"]["per_instance"]
    by_n = defaultdict(list)
    for r in sorted(per, key=lambda r: r["key"]):
        by_n[r["n"]].append(r)
    return {n: {k: np.array([r[k] for r in rs], dtype=float) for k in ("cce", "final_gap", "comparator", "uniform")}
            for n, rs in sorted(by_n.items())}


def _boot_metrics(arr: dict, idx: dict) -> dict:
    ratio = np.mean([arr[n]["cce"][idx[n]].sum(axis=1) / arr[n]["comparator"][idx[n]].sum(axis=1) for n in arr], axis=0)
    rel = np.concatenate([arr[n]["cce"][idx[n]] / arr[n]["uniform"][idx[n]] for n in arr], axis=1).mean(axis=1)
    gap = np.concatenate([arr[n]["final_gap"][idx[n]] / arr[n]["uniform"][idx[n]] for n in arr], axis=1).mean(axis=1)
    return {"comparator_exploitability_ratio": ratio, "relative_exploitability": rel, "relative_final_nash_gap": gap}


def paired_bootstrap(case_diag: dict, ref_diag: dict, rubric: dict, seed: int = 0) -> dict:
    """Bootstrap of the rank gain over instances (resampled within each player count, paired)."""
    rng = np.random.default_rng(seed)
    gain = np.zeros(BOOTSTRAP)
    primary = {}
    for exp, cfg in rubric["experiments"].items():
        c, r = _arrays(case_diag, exp), _arrays(ref_diag, exp)
        idx = {n: rng.integers(0, len(c[n]["cce"]), size=(BOOTSTRAP, len(c[n]["cce"]))) for n in c}
        mc, mr = _boot_metrics(c, idx), _boot_metrics(r, idx)
        for m, spec in cfg["metrics"].items():
            if m in mc:
                gain += cfg["weight"] * spec["coefficient"] * (mc[m] - mr[m])
        d = mc["comparator_exploitability_ratio"] - mr["comparator_exploitability_ratio"]
        primary[exp] = {"se": float(d.std()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return {"rank_gain_se": float(gain.std()),
            "rank_gain_ci95": [float(np.percentile(gain, 2.5)), float(np.percentile(gain, 97.5))],
            "primary_ratio_difference": primary}


def run_slice(case: str, split: str, folder: Path) -> dict:
    for name in (f"scores_{split}.json", f"diagnostics_{split}.json"):
        (ROOT / "scoring" / name).unlink(missing_ok=True)
    start = time.time()
    proc = subprocess.run(["bash", f"scripts/evaluate_{split}.sh"], cwd=ROOT, capture_output=True, text=True)
    out = {"exit": proc.returncode, "seconds": round(time.time() - start, 1)}
    (folder / f"evaluate_{split}.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
    if proc.returncode == 0:
        for name in (f"scores_{split}.json", f"diagnostics_{split}.json"):
            shutil.copy2(ROOT / "scoring" / name, folder / name)
        diag = json.loads((folder / f"diagnostics_{split}.json").read_text())
        out["status"] = {e: d["our_method"]["status"] for e, d in diag["experiments"].items()}
        out["failure_reasons"] = {e: d["our_method"]["failure_reasons"] for e, d in diag["experiments"].items()}
        out["failure_examples"] = {e: d["our_method"]["failure_examples"][:2] for e, d in diag["experiments"].items()}
        out["locality_audit"] = diag["_meta"]["locality_audit"].get("our_method")
        out["policy_run"] = {k: v for k, v in diag["_meta"]["policy_runs"].get("our_method", {}).items()
                             if k != "log_tail"}
    return out


def score_case(case: str) -> dict:
    folder = RESULTS / case
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    kind, module, factory, options, extra, description = CASES[case]
    install_case(case)
    shutil.copy2(ROOT / "outputs" / "candidate" / "policy.json", folder / "policy.json")
    before = evaluator_hash()
    result = {"case": case, "kind": kind, "description": description}
    for split in ("test", "train"):
        result[split] = run_slice(case, split, folder)
    result["evaluator_unchanged"] = evaluator_hash() == before
    (folder / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"case": case, "test": result["test"].get("status"), "train": result["train"].get("status"),
                      "seconds": [result["test"]["seconds"], result["train"]["seconds"]]}), flush=True)
    return result


def summarize() -> dict:
    rubric = json.loads((ROOT / "scoring" / "reference.json").read_text())
    ref = {s: json.loads((RESULTS / "reference" / f"scores_{s}.json").read_text()) for s in ("test", "train")}
    ref_diag = {s: json.loads((RESULTS / "reference" / f"diagnostics_{s}.json").read_text()) for s in ("test", "train")}
    summary = {}
    for folder in sorted(p for p in RESULTS.iterdir() if (p / "result.json").exists()):
        result = json.loads((folder / "result.json").read_text())
        entry = {"kind": result["kind"], "description": result["description"],
                 "evaluator_unchanged": result["evaluator_unchanged"]}
        for split in ("test", "train"):
            sfile = folder / f"scores_{split}.json"
            if not sfile.exists():
                entry[split] = {"exit": result[split]["exit"]}
                continue
            scores = json.loads(sfile.read_text())
            diag = json.loads((folder / f"diagnostics_{split}.json").read_text())
            block = {"rank_gain": rank_gain(scores, ref[split], rubric),
                     "metrics": {e: b["results"]["our_method"] for e, b in scores["experiments"].items()},
                     "status": result[split].get("status"), "failure_reasons": result[split].get("failure_reasons"),
                     "failure_examples": result[split].get("failure_examples"),
                     "locality_audit": result[split].get("locality_audit")}
            unaudited = {e: d["our_method"].get("unaudited_metrics") for e, d in diag["experiments"].items()}
            if all(unaudited.values()):
                ref_metrics = {e: b["results"]["our_method"] for e, b in ref[split]["experiments"].items()}
                block["unaudited_rank_gain"] = metrics_gain(unaudited, ref_metrics, rubric)
                block["unaudited_metrics"] = unaudited
            if all(s == "ok" for s in (block["status"] or {}).values()) and folder.name != "reference":
                block["paired_bootstrap"] = paired_bootstrap(diag, ref_diag[split], rubric)
            entry[split] = block
        summary[folder.name] = entry
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\n{'case':42s} {'test gain':>12s} {'(SE)':>8s} {'train gain':>12s}  status / unaudited test gain")
    for case, e in sorted(summary.items(), key=lambda kv: -(kv[1]["test"].get("rank_gain", -1e9))):
        t, tr = e["test"], e["train"]
        se = t.get("paired_bootstrap", {}).get("rank_gain_se")
        note = "ok" if all(s == "ok" for s in (t.get("status") or {"x": "?"}).values()) else \
            f"FAILED {sorted({k for v in (t.get('failure_reasons') or {}).values() for k in v})}"
        if "unaudited_rank_gain" in t:
            note += f"  unaudited {t['unaudited_rank_gain']:+.4f}"
        print(f"{case:42s} {t.get('rank_gain', float('nan')):+12.5f} {se if se is not None else float('nan'):8.4f} "
              f"{tr.get('rank_gain', float('nan')):+12.5f}  {note}", flush=True)
    return summary


def main(selected: list[str]) -> None:
    backup = HERE / "_backup"
    shutil.rmtree(backup, ignore_errors=True)
    backup.mkdir(parents=True)
    if (ROOT / "outputs").exists():
        shutil.move(str(ROOT / "outputs"), backup / "outputs")
    for name in SCORING_FILES:
        if (ROOT / "scoring" / name).exists():
            shutil.copy2(ROOT / "scoring" / name, backup / name)
    try:
        cases = selected or list(CASES)
        if "reference" not in cases and not (RESULTS / "reference" / "result.json").exists():
            cases = ["reference"] + cases
        for case in cases:
            score_case(case)
        summarize()
    finally:
        shutil.rmtree(ROOT / "outputs", ignore_errors=True)
        if (backup / "outputs").exists():
            shutil.move(str(backup / "outputs"), ROOT / "outputs")
        for name in SCORING_FILES:
            if (backup / name).exists():
                shutil.copy2(backup / name, ROOT / "scoring" / name)
            else:
                (ROOT / "scoring" / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1:])
