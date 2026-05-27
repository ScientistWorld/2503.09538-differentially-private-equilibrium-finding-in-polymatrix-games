"""Evaluate saved method artifacts without importing method/."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.polymatrix import cce_exploitability, load_game, nash_gap

SCORED_TOPOLOGIES = {"dense_er", "dense_cluster", "sparse_knn"}


def load_manifests(paths: list[str]) -> list[dict]:
    runs = []
    for path in paths:
        manifest = json.loads(Path(path).read_text())
        runs.extend(manifest["runs"])
    return runs


def summarize(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def _trend_slope(vals: list[tuple[int, float, float]], index: int) -> float:
    vals = sorted(vals)
    ns = np.array([v[0] for v in vals], dtype=np.float64)
    metric = np.array([max(v[index], 1e-12) for v in vals], dtype=np.float64)
    if len(vals) <= 1:
        return 0.0
    return float(np.polyfit(np.log(ns), np.log(metric), 1)[0])


def evaluate_runs(runs: list[dict]) -> tuple[dict, dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["topology"], run["method"], int(run["n_players"]))].append(run)

    by_experiment: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {}
    aggregate: dict[str, dict[str, list[tuple[int, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for (topology, method, n_players), group in sorted(grouped.items()):
        exp_name = f"{topology}_n{n_players}"
        method_name = "our_method" if method == "algorithm1" else method
        cce_vals = []
        gap_vals = []
        privacy_vals = []
        steps_vals = []
        for run in group:
            game = load_game(run["game_path"])
            data = np.load(run["trajectory_path"])
            traj = data["trajectory"]
            cce, _ = cce_exploitability(game, traj)
            gap, _ = nash_gap(game, traj[-1])
            cce_vals.append(cce)
            gap_vals.append(gap)
            privacy_vals.append(float(run["privacy_bound"]))
            steps_vals.append(float(run["steps"]))
        cce_mean = summarize(cce_vals)["mean"]
        privacy_mean = summarize(privacy_vals)["mean"]
        aggregate[topology][method_name].append((n_players, cce_mean, privacy_mean))

    for topology, methods in aggregate.items():
        if topology not in SCORED_TOPOLOGIES:
            continue
        exp_name = f"{topology}_scaling"
        by_experiment.setdefault(exp_name, {"results": {}})
        baseline_cce_by_n = {n: cce for n, cce, _ in methods.get("noisy_omd", [])}
        for method_name, vals in methods.items():
            if method_name != "our_method":
                continue
            vals = sorted(vals)
            ns = np.array([v[0] for v in vals], dtype=np.float64)
            cce = np.array([max(v[1], 1e-12) for v in vals], dtype=np.float64)
            privacy = np.array([max(v[2], 1e-12) for v in vals], dtype=np.float64)
            cce_slope = _trend_slope(vals, 1)
            privacy_slope = _trend_slope(vals, 2)
            ratios = [c / baseline_cce_by_n[n] for n, c, _ in vals if n in baseline_cce_by_n and baseline_cce_by_n[n] > 0]
            ratio = float(np.mean(ratios)) if ratios else 1.0
            diagnostics[exp_name] = {
                method_name: {
                    "raw_cce_slope": cce_slope,
                    "raw_privacy_slope": privacy_slope,
                    "raw_baseline_cce_ratio": ratio,
                }
            }
            # Paper figures report trend direction rather than tabulated slopes.
            # Score the qualitative claims as binary indicators.  Raw slopes are
            # emitted to scoring/diagnostics.json because scores.json must only
            # contain declared reference metrics.
            if topology.startswith("sparse"):
                by_experiment[exp_name]["results"][method_name] = {
                    "cce_stable_or_decreases_with_n": float(cce_slope <= 0.05),
                    "privacy_decreases_with_n": float(privacy_slope < 0.0),
                    "competitive_with_baseline": float(ratio <= 1.10),
                }
            else:
                by_experiment[exp_name]["results"][method_name] = {
                    "cce_decreases_with_n": float(cce_slope < 0.0),
                    "privacy_decreases_with_n": float(privacy_slope < 0.0),
                    "competitive_with_baseline": float(ratio <= 1.10),
                }
    return {"experiments": by_experiment}, {"experiments": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", default=["outputs/proposed/manifest.json", "outputs/baseline/manifest.json"])
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()
    existing = [p for p in args.manifest if Path(p).exists()]
    if not existing:
        raise FileNotFoundError("No manifest files found to evaluate.")
    scores, diagnostics = evaluate_runs(load_manifests(existing))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(scores, indent=2, sort_keys=True))
    Path(args.output).with_name("diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
