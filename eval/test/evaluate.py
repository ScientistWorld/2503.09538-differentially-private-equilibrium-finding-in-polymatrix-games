"""Held-out test-slice evaluator for saved polymatrix-game method outputs.

This module reads method artifacts from manifest files under outputs/.  It
does not import method/ or rerun any algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from data.polymatrix import cce_exploitability, load_game, nash_gap, theorem8_privacy_bound

SLICE = "test"
SCORED_TOPOLOGIES = {"dense_er", "dense_cluster", "sparse_knn"}
BASELINE_METHOD = "noisy_omd"
SEED_RE = re.compile(r"_seed(\d+)")


def load_manifests(paths: list[str]) -> list[dict]:
    runs = []
    for path in paths:
        manifest = json.loads(Path(path).read_text())
        runs.extend(manifest["runs"])
    return runs


def canonical_method_name(name: str) -> str:
    return "our_method" if name == "algorithm1" else name


def run_seed(run: dict) -> int:
    text = f"{run.get('game_path', '')} {run.get('trajectory_path', '')}"
    match = SEED_RE.search(text)
    if match:
        return int(match.group(1))
    digest = hashlib.sha256(f"{run['topology']}|{run['method']}|{run['n_players']}|{text}".encode()).hexdigest()
    return int(digest[:12], 16)


def deterministic_privacy_bound(game, run: dict) -> float:
    """Recompute the benchmark privacy bound instead of trusting manifest values."""
    count = min(8, len(game.edges))
    if count == 0:
        return 0.0
    rng = np.random.default_rng(4000 + 31 * run_seed(run) + int(run["n_players"]))
    indices = rng.choice(len(game.edges), size=count, replace=False)
    bounds = [
        theorem8_privacy_bound(
            game,
            float(run["eta"]),
            float(run.get("alpha", 2.0)),
            game.edges[int(idx)],
            int(run["steps"]),
        )
        for idx in indices
    ]
    return float(np.mean(bounds))


def simplex_violation(trajectory: np.ndarray) -> float:
    row_sum_error = np.abs(np.sum(trajectory, axis=-1) - 1.0)
    negative_mass = np.maximum(-trajectory, 0.0)
    excess_mass = np.maximum(trajectory - 1.0, 0.0)
    return float(row_sum_error.mean() + negative_mass.mean() + excess_mass.mean())


def select_slice_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for run in runs:
        key = (run["topology"], run["method"], int(run["n_players"]))
        grouped[key].append(run)

    selected = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: (run_seed(r), r.get("trajectory_path", "")))
        selected.extend(run for idx, run in enumerate(ordered) if idx % 2 == 1)
    return selected


def summarize(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def trend_slope(vals: list[tuple[int, float]], floor: float = 1e-12) -> float:
    vals = sorted(vals)
    if len(vals) <= 1:
        return 0.0
    ns = np.array([v[0] for v in vals], dtype=np.float64)
    metric = np.array([max(v[1], floor) for v in vals], dtype=np.float64)
    return float(np.polyfit(np.log(ns), np.log(metric), 1)[0])


def trend_score(slope: float) -> float:
    return float(np.clip(1.0 - 0.1 * slope, 0.0, 2.0))


def stable_score(slope: float) -> float:
    return trend_score(slope)


def expected_methods_from_train(path: str | None) -> dict[str, set[str]]:
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    experiments = data.get("experiments", data)
    expected: dict[str, set[str]] = {}
    for exp_name, exp_data in experiments.items():
        if exp_name.startswith("_") or not isinstance(exp_data, dict):
            continue
        results = exp_data.get("results", exp_data)
        if isinstance(results, dict):
            expected[exp_name] = {name for name in results if not name.startswith("_")}
    return expected


def enforce_expected_methods(scores: dict, expected: dict[str, set[str]]) -> None:
    if not expected:
        return
    experiments = scores.get("experiments", {})
    errors = []
    for exp_name, method_names in expected.items():
        results = experiments.get(exp_name, {}).get("results", {})
        missing = sorted(method_names - set(results))
        if missing:
            errors.append(f"{exp_name}: missing method(s) from test slice: {missing}")
        extra = sorted(set(results) - method_names)
        for method in extra:
            del results[method]
    if errors:
        raise ValueError("; ".join(errors))


def evaluate_runs(runs: list[dict]) -> tuple[dict, dict]:
    selected_runs = select_slice_runs(runs)
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for run in selected_runs:
        grouped[(run["topology"], run["method"], int(run["n_players"]))].append(run)

    aggregate: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    per_group: dict[str, dict] = {}
    for (topology, method, n_players), group in sorted(grouped.items()):
        cce_vals = []
        gap_vals = []
        privacy_vals = []
        simplex_vals = []
        for run in group:
            game = load_game(run["game_path"])
            data = np.load(run["trajectory_path"])
            trajectory = data["trajectory"]
            if trajectory.ndim != 3 or trajectory.shape[1:] != (game.n_players, game.n_actions):
                raise ValueError(
                    f"bad trajectory shape for {run['trajectory_path']}: "
                    f"expected (*, {game.n_players}, {game.n_actions}), got {trajectory.shape}"
                )
            if not np.isfinite(trajectory).all():
                raise ValueError(f"non-finite trajectory values in {run['trajectory_path']}")
            cce, _ = cce_exploitability(game, trajectory)
            gap, _ = nash_gap(game, trajectory[-1])
            cce_vals.append(cce)
            gap_vals.append(gap)
            privacy_vals.append(deterministic_privacy_bound(game, run))
            simplex_vals.append(simplex_violation(trajectory))

        method_name = canonical_method_name(method)
        cce_summary = summarize(cce_vals)
        gap_summary = summarize(gap_vals)
        privacy_summary = summarize(privacy_vals)
        simplex_summary = summarize(simplex_vals)
        aggregate[topology][method_name].append(
            {
                "n_players": n_players,
                "cce": cce_summary["mean"],
                "nash_gap": gap_summary["mean"],
                "privacy": privacy_summary["mean"],
                "simplex_violation": simplex_summary["mean"],
                "runs": cce_summary["n"],
            }
        )
        per_group[f"{topology}/{method_name}/n{n_players}"] = {
            "slice": SLICE,
            "runs": cce_summary["n"],
            "cce_exploitability": cce_summary,
            "final_nash_gap": gap_summary,
            "privacy_budget": privacy_summary,
            "strategy_simplex_violation": simplex_summary,
        }

    by_experiment: dict[str, dict] = {}
    diagnostics: dict[str, dict] = {"_meta": {"slice": SLICE, "selected_runs": len(selected_runs)}, "groups": per_group}
    for topology, methods in aggregate.items():
        exp_name = f"{topology}_scaling"
        if topology not in SCORED_TOPOLOGIES:
            continue

        baseline_cce_by_n = {
            int(v["n_players"]): float(v["cce"]) for v in methods.get(BASELINE_METHOD, [])
        }
        by_experiment.setdefault(exp_name, {"results": {}})
        diagnostics.setdefault("experiments", {}).setdefault(exp_name, {"slice": SLICE, "methods": {}})

        for method_name, vals in sorted(methods.items()):
            if method_name == BASELINE_METHOD:
                continue
            vals = sorted(vals, key=lambda x: int(x["n_players"]))
            if len(vals) < 2:
                continue

            cce_slope = trend_slope([(int(v["n_players"]), float(v["cce"])) for v in vals])
            privacy_slope = trend_slope([(int(v["n_players"]), float(v["privacy"])) for v in vals])
            gap_slope = trend_slope([(int(v["n_players"]), float(v["nash_gap"])) for v in vals])
            ratios = [
                float(v["cce"]) / baseline_cce_by_n[int(v["n_players"])]
                for v in vals
                if int(v["n_players"]) in baseline_cce_by_n and baseline_cce_by_n[int(v["n_players"])] > 0
            ]
            baseline_ratio = float(np.mean(ratios)) if ratios else 1.0
            cce_mean = float(np.mean([float(v["cce"]) for v in vals]))
            privacy_mean = float(np.mean([float(v["privacy"]) for v in vals]))
            gap_mean = float(np.mean([float(v["nash_gap"]) for v in vals]))
            simplex_mean = float(np.mean([float(v["simplex_violation"]) for v in vals]))

            if topology.startswith("sparse"):
                primary_cce_metric = {"cce_stable_or_decreases_with_n": stable_score(cce_slope)}
            else:
                primary_cce_metric = {"cce_decreases_with_n": trend_score(cce_slope)}

            by_experiment[exp_name]["results"][method_name] = {
                **primary_cce_metric,
                "privacy_decreases_with_n": trend_score(privacy_slope),
                "competitive_with_baseline": float(baseline_ratio <= 1.10),
                "cce_exploitability_mean": cce_mean,
                "privacy_budget_mean": privacy_mean,
                "final_nash_gap_mean": gap_mean,
                "baseline_cce_ratio": baseline_ratio,
                "strategy_simplex_violation_mean": simplex_mean,
            }
            diagnostics["experiments"][exp_name]["methods"][method_name] = {
                "slice": SLICE,
                "raw_cce_slope": cce_slope,
                "raw_privacy_slope": privacy_slope,
                "raw_nash_gap_slope": gap_slope,
                "raw_baseline_cce_ratio": baseline_ratio,
            }

    return {"experiments": by_experiment}, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--match-train", default="scoring/scores_train.json")
    args = parser.parse_args()

    existing = [p for p in args.manifest if Path(p).exists()]
    if not existing:
        raise FileNotFoundError("No manifest files found to evaluate.")

    scores, diagnostics = evaluate_runs(load_manifests(existing))
    enforce_expected_methods(scores, expected_methods_from_train(args.match_train))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2, sort_keys=True))
    output_path.with_name("diagnostics_test.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
