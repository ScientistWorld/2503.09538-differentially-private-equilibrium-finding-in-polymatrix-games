"""Visible train-slice evaluator for private polymatrix-game equilibrium finding.

Self-contained (numpy only).  It regenerates the train games, the privacy
operating point and the fixed broadcast-noise stream itself, verifies every
submitted transcript against the communication protocol, and scores
exploitability against comparator values stored next to this file.  It never
imports method/, data/ or eval/test/, and it reads nothing a method writes
except the instance identity and the ``trajectory``/``broadcasts`` arrays.

Usage:
    python -m eval.train.evaluate --manifest outputs/*/*/manifest.json
    python -m eval.train.evaluate --build-comparator      # regenerate fixed values
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SLICE = "train"
SEEDS = (0, 2, 4, 6, 8, 10)
COMPARATOR_PATH = Path(__file__).resolve().parent / "comparator_train.json"

EXPERIMENTS = {
    "dense_er_scaling": "dense_er",
    "dense_cluster_scaling": "dense_cluster",
    "sparse_knn_scaling": "sparse_knn",
}
SIZES = (64, 128, 256, 512)
N_ACTIONS = 5
EDGE_PROB = 0.12
SPARSE_DEGREE = 4
RENYI_ORDER = 2.0
NOISE_BASE = 20250311
TOPOLOGY_CODE = {"dense_er": 1, "dense_cluster": 2, "sparse_random": 3, "sparse_knn": 4}

# A failed experiment scores FAIL_VALUE on every exploitability metric; valid
# values are clipped to it, so no valid submission can score below a failed one.
FAIL_VALUE = 10.0
SIMPLEX_TOL = 1e-6
NOISE_TOL = 1e-8
MOVE_REL_TOL = 1e-6
MOVE_ABS_TOL = 1e-8
METRICS = (
    "comparator_exploitability_ratio",
    "relative_exploitability",
    "relative_final_nash_gap",
    "privacy_protocol_compliance",
)


# ---------------------------------------------------------------------------
# Instances: games, operating point, broadcast noise
# ---------------------------------------------------------------------------
class Game:
    def __init__(self, n: int, a: int, edges: np.ndarray, utilities: np.ndarray):
        self.n, self.a, self.edges = n, a, edges
        deg = np.zeros(n, dtype=np.int64)
        np.add.at(deg, edges[:, 0], 1)
        np.add.at(deg, edges[:, 1], 1)
        self.deg = np.maximum(deg.astype(np.float64), 1.0)
        self.fingerprint = hashlib.sha256(edges.astype(np.int64).tobytes() + utilities.tobytes()).hexdigest()[:20]
        self.op = np.ascontiguousarray(utilities.transpose(0, 2, 1, 3).reshape(n * a, n * a))


def _no_isolates(n: int, edges: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    edge_set = {tuple(map(int, e)) for e in np.asarray(edges, dtype=np.int64).reshape(-1, 2)}
    deg = np.zeros(n, dtype=np.int64)
    for i, j in edge_set:
        deg[i] += 1
        deg[j] += 1
    for i in np.where(deg == 0)[0]:
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        edge_set.add((min(i, j), max(i, j)))
    return np.array(sorted(edge_set), dtype=np.int64)


def _erdos_renyi(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    rows, cols = np.triu_indices(n, k=1)
    mask = rng.random(rows.shape[0]) < p
    return _no_isolates(n, np.stack([rows[mask], cols[mask]], axis=1), rng)


def _clustered(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    n_clusters = max(1, int(np.floor(1.0 / p)))
    labels = rng.integers(0, n_clusters, size=n)
    edge_set: set[tuple[int, int]] = set()
    cross_p = min(1.0, 10.0 * p / n)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j] or rng.random() < cross_p:
                edge_set.add((i, j))
    return _no_isolates(n, np.array(sorted(edge_set), dtype=np.int64), rng)


def _knn(n: int, c: int, rng: np.random.Generator) -> np.ndarray:
    points = rng.random((n, 2))
    dist = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(dist, np.inf)
    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        for j in np.argpartition(dist[i], c)[:c]:
            edge_set.add((min(i, int(j)), max(i, int(j))))
    return _no_isolates(n, np.array(sorted(edge_set), dtype=np.int64), rng)


def make_game(topology: str, n: int, seed: int) -> Game:
    rng = np.random.default_rng(1000 + 17 * seed + n)
    if topology == "dense_er":
        edges = _erdos_renyi(n, EDGE_PROB, rng)
    elif topology == "dense_cluster":
        edges = _clustered(n, EDGE_PROB, rng)
    elif topology == "sparse_knn":
        edges = _knn(n, SPARSE_DEGREE, rng)
    else:
        raise ValueError(topology)
    utilities = rng.uniform(-1.0, 1.0, size=(n, n, N_ACTIONS, N_ACTIONS))
    mask = np.zeros((n, n), dtype=bool)
    mask[edges[:, 0], edges[:, 1]] = True
    mask[edges[:, 1], edges[:, 0]] = True
    utilities[~mask] = 0.0
    return Game(n, N_ACTIONS, edges, utilities)


def operating_point(n: int, topology: str) -> tuple[int, float, float]:
    if topology.startswith("dense"):
        exponent = max(0.2, min(1.0, np.log(max(EDGE_PROB * n, 2.0)) / np.log(n)))
        steps = max(8, int(round((n ** (8.0 * exponent / 9.0)) / (np.log(n) ** 2))))
    else:
        sparse_steps = (1.0 - np.log(np.log(n)) / np.log(n)) * np.log(n) / np.log(max(SPARSE_DEGREE, 2))
        steps = max(2, int(round(sparse_steps)))
    return steps, 1.0 / (steps ** (1.0 / 3.0)), 1.0 / np.sqrt(steps)


def noise_stream(topology: str, n: int, seed: int, steps: int, sigma: float, key: int = 0) -> np.ndarray:
    rng = np.random.default_rng([NOISE_BASE, int(key), TOPOLOGY_CODE[topology], int(n), int(seed)])
    return rng.normal(0.0, sigma, size=(steps, n, N_ACTIONS))


def noise_fingerprint(noise: np.ndarray) -> str:
    return hashlib.sha256(noise.tobytes()).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Game quantities
# ---------------------------------------------------------------------------
def project_simplex(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    u = np.sort(flat, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, flat.shape[1] + 1)
    cond = u - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(flat.shape[0]), rho] / (rho + 1)
    return np.maximum(flat - theta[:, None], 0.0).reshape(arr.shape)


def gradients(game: Game, strategies: np.ndarray) -> np.ndarray:
    s = np.asarray(strategies, dtype=np.float64)
    flat = s.reshape(-1, game.n * game.a)
    return -(flat @ game.op.T).reshape(s.shape) / game.deg[:, None]


def cce_exploitability(game: Game, trajectory: np.ndarray) -> float:
    grads = gradients(game, trajectory)
    average_current = np.mean(np.sum(grads * trajectory, axis=2), axis=0)
    best_fixed = np.mean(grads, axis=0).min(axis=1)
    return float(np.maximum(average_current - best_fixed, 0.0).mean())


def nash_gap(game: Game, strategies: np.ndarray) -> float:
    grads = gradients(game, strategies)
    return float(np.maximum(np.sum(grads * strategies, axis=1) - grads.min(axis=1), 0.0).mean())


def uniform_profile(game: Game) -> np.ndarray:
    return np.full((game.n, game.a), 1.0 / game.a)


# ---------------------------------------------------------------------------
# Protocol verification
# ---------------------------------------------------------------------------
def check_transcript(game: Game, steps: int, eta: float, noise: np.ndarray,
                     trajectory: np.ndarray, broadcasts: np.ndarray) -> tuple[bool, str, dict]:
    """Verify that a transcript is an execution of the private protocol.

    1. exactly ``steps`` rounds of strategies and broadcasts for every player;
    2. every strategy is a probability vector;
    3. every broadcast equals the profile played before that round (uniform
       before round 0) plus the fixed instance noise;
    4. every update stays within ``eta`` times the norm of the player's
       centred payoff gradient (against all projected broadcasts) of the
       segment from its own projected broadcast P towards the uniform
       strategy u, ending at (P + eta * u) / (1 + eta), i.e. a payoff-free
       shrink of at most a proximal weight of one.  This bounds how much a
       player's next broadcast can depend on its private payoffs beyond public
       information to O(eta) per round, which together with the fixed noise
       fixes the privacy budget of every compliant submission.
    """
    shape = (steps, game.n, game.a)
    if trajectory.shape != shape or broadcasts.shape != shape:
        return False, f"shape: expected {shape}, got {trajectory.shape}/{broadcasts.shape}", {}
    if not (np.isfinite(trajectory).all() and np.isfinite(broadcasts).all()):
        return False, "non-finite values", {}
    if trajectory.min() < -SIMPLEX_TOL or np.abs(trajectory.sum(axis=2) - 1.0).max() > SIMPLEX_TOL:
        return False, "strategy outside the probability simplex", {}
    uniform = uniform_profile(game)
    played_before = np.concatenate([uniform[None], trajectory[:-1]], axis=0)
    noise_error = float(np.abs(broadcasts - played_before - noise).max())
    if noise_error > NOISE_TOL:
        return False, f"broadcast is not played strategy + instance noise (max error {noise_error:.3g})", {}
    anchors = project_simplex(broadcasts)
    grads = gradients(game, anchors)
    centred = grads - grads.mean(axis=2, keepdims=True)
    cap = eta * np.linalg.norm(centred, axis=2)
    min_keep = 1.0 / (1.0 + eta)
    shrunk = min_keep * anchors + (1.0 - min_keep) * uniform
    direction = anchors - shrunk
    offset = trajectory - shrunk
    dd = np.sum(direction * direction, axis=2)
    lam = np.clip(np.sum(offset * direction, axis=2) / np.where(dd > 0, dd, 1.0), 0.0, 1.0)
    lam = np.where(dd > 0, lam, 0.0)
    distance = np.linalg.norm(offset - lam[..., None] * direction, axis=2)
    utilization = distance / np.maximum(cap, 1e-12)
    stats = {"max_step_utilization": float(utilization.max()), "mean_step_utilization": float(utilization.mean())}
    excess = distance - (cap * (1.0 + MOVE_REL_TOL) + MOVE_ABS_TOL)
    if (excess > 0).any():
        t, i = np.unravel_index(int(np.argmax(excess)), excess.shape)
        return False, (f"update exceeds the step allowance (round {t}, player {i}: "
                       f"{distance[t, i]:.4g} > {cap[t, i]:.4g})"), stats
    return True, "ok", stats


# ---------------------------------------------------------------------------
# Comparator (fixed reference values) and privacy accounting diagnostics
# ---------------------------------------------------------------------------
def _locality_fraction_max(game: Game, steps: int) -> float:
    n = game.n
    adj = np.zeros((n, n), dtype=np.float32)
    adj[game.edges[:, 0], game.edges[:, 1]] = 1.0
    adj[game.edges[:, 1], game.edges[:, 0]] = 1.0
    dist = np.full((n, n), steps + 1, dtype=np.int16)
    reached = np.eye(n, dtype=bool)
    dist[reached] = 0
    for depth in range(1, steps + 1):
        new = ((reached.astype(np.float32) @ adj) > 0) & ~reached
        if not new.any():
            break
        dist[new] = depth
        reached |= new
    near = np.minimum(dist[game.edges[:, 0]], dist[game.edges[:, 1]]) < steps
    return float(near.sum(axis=1).max() / n)


def _structure_free_run(game: Game, steps: int, step: float, noise: np.ndarray) -> np.ndarray:
    strategies = uniform_profile(game)
    trajectory = []
    for t in range(steps):
        anchor = project_simplex(strategies + noise[t])
        strategies = project_simplex(anchor - step * gradients(game, anchor))
        trajectory.append(strategies.copy())
    return np.stack(trajectory, axis=0)


def build_comparator(path: Path) -> None:
    instances = {}
    for topology in EXPERIMENTS.values():
        for n in SIZES:
            steps, eta, sigma = operating_point(n, topology)
            for seed in SEEDS:
                game = make_game(topology, n, seed)
                noise = noise_stream(topology, n, seed, steps, sigma)
                a = game.a
                hdeg = float(n / np.sum(1.0 / game.deg))
                club = float(16.0 * a ** 3 * np.log(n) ** 2 / hdeg ** (4.0 / 9.0) + 4.0 * a / n)
                spade_max = 2.0 * a * _locality_fraction_max(game, steps)
                step = float(eta * min(1.0, np.sqrt(min(club, spade_max) / (2.0 * a))))
                trajectory = _structure_free_run(game, steps, step, noise)
                instances[f"{topology}/n{n}/seed{seed}"] = {
                    "game_fingerprint": game.fingerprint,
                    "noise_fingerprint": noise_fingerprint(noise),
                    "steps": steps, "eta": eta, "sigma": sigma,
                    "comparator_step": step,
                    "comparator_cce": cce_exploitability(game, trajectory),
                    "comparator_final_nash_gap": nash_gap(game, trajectory[-1]),
                    "uniform_exploitability": nash_gap(game, uniform_profile(game)),
                    "dense_sensitivity_term": club,
                    "locality_term_max": spade_max,
                    "renyi_budget_theorem": float(RENYI_ORDER * eta ** 2 / sigma ** 2 * min(club, spade_max) * steps),
                    "renyi_budget_regime_branch": float(RENYI_ORDER * eta ** 2 * (club if hdeg > np.log(n) else spade_max)),
                }
                print(f"comparator {topology} n={n} seed={seed} step={step:.4f} "
                      f"cce={instances[f'{topology}/n{n}/seed{seed}']['comparator_cce']:.5f}", flush=True)
    payload = {
        "slice": SLICE,
        "about": ("Fixed comparator: structure-free private projected-gradient play on the same noise and "
                  "rounds, step calibrated to the operating point's per-instance privacy budget."),
        "instances": instances,
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------
def load_runs(manifest_paths: list[str]) -> tuple[dict, dict]:
    by_method: dict[str, dict] = defaultdict(dict)
    duplicates: dict[str, set] = defaultdict(set)
    for manifest_path in manifest_paths:
        mpath = Path(manifest_path)
        try:
            manifest = json.loads(mpath.read_text())
        except (OSError, ValueError) as exc:
            print(f"warning: unreadable manifest {mpath}: {exc}")
            continue
        if not isinstance(manifest, dict):
            continue
        default_name = manifest.get("method") or mpath.parent.name
        for run in manifest.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            name = str(run.get("method", default_name))
            try:
                key = (str(run["topology"]), int(run["n_players"]), int(run["seed"]))
            except (KeyError, TypeError, ValueError):
                continue
            raw = Path(str(run.get("trajectory_path", "")))
            path = raw if (raw.is_absolute() or raw.exists()) else mpath.parent / raw
            if key in by_method[name]:
                duplicates[name].add(key)
            by_method[name][key] = path
    return by_method, duplicates


def _load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = []
        for name in ("trajectory", "broadcasts"):
            arr = data[name]
            if arr.dtype.kind not in "fiu":
                raise TypeError(f"{name} has dtype {arr.dtype}")
            arrays.append(np.array(arr, dtype=np.float64))
    return arrays[0], arrays[1]


def _slope(xs: list[float], ys: list[float]) -> float:
    ys = np.maximum(np.asarray(ys, dtype=np.float64), 1e-12)
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.log(np.asarray(xs, dtype=np.float64)), np.log(ys), 1)[0])


def summarize(records: list[dict], comparator: dict, topology: str) -> tuple[dict, dict]:
    required = len(SIZES) * len(SEEDS)
    valid = [r for r in records if r.get("ok")]
    compliance = len(valid) / required
    by_n = defaultdict(list)
    for r in valid:
        by_n[r["n"]].append(r)
    per_n = {}
    for n, rs in sorted(by_n.items()):
        cce = np.array([r["cce"] for r in rs])
        comp = np.array([r["comparator"] for r in rs])
        uni = np.array([r["uniform"] for r in rs])
        ref = [comparator["instances"][f"{topology}/n{n}/seed{r['seed']}"] for r in rs]
        per_n[str(n)] = {
            "runs": len(rs),
            "cce_exploitability_mean": float(cce.mean()),
            "comparator_cce_mean": float(comp.mean()),
            "uniform_exploitability_mean": float(uni.mean()),
            "comparator_ratio": float(cce.sum() / comp.sum()),
            "relative_exploitability": float(np.mean(cce / uni)),
            "operating_point": {"steps": ref[0]["steps"], "eta": ref[0]["eta"], "sigma": ref[0]["sigma"]},
            "comparator_step_mean": float(np.mean([x["comparator_step"] for x in ref])),
            "renyi_budget_theorem_mean": float(np.mean([x["renyi_budget_theorem"] for x in ref])),
            "renyi_budget_regime_branch_mean": float(np.mean([x["renyi_budget_regime_branch"] for x in ref])),
        }
    reasons = Counter(str(r.get("reason", "?")).split(" (")[0] for r in records if not r.get("ok"))
    examples = [f"n={r['n']} seed={r['seed']}: {r.get('reason')}" for r in records if not r.get("ok")][:5]
    diag = {
        "slice": SLICE,
        "required_instances": required,
        "valid_instances": len(valid),
        "failure_reasons": dict(reasons),
        "failure_examples": examples,
        "per_n": per_n,
        "max_step_utilization": float(max((r.get("max_step_utilization", 0.0) for r in records), default=0.0)),
        "mean_step_utilization": float(np.mean([r["mean_step_utilization"] for r in valid])) if valid else None,
    }
    if compliance < 1.0:
        metrics = {m: FAIL_VALUE for m in METRICS}
        metrics["privacy_protocol_compliance"] = float(compliance)
        diag["status"] = "failed"
        return metrics, diag

    ns = sorted(by_n)
    ratio = float(np.mean([per_n[str(n)]["comparator_ratio"] for n in ns]))
    rel = float(np.mean([r["cce"] / r["uniform"] for r in valid]))
    rel_gap = float(np.mean([r["final_gap"] / r["uniform"] for r in valid]))
    metrics = {
        "comparator_exploitability_ratio": ratio,
        "relative_exploitability": rel,
        "relative_final_nash_gap": rel_gap,
        "privacy_protocol_compliance": 1.0,
    }
    for name in METRICS[:3]:
        value = metrics[name]
        metrics[name] = float(min(value, FAIL_VALUE)) if np.isfinite(value) else FAIL_VALUE
    diag["status"] = "ok"
    diag["cce_slope_vs_n"] = _slope(ns, [per_n[str(n)]["cce_exploitability_mean"] for n in ns])
    diag["relative_exploitability_slope_vs_n"] = _slope(ns, [per_n[str(n)]["relative_exploitability"] for n in ns])
    diag["comparator_ratio_slope_vs_n"] = _slope(ns, [per_n[str(n)]["comparator_ratio"] for n in ns])
    diag["renyi_budget_theorem_slope_vs_n"] = _slope(ns, [per_n[str(n)]["renyi_budget_theorem_mean"] for n in ns])
    diag["renyi_budget_regime_branch_slope_vs_n"] = _slope(ns, [per_n[str(n)]["renyi_budget_regime_branch_mean"] for n in ns])
    return metrics, diag


def evaluate(manifest_paths: list[str], comparator: dict, noise_key: int = 0) -> tuple[dict, dict]:
    by_method, duplicates = load_runs(manifest_paths)
    methods = sorted(by_method)
    records = {m: {exp: [] for exp in EXPERIMENTS} for m in methods}
    for exp, topology in EXPERIMENTS.items():
        for n in SIZES:
            steps, eta, sigma = operating_point(n, topology)
            for seed in SEEDS:
                key = f"{topology}/n{n}/seed{seed}"
                ref = comparator["instances"].get(key)
                if ref is None:
                    raise SystemExit(f"fixed comparator values missing for {key}; run --build-comparator")
                game = make_game(topology, n, seed)
                if game.fingerprint != ref["game_fingerprint"]:
                    raise SystemExit(f"regenerated game {key} differs from the fixed instance (numpy RNG drift?)")
                noise = noise_stream(topology, n, seed, steps, sigma, noise_key)
                if noise_key == 0 and noise_fingerprint(noise) != ref["noise_fingerprint"]:
                    raise SystemExit(f"regenerated noise for {key} differs from the fixed stream")
                uniform_gap = nash_gap(game, uniform_profile(game))
                for method in methods:
                    rec = {"n": n, "seed": seed, "uniform": uniform_gap, "comparator": ref["comparator_cce"]}
                    path = by_method[method].get((topology, n, seed))
                    if path is None:
                        rec.update(ok=False, reason="missing run")
                    elif (topology, n, seed) in duplicates[method]:
                        rec.update(ok=False, reason="duplicate run")
                    else:
                        try:
                            trajectory, broadcasts = _load_arrays(path)
                        except Exception as exc:  # noqa: BLE001 - any unreadable artifact fails the instance
                            rec.update(ok=False, reason=f"unreadable artifact ({type(exc).__name__})")
                        else:
                            ok, reason, stats = check_transcript(game, steps, eta, noise, trajectory, broadcasts)
                            rec.update(ok=ok, reason=reason, **stats)
                            if ok:
                                rec["cce"] = cce_exploitability(game, trajectory)
                                rec["final_gap"] = nash_gap(game, trajectory[-1])
                    records[method][exp].append(rec)
                del game

    scores: dict = {"experiments": {}}
    diagnostics: dict = {"_meta": {"slice": SLICE, "seeds": list(SEEDS), "sizes": list(SIZES),
                                   "noise_key": noise_key, "methods": methods}, "experiments": {}}
    for exp, topology in EXPERIMENTS.items():
        scores["experiments"][exp] = {"results": {}}
        diagnostics["experiments"][exp] = {}
        for method in methods:
            metrics, diag = summarize(records[method][exp], comparator, topology)
            scores["experiments"][exp]["results"][method] = metrics
            diagnostics["experiments"][exp][method] = diag
    return scores, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="*", default=[])
    parser.add_argument("--output", default="scoring/scores_train.json")
    parser.add_argument("--diagnostics", default="scoring/diagnostics_train.json")
    parser.add_argument("--comparator", default=str(COMPARATOR_PATH))
    parser.add_argument("--noise-key", type=int, default=0, help="robustness studies only; the benchmark uses 0")
    parser.add_argument("--build-comparator", action="store_true")
    args = parser.parse_args()

    if args.build_comparator:
        build_comparator(Path(args.comparator))
        return
    existing = [p for p in args.manifest if Path(p).is_file()]
    if not existing:
        raise SystemExit("No manifest files found to evaluate.")
    comparator = json.loads(Path(args.comparator).read_text())
    scores, diagnostics = evaluate(existing, comparator, args.noise_key)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True))
    Path(args.diagnostics).write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    for exp, block in scores["experiments"].items():
        for method, metrics in sorted(block["results"].items()):
            print(f"[{SLICE}] {exp:24s} {method:28s} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())))


if __name__ == "__main__":
    main()
