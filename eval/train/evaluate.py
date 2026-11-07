"""Visible train-slice evaluator for private polymatrix-game equilibrium finding.

A submission is a *player policy* (``outputs/<dir>/policy.json`` naming a
module with a factory ``make_player(view)``; interface in
briefing/evaluation.md).  The train slice consists of the public development
instances with even seeds (games of ``data/polymatrix.benchmark_game``, noise
``data/polymatrix.broadcast_noise`` with key 0); this file keeps its own copy of
the generator.  The hidden test runs the same protocol, checks, audit and
metrics on its own instances.  For every policy found under outputs/ it

1. runs the private protocol on every instance of this slice in a child process
   (``--run-policy``).  Each player is built from its own local view (own
   payoff matrices, public graph, operating point) and in each round receives
   only its own noisy message and its neighbours' noisy messages;
2. in this (scoring) process, which never executes submitted code, regenerates
   every game and noise stream, recomputes every broadcast from the returned
   trajectory, and verifies the protocol (exact rounds, probability vectors,
   bounded payoff-dependent movement per round);
3. re-executes randomly chosen players of every instance in a fresh process
   (``--replay``) from their local view and the messages they received alone.
   A player whose transcript cannot be reproduced that way used information
   it does not have, and the instance fails (locality audit);
4. scores exploitability against fixed comparator values stored next to this
   file.

It never imports method/, data/ or eval/test/, and it reads nothing a policy
writes except the trajectory arrays of step 1 and the replayed strategies of
step 3, which it checks against its own recomputation.

Usage:
    python -m eval.train.evaluate                       # every outputs/*/policy.json
    python -m eval.train.evaluate --policy outputs/x/policy.json
    python -m eval.train.evaluate --build-comparator    # regenerate fixed values
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SLICE = "train"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULE = f"eval.{SLICE}.evaluate"
COMPARATOR_PATH = HERE / "comparator_train.json"

EXPERIMENTS = {
    "dense_er_scaling": "dense_er",
    "dense_cluster_scaling": "dense_cluster",
    "sparse_knn_scaling": "sparse_knn",
}
SIZES = (64, 128, 256, 512)
INSTANCE_IDS = (0, 2, 4, 6, 8, 10)
N_ACTIONS = 5
EDGE_PROB = 0.12
SPARSE_DEGREE = 4
RENYI_ORDER = 2.0
TOPOLOGY_CODE = {"dense_er": 1, "dense_cluster": 2, "sparse_random": 3, "sparse_knn": 4}

# Public development instances: game seed 1000 + 17 * seed + N, noise key 0.
NOISE_BASE = 20250311
_SEED_SALT = 0x0B1D5EED
_AUDIT_SALT = 0x0A0D17ED

# A failed experiment scores FAIL_VALUE on every exploitability metric; valid
# values are clipped to it, so no valid submission can score below a failed one.
FAIL_VALUE = 10.0
SIMPLEX_TOL = 1e-6
MOVE_REL_TOL = 1e-6
MOVE_ABS_TOL = 1e-8
AUDITED_PLAYERS = 8
AUDIT_TOL = 1e-7
RUN_TIMEOUT = 3 * 3600
REPLAY_TIMEOUT = 3600
METRICS = (
    "comparator_exploitability_ratio",
    "relative_exploitability",
    "relative_final_nash_gap",
    "privacy_protocol_compliance",
)


# ---------------------------------------------------------------------------
# Instances: games, operating point, broadcast noise
# ---------------------------------------------------------------------------
def instances() -> list[tuple[str, str, int, int]]:
    return [(exp, topology, n, idx) for exp, topology in EXPERIMENTS.items() for n in SIZES for idx in INSTANCE_IDS]


def instance_key(topology: str, n: int, idx: int) -> str:
    return f"{topology}/n{n}/seed{idx}"


class Game:
    def __init__(self, n: int, a: int, edges: np.ndarray, utilities: np.ndarray):
        self.n, self.a = n, a
        self.edges = np.asarray(edges, dtype=np.int64)
        self.utilities = utilities
        adj = np.zeros((n, n), dtype=bool)
        adj[self.edges[:, 0], self.edges[:, 1]] = True
        adj[self.edges[:, 1], self.edges[:, 0]] = True
        self.nbrs = [np.flatnonzero(adj[i]).astype(np.int64) for i in range(n)]
        self.deg = np.maximum(np.array([len(x) for x in self.nbrs], dtype=np.float64), 1.0)
        self.fingerprint = hashlib.sha256(self.edges.tobytes() + utilities.tobytes()).hexdigest()[:20]
        self._op = None

    @property
    def op(self) -> np.ndarray:
        if self._op is None:
            n, a = self.n, self.a
            self._op = np.ascontiguousarray(self.utilities.transpose(0, 2, 1, 3).reshape(n * a, n * a))
        return self._op


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


def make_game(topology: str, n: int, idx: int) -> Game:
    rng = np.random.default_rng(1000 + 17 * idx + n)
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


def noise_stream(topology: str, n: int, idx: int, steps: int, sigma: float) -> np.ndarray:
    rng = np.random.default_rng([NOISE_BASE, 0, TOPOLOGY_CODE[topology], int(n), int(idx)])
    return rng.normal(0.0, sigma, size=(steps, n, N_ACTIONS))


def noise_fingerprint(noise: np.ndarray) -> str:
    return hashlib.sha256(noise.tobytes()).hexdigest()[:20]


def player_seed(topology: str, n: int, idx: int, player: int) -> int:
    seq = np.random.SeedSequence([_SEED_SALT, TOPOLOGY_CODE[topology], int(n), int(idx), int(player)])
    return int(seq.generate_state(1)[0] & 0x7FFFFFFF)


def audited_players(topology: str, n: int, idx: int) -> list[int]:
    rng = np.random.default_rng([_AUDIT_SALT, TOPOLOGY_CODE[topology], int(n), int(idx)])
    return sorted(int(i) for i in rng.choice(n, size=min(AUDITED_PLAYERS, n), replace=False))


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
# Policies: manifest, loading, protocol runner (child processes only)
# ---------------------------------------------------------------------------
_MODULE_RE = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")


def read_policy(path: Path) -> dict:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("policy manifest is not a JSON object")
    module, factory, options = raw.get("module"), raw.get("factory", "make_player"), raw.get("options", {})
    if not isinstance(module, str) or not _MODULE_RE.fullmatch(module):
        raise ValueError("'module' must be a dotted module name importable from the workspace root")
    if not isinstance(factory, str) or not factory.isidentifier():
        raise ValueError("'factory' must be an identifier")
    if not isinstance(options, dict):
        raise ValueError("'options' must be a JSON object")
    return {"module": module, "factory": factory, "options": options}


def load_factory(spec: dict):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    factory = getattr(importlib.import_module(spec["module"]), spec["factory"])
    if not callable(factory):
        raise TypeError("factory is not callable")
    return factory


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _as_strategy(out, a: int) -> np.ndarray:
    return np.asarray(out, dtype=np.float64).reshape(a)


def _view(player: int, n: int, a: int, edges: np.ndarray, neighbors: np.ndarray, utilities: np.ndarray,
          steps: int, eta: float, sigma: float, seed: int, options: dict) -> dict:
    return {"player": int(player), "n_players": int(n), "n_actions": int(a), "edges": edges,
            "neighbors": np.array(neighbors, dtype=np.int64), "utilities": np.array(utilities, dtype=np.float64),
            "rounds": int(steps), "eta": float(eta), "sigma": float(sigma), "seed": int(seed),
            "options": copy.deepcopy(options)}


def execute_policy(factory, topology: str, n: int, idx: int, options: dict) -> np.ndarray:
    """Run the private protocol on one instance; returns the (T, N, A) trajectory."""
    steps, eta, sigma = operating_point(n, topology)
    game = make_game(topology, n, idx)
    noise = noise_stream(topology, n, idx, steps, sigma)
    edges = np.array(game.edges, dtype=np.int64)
    edges.setflags(write=False)
    players = [factory(_view(i, n, game.a, edges, game.nbrs[i], game.utilities[i, game.nbrs[i]], steps, eta,
                             sigma, player_seed(topology, n, idx, i), options)) for i in range(n)]
    current = uniform_profile(game)
    trajectory = np.empty((steps, n, game.a))
    for t in range(steps):
        messages = current + noise[t]
        for i, player in enumerate(players):
            trajectory[t, i] = _as_strategy(player.step(t, messages[i].copy(), messages[game.nbrs[i]]), game.a)
        current = trajectory[t]
    return trajectory


def _artifact(key: str) -> str:
    return key.replace("/", "__") + ".npy"


def run_policy_child(policy_path: Path, workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    status_path = workdir / "status.json"
    try:
        spec = read_policy(policy_path)
        factory = load_factory(spec)
    except BaseException as exc:  # noqa: BLE001 - any failure to load fails every instance
        status["_policy"] = f"policy could not be loaded: {_describe(exc)}"
        status_path.write_text(json.dumps(status))
        return
    for _, topology, n, idx in instances():
        key = instance_key(topology, n, idx)
        start = time.time()
        try:
            trajectory = execute_policy(factory, topology, n, idx, spec["options"])
            np.save(workdir / _artifact(key), trajectory, allow_pickle=False)
            status[key] = "ok"
        except BaseException as exc:  # noqa: BLE001 - a crashing policy fails the instance
            status[key] = f"policy error: {_describe(exc)}"
        status_path.write_text(json.dumps(status))
        print(f"[{SLICE}:run] {key} {status[key]} ({time.time() - start:.1f}s)", flush=True)


def replay_child(policy_path: Path, iodir: Path) -> None:
    """Re-execute audited players from their local views and received messages only."""
    errors: dict[str, str] = {}
    outputs: dict[int, np.ndarray] = {}
    try:
        spec = read_policy(policy_path)
        factory = load_factory(spec)
        meta = json.loads((iodir / "replay_in.json").read_text())
        with np.load(iodir / "replay_in.npz", allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
    except BaseException as exc:  # noqa: BLE001
        (iodir / "replay_status.json").write_text(json.dumps({"_policy": _describe(exc)}))
        return
    for inst in meta["instances"]:
        edges = np.array(data[f"edges{inst['j']}"], dtype=np.int64)
        edges.setflags(write=False)
        steps, a = int(inst["steps"]), int(inst["a"])
        players = {}
        for item in inst["items"]:
            m = int(item["m"])
            view = _view(item["player"], inst["n"], a, edges, data[f"nb{m}"], data[f"u{m}"], steps, inst["eta"],
                         inst["sigma"], item["seed"], spec["options"])
            try:
                players[m] = factory(view)
                outputs[m] = np.full((steps, a), np.nan)
            except BaseException as exc:  # noqa: BLE001
                errors[str(m)] = _describe(exc)
        for t in range(steps):
            for m in list(players):
                try:
                    out = players[m].step(t, data[f"own{m}"][t].copy(), data[f"rx{m}"][t].copy())
                    outputs[m][t] = _as_strategy(out, a)
                except BaseException as exc:  # noqa: BLE001
                    errors[str(m)] = _describe(exc)
                    del players[m]
    np.savez(iodir / "replay_out.npz", **{f"out{m}": v for m, v in outputs.items()})
    (iodir / "replay_status.json").write_text(json.dumps(errors))


# ---------------------------------------------------------------------------
# Protocol verification (scoring process)
# ---------------------------------------------------------------------------
def check_transcript(game: Game, steps: int, eta: float, noise: np.ndarray, trajectory: np.ndarray):
    """Verify that a trajectory is an execution of the private protocol.

    The broadcasts are recomputed here: round t's messages are the profile
    played before round t (uniform before round 0) plus the instance noise.

    1. exactly ``steps`` rounds for every player, finite values;
    2. every strategy is a probability vector;
    3. every update has bounded payoff dependence: with P the player's own
       projected broadcast, u uniform and g its payoff gradient against its
       neighbours' projected broadcasts (centred over actions), the new
       strategy lies within ``eta * ||g||`` of the segment from P to
       (P + eta * u) / (1 + eta).  This bounds how much a message can depend on
       private payoffs beyond public information to O(eta) per round, which
       with the fixed noise fixes the privacy budget of every compliant policy.
    Returns (ok, reason, stats, broadcasts).
    """
    shape = (steps, game.n, game.a)
    if trajectory.shape != shape:
        return False, f"shape: expected {shape}, got {trajectory.shape}", {}, None
    if not np.isfinite(trajectory).all():
        return False, "non-finite values", {}, None
    if trajectory.min() < -SIMPLEX_TOL or np.abs(trajectory.sum(axis=2) - 1.0).max() > SIMPLEX_TOL:
        return False, "strategy outside the probability simplex", {}, None
    uniform = uniform_profile(game)
    broadcasts = np.concatenate([uniform[None], trajectory[:-1]], axis=0) + noise
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
                       f"{distance[t, i]:.4g} > {cap[t, i]:.4g})"), stats, None
    return True, "ok", stats, broadcasts


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
    out = {}
    for _, topology, n, idx in instances():
        key = instance_key(topology, n, idx)
        steps, eta, sigma = operating_point(n, topology)
        game = make_game(topology, n, idx)
        noise = noise_stream(topology, n, idx, steps, sigma)
        a = game.a
        hdeg = float(n / np.sum(1.0 / game.deg))
        club = float(16.0 * a ** 3 * np.log(n) ** 2 / hdeg ** (4.0 / 9.0) + 4.0 * a / n)
        spade_max = 2.0 * a * _locality_fraction_max(game, steps)
        step = float(eta * min(1.0, np.sqrt(min(club, spade_max) / (2.0 * a))))
        trajectory = _structure_free_run(game, steps, step, noise)
        out[key] = {
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
        print(f"comparator {key} step={step:.4f} cce={out[key]['comparator_cce']:.5f}", flush=True)
    payload = {
        "slice": SLICE,
        "about": ("Fixed comparator: structure-free private projected-gradient play on the same noise and "
                  "rounds, step calibrated to the operating point's per-instance privacy budget."),
        "instances": out,
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))


# ---------------------------------------------------------------------------
# Scoring process
# ---------------------------------------------------------------------------
def discover_policies() -> list[str]:
    found = sorted(set((ROOT / "outputs").glob("*/policy.json")) | set((ROOT / "outputs").glob("*/*/policy.json")))
    return [str(p) for p in found]


def collect_methods(policy_paths: list[str]) -> tuple[dict, dict]:
    """Map method name -> policy manifest; invalid or duplicated names -> failure reason."""
    methods: dict[str, Path] = {}
    problems: dict[str, str] = {}
    for p in policy_paths:
        path = Path(p).resolve()
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            raw = None
        name = raw.get("method") if isinstance(raw, dict) else None
        name = name if isinstance(name, str) and name else path.parent.name
        if name in methods or name in problems:
            methods.pop(name, None)
            problems[name] = "method name registered by more than one policy manifest"
            continue
        try:
            read_policy(path)
        except (OSError, ValueError) as exc:
            problems[name] = f"invalid policy manifest: {exc}"
            continue
        methods[name] = path
    return methods, problems


def run_children(methods: dict, workroot: Path) -> tuple[dict, dict]:
    """Run every policy on every instance, one child process per policy (concurrently)."""
    procs = {}
    for k, (name, path) in enumerate(sorted(methods.items())):
        workdir = workroot / f"run{k}"
        log_path = workroot / f"run{k}.log"
        log = open(log_path, "w")
        cmd = [sys.executable, "-m", MODULE, "--run-policy", str(path), "--workdir", str(workdir)]
        procs[name] = (subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), workdir, log, log_path)
    start = time.time()
    info = {}
    for name, (proc, _, log, log_path) in procs.items():
        try:
            proc.wait(timeout=max(1.0, RUN_TIMEOUT - (time.time() - start)))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        tail = log_path.read_text(errors="replace")[-1500:] if log_path.is_file() else ""
        info[name] = {"exit_code": proc.returncode, "seconds": round(time.time() - start, 1), "log_tail": tail}
        print(f"[{SLICE}] ran policy {name} (exit {proc.returncode}, {info[name]['seconds']}s)", flush=True)
    return {name: workdir for name, (_, workdir, _, _) in procs.items()}, info


def _load_trajectory(workdir: Path, key: str, status: dict):
    path = workdir / _artifact(key)
    if not path.is_file():
        why = status.get(key) or status.get("_policy") or "policy run did not finish"
        return None, f"no trajectory ({why})"
    try:
        arr = np.load(path, allow_pickle=False)
        if not isinstance(arr, np.ndarray) or arr.dtype.kind not in "fiu":
            raise TypeError("not a numeric array")
        return np.array(arr, dtype=np.float64), ""
    except Exception as exc:  # noqa: BLE001 - unreadable artifact fails the instance
        return None, f"unreadable trajectory ({type(exc).__name__})"


def run_replays(path: Path, jobs: list[dict], workroot: Path) -> tuple[dict, dict]:
    """Isolated re-execution of the audited players; returns (failures by instance, stats)."""
    iodir = Path(tempfile.mkdtemp(prefix="replay_", dir=workroot))
    arrays, meta, expected = {}, {"instances": []}, {}
    m = 0
    for j, job in enumerate(jobs):
        arrays[f"edges{j}"] = job["edges"]
        inst = {"j": j, "n": job["n"], "a": job["a"], "steps": job["steps"], "eta": job["eta"],
                "sigma": job["sigma"], "items": []}
        for item in job["items"]:
            arrays[f"u{m}"], arrays[f"nb{m}"] = item["utilities"], item["neighbors"]
            arrays[f"own{m}"], arrays[f"rx{m}"] = item["own"], item["received"]
            inst["items"].append({"m": m, "player": item["player"], "seed": item["seed"]})
            expected[m] = (job["key"], item["player"], item["expected"])
            m += 1
        meta["instances"].append(inst)
    np.savez(iodir / "replay_in.npz", **arrays)
    (iodir / "replay_in.json").write_text(json.dumps(meta))
    with open(iodir / "replay.log", "w") as log:
        try:
            subprocess.run([sys.executable, "-m", MODULE, "--replay", str(path), "--workdir", str(iodir)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=REPLAY_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
    replayed = {}
    try:
        with np.load(iodir / "replay_out.npz", allow_pickle=False) as npz:
            replayed = {k: np.array(npz[k], dtype=np.float64) for k in npz.files}
    except Exception:  # noqa: BLE001 - nothing replayed: every audited instance fails
        replayed = {}
    try:
        errors = json.loads((iodir / "replay_status.json").read_text())
        errors = errors if isinstance(errors, dict) else {}
    except (OSError, ValueError):
        errors = {"_policy": "replay did not finish"}
    failures, max_dev = {}, 0.0
    for m, (key, player, exp) in expected.items():
        got = replayed.get(f"out{m}")
        if got is None or got.shape != exp.shape:
            why = errors.get(str(m)) or errors.get("_policy") or "no replayed output"
            failures.setdefault(key, f"locality audit: player {player} could not be re-executed ({why})"[:300])
            max_dev = np.inf
            continue
        dev = float(np.nan_to_num(np.abs(got - exp), nan=np.inf).max())
        max_dev = max(max_dev, dev)
        if dev > AUDIT_TOL:
            failures.setdefault(key, (f"locality audit: player {player}'s strategies are not reproduced from its "
                                      f"local view and received messages (max deviation {dev:.3g})"))
    shutil.rmtree(iodir, ignore_errors=True)
    stats = {"audited_players": len(expected), "audited_instances": len(jobs), "failed_instances": len(failures),
             "max_deviation": max_dev if np.isfinite(max_dev) else "inf"}
    return failures, stats


def _metric_block(records: list[dict]) -> dict:
    by_n = defaultdict(list)
    for r in records:
        by_n[r["n"]].append(r)
    ratio = float(np.mean([sum(r["cce"] for r in rs) / sum(r["comparator"] for r in rs) for rs in by_n.values()]))
    metrics = {
        "comparator_exploitability_ratio": ratio,
        "relative_exploitability": float(np.mean([r["cce"] / r["uniform"] for r in records])),
        "relative_final_nash_gap": float(np.mean([r["final_gap"] / r["uniform"] for r in records])),
    }
    for name, value in metrics.items():
        metrics[name] = float(min(value, FAIL_VALUE)) if np.isfinite(value) else FAIL_VALUE
    metrics["privacy_protocol_compliance"] = 1.0
    return metrics


def _slope(xs: list[float], ys: list[float]) -> float:
    ys = np.maximum(np.asarray(ys, dtype=np.float64), 1e-12)
    if len(xs) < 2:
        return 0.0
    return float(np.polyfit(np.log(np.asarray(xs, dtype=np.float64)), np.log(ys), 1)[0])


def summarize(records: list[dict], comparator: dict) -> tuple[dict, dict]:
    required = len(SIZES) * len(INSTANCE_IDS)
    valid = [r for r in records if r["ok"]]
    compliance = len(valid) / required
    by_n = defaultdict(list)
    for r in valid:
        by_n[r["n"]].append(r)
    per_n = {}
    for n, rs in sorted(by_n.items()):
        cce = np.array([r["cce"] for r in rs])
        comp = np.array([r["comparator"] for r in rs])
        uni = np.array([r["uniform"] for r in rs])
        ref = [comparator["instances"][r["key"]] for r in rs]
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
    reasons = Counter(str(r.get("reason", "?")).split(" (")[0] for r in records if not r["ok"])
    diag = {
        "slice": SLICE,
        "required_instances": required,
        "valid_instances": len(valid),
        "transcript_valid_instances": sum(1 for r in records if r["transcript_ok"]),
        "failure_reasons": dict(reasons),
        "failure_examples": [f"{r['key']}: {r.get('reason')}" for r in records if not r["ok"]][:5],
        "per_n": per_n,
        "max_step_utilization": float(max((r.get("max_step_utilization", 0.0) for r in records), default=0.0)),
        "mean_step_utilization": float(np.mean([r["mean_step_utilization"] for r in valid])) if valid else None,
        "per_instance": [{k: r.get(k) for k in ("key", "n", "ok", "cce", "final_gap", "comparator", "uniform")}
                         for r in records],
    }
    if compliance < 1.0:
        metrics = {m: FAIL_VALUE for m in METRICS}
        metrics["privacy_protocol_compliance"] = float(compliance)
        diag["status"] = "failed"
        transcript_valid = [r for r in records if r["transcript_ok"]]
        if len(transcript_valid) == required:
            # What the transcripts would have scored had they passed the locality audit (analysis only).
            diag["unaudited_metrics"] = _metric_block(transcript_valid)
        return metrics, diag

    metrics = _metric_block(valid)
    ns = sorted(by_n)
    diag["status"] = "ok"
    diag["cce_slope_vs_n"] = _slope(ns, [per_n[str(n)]["cce_exploitability_mean"] for n in ns])
    diag["relative_exploitability_slope_vs_n"] = _slope(ns, [per_n[str(n)]["relative_exploitability"] for n in ns])
    diag["comparator_ratio_slope_vs_n"] = _slope(ns, [per_n[str(n)]["comparator_ratio"] for n in ns])
    diag["renyi_budget_theorem_slope_vs_n"] = _slope(ns, [per_n[str(n)]["renyi_budget_theorem_mean"] for n in ns])
    diag["renyi_budget_regime_branch_slope_vs_n"] = _slope(ns, [per_n[str(n)]["renyi_budget_regime_branch_mean"] for n in ns])
    return metrics, diag


def evaluate(policy_paths: list[str], comparator: dict) -> tuple[dict, dict]:
    methods, problems = collect_methods(policy_paths)
    names = sorted(set(methods) | set(problems))
    records = {m: {exp: [] for exp in EXPERIMENTS} for m in names}
    audit_jobs: dict[str, list] = defaultdict(list)
    audit_stats: dict[str, dict] = {}
    run_info: dict[str, dict] = {}
    workroot = Path(tempfile.mkdtemp(prefix=f"polymatrix_{SLICE}_"))
    try:
        workdirs, run_info = run_children(methods, workroot)
        statuses = {}
        for name, workdir in workdirs.items():
            try:
                statuses[name] = json.loads((workdir / "status.json").read_text())
            except (OSError, ValueError):
                statuses[name] = {}
            if not isinstance(statuses[name], dict):
                statuses[name] = {}
        for exp, topology, n, idx in instances():
            key = instance_key(topology, n, idx)
            ref = comparator["instances"].get(key)
            if ref is None:
                raise SystemExit(f"fixed comparator values missing for {key}; run --build-comparator")
            steps, eta, sigma = operating_point(n, topology)
            game = make_game(topology, n, idx)
            if game.fingerprint != ref["game_fingerprint"]:
                raise SystemExit(f"regenerated game {key} differs from the fixed instance (numpy RNG drift?)")
            noise = noise_stream(topology, n, idx, steps, sigma)
            if noise_fingerprint(noise) != ref["noise_fingerprint"]:
                raise SystemExit(f"regenerated noise for {key} differs from the fixed stream")
            uniform_gap = nash_gap(game, uniform_profile(game))
            audit_ids = audited_players(topology, n, idx)
            for name in names:
                rec = {"key": key, "n": n, "idx": idx, "uniform": uniform_gap, "comparator": ref["comparator_cce"],
                       "ok": False, "transcript_ok": False}
                if name in problems:
                    rec["reason"] = problems[name]
                else:
                    trajectory, why = _load_trajectory(workdirs[name], key, statuses[name])
                    if trajectory is None:
                        rec["reason"] = why
                    else:
                        ok, reason, stats, broadcasts = check_transcript(game, steps, eta, noise, trajectory)
                        rec.update(reason=reason, **stats)
                        if ok:
                            rec.update(ok=True, transcript_ok=True, cce=cce_exploitability(game, trajectory),
                                       final_gap=nash_gap(game, trajectory[-1]))
                            audit_jobs[name].append({
                                "key": key, "n": n, "a": game.a, "steps": steps, "eta": eta, "sigma": sigma,
                                "edges": game.edges,
                                "items": [{"player": i, "seed": player_seed(topology, n, idx, i),
                                           "utilities": game.utilities[i, game.nbrs[i]], "neighbors": game.nbrs[i],
                                           "own": broadcasts[:, i, :], "received": broadcasts[:, game.nbrs[i], :],
                                           "expected": trajectory[:, i, :]} for i in audit_ids],
                            })
                records[name][exp].append(rec)
            del game
        # The policies' own files are gone before any replay starts.
        for workdir in workdirs.values():
            shutil.rmtree(workdir, ignore_errors=True)
        for name, jobs in audit_jobs.items():
            failures, audit_stats[name] = run_replays(methods[name], jobs, workroot)
            for exp in EXPERIMENTS:
                for rec in records[name][exp]:
                    if rec["ok"] and rec["key"] in failures:
                        rec["ok"] = False
                        rec["reason"] = failures[rec["key"]]
            print(f"[{SLICE}] locality audit {name}: {audit_stats[name]}", flush=True)
    finally:
        shutil.rmtree(workroot, ignore_errors=True)

    scores: dict = {"experiments": {}}
    diagnostics: dict = {"_meta": {"slice": SLICE, "instances_per_size": len(INSTANCE_IDS), "sizes": list(SIZES),
                                   "methods": names, "audited_players_per_instance": AUDITED_PLAYERS,
                                   "audit_tolerance": AUDIT_TOL, "locality_audit": audit_stats,
                                   "policy_runs": run_info},
                         "experiments": {}}
    for exp in EXPERIMENTS:
        scores["experiments"][exp] = {"results": {}}
        diagnostics["experiments"][exp] = {}
        for name in names:
            metrics, diag = summarize(records[name][exp], comparator)
            scores["experiments"][exp]["results"][name] = metrics
            diagnostics["experiments"][exp][name] = diag
    return scores, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", nargs="*", default=None, help="policy manifests (default: outputs/*/policy.json)")
    parser.add_argument("--output", default=f"scoring/scores_{SLICE}.json")
    parser.add_argument("--diagnostics", default=f"scoring/diagnostics_{SLICE}.json")
    parser.add_argument("--comparator", default=str(COMPARATOR_PATH))
    parser.add_argument("--build-comparator", action="store_true")
    parser.add_argument("--run-policy", help=argparse.SUPPRESS)
    parser.add_argument("--replay", help=argparse.SUPPRESS)
    parser.add_argument("--workdir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_policy:
        run_policy_child(Path(args.run_policy), Path(args.workdir))
        return
    if args.replay:
        replay_child(Path(args.replay), Path(args.workdir))
        return
    if args.build_comparator:
        build_comparator(Path(args.comparator))
        return
    paths = args.policy if args.policy else discover_policies()
    existing = [p for p in paths if Path(p).is_file()]
    if not existing:
        raise SystemExit("No policy manifests found to evaluate (expected outputs/<dir>/policy.json).")
    comparator = json.loads(Path(args.comparator).read_text())
    scores, diagnostics = evaluate(existing, comparator)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True))
    Path(args.diagnostics).write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    for exp, block in scores["experiments"].items():
        for method, metrics in sorted(block["results"].items()):
            print(f"[{SLICE}] {exp:24s} {method:28s} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())))


if __name__ == "__main__":
    main()
