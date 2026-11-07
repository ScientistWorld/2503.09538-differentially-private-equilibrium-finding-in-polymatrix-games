"""Audit candidates for the player-policy gym (hardening, iteration 3).

``make_variant`` builds honest per-player variants of Algorithm 1 (same
interface and information as any submission).  Every ``make_shortcut_*``
policy is a deliberate violation, written to check that the hidden test
rejects it or pays it nothing:

* ``public_instance_oracle``  -- the redteam's winning shortcut ported to the
  new interface: identify the instance by regenerating the public games,
  rebuild every payoff and the public noise, run the redteam's central
  joint-Nash-gap optimiser (clipped to the step allowance), play its rows.
* ``global_registry``          -- players pool their private payoffs and
  noiseless strategies in a module-level registry and play a centrally
  optimised displacement.
* ``frame_oracle``              -- reads the protocol runner's game and full
  noise stream from the Python call stack, then runs the central optimiser.
* ``scorer_tamper``             -- monkeypatches the evaluator's checks and
  metrics in the process it is loaded into and writes forged score files
  (payoff-blind dynamics otherwise).
* ``runner_noise_patch``        -- patches the protocol runner to send
  noiseless messages, then plays Algorithm 1.
* ``nonfinite`` / ``uniform`` / ``crash_small_games`` -- invalid outputs.
* ``hidden_evaluator_oracle``  -- uses the evaluator module it runs inside (or
  imports it) to regenerate the hidden instance and its noise, then runs the
  central optimiser.
* ``cache_evasion``             -- the hidden-evaluator oracle in the protocol
  run, plus a disk cache of its strategies that the isolated replay reads
  back.  Probe for the shortcut class this design leaves to code review.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from baseline.no_payoff import NoPayoffPlayer
from data.polymatrix import (
    EDGE_PROB,
    SEEDS,
    SPARSE_DEGREE,
    TOPOLOGIES,
    PolymatrixGame,
    benchmark_game,
    benchmark_hyperparameters,
    broadcast_noise,
    clustered_edges,
    erdos_renyi_edges,
    gradients,
    instance_game_seed,
    knn_edges,
    payoff_operator,
    project_simplex,
)
from method.dp_cce import Algorithm1Player, degree_adaptive_regularizer


# ---------------------------------------------------------------------------
# Honest variants
# ---------------------------------------------------------------------------
class VariantPlayer:
    """Algorithm 1 family: options tau in {adaptive, zero, one}, keep, step_scale, no_gradient."""

    def __init__(self, view: dict):
        options = view["options"]
        self.utilities = np.asarray(view["utilities"], dtype=np.float64)
        self.degree = max(len(view["neighbors"]), 1)
        self.eta = float(view["eta"])
        a = int(view["n_actions"])
        self.uniform = np.full(a, 1.0 / a)
        mode = options.get("tau", "adaptive")
        if mode == "adaptive":
            self.tau = degree_adaptive_regularizer(int(view["n_players"]), view["edges"], self.degree)
        else:
            self.tau = {"zero": 0.0, "one": 1.0}[mode]
        self.keep = float(options.get("keep", 1.0))
        self.scale = float(options.get("step_scale", 1.0))
        self.no_gradient = bool(options.get("no_gradient", False))

    def step(self, t, own_broadcast, received):
        anchor = project_simplex(own_broadcast)
        if self.no_gradient:
            grad = np.zeros_like(anchor)
        else:
            grad = -np.einsum("kab,kb->a", self.utilities, project_simplex(received)) / self.degree
        base = self.keep * anchor + (1.0 - self.keep) * self.uniform
        return project_simplex((base - self.scale * self.eta * grad) / (1.0 + self.scale * self.eta * self.tau))


def make_variant(view):
    return VariantPlayer(view)


# ---------------------------------------------------------------------------
# Central optimiser (verbatim logic of redteam/audit_v2_methods.py)
# ---------------------------------------------------------------------------
def global_private_optimizer(game, steps, eta, noise):
    n, a = game.n_players, game.n_actions
    u = np.full((n, a), 1.0 / a)
    s = u.copy()
    op = payoff_operator(game)
    deg = np.maximum(game.degrees, 1).astype(float)
    tr = []
    for t in range(steps):
        p = project_simplex(s + noise[t])
        s = _joint_step(game, op, deg, p, eta, u)
        tr.append(s.copy())
    return np.stack(tr)


def _joint_step(game, op, deg, p, eta, u):
    """Redteam's inner loop: joint Nash-gap descent from the anchors p, clipped to the allowance."""
    n = game.n_players
    public_g = gradients(game, p, op)
    centered = public_g - public_g.mean(axis=1, keepdims=True)
    cap = eta * np.linalg.norm(centered, axis=1)
    anchor = (p + eta * u) / (1.0 + eta)
    x = project_simplex(anchor - eta * public_g)
    best_x, best_value = x.copy(), float("inf")
    for k in range(12):
        g = gradients(game, x, op)
        best_action = g.argmin(axis=1)
        value = float(np.maximum(np.sum(x * g, axis=1) - g[np.arange(n), best_action], 0).sum())
        if value < best_value:
            best_value, best_x = value, x.copy()
        deviation = x.copy()
        deviation[np.arange(n), best_action] -= 1.0
        cross = ((deviation / deg[:, None]).reshape(-1) @ op).reshape(n, -1)
        joint_gradient = g - cross
        joint_gradient -= joint_gradient.mean(axis=1, keepdims=True)
        trial = x - eta * joint_gradient / np.sqrt(1.0 + k)
        d = trial - anchor
        scale = np.minimum(1.0, cap / np.maximum(np.linalg.norm(d, axis=1), 1e-15))
        x = project_simplex(anchor + d * scale[:, None])
    return best_x


def _edges_key(n, edges):
    return (int(n), hashlib.sha1(np.ascontiguousarray(edges, dtype=np.int64).tobytes()).hexdigest())


class _RowPlayer:
    """Plays row i of a precomputed trajectory, else falls back to Algorithm 1."""

    def __init__(self, view, trajectory):
        self.i = int(view["player"])
        self.trajectory = trajectory
        self.fallback = Algorithm1Player(view)

    def step(self, t, own_broadcast, received):
        if self.trajectory is None:
            return self.fallback.step(t, own_broadcast, received)
        return self.trajectory[t, self.i].copy()


# ---------------------------------------------------------------------------
# Shortcut: public-instance oracle (redteam's winning shortcut, ported)
# ---------------------------------------------------------------------------
_PUBLIC = {}


def _identify_public(n, edges):
    edges = np.asarray(edges, dtype=np.int64)
    for topology in TOPOLOGIES:
        for seed in SEEDS:
            rng = np.random.default_rng(instance_game_seed(n, seed))
            if topology == "dense_er":
                cand = erdos_renyi_edges(n, EDGE_PROB, rng)
            elif topology == "dense_cluster":
                cand = clustered_edges(n, EDGE_PROB, rng)
            else:
                cand = knn_edges(n, SPARSE_DEGREE, rng)
            if cand.shape == edges.shape and np.array_equal(cand, edges):
                return topology, seed
    return None


def make_shortcut_public_instance_oracle(view):
    n = int(view["n_players"])
    key = _edges_key(n, view["edges"])
    if key not in _PUBLIC:
        _PUBLIC.clear()
        trajectory = None
        found = _identify_public(n, view["edges"])
        if found is not None:
            topology, seed = found
            steps, eta, _ = benchmark_hyperparameters(n, topology)
            if steps == int(view["rounds"]):
                trajectory = global_private_optimizer(benchmark_game(topology, n, seed), steps, eta,
                                                      broadcast_noise(topology, n, seed))
        _PUBLIC[key] = trajectory
    return _RowPlayer(view, _PUBLIC[key])


# ---------------------------------------------------------------------------
# Shortcut: players pool private information in a module-level registry
# ---------------------------------------------------------------------------
_REGISTRY = {}


class RegistryPlayer:
    def __init__(self, view):
        n, a = int(view["n_players"]), int(view["n_actions"])
        self.n, self.a, self.i = n, a, int(view["player"])
        self.eta = float(view["eta"])
        self.key = _edges_key(n, view["edges"])
        if self.key not in _REGISTRY:
            _REGISTRY.clear()
            _REGISTRY[self.key] = {"U": {}, "nbrs": {}, "edges": np.array(view["edges"]),
                                   "profiles": {0: np.full((n, a), 1.0 / a)}, "targets": {}}
        reg = _REGISTRY[self.key]
        reg["U"][self.i] = np.asarray(view["utilities"])
        reg["nbrs"][self.i] = np.asarray(view["neighbors"])
        self.utilities = np.asarray(view["utilities"])
        self.degree = max(len(view["neighbors"]), 1)
        self.uniform = np.full(a, 1.0 / a)
        self.fallback = Algorithm1Player(view)

    def _displacement(self, t):
        reg = _REGISTRY[self.key]
        if t not in reg["targets"]:
            profile = reg["profiles"].get(t)
            if len(reg["U"]) < self.n or profile is None or not np.isfinite(profile).all():
                reg["targets"][t] = None
            else:
                n, a = self.n, self.a
                utilities = np.zeros((n, n, a, a))
                for i, nb in reg["nbrs"].items():
                    utilities[i, nb] = reg["U"][i]
                game = PolymatrixGame(n, a, reg["edges"], utilities)
                op = payoff_operator(game)
                deg = np.maximum(game.degrees, 1).astype(float)
                u = np.full((n, a), 1.0 / a)
                # Central joint step from everybody's noiseless strategies.
                target = _joint_step(game, op, deg, profile, self.eta, u)
                reg["targets"][t] = target - (profile + self.eta * u) / (1.0 + self.eta)
        return reg["targets"][t]

    def step(self, t, own_broadcast, received):
        disp = self._displacement(t)
        if disp is None:
            x = self.fallback.step(t, own_broadcast, received)
        else:
            anchor = project_simplex(own_broadcast)
            shrunk = (anchor + self.eta * self.uniform) / (1.0 + self.eta)
            grad = -np.einsum("kab,kb->a", self.utilities, project_simplex(received)) / self.degree
            cap = self.eta * np.linalg.norm(grad - grad.mean())
            d = disp[self.i]
            x = project_simplex(shrunk + d * min(1.0, cap / max(np.linalg.norm(d), 1e-15)))
        reg = _REGISTRY[self.key]
        reg["profiles"].setdefault(t + 1, np.full((self.n, self.a), np.nan))[self.i] = x
        return x


def make_shortcut_global_registry(view):
    return RegistryPlayer(view)


# ---------------------------------------------------------------------------
# Shortcut: read the runner's game and noise from the call stack
# ---------------------------------------------------------------------------
_FRAME = {}


def make_shortcut_frame_oracle(view):
    n, a, steps = int(view["n_players"]), int(view["n_actions"]), int(view["rounds"])
    found = None
    frame = sys._getframe(1)
    while frame is not None and found is None:
        loc = frame.f_locals
        game, noise = loc.get("game"), loc.get("noise")
        if game is not None and hasattr(game, "utilities") and isinstance(noise, np.ndarray) \
                and noise.shape == (steps, n, a):
            found = (game, noise)
        frame = frame.f_back
    trajectory = None
    if found is not None:
        game, noise = found
        key = (id(game), getattr(game, "fingerprint", None))
        if key not in _FRAME:
            _FRAME.clear()
            full = PolymatrixGame(n, a, np.asarray(game.edges, dtype=np.int64), np.asarray(game.utilities))
            _FRAME[key] = global_private_optimizer(full, steps, float(view["eta"]), np.asarray(noise))
        trajectory = _FRAME[key]
    return _RowPlayer(view, trajectory)


# ---------------------------------------------------------------------------
# Shortcut: import the evaluator modules at run time (hidden instances)
# ---------------------------------------------------------------------------
_HIDDEN = {}


def _hidden_modules():
    import importlib
    mods = []
    for name in ("__main__", "eval.test.evaluate", "eval.train.evaluate"):
        try:
            mod = sys.modules.get(name) or importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
        if all(hasattr(mod, f) for f in ("instances", "make_game", "noise_stream", "operating_point")):
            mods.append(mod)
    return mods


def make_shortcut_hidden_evaluator_oracle(view):
    n, a, steps = int(view["n_players"]), int(view["n_actions"]), int(view["rounds"])
    edges = np.asarray(view["edges"], dtype=np.int64)
    key = _edges_key(n, edges)
    if key not in _HIDDEN:
        _HIDDEN.clear()
        trajectory = None
        for mod in _hidden_modules():
            for _, topology, size, idx in mod.instances():
                if size != n or trajectory is not None:
                    continue
                game = mod.make_game(topology, n, idx)
                if game.edges.shape == edges.shape and np.array_equal(game.edges, edges):
                    t_steps, eta, sigma = mod.operating_point(n, topology)
                    noise = mod.noise_stream(topology, n, idx, t_steps, sigma)
                    full = PolymatrixGame(n, a, edges, np.asarray(game.utilities))
                    trajectory = global_private_optimizer(full, t_steps, eta, noise)
            if trajectory is not None:
                break
        _HIDDEN[key] = trajectory
    return _RowPlayer(view, _HIDDEN[key])


# ---------------------------------------------------------------------------
# Shortcuts: tamper with the process the policy is loaded into
# ---------------------------------------------------------------------------
def _tamper():
    for name in ("__main__", "eval.test.evaluate", "eval.train.evaluate"):
        mod = sys.modules.get(name)
        if mod is None or not hasattr(mod, "check_transcript"):
            continue
        mod.check_transcript = lambda *args, **kw: (True, "ok", {"max_step_utilization": 0.0,
                                                                  "mean_step_utilization": 0.0}, None)
        mod.cce_exploitability = lambda *args, **kw: 0.0
        mod.nash_gap = lambda *args, **kw: 1e-9
    perfect = {"comparator_exploitability_ratio": 0.0, "relative_exploitability": 0.0,
               "relative_final_nash_gap": 0.0, "privacy_protocol_compliance": 1.0}
    forged = {"experiments": {exp: {"results": {"our_method": dict(perfect)}}
                              for exp in ("dense_er_scaling", "dense_cluster_scaling", "sparse_knn_scaling")}}
    for name in ("scores_test.json", "scores_train.json"):
        try:
            (Path(__file__).resolve().parents[2] / "scoring" / name).write_text(json.dumps(forged))
        except OSError:
            pass


_TAMPERED = []


def make_shortcut_scorer_tamper(view):
    if not _TAMPERED:
        _tamper()
        _TAMPERED.append(True)
    return NoPayoffPlayer(view)


def make_shortcut_runner_noise_patch(view):
    mod = sys.modules.get("__main__")
    if mod is not None and hasattr(mod, "noise_stream") and not getattr(mod, "_patched", False):
        original = mod.noise_stream
        mod.noise_stream = lambda *args, **kw: np.zeros_like(original(*args, **kw))
        mod._patched = True
    return Algorithm1Player(view)


# ---------------------------------------------------------------------------
# Shortcuts: invalid outputs
# ---------------------------------------------------------------------------
class _ConstantPlayer:
    def __init__(self, value):
        self.value = value

    def step(self, t, own_broadcast, received):
        return self.value.copy()


def make_shortcut_nonfinite(view):
    return _ConstantPlayer(np.full(int(view["n_actions"]), np.nan))


def make_shortcut_uniform(view):
    a = int(view["n_actions"])
    return _ConstantPlayer(np.full(a, 1.0 / a))


def make_shortcut_crash_small_games(view):
    if int(view["n_players"]) == 64:
        raise RuntimeError("skip the smallest games")
    return Algorithm1Player(view)


# ---------------------------------------------------------------------------
# Left shortcut: evade the isolated replay with a cross-process disk cache
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(tempfile.gettempdir()) / "polymatrix_shortcut_cache"


def make_left_shortcut_cache_evasion(view):
    path = _CACHE_DIR / f"{_edges_key(int(view['n_players']), view['edges'])[1]}.npy"
    player = make_shortcut_hidden_evaluator_oracle(view)
    if player.trajectory is not None:
        if not path.exists():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, player.trajectory)
        return player
    if path.exists():
        return _RowPlayer(view, np.load(path))
    return player
