"""Problem-side utilities for private polymatrix-game equilibrium finding.

This module is intentionally independent of method/.  It defines the public
benchmark: procedural game generation, the instance grid, the privacy
operating point (rounds, noise level, step scale), the fixed broadcast-noise
stream every method must use, and basic game quantities.  The evaluators in
eval/ keep their own copies of everything they score with, so editing this
file cannot change a score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Public benchmark grid
# ---------------------------------------------------------------------------
TOPOLOGIES = ("dense_er", "dense_cluster", "sparse_knn")
SIZES = (64, 128, 256, 512)
SEEDS = tuple(range(12))
N_ACTIONS = 5
EDGE_PROB = 0.12
SPARSE_DEGREE = 4
RENYI_ORDER = 2.0
_NOISE_BASE = 20250311
_TOPOLOGY_CODE = {"dense_er": 1, "dense_cluster": 2, "sparse_random": 3, "sparse_knn": 4}


@dataclass(frozen=True)
class PolymatrixGame:
    n_players: int
    n_actions: int
    edges: np.ndarray
    utilities: np.ndarray

    @property
    def degrees(self) -> np.ndarray:
        deg = np.zeros(self.n_players, dtype=np.int64)
        np.add.at(deg, self.edges[:, 0], 1)
        np.add.at(deg, self.edges[:, 1], 1)
        return deg

    @property
    def neighbors(self) -> list[list[int]]:
        nbrs: list[list[int]] = [[] for _ in range(self.n_players)]
        for i, j in self.edges:
            nbrs[int(i)].append(int(j))
            nbrs[int(j)].append(int(i))
        return nbrs


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of vectors on the last axis onto the simplex."""
    arr = np.asarray(v, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    u = np.sort(flat, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, flat.shape[1] + 1)
    cond = u - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(flat.shape[0]), rho] / (rho + 1)
    w = np.maximum(flat - theta[:, None], 0.0)
    return w.reshape(arr.shape)


# ---------------------------------------------------------------------------
# Graph families and game generation
# ---------------------------------------------------------------------------
def erdos_renyi_edges(n_players: int, p: float, rng: np.random.Generator) -> np.ndarray:
    rows, cols = np.triu_indices(n_players, k=1)
    mask = rng.random(rows.shape[0]) < p
    edges = np.stack([rows[mask], cols[mask]], axis=1)
    return _ensure_no_isolates(n_players, edges, rng)


def clustered_edges(n_players: int, p: float, rng: np.random.Generator) -> np.ndarray:
    n_clusters = max(1, int(np.floor(1.0 / p)))
    labels = rng.integers(0, n_clusters, size=n_players)
    edge_set: set[tuple[int, int]] = set()
    cross_p = min(1.0, 10.0 * p / n_players)
    for i in range(n_players):
        for j in range(i + 1, n_players):
            if labels[i] == labels[j] or rng.random() < cross_p:
                edge_set.add((i, j))
    return _ensure_no_isolates(n_players, np.array(sorted(edge_set), dtype=np.int64), rng)


def sparse_random_edges(n_players: int, c: int, rng: np.random.Generator) -> np.ndarray:
    edge_set: set[tuple[int, int]] = set()
    for i in range(n_players):
        choices = rng.choice([j for j in range(n_players) if j != i], size=c, replace=False)
        for j in choices:
            edge_set.add((min(i, int(j)), max(i, int(j))))
    return _ensure_no_isolates(n_players, np.array(sorted(edge_set), dtype=np.int64), rng)


def knn_edges(n_players: int, c: int, rng: np.random.Generator) -> np.ndarray:
    points = rng.random((n_players, 2))
    dist = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(dist, np.inf)
    edge_set: set[tuple[int, int]] = set()
    for i in range(n_players):
        for j in np.argpartition(dist[i], c)[:c]:
            edge_set.add((min(i, int(j)), max(i, int(j))))
    return _ensure_no_isolates(n_players, np.array(sorted(edge_set), dtype=np.int64), rng)


def _ensure_no_isolates(n_players: int, edges: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    edge_set = {tuple(map(int, e)) for e in np.asarray(edges, dtype=np.int64).reshape(-1, 2)}
    deg = np.zeros(n_players, dtype=np.int64)
    for i, j in edge_set:
        deg[i] += 1
        deg[j] += 1
    for i in np.where(deg == 0)[0]:
        j = int(rng.integers(0, n_players - 1))
        if j >= i:
            j += 1
        edge_set.add((min(i, j), max(i, j)))
    return np.array(sorted(edge_set), dtype=np.int64)


def generate_game(
    n_players: int,
    n_actions: int,
    topology: str,
    seed: int,
    p: float = EDGE_PROB,
    c: int = SPARSE_DEGREE,
) -> PolymatrixGame:
    rng = np.random.default_rng(seed)
    if topology == "dense_er":
        edges = erdos_renyi_edges(n_players, p, rng)
    elif topology == "dense_cluster":
        edges = clustered_edges(n_players, p, rng)
    elif topology == "sparse_random":
        edges = sparse_random_edges(n_players, c, rng)
    elif topology == "sparse_knn":
        edges = knn_edges(n_players, c, rng)
    else:
        raise ValueError(f"unknown topology: {topology}")

    utilities = rng.uniform(-1.0, 1.0, size=(n_players, n_players, n_actions, n_actions))
    mask = np.zeros((n_players, n_players), dtype=bool)
    mask[edges[:, 0], edges[:, 1]] = True
    mask[edges[:, 1], edges[:, 0]] = True
    utilities[~mask] = 0.0
    return PolymatrixGame(n_players, n_actions, edges, utilities)


def instance_game_seed(n_players: int, seed: int) -> int:
    """Generator seed of benchmark instance ``seed`` at size ``n_players``."""
    return 1000 + 17 * int(seed) + int(n_players)


def benchmark_game(topology: str, n_players: int, seed: int) -> PolymatrixGame:
    return generate_game(n_players, N_ACTIONS, topology, seed=instance_game_seed(n_players, seed))


def benchmark_instances(topologies: Iterable[str] = TOPOLOGIES, sizes: Iterable[int] = SIZES,
                        seeds: Iterable[int] = SEEDS) -> list[tuple[str, int, int]]:
    """Every (topology, n_players, seed) instance a submission must solve."""
    return [(t, int(n), int(s)) for t in topologies for n in sizes for s in seeds]


# ---------------------------------------------------------------------------
# Privacy operating point and the fixed broadcast-noise stream
# ---------------------------------------------------------------------------
def benchmark_hyperparameters(n_players: int, topology: str, p: float = EDGE_PROB,
                              c: int = SPARSE_DEGREE) -> tuple[int, float, float]:
    """Operating point (rounds T, step scale eta, broadcast noise sigma)."""
    if topology.startswith("dense"):
        exponent = max(0.2, min(1.0, np.log(max(p * n_players, 2.0)) / np.log(n_players)))
        steps = max(8, int(round((n_players ** (8.0 * exponent / 9.0)) / (np.log(n_players) ** 2))))
    else:
        sparse_steps = (1.0 - np.log(np.log(n_players)) / np.log(n_players)) * np.log(n_players) / np.log(max(c, 2))
        steps = max(2, int(round(sparse_steps)))
    eta = 1.0 / (steps ** (1.0 / 3.0))
    sigma = 1.0 / np.sqrt(steps)
    return steps, eta, sigma


def broadcast_noise(topology: str, n_players: int, seed: int, key: int = 0) -> np.ndarray:
    """The fixed Gaussian broadcast noise of one instance, shape (T, N, A).

    Round ``t`` broadcasts ``strategy_t + noise[t]`` (``strategy_0`` is the
    uniform profile).  Every method uses this exact stream, so methods are
    compared under common noise and cannot select or shape their noise.
    """
    steps, _, sigma = benchmark_hyperparameters(n_players, topology)
    rng = np.random.default_rng([_NOISE_BASE, int(key), _TOPOLOGY_CODE[topology], int(n_players), int(seed)])
    return rng.normal(0.0, sigma, size=(steps, n_players, N_ACTIONS))


# ---------------------------------------------------------------------------
# Game quantities
# ---------------------------------------------------------------------------
def payoff_operator(game: PolymatrixGame) -> np.ndarray:
    """Matrix M with M[(i,a),(j,b)] = U[i,j,a,b], for batched gradients."""
    n, a = game.n_players, game.n_actions
    return np.ascontiguousarray(game.utilities.transpose(0, 2, 1, 3).reshape(n * a, n * a))


def gradients(game: PolymatrixGame, strategies: np.ndarray, operator: np.ndarray | None = None) -> np.ndarray:
    """Loss gradients -(1/deg_i) sum_j U_ij pi_j for one profile (N, A) or a batch (T, N, A)."""
    op = payoff_operator(game) if operator is None else operator
    n, a = game.n_players, game.n_actions
    deg = np.maximum(game.degrees.astype(np.float64), 1.0)
    s = np.asarray(strategies, dtype=np.float64)
    flat = s.reshape(-1, n * a)
    g = -(flat @ op.T).reshape(s.shape) / deg[:, None]
    return g


def cce_exploitability(game: PolymatrixGame, trajectory: np.ndarray) -> tuple[float, np.ndarray]:
    """Average fixed-deviation regret of the empirical distribution of iterates."""
    traj = np.asarray(trajectory, dtype=np.float64)
    grads = gradients(game, traj)
    average_current = np.mean(np.sum(grads * traj, axis=2), axis=0)
    best_fixed = np.mean(grads, axis=0).min(axis=1)
    per_player = np.maximum(average_current - best_fixed, 0.0)
    return float(per_player.mean()), per_player


def nash_gap(game: PolymatrixGame, strategies: np.ndarray) -> tuple[float, np.ndarray]:
    grads = gradients(game, strategies)
    per_player = np.maximum(np.sum(grads * strategies, axis=1) - grads.min(axis=1), 0.0)
    return float(per_player.mean()), per_player


def harmonic_mean_degree(game: PolymatrixGame) -> float:
    deg = np.maximum(game.degrees.astype(np.float64), 1.0)
    return float(game.n_players / np.sum(1.0 / deg))


# ---------------------------------------------------------------------------
# Graph distances and privacy accounting helpers
# ---------------------------------------------------------------------------
def graph_distances(n_players: int, edges: np.ndarray, sources: list[int]) -> np.ndarray:
    nbrs: list[list[int]] = [[] for _ in range(n_players)]
    for i, j in edges:
        nbrs[int(i)].append(int(j))
        nbrs[int(j)].append(int(i))
    dist = np.full(n_players, np.inf)
    frontier = list(sources)
    for s in frontier:
        dist[s] = 0
    head = 0
    while head < len(frontier):
        cur = frontier[head]
        head += 1
        for nxt in nbrs[cur]:
            if np.isinf(dist[nxt]):
                dist[nxt] = dist[cur] + 1
                frontier.append(nxt)
    return dist


def all_pairs_distances(game: PolymatrixGame, max_depth: int) -> np.ndarray:
    """Shortest-path distances truncated at ``max_depth`` (larger values -> max_depth + 1)."""
    n = game.n_players
    adj = np.zeros((n, n), dtype=np.float32)
    adj[game.edges[:, 0], game.edges[:, 1]] = 1.0
    adj[game.edges[:, 1], game.edges[:, 0]] = 1.0
    dist = np.full((n, n), max_depth + 1, dtype=np.int16)
    reached = np.eye(n, dtype=bool)
    dist[reached] = 0
    for depth in range(1, max_depth + 1):
        new = ((reached.astype(np.float32) @ adj) > 0) & ~reached
        if not new.any():
            break
        dist[new] = depth
        reached |= new
    return dist


def locality_fraction_max(game: PolymatrixGame, steps: int) -> float:
    """max over edges (v1,v2) of the fraction of players with min distance < steps."""
    dist = all_pairs_distances(game, steps)
    near = np.minimum(dist[game.edges[:, 0]], dist[game.edges[:, 1]]) < steps
    return float(near.sum(axis=1).max() / game.n_players)


def dense_sensitivity_term(game: PolymatrixGame) -> float:
    n, a = game.n_players, game.n_actions
    hdeg = harmonic_mean_degree(game)
    return float(16.0 * (a ** 3) * (np.log(n) ** 2) / (hdeg ** (4.0 / 9.0)) + 4.0 * a / n)


def privacy_bound(game: PolymatrixGame, eta: float, sigma: float, steps: int, alpha: float,
                  adjacent_edge: Iterable[int]) -> float:
    """Average single-channel Renyi-DP upper bound for one adjacent edge."""
    v1, v2 = map(int, adjacent_edge)
    n = game.n_players
    a = game.n_actions
    club = dense_sensitivity_term(game)
    dists = graph_distances(n, game.edges, [v1, v2])
    spade = 2.0 * a / n * float(np.sum(steps > dists))
    return float(alpha * eta * eta / (sigma * sigma) * min(club, spade) * steps)


def save_game(path: str | Path, game: PolymatrixGame) -> None:
    np.savez_compressed(path, n_players=game.n_players, n_actions=game.n_actions, edges=game.edges, utilities=game.utilities)


def load_game(path: str | Path) -> PolymatrixGame:
    data = np.load(path)
    return PolymatrixGame(int(data["n_players"]), int(data["n_actions"]), data["edges"], data["utilities"])
