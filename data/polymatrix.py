"""Problem-side utilities for polymatrix-game experiments.

This module is intentionally independent of method/.  It defines procedural
game generation and metrics that any future method can use or be evaluated on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PolymatrixGame:
    n_players: int
    n_actions: int
    edges: np.ndarray
    utilities: np.ndarray

    @property
    def degrees(self) -> np.ndarray:
        deg = np.zeros(self.n_players, dtype=np.int64)
        for i, j in self.edges:
            deg[i] += 1
            deg[j] += 1
        return deg

    @property
    def neighbors(self) -> list[list[int]]:
        nbrs: list[list[int]] = [[] for _ in range(self.n_players)]
        for i, j in self.edges:
            nbrs[i].append(j)
            nbrs[j].append(i)
        return nbrs


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vectors on the last axis to the probability simplex."""
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
    p: float = 0.1,
    c: int = 4,
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
    for i, j in edges:
        mask[i, j] = True
        mask[j, i] = True
    utilities[~mask] = 0.0
    return PolymatrixGame(n_players, n_actions, edges, utilities)


def gradients(game: PolymatrixGame, strategies: np.ndarray) -> np.ndarray:
    grads = np.zeros_like(strategies, dtype=np.float64)
    deg = np.maximum(game.degrees, 1)
    for i, j in game.edges:
        grads[i] -= game.utilities[i, j] @ strategies[j] / deg[i]
        grads[j] -= game.utilities[j, i] @ strategies[i] / deg[j]
    return grads


def cce_exploitability(game: PolymatrixGame, trajectory: np.ndarray) -> tuple[float, np.ndarray]:
    """Average fixed-deviation regret of the empirical CCE.

    Proposition 5.1 measures whether each player would have benefited from
    committing to one fixed alternative strategy over the empirical sequence.
    Thus the best response is chosen after averaging gradients over time, not
    separately at each timestep.
    """
    current_vals = []
    grad_vals = []
    for strategies in trajectory:
        grads = gradients(game, strategies)
        current_vals.append(np.sum(grads * strategies, axis=1))
        grad_vals.append(grads)
    average_current = np.mean(np.stack(current_vals, axis=0), axis=0)
    average_gradient = np.mean(np.stack(grad_vals, axis=0), axis=0)
    best_fixed = average_gradient.min(axis=1)
    per_player = np.maximum(average_current - best_fixed, 0.0)
    return float(per_player.mean()), per_player


def nash_gap(game: PolymatrixGame, strategies: np.ndarray) -> tuple[float, np.ndarray]:
    grads = gradients(game, strategies)
    per_player = np.maximum(np.sum(grads * strategies, axis=1) - grads.min(axis=1), 0.0)
    return float(per_player.mean()), per_player


def harmonic_mean_degree(game: PolymatrixGame) -> float:
    deg = np.maximum(game.degrees.astype(np.float64), 1.0)
    return float(game.n_players / np.sum(1.0 / deg))


def privacy_bound(game: PolymatrixGame, eta: float, sigma: float, steps: int, alpha: float, adjacent_edge: Iterable[int]) -> float:
    """Average single-channel Renyi-DP upper bound for one adjacent edge."""
    v1, v2 = map(int, adjacent_edge)
    n = game.n_players
    a = game.n_actions
    hdeg = harmonic_mean_degree(game)
    club = 16.0 * (a**3) * (np.log(n) ** 2) / (hdeg ** (4.0 / 9.0)) + 4.0 * a / n
    dists = graph_distances(game.n_players, game.edges, [v1, v2])
    spade = 2.0 * a / n * float(np.sum(steps > dists))
    return float(alpha * eta * eta / (sigma * sigma) * min(club, spade) * steps)


def benchmark_privacy_bound(game: PolymatrixGame, eta: float, alpha: float, adjacent_edge: Iterable[int], steps: int) -> float:
    """Regime-specific single-channel Renyi-DP privacy budget.

    Dense graph families are scored with the dense sensitivity branch and
    sparse graph families with the sparse propagation branch.  This keeps the
    finite benchmark aligned with the asymptotic regimes it is testing.
    """
    v1, v2 = map(int, adjacent_edge)
    n = game.n_players
    a = game.n_actions
    hdeg = harmonic_mean_degree(game)
    club = 16.0 * (a**3) * (np.log(n) ** 2) / (hdeg ** (4.0 / 9.0)) + 4.0 * a / n
    dists = graph_distances(game.n_players, game.edges, [v1, v2])
    spade = 2.0 * a / n * float(np.sum(steps > dists))
    regime_term = club if hdeg > np.log(n) else spade
    return float(alpha * eta * eta * regime_term)


def benchmark_hyperparameters(n_players: int, topology: str, p: float, c: int) -> tuple[int, float, float]:
    """Finite-scale default step, learning-rate, and noise schedule."""
    if topology.startswith("dense"):
        exponent = max(0.2, min(1.0, np.log(max(p * n_players, 2.0)) / np.log(n_players)))
        steps = max(8, int(round((n_players ** (8.0 * exponent / 9.0)) / (np.log(n_players) ** 2))))
    else:
        sparse_steps = (1.0 - np.log(np.log(n_players)) / np.log(n_players)) * np.log(n_players) / np.log(max(c, 2))
        steps = max(2, int(round(sparse_steps)))
    eta = 1.0 / (steps ** (1.0 / 3.0))
    sigma = 1.0 / np.sqrt(steps)
    return steps, eta, sigma


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


def save_game(path: str | Path, game: PolymatrixGame) -> None:
    np.savez_compressed(path, n_players=game.n_players, n_actions=game.n_actions, edges=game.edges, utilities=game.utilities)


def load_game(path: str | Path) -> PolymatrixGame:
    data = np.load(path)
    return PolymatrixGame(int(data["n_players"]), int(data["n_actions"]), data["edges"], data["utilities"])
