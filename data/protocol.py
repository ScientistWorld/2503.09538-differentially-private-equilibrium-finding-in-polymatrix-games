"""The private communication protocol and the player-policy interface (public).

A submission is a *player policy*: a factory ``make_player(view)`` that every
player of a game calls once with its own local view, returning an object whose
``step`` method is called once per round:

    player = make_player(view)
    x = player.step(t, own_broadcast, received)      # t = 0 .. T-1

``view`` (one per player, built by the protocol runner):

    player      int               the player's own index
    n_players   int               N
    n_actions   int               A (5 in the benchmark)
    edges       (E, 2) int64      the public interaction graph (read-only)
    neighbors   (d,) int64        the player's neighbours, ascending
    utilities   (d, A, A) float   the player's own payoff matrices:
                                  utilities[k][a, b] = payoff when it plays a and
                                  neighbors[k] plays b; its payoff is the
                                  average over neighbours of pi_i^T U_ik pi_k
    rounds, eta, sigma            the fixed operating point (T, step scale, noise)
    seed        int               seed for any randomness the policy needs
    options     dict              the ``options`` of the policy manifest

``step`` receives, for round ``t``: ``own_broadcast`` (A,) -- the player's own
message of this round, i.e. the strategy it is currently playing plus its
Gaussian noise -- and ``received`` (d, A) -- the messages of its neighbours
(their current strategies plus noise), in ``neighbors`` order.  It returns the
strategy (a probability vector of length A) the player plays next.  Before
round 0 every player plays the uniform strategy.

Nothing else is available to a player: not the other players' payoffs, not
anybody's noiseless strategy, not future noise.  The evaluators run this
protocol themselves (games and noise are theirs), verify the transcript and
re-execute randomly chosen players in a separate process from their local view
alone; a player whose output cannot be reproduced that way fails the instance.
So a step must be a deterministic function of the view and the messages
received so far (use ``view['seed']`` for randomness) and must not share state
between players.

This module runs the same protocol on the public development instances, for
development only; scores come from ``scripts/evaluate*.sh``.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from data.polymatrix import (
    PolymatrixGame,
    benchmark_game,
    benchmark_hyperparameters,
    broadcast_noise,
)


def neighbor_lists(n_players: int, edges: np.ndarray) -> list[np.ndarray]:
    adj = np.zeros((n_players, n_players), dtype=bool)
    edges = np.asarray(edges, dtype=np.int64)
    adj[edges[:, 0], edges[:, 1]] = True
    adj[edges[:, 1], edges[:, 0]] = True
    return [np.flatnonzero(adj[i]).astype(np.int64) for i in range(n_players)]


def player_views(game: PolymatrixGame, steps: int, eta: float, sigma: float,
                 options: dict | None = None, seed_base: int = 0) -> list[dict]:
    """One local view per player (same layout the evaluators use)."""
    edges = np.array(game.edges, dtype=np.int64)
    edges.setflags(write=False)
    nbrs = neighbor_lists(game.n_players, edges)
    views = []
    for i in range(game.n_players):
        views.append({
            "player": i,
            "n_players": game.n_players,
            "n_actions": game.n_actions,
            "edges": edges,
            "neighbors": nbrs[i].copy(),
            "utilities": game.utilities[i, nbrs[i]].copy(),
            "rounds": int(steps),
            "eta": float(eta),
            "sigma": float(sigma),
            "seed": int(seed_base) * 1_000_003 + i,
            "options": copy.deepcopy(options or {}),
        })
    return views


def read_policy(path: str | Path) -> dict:
    """Read a policy manifest: {"method", "module", "factory", "options"}."""
    spec = json.loads(Path(path).read_text())
    if not isinstance(spec, dict) or not isinstance(spec.get("module"), str):
        raise ValueError(f"{path}: a policy manifest needs a 'module' string")
    spec.setdefault("factory", "make_player")
    spec.setdefault("options", {})
    return spec


def load_policy(spec: dict) -> Callable[[dict], object]:
    module = importlib.import_module(spec["module"])
    return getattr(module, spec.get("factory", "make_player"))


def run_protocol(factory: Callable[[dict], object], game: PolymatrixGame, steps: int, eta: float,
                 sigma: float, noise: np.ndarray, options: dict | None = None,
                 seed_base: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run the private protocol; returns (trajectory, broadcasts), each (T, N, A).

    ``trajectory[t]`` is the profile played after round t, ``broadcasts[t]``
    the messages of round t (the profile played before round t plus noise[t]).
    """
    n, a = game.n_players, game.n_actions
    views = player_views(game, steps, eta, sigma, options, seed_base)
    nbrs = [v["neighbors"] for v in views]
    players = [factory(v) for v in views]
    current = np.full((n, a), 1.0 / a)
    trajectory = np.empty((steps, n, a))
    broadcasts = np.empty((steps, n, a))
    for t in range(steps):
        messages = current + noise[t]
        broadcasts[t] = messages
        for i, player in enumerate(players):
            out = player.step(t, messages[i].copy(), messages[nbrs[i]])
            trajectory[t, i] = np.asarray(out, dtype=np.float64).reshape(a)
        current = trajectory[t]
    return trajectory, broadcasts


def run_public_instance(factory: Callable[[dict], object], topology: str, n_players: int, seed: int,
                        options: dict | None = None) -> tuple[PolymatrixGame, np.ndarray, np.ndarray]:
    """Run a policy on one public development instance with its key-0 noise."""
    steps, eta, sigma = benchmark_hyperparameters(n_players, topology)
    game = benchmark_game(topology, n_players, seed)
    noise = broadcast_noise(topology, n_players, seed)
    trajectory, broadcasts = run_protocol(factory, game, steps, eta, sigma, noise, options, seed_base=seed)
    return game, trajectory, broadcasts
