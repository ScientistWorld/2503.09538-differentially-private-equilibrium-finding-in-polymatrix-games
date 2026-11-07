"""Prior-work-style private baseline at the same privacy budget (player policy).

The paper compares against a private distributed learner adapted from
single-objective DP optimization (Huang et al., 2015) run "under the same
differential privacy budget".  Such a learner cannot exploit the polymatrix
graph structure in its privacy accounting.  Each player of this baseline:

* follows the same protocol (same rounds, same broadcast noise),
* takes plain projected gradient steps anchored at its projected noisy
  broadcast, with no graph-aware regularization, and
* uses a step size calibrated so that a structure-free single-edge accounting
  (every observed player treated as affected by the changed edge, i.e. the
  locality term with all N players counted, 2A) equals the per-instance
  budget that the paper's accounting min{dense term, locality term} assigns
  to the operating point:  gamma = eta * sqrt(min(club, spade_max) / (2A)).

The step depends only on the public graph and operating point.  The
evaluators hold the fixed comparator values of this learner on their own
instances; submitting it as a method reproduces them.
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np

from data.polymatrix import (
    dense_sensitivity_term_from_edges,
    locality_fraction_max_from_edges,
    project_simplex,
)
from data.submission import write_policy

# Public-graph computations are shared between the players of one game; the
# cache is keyed by everything the value depends on.
_STEP_CACHE: dict[tuple, float] = {}


def budget_matched_step(n_players: int, n_actions: int, edges: np.ndarray, steps: int, eta: float) -> float:
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    key = (n_players, n_actions, steps, float(eta), hashlib.sha1(edges.tobytes()).hexdigest())
    if key not in _STEP_CACHE:
        club = dense_sensitivity_term_from_edges(n_players, n_actions, edges)
        spade_max = 2.0 * n_actions * locality_fraction_max_from_edges(n_players, edges, steps)
        _STEP_CACHE[key] = float(eta * min(1.0, np.sqrt(min(club, spade_max) / (2.0 * n_actions))))
    return _STEP_CACHE[key]


class StructureFreePlayer:
    def __init__(self, view: dict):
        self.utilities = np.asarray(view["utilities"], dtype=np.float64)
        self.degree = max(len(view["neighbors"]), 1)
        self.step_size = budget_matched_step(int(view["n_players"]), int(view["n_actions"]), view["edges"],
                                             int(view["rounds"]), float(view["eta"]))

    def step(self, t: int, own_broadcast: np.ndarray, received: np.ndarray) -> np.ndarray:
        anchor = project_simplex(own_broadcast)
        grad = -np.einsum("kab,kb->a", self.utilities, project_simplex(received)) / self.degree
        return project_simplex(anchor - self.step_size * grad)


def make_player(view: dict) -> StructureFreePlayer:
    return StructureFreePlayer(view)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the structure-free private baseline as a player policy.")
    parser.add_argument("--output-dir", default="outputs/structure_free_baseline")
    parser.add_argument("--method-name", default="structure_free_baseline")
    args = parser.parse_args()
    path = write_policy(args.output_dir, args.method_name, "baseline.structure_free", "make_player")
    print(f"registered {args.method_name} -> {path}")


if __name__ == "__main__":
    main()
