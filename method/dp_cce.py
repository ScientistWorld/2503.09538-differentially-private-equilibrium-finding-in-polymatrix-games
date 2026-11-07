"""Algorithm 1: differentially private CCE computation in polymatrix games.

Per-player policy (interface in data/protocol.py).  In every round player i

  1. broadcasts its strategy plus Gaussian noise (the protocol runner adds the
     noise and hands the player its own noisy message ``own_broadcast``),
  2. projects the noisy strategies received from its neighbours onto the
     simplex, bar pi_j = Proj(pi_j + n_j),
  3. computes its loss gradient g_i = -(1/|N(i)|) sum_j U_ij bar pi_j, and
  4. takes a proximal projected-gradient step that starts from its own
     projected noisy broadcast bar pi_i, with the degree-adaptive regularizer
     tau_i = hdeg^(5/9) / (|N(i)| log N), hdeg the harmonic mean of degrees:

         pi_i <- Proj( (bar pi_i - eta g_i) / (1 + eta tau_i) ).

Only the player's own payoff matrices, the public graph and the noisy
messages are used.
"""

from __future__ import annotations

import argparse

import numpy as np

from data.polymatrix import harmonic_mean_degree_from_edges, project_simplex
from data.submission import write_policy


def degree_adaptive_regularizer(n_players: int, edges: np.ndarray, degree: int) -> float:
    hdeg = harmonic_mean_degree_from_edges(n_players, edges)
    return float(hdeg ** (5.0 / 9.0) / (max(degree, 1) * np.log(n_players)))


class Algorithm1Player:
    def __init__(self, view: dict):
        self.utilities = np.asarray(view["utilities"], dtype=np.float64)
        self.degree = max(len(view["neighbors"]), 1)
        self.eta = float(view["eta"])
        self.tau = degree_adaptive_regularizer(int(view["n_players"]), view["edges"], self.degree)

    def gradient(self, received: np.ndarray) -> np.ndarray:
        projected = project_simplex(received)
        return -np.einsum("kab,kb->a", self.utilities, projected) / self.degree

    def step(self, t: int, own_broadcast: np.ndarray, received: np.ndarray) -> np.ndarray:
        anchor = project_simplex(own_broadcast)
        grad = self.gradient(received)
        return project_simplex((anchor - self.eta * grad) / (1.0 + self.eta * self.tau))


def make_player(view: dict) -> Algorithm1Player:
    return Algorithm1Player(view)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register Algorithm 1 as a player policy.")
    parser.add_argument("--output-dir", default="outputs/proposed")
    parser.add_argument("--method-name", default="our_method")
    args = parser.parse_args()
    path = write_policy(args.output_dir, args.method_name, "method.dp_cce", "make_player")
    print(f"registered {args.method_name} -> {path}")


if __name__ == "__main__":
    main()
