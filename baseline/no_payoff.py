"""Simple baseline: follow the private protocol without ever using payoffs.

Each round the player moves to the most-regularized point the protocol allows
from its projected noisy broadcast P, namely (P + eta * uniform) / (1 + eta).
No utility is ever read, so the run is trivially private; it shows what the
noisy protocol alone achieves and is the floor any payoff-using method should
clear.
"""

from __future__ import annotations

import argparse

import numpy as np

from data.polymatrix import project_simplex
from data.submission import write_policy


class NoPayoffPlayer:
    def __init__(self, view: dict):
        a = int(view["n_actions"])
        self.uniform = np.full(a, 1.0 / a)
        self.eta = float(view["eta"])

    def step(self, t: int, own_broadcast: np.ndarray, received: np.ndarray) -> np.ndarray:
        return (project_simplex(own_broadcast) + self.eta * self.uniform) / (1.0 + self.eta)


def make_player(view: dict) -> NoPayoffPlayer:
    return NoPayoffPlayer(view)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the payoff-blind protocol baseline as a player policy.")
    parser.add_argument("--output-dir", default="outputs/no_payoff_baseline")
    parser.add_argument("--method-name", default="no_payoff_baseline")
    args = parser.parse_args()
    path = write_policy(args.output_dir, args.method_name, "baseline.no_payoff", "make_player")
    print(f"registered {args.method_name} -> {path}")


if __name__ == "__main__":
    main()
