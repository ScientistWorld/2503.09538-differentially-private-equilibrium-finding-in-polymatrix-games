"""Simple baseline: follow the private protocol without ever using payoffs.

Each round the player broadcasts its strategy plus the fixed noise and then
moves to the most-regularized point the protocol allows from its projected
noisy broadcast P, namely (P + eta * uniform) / (1 + eta).  No utility is ever
read, so the run is trivially private; it shows what the noisy protocol alone
achieves and is the floor any payoff-using method should clear.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.polymatrix import SEEDS, SIZES, TOPOLOGIES, N_ACTIONS, benchmark_hyperparameters, broadcast_noise, project_simplex
from data.submission import save_run, write_manifest


def run_no_payoff(n_players: int, steps: int, eta: float, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniform = np.full((n_players, N_ACTIONS), 1.0 / N_ACTIONS)
    strategies = uniform.copy()
    trajectory, broadcasts = [], []
    for t in range(steps):
        noisy = strategies + noise[t]
        strategies = (project_simplex(noisy) + eta * uniform) / (1.0 + eta)
        trajectory.append(strategies.copy())
        broadcasts.append(noisy.copy())
    return np.stack(trajectory, axis=0), np.stack(broadcasts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/no_payoff_baseline")
    parser.add_argument("--method-name", default="no_payoff_baseline")
    parser.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES))
    parser.add_argument("--n-values", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()

    for topology in args.topologies:
        out = Path(args.output_dir) / topology
        runs = []
        for n in args.n_values:
            steps, eta, _ = benchmark_hyperparameters(n, topology)
            for seed in args.seeds:
                traj, broadcasts = run_no_payoff(n, steps, eta, broadcast_noise(topology, n, seed))
                runs.append(save_run(out, args.method_name, topology, n, seed, traj, broadcasts, steps=steps, eta=eta))
        write_manifest(out, args.method_name, runs)


if __name__ == "__main__":
    main()
