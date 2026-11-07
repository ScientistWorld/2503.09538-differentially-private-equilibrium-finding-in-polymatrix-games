"""Prior-work-style private baseline at the same privacy budget.

The paper compares against a private distributed learner adapted from
single-objective DP optimization (Huang et al., 2015) run "under the same
differential privacy budget".  Such a learner cannot exploit the polymatrix
graph structure in its privacy accounting.  This baseline therefore:

* broadcasts with the benchmark's fixed noise for the same T rounds,
* takes plain projected gradient steps anchored at its projected noisy
  broadcast, with no graph-aware regularization, and
* calibrates its step size so that a structure-free single-edge accounting
  (every observed player treated as affected by the changed edge, i.e. the
  locality term with all N players counted, 2A) equals the per-instance
  budget that the paper's accounting min{dense term, locality term} assigns
  to the operating point:  gamma = eta * sqrt(min(club, spade_max) / (2A)).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.polymatrix import (
    SEEDS,
    SIZES,
    TOPOLOGIES,
    benchmark_game,
    benchmark_hyperparameters,
    broadcast_noise,
    dense_sensitivity_term,
    gradients,
    locality_fraction_max,
    payoff_operator,
    project_simplex,
)
from data.submission import save_run, write_manifest


def budget_matched_step(game, steps: int, eta: float) -> float:
    a = game.n_actions
    club = dense_sensitivity_term(game)
    spade_max = 2.0 * a * locality_fraction_max(game, steps)
    return float(eta * min(1.0, np.sqrt(min(club, spade_max) / (2.0 * a))))


def run_structure_free(game, steps: int, step: float, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, a = game.n_players, game.n_actions
    strategies = np.full((n, a), 1.0 / a, dtype=np.float64)
    operator = payoff_operator(game)
    trajectory, broadcasts = [], []
    for t in range(steps):
        noisy = strategies + noise[t]
        projected_noisy = project_simplex(noisy)
        grads = gradients(game, projected_noisy, operator)
        strategies = project_simplex(projected_noisy - step * grads)
        trajectory.append(strategies.copy())
        broadcasts.append(noisy.copy())
    return np.stack(trajectory, axis=0), np.stack(broadcasts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/structure_free_baseline")
    parser.add_argument("--method-name", default="structure_free_baseline")
    parser.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES))
    parser.add_argument("--n-values", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()

    for topology in args.topologies:
        out = Path(args.output_dir) / topology
        runs = []
        for n in args.n_values:
            steps, eta, sigma = benchmark_hyperparameters(n, topology)
            for seed in args.seeds:
                game = benchmark_game(topology, n, seed)
                step = budget_matched_step(game, steps, eta)
                traj, broadcasts = run_structure_free(game, steps, step, broadcast_noise(topology, n, seed))
                runs.append(save_run(out, args.method_name, topology, n, seed, traj, broadcasts,
                                     steps=steps, eta=eta, sigma=sigma, budget_matched_step=step))
                print(f"{args.method_name} {topology} n={n} seed={seed} step={step:.4f}", flush=True)
        write_manifest(out, args.method_name, runs)


if __name__ == "__main__":
    main()
