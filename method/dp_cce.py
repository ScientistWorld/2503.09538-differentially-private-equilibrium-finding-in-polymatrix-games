"""Algorithm 1: differentially private CCE computation in polymatrix games.

Each player broadcasts its mixed strategy plus Gaussian noise, projects the
received noisy strategies back to the simplex, and takes a proximal projected
gradient step that starts from its own projected noisy broadcast and uses the
degree-adaptive regularizer tau_i = (harmonic-mean degree)^(5/9) / (deg_i log N).
The broadcast noise is the benchmark's fixed per-instance stream.
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
    gradients,
    harmonic_mean_degree,
    payoff_operator,
    project_simplex,
)
from data.submission import save_run, write_manifest


def degree_adaptive_regularizer(game) -> np.ndarray:
    deg = np.maximum(game.degrees.astype(np.float64), 1.0)
    return (harmonic_mean_degree(game) ** (5.0 / 9.0)) / (deg * np.log(game.n_players))


def run_algorithm1(game, steps: int, eta: float, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, a = game.n_players, game.n_actions
    strategies = np.full((n, a), 1.0 / a, dtype=np.float64)
    tau = degree_adaptive_regularizer(game)
    operator = payoff_operator(game)
    trajectory, broadcasts = [], []
    for t in range(steps):
        noisy = strategies + noise[t]
        projected_noisy = project_simplex(noisy)
        grads = gradients(game, projected_noisy, operator)
        strategies = project_simplex((projected_noisy - eta * grads) / (1.0 + eta * tau[:, None]))
        trajectory.append(strategies.copy())
        broadcasts.append(noisy.copy())
    return np.stack(trajectory, axis=0), np.stack(broadcasts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/proposed")
    parser.add_argument("--method-name", default="our_method")
    parser.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES))
    parser.add_argument("--n-values", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--noise-key", type=int, default=0,
                        help="Alternative noise stream for robustness studies; the benchmark uses 0.")
    args = parser.parse_args()

    for topology in args.topologies:
        out = Path(args.output_dir) / topology
        runs = []
        for n in args.n_values:
            steps, eta, sigma = benchmark_hyperparameters(n, topology)
            for seed in args.seeds:
                game = benchmark_game(topology, n, seed)
                noise = broadcast_noise(topology, n, seed, key=args.noise_key)
                traj, broadcasts = run_algorithm1(game, steps, eta, noise)
                runs.append(save_run(out, args.method_name, topology, n, seed, traj, broadcasts,
                                     steps=steps, eta=eta, sigma=sigma))
                print(f"{args.method_name} {topology} n={n} seed={seed} done", flush=True)
        write_manifest(out, args.method_name, runs)


if __name__ == "__main__":
    main()
