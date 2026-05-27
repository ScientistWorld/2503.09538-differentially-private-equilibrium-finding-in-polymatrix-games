"""Baseline: noisy online projected gradient without degree-adaptive decay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.polymatrix import generate_game, harmonic_mean_degree, project_simplex, save_game, theorem8_hyperparameters, theorem8_privacy_bound


def run_baseline(game, steps: int, eta: float, sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, a = game.n_players, game.n_actions
    strategies = np.full((n, a), 1.0 / a, dtype=np.float64)
    trajectory = []
    broadcast_trajectory = []
    deg = np.maximum(game.degrees.astype(np.float64), 1.0)
    tau = (harmonic_mean_degree(game) ** (5.0 / 9.0)) / (float(deg.min()) * np.log(n))
    for _ in range(steps):
        noisy = strategies + rng.normal(0.0, sigma, size=strategies.shape)
        projected_noisy = project_simplex(noisy)
        grads = np.zeros_like(strategies)
        for i, j in game.edges:
            grads[i] -= game.utilities[i, j] @ projected_noisy[j] / deg[i]
            grads[j] -= game.utilities[j, i] @ projected_noisy[i] / deg[j]
        strategies = project_simplex((projected_noisy - eta * grads) / (1.0 + eta * tau))
        trajectory.append(strategies.copy())
        broadcast_trajectory.append(noisy.copy())
    return np.stack(trajectory, axis=0), np.stack(broadcast_trajectory, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--topology", choices=["dense_er", "dense_cluster", "sparse_random", "sparse_knn"], default="dense_er")
    parser.add_argument("--n-values", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--actions", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.12)
    parser.add_argument("--c", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=2.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for n in args.n_values:
        steps, eta, sigma = theorem8_hyperparameters(n, args.topology, args.p, args.c)
        for seed in args.seeds:
            game = generate_game(n, args.actions, args.topology, seed=1000 + 17 * seed + n, p=args.p, c=args.c)
            edge = game.edges[0].tolist()
            traj, broadcasts = run_baseline(game, steps, eta, sigma, seed=3000 + seed)
            name = f"noisy_omd_{args.topology}_n{n}_seed{seed}"
            game_path = out / f"{name}_game.npz"
            traj_path = out / f"{name}_trajectory.npz"
            save_game(game_path, game)
            np.savez_compressed(traj_path, trajectory=traj, broadcasts=broadcasts)
            runs.append({
                "method": "noisy_omd",
                "topology": args.topology,
                "n_players": n,
                "n_actions": args.actions,
                "steps": steps,
                "eta": eta,
                "sigma": sigma,
                "alpha": args.alpha,
                "adjacent_edge": edge,
                "game_path": str(game_path),
                "trajectory_path": str(traj_path),
                "privacy_bound": theorem8_privacy_bound(game, eta, args.alpha, edge, steps),
            })
    (out / "manifest.json").write_text(json.dumps({"runs": runs}, indent=2))


if __name__ == "__main__":
    main()
