"""Output contract helpers shared by any method (problem-side, method-agnostic).

A submission is one directory per (method, topology) holding ``manifest.json``
and one ``.npz`` file per instance with two arrays of shape (T, N, A):

* ``trajectory`` -- the strategy profile played after each of the T rounds;
* ``broadcasts`` -- the message each player broadcast in each round, i.e. the
  profile played before that round (uniform before round 0) plus the fixed
  instance noise from ``data.polymatrix.broadcast_noise``.

Only these arrays and the instance identity (topology, n_players, seed) are
read by the evaluators; every other manifest field is informational.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_run(out_dir: str | Path, method: str, topology: str, n_players: int, seed: int,
             trajectory: np.ndarray, broadcasts: np.ndarray, **info) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{method}_{topology}_n{n_players}_seed{seed}.npz"
    np.savez_compressed(path, trajectory=np.asarray(trajectory, dtype=np.float64),
                        broadcasts=np.asarray(broadcasts, dtype=np.float64))
    run = {"method": method, "topology": topology, "n_players": int(n_players), "seed": int(seed),
           "trajectory_path": str(path)}
    run.update(info)
    return run


def write_manifest(out_dir: str | Path, method: str, runs: list[dict]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    path.write_text(json.dumps({"method": method, "runs": runs}, indent=2))
    return path
