"""Locality audit: isolated re-execution of audited players (train slice).

Started by ``eval/train/evaluate.py`` in a fresh process for one policy.  It
reads the audited players' local views and received messages from
``--workdir``, builds each player with the submitted factory, replays its
rounds, and writes the strategies it returns.  The scoring process compares
them with the transcript.

This module holds no instance generator, noise or secret, and it blocks
imports of the evaluator package before the policy is loaded.  A replayed
player therefore has nothing but its own view and the messages it received;
a player whose transcript depended on anything else is not reproduced.

Usage (internal):  python -m eval.train.replay --policy P --workdir D
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import sys
from importlib.abc import MetaPathFinder
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_MODULE_RE = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")


class _NoEvaluator(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "eval" or name.startswith("eval."):
            raise ImportError("evaluator modules are not available to submitted code")
        return None


def _isolate() -> None:
    for name in [k for k in sys.modules if k == "eval" or k.startswith("eval.")]:
        del sys.modules[name]
    sys.meta_path.insert(0, _NoEvaluator())


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _load_factory(policy_path: Path):
    raw = json.loads(Path(policy_path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("policy manifest is not a JSON object")
    module, factory, options = raw.get("module"), raw.get("factory", "make_player"), raw.get("options", {})
    if not isinstance(module, str) or not _MODULE_RE.fullmatch(module) or module.split(".")[0] == "eval":
        raise ValueError("'module' must be a dotted module name importable from the workspace root")
    if not isinstance(factory, str) or not factory.isidentifier() or not isinstance(options, dict):
        raise ValueError("invalid 'factory' or 'options'")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    fn = getattr(importlib.import_module(module), factory)
    if not callable(fn):
        raise TypeError("factory is not callable")
    return fn, options


def replay(policy_path: Path, iodir: Path) -> None:
    errors: dict[str, str] = {}
    outputs: dict[int, np.ndarray] = {}
    try:
        meta = json.loads((iodir / "replay_in.json").read_text())
        with np.load(iodir / "replay_in.npz", allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
        _isolate()
        factory, options = _load_factory(policy_path)
    except BaseException as exc:  # noqa: BLE001
        (iodir / "replay_status.json").write_text(json.dumps({"_policy": _describe(exc)}))
        return
    for inst in meta["instances"]:
        edges = np.array(data[f"edges{inst['j']}"], dtype=np.int64)
        edges.setflags(write=False)
        steps, a = int(inst["steps"]), int(inst["a"])
        players = {}
        for item in inst["items"]:
            m = int(item["m"])
            view = {"player": int(item["player"]), "n_players": int(inst["n"]), "n_actions": a, "edges": edges,
                    "neighbors": np.array(data[f"nb{m}"], dtype=np.int64),
                    "utilities": np.array(data[f"u{m}"], dtype=np.float64),
                    "rounds": steps, "eta": float(inst["eta"]), "sigma": float(inst["sigma"]),
                    "seed": int(item["seed"]), "options": copy.deepcopy(options)}
            try:
                players[m] = factory(view)
                outputs[m] = np.full((steps, a), np.nan)
            except BaseException as exc:  # noqa: BLE001
                errors[str(m)] = _describe(exc)
        for t in range(steps):
            for m in list(players):
                try:
                    out = players[m].step(t, data[f"own{m}"][t].copy(), data[f"rx{m}"][t].copy())
                    outputs[m][t] = np.asarray(out, dtype=np.float64).reshape(a)
                except BaseException as exc:  # noqa: BLE001
                    errors[str(m)] = _describe(exc)
                    del players[m]
    np.savez(iodir / "replay_out.npz", **{f"out{m}": v for m, v in outputs.items()})
    (iodir / "replay_status.json").write_text(json.dumps(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    replay(Path(args.policy), Path(args.workdir))


if __name__ == "__main__":
    main()
