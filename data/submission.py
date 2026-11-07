"""Submission contract helpers shared by any method (problem-side, method-agnostic).

A submission is a *player policy* registered by one ``policy.json`` file in its
own directory under ``outputs/`` (``outputs/<dir>/policy.json``):

    {"method": "our_method",            # name of the scored entry
     "module": "method.my_policy",      # importable from the workspace root
     "factory": "make_player",          # make_player(view) -> object with .step(...)
     "options": {}}                     # JSON passed to every player as view["options"]

The evaluators run the private protocol themselves with this policy on their
own instances (see ``data/protocol.py`` for the interface).  ``our_method`` is
the entry ranked against the reference.  Any other field is ignored.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_policy(out_dir: str | Path, method: str, module: str, factory: str = "make_player",
                 options: dict | None = None, **info) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "policy.json"
    payload = {"method": method, "module": module, "factory": factory, "options": dict(options or {})}
    payload.update(info)
    path.write_text(json.dumps(payload, indent=2))
    return path
