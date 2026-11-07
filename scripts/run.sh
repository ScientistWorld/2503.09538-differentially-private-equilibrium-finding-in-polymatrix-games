#!/bin/bash
# Hardening job (CPU): smoke-test the player-policy interface, build the fixed
# comparator values of the hidden test instances, reproduce the reference
# method and baselines end to end (train + hidden test), run the hardening
# audit (variants and shortcuts through the no-argument hidden test), validate.

set -uo pipefail

cd /home/user
mkdir -p tmp

stage() { echo; echo "=== $1 [$(date +%H:%M:%S)]"; }

stage "smoke test (policies vs vectorized dynamics, protocol runner, replay audit)"
python - <<'PY' || { echo "SMOKE TEST FAILED"; exit 1; }
import json, tempfile
from pathlib import Path
import numpy as np
import eval.train.evaluate as T
from data.polymatrix import (benchmark_game, benchmark_hyperparameters, broadcast_noise, gradients,
                             payoff_operator, project_simplex, harmonic_mean_degree)
from data.protocol import run_public_instance
from method.dp_cce import make_player as alg1
from baseline.structure_free import make_player as sfree, budget_matched_step
from baseline.no_payoff import make_player as nopay

def vec_alg1(game, steps, eta, noise, tau=None, step=None):
    n, a = game.n_players, game.n_actions
    deg = np.maximum(game.degrees.astype(float), 1.0)
    if tau is None:
        tau = harmonic_mean_degree(game) ** (5 / 9) / (deg * np.log(n))
    s, op, out = np.full((n, a), 1 / a), payoff_operator(game), []
    for t in range(steps):
        p = project_simplex(s + noise[t])
        g = gradients(game, p, op)
        s = project_simplex((p - (eta if step is None else step) * g) / (1 + (eta if step is None else 0) * tau[:, None]))
        out.append(s.copy())
    return np.stack(out)

for topology in ("dense_er", "dense_cluster", "sparse_knn"):
    n, seed = 64, 0
    steps, eta, sigma = benchmark_hyperparameters(n, topology)
    assert (steps, eta, sigma) == T.operating_point(n, topology)
    game, tr, bc = run_public_instance(alg1, topology, n, seed)
    noise = broadcast_noise(topology, n, seed)
    ref = vec_alg1(game, steps, eta, noise)
    print(topology, "alg1 policy vs vectorized max diff", float(np.abs(tr - ref).max()))
    assert np.abs(tr - ref).max() < 1e-10
    _, tr_sf, _ = run_public_instance(sfree, topology, n, seed)
    step = budget_matched_step(n, game.n_actions, game.edges, steps, eta)
    ref_sf = vec_alg1(game, steps, eta, noise, tau=np.zeros(n), step=step)
    print(topology, "structure-free policy vs vectorized max diff", float(np.abs(tr_sf - ref_sf).max()), "step", step)
    assert np.abs(tr_sf - ref_sf).max() < 1e-10
    g_ev = T.make_game(topology, n, seed)
    assert np.array_equal(g_ev.edges, game.edges) and np.array_equal(g_ev.utilities, game.utilities)
    assert np.array_equal(noise, T.noise_stream(topology, n, seed, steps, sigma))
    ok, reason, stats, bcast = T.check_transcript(g_ev, steps, eta, noise, tr)
    print(topology, "reference transcript", ok, reason, stats)
    assert ok and np.allclose(bcast, bc)
    tr_ev = T.execute_policy(alg1, topology, n, seed, {})
    assert np.abs(tr_ev - tr).max() < 1e-12
    _, tr_np, _ = run_public_instance(nopay, topology, n, seed)
    assert T.check_transcript(g_ev, steps, eta, noise, tr_np)[0]
    # isolated replay of audited players through a real child process
    spec = Path(tempfile.mkdtemp()) / "policy.json"
    spec.write_text(json.dumps({"method": "x", "module": "method.dp_cce", "factory": "make_player"}))
    job = {"key": T.instance_key(topology, n, seed), "n": n, "a": 5, "steps": steps, "eta": eta, "sigma": sigma,
           "edges": g_ev.edges,
           "items": [{"player": i, "seed": T.player_seed(topology, n, seed, i), "utilities": g_ev.utilities[i, g_ev.nbrs[i]],
                      "neighbors": g_ev.nbrs[i], "own": bcast[:, i, :], "received": bcast[:, g_ev.nbrs[i], :],
                      "expected": tr_ev[:, i, :]} for i in T.audited_players(topology, n, seed)]}
    failures, astats = T.run_replays(spec, [job], Path(tempfile.mkdtemp()))
    print(topology, "replay audit", failures, astats)
    assert not failures and float(astats["max_deviation"]) < 1e-12
    job["items"][0]["expected"] = job["items"][0]["expected"] + 1e-3
    failures, _ = T.run_replays(spec, [job], Path(tempfile.mkdtemp()))
    assert failures, "audit must reject a non-reproducible player"
print("SMOKE OK")
PY

stage "fixed comparator values (train kept; hidden test instances built if missing)"
for split in train test; do
  if [ ! -f "eval/$split/comparator_$split.json" ]; then
    python -m "eval.$split.evaluate" --build-comparator > "tmp/comparator_$split.log" 2>&1 || { tail -30 "tmp/comparator_$split.log"; exit 1; }
    tail -3 "tmp/comparator_$split.log"
  fi
done

stage "reproduce reference method + baselines and score (train + hidden test)"
git show HEAD:scoring/scores_train.json > tmp/scores_train_before.json 2>/dev/null || true
bash scripts/reproduce.sh > tmp/reproduce.log 2>&1 || { tail -60 tmp/reproduce.log; exit 1; }
grep -E "^\[(train|test)\]" tmp/reproduce.log
python - <<'PY'
import json
old = json.load(open("tmp/scores_train_before.json"))
new = json.load(open("scoring/scores_train.json"))
diff = max(abs(new["experiments"][e]["results"][m][k] - v)
           for e, b in old["experiments"].items() for m, r in b["results"].items() for k, v in r.items())
print("train scores through the player interface vs. the previous transcript evaluator: max |diff| =", diff)
PY

stage "hardening audit"
python -m redteam.hardening_v3.harness > tmp/harness.log 2>&1 || { echo "AUDIT FAILED"; tail -80 tmp/harness.log; }
tail -30 tmp/harness.log

stage "validate"
python validate.py --post-design --compare
echo "=== done [$(date +%H:%M:%S)]"
