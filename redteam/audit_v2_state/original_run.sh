#!/bin/bash
# Hardening job (CPU): build the fixed comparator values, reproduce the reference
# method and baselines end to end, score train/test, run the hardening audit
# (variants + shortcuts through the no-argument hidden test), validate.
#
# Stages are resumable: the comparator files and the audit's generated runs are
# reused if present.

set -uo pipefail

cd /home/user
mkdir -p tmp

stage() { echo; echo "=== $1 [$(date +%H:%M:%S)]"; }

stage "smoke test (tiny instances, checks reference passes the protocol gate)"
python - <<'PY' || { echo "SMOKE TEST FAILED"; exit 1; }
import numpy as np
import eval.test.evaluate as T
from data.polymatrix import benchmark_game, benchmark_hyperparameters, broadcast_noise
from method.dp_cce import run_algorithm1
from baseline.structure_free import budget_matched_step, run_structure_free
from baseline.no_payoff import run_no_payoff
for topology in ("dense_er", "dense_cluster", "sparse_knn"):
    n, seed = 64, 1
    steps, eta, sigma = benchmark_hyperparameters(n, topology)
    assert (steps, eta, sigma) == T.operating_point(n, topology)
    g_pub = benchmark_game(topology, n, seed)
    g_ev = T.make_game(topology, n, seed)
    assert np.array_equal(g_pub.edges, g_ev.edges)
    noise = broadcast_noise(topology, n, seed)
    assert np.array_equal(noise, T.noise_stream(topology, n, seed, steps, sigma))
    for name, (tr, bc) in {
        "reference": run_algorithm1(g_pub, steps, eta, noise),
        "structure_free": run_structure_free(g_pub, steps, budget_matched_step(g_pub, steps, eta), noise),
        "no_payoff": run_no_payoff(n, steps, eta, noise),
    }.items():
        ok, reason, stats = T.check_transcript(g_ev, steps, eta, noise, tr, bc)
        print(topology, name, ok, reason, stats, T.cce_exploitability(g_ev, tr))
        assert ok, (topology, name, reason)
    bad = run_algorithm1(g_pub, steps, 3 * eta, noise)
    print(topology, "3x step", T.check_transcript(g_ev, steps, eta, noise, *bad)[:2])
print("SMOKE OK")
PY

stage "fixed comparator values"
for split in train test; do
  if [ ! -f "eval/$split/comparator_$split.json" ]; then
    python -m "eval.$split.evaluate" --build-comparator > "tmp/comparator_$split.log" 2>&1 || { tail -30 "tmp/comparator_$split.log"; exit 1; }
    tail -3 "tmp/comparator_$split.log"
  fi
done

stage "reproduce reference method + baselines and score"
bash scripts/reproduce.sh > tmp/reproduce.log 2>&1 || { tail -40 tmp/reproduce.log; exit 1; }
grep -E "^\[(train|test)\]" tmp/reproduce.log

stage "hardening audit"
python -m redteam.hardening.harness > tmp/harness.log 2>&1 || { echo "AUDIT FAILED"; tail -60 tmp/harness.log; }
tail -45 tmp/harness.log

stage "final scores from the restored reference outputs"
bash scripts/evaluate.sh > /dev/null && bash scripts/evaluate_train.sh && bash scripts/evaluate_test.sh

stage "validate"
python validate.py --post-design --compare
echo "=== done [$(date +%H:%M:%S)]"
