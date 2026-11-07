"""Verify saved audit evidence and summarize it without rerunning experiments.

This is audit reporting code, never part of candidate generation or evaluation.
Only Python's standard library and small JSON files are used.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from redteam.audit import CASES, rank, write

HERE = Path(__file__).resolve().parent


def summarize():
    rubric = json.loads((HERE / '_audit/reference.json').read_text())['experiments']
    reference = json.loads((HERE / '_audit/reference_scores_test.json').read_text())
    results = json.loads((HERE / 'results.json').read_text())
    assert set(results) == set(CASES)
    metrics = ('cce_exploitability_mean', 'final_nash_gap_mean',
               'privacy_budget_mean', 'baseline_cce_ratio',
               'strategy_simplex_violation_mean')
    rows = {}
    for name in CASES:
        folder = HERE / name
        saved = json.loads((folder / 'result.json').read_text())
        assert saved == results[name], name
        assert (folder / 'run.sh').is_file() and (folder / 'method.py').is_file()
        assert saved['preparation_exit'] == 0
        if saved['test_exit']:
            assert not (folder / 'scores_test.json').exists()
            rows[name] = {'test_exit': saved['test_exit'], 'rank_gain': None}
            continue
        scores = json.loads((folder / 'scores_test.json').read_text())
        gain, components = rank(scores)
        assert math.isclose(gain, saved['rank_gain'], rel_tol=1e-13, abs_tol=1e-13), name
        assert components == saved['rank_components'], name
        if name in ('reference_seed0', 'shortcut_reported_bound'):
            assert scores == reference, name
        means = {
            metric: sum(config['weight'] * scores['experiments'][exp]['results']['our_method'][metric]
                        for exp, config in rubric.items())
            for metric in metrics
        }
        rows[name] = {'test_exit': 0, 'rank_gain': gain, 'weighted_metrics': means}

    seeds = [rows[f'reference_seed{i}']['rank_gain'] for i in range(3)]
    spread = max(seeds) - min(seeds)
    assert rows['better_fixed_transcript_readout']['rank_gain'] > spread
    assert rows['weaker_no_gradient']['rank_gain'] < -spread
    assert rows['shortcut_zero_outputs_zero_eta']['weighted_metrics']['strategy_simplex_violation_mean'] == 1
    shortcuts = [value for name, value in rows.items() if name.startswith('shortcut_')]
    report = {
        'rank_definition': 'sum_experiments(weight * sum_metrics(coefficient * (score - reference)))',
        'seed_rank_gains': seeds,
        'seed_rank_range': spread,
        'seed_rank_population_std': statistics.pstdev(seeds),
        'seed_scope': 'algorithm RNG changes; the same fixed test-game grid',
        'shortcut_counts': {
            'above_reference': sum(v['rank_gain'] is not None and v['rank_gain'] > 0 for v in shortcuts),
            'tied_reference': sum(v['rank_gain'] == 0 for v in shortcuts),
            'below_reference': sum(v['rank_gain'] is not None and v['rank_gain'] < 0 for v in shortcuts),
            'rejected': sum(v['test_exit'] != 0 for v in shortcuts),
        },
        'cases': rows,
    }
    write(HERE / 'audit_summary.json', report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    summarize()
