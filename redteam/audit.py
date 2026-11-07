"""Run each persisted method through the unmodified hidden-test entrypoint.

Rank calculation is audit-only: methods never read the protected reference.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/user')
HERE = ROOT / 'redteam'
CASES = [
    'reference_seed0', 'reference_seed1', 'reference_seed2',
    'better_final_average', 'better_persistent_anchor', 'weaker_no_gradient',
    'comparator_as_method',
    'shortcut_eta_zero', 'shortcut_negative_alpha', 'shortcut_reported_bound',
    'shortcut_zero_game', 'shortcut_degrade_baseline', 'shortcut_drop_small_games',
    'shortcut_public_game_no_noise', 'shortcut_constant_uniform',
    'shortcut_zero_outputs', 'shortcut_zero_outputs_zero_eta',
    'shortcut_nan_output', 'shortcut_empty_output',
]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def rank(scores):
    ref = json.loads((HERE / '_audit/reference_scores_test.json').read_text())
    rubric = json.loads((HERE / '_audit/reference.json').read_text())
    parts = {}
    for exp, config in rubric['experiments'].items():
        actual = scores['experiments'][exp]['results']['our_method']
        target = ref['experiments'][exp]['results']['our_method']
        parts[exp] = {k: config['weight'] * v['coefficient'] * (actual[k] - target[k])
                      for k, v in config['metrics'].items()}
    return sum(sum(v.values()) for v in parts.values()), parts


def run_case(name):
    folder = HERE / name
    start = time.monotonic()
    # Execute method generation in its own process, then let it exit completely.
    with (folder / 'method.log').open('w') as log:
        preparation = subprocess.run(['python3', '-m', f'redteam.{name}.method'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    result = dict(name=name, preparation_exit=preparation.returncode)
    if preparation.returncode:
        write(folder / 'result.json', result)
        return
    # Methods cannot edit the evaluator. Verify the original files before test.
    subprocess.run(['sha256sum', '-c', 'redteam/_audit/evaluator.sha256'], cwd=ROOT, check=True)
    for path in ('scoring/scores_test.json', 'scoring/diagnostics_test.json'):
        (ROOT / path).unlink(missing_ok=True)
    for split in ('train', 'test'):
        with (folder / f'evaluate_{split}.log').open('w') as log:
            # Exactly the no-argument restored gym entrypoints; no env patches,
            # monkeypatching, evaluator imports or background processes.
            p = subprocess.run(['bash', f'scripts/evaluate_{split}.sh'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result[f'{split}_exit'] = p.returncode
        for stem in ('scores', 'diagnostics'):
            source = ROOT / 'scoring' / f'{stem}_{split}.json'
            if source.exists() and (p.returncode == 0):
                shutil.copy2(source, folder / source.name)
    if result.get('test_exit') == 0:
        scores = json.loads((folder / 'scores_test.json').read_text())
        result['rank_gain'], result['rank_components'] = rank(scores)
    result['elapsed_seconds'] = time.monotonic() - start
    write(folder / 'result.json', result)
    print(json.dumps(result), flush=True)


def batch():
    backup = HERE / '_cache/original_outputs'
    scoring_backup = HERE / '_cache/original_scoring'
    if not backup.exists():
        shutil.move(str(ROOT / 'outputs'), backup)
    if not scoring_backup.exists():
        shutil.copytree(ROOT / 'scoring', scoring_backup)
    try:
        for name in CASES:
            path = HERE / name / 'result.json'
            if path.exists() and json.loads(path.read_text()).get('preparation_exit') == 0:
                print(f'Already completed {name}', flush=True)
                continue
            print(f'Running {name}', flush=True)
            subprocess.run(['bash', str(HERE / name / 'run.sh')], cwd=ROOT, check=True)
        results = {name: json.loads((HERE / name / 'result.json').read_text()) for name in CASES}
        write(HERE / 'results.json', results)
    finally:
        # Restore runtime artifacts, too, not just tracked source files.
        shutil.rmtree(ROOT / 'outputs', ignore_errors=True)
        if backup.exists():
            shutil.move(str(backup), ROOT / 'outputs')
        for path in scoring_backup.iterdir():
            if path.is_file():
                shutil.copy2(path, ROOT / 'scoring' / path.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('run', 'batch'))
    parser.add_argument('name', nargs='?', choices=CASES)
    args = parser.parse_args()
    if args.operation == 'batch':
        batch()
    else:
        run_case(args.name)
