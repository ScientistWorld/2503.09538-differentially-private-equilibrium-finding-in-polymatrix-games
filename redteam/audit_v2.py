"""Run each audit candidate, then the restored no-argument hidden-test command.

All method computation finishes in a separate process before evaluation starts.
No evaluator modules are imported here. The test script is never modified, and
the process environment is inherited unchanged. Candidate preparation imports
only public data, baseline and method code. JSON scoring happens afterwards.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
STATE = ROOT / 'redteam/audit_v2_state'
CASES = {
    'reference_seed0': 'Reference rerun with process seed 0 and contract noise key 0.',
    'reference_seed1': 'Independent reference rerun with process seed 1 and contract noise key 0.',
    'reference_seed2': 'Independent reference rerun with process seed 2 and contract noise key 0.',
    'better_max_regularization': 'Real fix: proximal weight 1, still within the allowed shrink segment.',
    'better_shrink_075': 'Real fix: anchor .75*projected own message + .25*uniform, public payoff gradient.',
    'weaker_no_gradient': 'Remove the payoff gradient, preserve reference regularization and communications.',
    'simple_baseline': 'Payoff-blind protocol baseline from baseline/no_payoff.py.',
    'comparator_baseline': 'Budget-matched structure-free baseline from the public baseline code.',
    'shortcut_global_private_optimizer': 'Shortcut: centrally minimize joint true-payoff Nash gap using other players\' private utilities and unnoised strategies, clipped to the transcript cap.',
    'shortcut_oracle_selection': 'Shortcut: reconstruct all games and future noise; select an entire transcript using global private payoff scores.',
    'shortcut_forged_claims': 'Shortcut: reuse reference arrays, forge scores and manifest privacy, quality, timing, size and success claims.',
    'shortcut_move_baseline': 'Shortcut: reuse reference arrays; replace on-disk baseline outputs with invalid zero strategies.',
    'shortcut_zero_game_artifact': 'Shortcut: reuse reference arrays and point game_path at replacement zero-utility game artifacts.',
    'shortcut_drop_small_games': 'Shortcut: omit every N=64 instance from the manifest.',
    'shortcut_failure_compensation': 'Shortcut: use oracle outputs except omit one smallest sparse instance in each split, seeking compensation from the other experiments.',
    'shortcut_constant_uniform': 'Shortcut: output constant uniform profiles for the correct number of rounds, with matching noisy broadcasts.',
    'shortcut_zero_outputs': 'Shortcut: output zero strategies for the correct number of rounds, with matching noisy broadcasts.',
    'shortcut_nonfinite': 'Shortcut: output NaN on odd seeds and infinity on even seeds.',
    'shortcut_empty': 'Shortcut: output zero rounds.',
    'shortcut_remove_noise': 'Shortcut: run the original updates without any privacy noise.',
}


def fingerprint():
    files = sorted((ROOT / 'eval').rglob('*.py')) + sorted((ROOT / 'eval').rglob('*.json'))
    files += [ROOT / 'scripts/evaluate_test.sh', ROOT / 'scripts/evaluate_train.sh']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def rank(scores, method='our_method'):
    reference = json.loads((STATE / 'reference_scores_test.json').read_text())
    rubric = json.loads((STATE / 'rubric.json').read_text())
    parts = {}
    for exp, spec in rubric['experiments'].items():
        actual = scores['experiments'][exp]['results'][method]
        base = reference['experiments'][exp]['results']['our_method']
        parts[exp] = {name: spec['weight'] * cfg['coefficient'] * (actual[name] - base[name])
                      for name, cfg in spec['metrics'].items()}
    return sum(sum(p.values()) for p in parts.values()), parts


def summarize():
    results = {}
    for case in CASES:
        folder = ROOT / 'redteam' / f'audit_v2_{case}'
        if (folder / 'result.json').exists():
            results[case] = json.loads((folder / 'result.json').read_text())
    rubric = json.loads((STATE / 'rubric.json').read_text())
    reference = json.loads((STATE / 'reference_scores_test.json').read_text())
    ceiling = 0.0
    failures = {}
    for exp, spec in rubric['experiments'].items():
        ref = reference['experiments'][exp]['results']['our_method']
        ceiling += spec['weight'] * sum(cfg['coefficient'] * ((1.0 if name == 'privacy_protocol_compliance' else 0.0) - ref[name])
                                       for name, cfg in spec['metrics'].items())
    for exp, spec in rubric['experiments'].items():
        # Optimistic upper bound for a task failure: all other errors zero,
        # all other compliance 1, and only one of 24 required instances fails.
        failures[exp] = ceiling - spec['weight'] * (10 * (1 + .5 + .25) + 1 / 24)
    seed_gains = [results[f'reference_seed{s}']['rank_gain'] for s in range(3)
                  if f'reference_seed{s}' in results and 'rank_gain' in results[f'reference_seed{s}']]
    hashes = []
    for seed in range(3):
        path = ROOT / 'redteam' / f'audit_v2_reference_seed{seed}/artifact_hashes.json'
        if path.exists():
            hashes.append(json.loads(path.read_text()))
    report = dict(results=results, best_mathematically_possible_gain=ceiling,
                  upper_bound_gain_with_one_failure=failures,
                  seed_rank_gains=seed_gains,
                  seed_rank_spread=max(seed_gains) - min(seed_gains) if seed_gains else None,
                  seed_transcripts_identical=len(hashes) == 3 and hashes[0] == hashes[1] == hashes[2])
    (STATE / 'summary.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    for case, res in results.items():
        print(case, res.get('rank_gain'), res.get('status'), flush=True)


def run_case(case):
    folder = ROOT / 'redteam' / f'audit_v2_{case}'
    folder.mkdir(exist_ok=True)
    backup = STATE / 'runtime/backup'
    if backup.exists():
        raise RuntimeError('A previous run left a backup; restore it before rerunning.')
    backup.mkdir(parents=True)
    scoring_snapshot = {p.name: p.read_bytes() for p in (ROOT / 'scoring').iterdir() if p.is_file()}
    if (ROOT / 'outputs').exists():
        shutil.move(ROOT / 'outputs', backup / 'outputs')
    (ROOT / 'outputs').mkdir()
    # Different source wrappers can have equal size and mtime. Reusing one
    # module filename can therefore reuse an earlier wrapper's valid .pyc.
    # A distinct module name per case prevents that ordinary import-cache bug.
    installed = ROOT / f'method/audit_v2_{case}.py'
    if installed.exists():
        raise RuntimeError(f'Refusing to overwrite existing {installed}')
    started = time.time()
    protected_before = fingerprint()
    result = dict(case=case, description=CASES[case], evaluator_command=['bash', 'scripts/evaluate_test.sh'])
    try:
        shutil.copy2(folder / 'method.py', installed)
        with (folder / 'generation.log').open('w') as log:
            proc = subprocess.run([sys.executable, '-m', f'method.audit_v2_{case}'], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT)
        result['generation_exit'] = proc.returncode
        if proc.returncode:
            raise RuntimeError(f'candidate generation failed: {case}')
        result['preparation_process_exited_before_evaluation'] = True
        if case.startswith('reference_seed'):
            expected = f'{case}: dense_er N=64 done'
            if expected not in (folder / 'generation.log').read_text():
                raise RuntimeError(f'Wrong reference wrapper executed for {case}')
        for split in ('train', 'test'):
            with (folder / f'evaluate_{split}.log').open('w') as log:
                proc = subprocess.run(['bash', f'scripts/evaluate_{split}.sh'], cwd=ROOT,
                                      stdout=log, stderr=subprocess.STDOUT)
            result[f'{split}_exit'] = proc.returncode
            for kind in ('scores', 'diagnostics'):
                src = ROOT / f'scoring/{kind}_{split}.json'
                if proc.returncode == 0 and src.exists():
                    shutil.copy2(src, folder / src.name)
        if result.get('test_exit') == 0:
            scores = json.loads((folder / 'scores_test.json').read_text())
            diag = json.loads((folder / 'diagnostics_test.json').read_text())
            result['rank_gain'], result['rank_components'] = rank(scores)
            result['metrics'] = {exp: block['results']['our_method'] for exp, block in scores['experiments'].items()}
            result['status'] = {exp: block['our_method']['status'] for exp, block in diag['experiments'].items()}
            result['failure_reasons'] = {exp: block['our_method']['failure_reasons'] for exp, block in diag['experiments'].items()}
    finally:
        result['evaluator_unchanged'] = protected_before == fingerprint()
        result['elapsed_seconds'] = round(time.time() - started, 2)
        (folder / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True))
        installed.unlink(missing_ok=True)
        for cached_wrapper in (ROOT / 'method/__pycache__').glob(f'audit_v2_{case}.*.pyc'):
            cached_wrapper.unlink()
        shutil.rmtree(ROOT / 'outputs')
        if (backup / 'outputs').exists():
            shutil.move(backup / 'outputs', ROOT / 'outputs')
        for p in (ROOT / 'scoring').iterdir():
            if p.is_file() and p.name not in scoring_snapshot:
                p.unlink()
        for name, payload in scoring_snapshot.items():
            (ROOT / 'scoring' / name).write_bytes(payload)
        shutil.rmtree(backup)
    if not result['evaluator_unchanged']:
        raise RuntimeError('An evaluator changed during the run')
    summarize()


if __name__ == '__main__':
    if sys.argv[1] == 'batch':
        for case in CASES:
            done = ROOT / 'redteam' / f'audit_v2_{case}/result.json'
            if done.exists() and json.loads(done.read_text()).get('test_exit') == 0:
                continue
            run_case(case)
        summarize()
    elif sys.argv[1] == 'summarize':
        summarize()
    else:
        run_case(sys.argv[1])
