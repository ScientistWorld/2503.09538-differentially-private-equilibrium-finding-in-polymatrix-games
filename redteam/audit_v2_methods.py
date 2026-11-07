"""Audit candidates. Imports only sources available to the future scientist.

The shortcut oracle deliberately violates the documented information restriction:
it evaluates complete transcripts with every player's private payoffs and future
noise before selecting the transcript to submit. It does not inspect eval/test/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from baseline.no_payoff import run_no_payoff
from baseline.structure_free import budget_matched_step, run_structure_free
from data.polymatrix import (
    SEEDS, SIZES, TOPOLOGIES, benchmark_game, benchmark_hyperparameters,
    broadcast_noise, gradients, payoff_operator, project_simplex,
)
from data.submission import save_run, write_manifest
from method.dp_cce import degree_adaptive_regularizer, run_algorithm1

ROOT = Path('/home/user')
STATE = ROOT / 'redteam/audit_v2_state'
RUNTIME = STATE / 'runtime'
DEST = ROOT / 'outputs/candidate'


def simulate(game, steps, eta, noise, *, keep=None, tau=None, no_gradient=False):
    n, a = game.n_players, game.n_actions
    u = np.full((n, a), 1.0 / a)
    s = u.copy()
    op = payoff_operator(game)
    tau = np.zeros(n) if tau is None else np.asarray(tau)
    trajectory, broadcasts = [], []
    for t in range(steps):
        b = s + noise[t]
        p = project_simplex(b)
        g = np.zeros_like(s) if no_gradient else gradients(game, p, op)
        anchor = p if keep is None else keep * p + (1.0 - keep) * u
        s = project_simplex((anchor - eta * g) / (1.0 + eta * tau[:, None]))
        broadcasts.append(b.copy())
        trajectory.append(s.copy())
    return np.stack(trajectory), np.stack(broadcasts)


def quality(game, tr, op):
    g = gradients(game, tr, op)
    current = np.mean(np.sum(g * tr, axis=2), axis=0)
    cce = np.maximum(current - g.mean(axis=0).min(axis=1), 0).mean()
    last = g[-1]
    gap = np.maximum(np.sum(last * tr[-1], axis=1) - last.min(axis=1), 0).mean()
    return float(cce), float(gap)


def shortcut_global_private_optimizer(game, steps, eta, noise):
    """Shortcut: optimize the joint true-payoff Nash objective centrally.

    The cross-player derivative explicitly reads OTHER players' utility rows
    and their unnoised strategies. Clipping its movement hides this information
    violation from the transcript's pointwise distance check.
    """
    n, a = game.n_players, game.n_actions
    u = np.full((n, a), 1.0 / a)
    s = u.copy()
    op = payoff_operator(game)
    deg = np.maximum(game.degrees, 1).astype(float)
    tr, bc = [], []
    for t in range(steps):
        b = s + noise[t]
        p = project_simplex(b)
        public_g = gradients(game, p, op)
        centered = public_g - public_g.mean(axis=1, keepdims=True)
        cap = eta * np.linalg.norm(centered, axis=1)
        anchor = (p + eta * u) / (1.0 + eta)
        x = project_simplex(anchor - eta * public_g)
        best_x = x.copy()
        best_value = float('inf')
        for k in range(12):
            g = gradients(game, x, op)
            best_action = g.argmin(axis=1)
            value = float(np.maximum(np.sum(x * g, axis=1) - g[np.arange(n), best_action], 0).sum())
            if value < best_value:
                best_value, best_x = value, x.copy()
            deviation = x.copy()
            deviation[np.arange(n), best_action] -= 1.0
            cross = ((deviation / deg[:, None]).reshape(-1) @ op).reshape(n, a)
            # Gradient of sum_i (x_i - best_response_i)^T g_i(x).
            joint_gradient = g - cross
            joint_gradient -= joint_gradient.mean(axis=1, keepdims=True)
            trial = x - eta * joint_gradient / np.sqrt(1.0 + k)
            d = trial - anchor
            scale = np.minimum(1.0, cap / np.maximum(np.linalg.norm(d, axis=1), 1e-15))
            x = project_simplex(anchor + d * scale[:, None])
        s = best_x
        tr.append(s.copy())
        bc.append(b.copy())
    return np.stack(tr), np.stack(bc)


def shortcut_oracle(game, steps, eta, noise):
    """Shortcut: select by exact global payoffs after seeing the whole future.

    A real distributed player cannot make this selection. Each candidate's
    individual projected step is nevertheless within the public movement cap.
    The pool uses only the public operating point, never hidden scores.
    """
    n = game.n_players
    pool = {'reference': run_algorithm1(game, steps, eta, noise)}
    for keep in (1.0 / (1.0 + eta), .70, .75, .80, .85, .90, 1.0):
        pool[f'keep_{keep:.6f}'] = simulate(game, steps, eta, noise, keep=keep)
    pool['tau_1'] = simulate(game, steps, eta, noise, tau=np.ones(n))
    pool['shortcut_global_private_optimizer'] = shortcut_global_private_optimizer(game, steps, eta, noise)
    op = payoff_operator(game)
    comp, _ = run_structure_free(game, steps, budget_matched_step(game, steps, eta), noise)
    comp_cce, _ = quality(game, comp, op)
    uniform = np.full((1, n, game.n_actions), 1.0 / game.n_actions)
    uniform_cce, _ = quality(game, uniform, op)
    objectives = {}
    for name, (tr, _) in pool.items():
        cce, gap = quality(game, tr, op)
        objectives[name] = cce / comp_cce + .5 * cce / uniform_cce + .25 * gap / uniform_cce
    selected = min(objectives, key=objectives.get)
    return *pool[selected], {'selected': selected, 'objectives': objectives}


def generate(case):
    records, choices = [], []
    for topology in TOPOLOGIES:
        runs = []
        folder = DEST / topology
        for n in SIZES:
            steps, eta, _ = benchmark_hyperparameters(n, topology)
            for seed in SEEDS:
                game = benchmark_game(topology, n, seed)
                noise = broadcast_noise(topology, n, seed)
                if case.startswith('reference_seed'):
                    # Algorithm 1's only RNG is broadcast_noise, fixed by contract.
                    # Reseeding the process is an actual reference rerun, not a
                    # change of the evaluator's fixed noise or instance seeds.
                    np.random.seed(int(case.rsplit('seed', 1)[1]))
                    tr, bc = run_algorithm1(game, steps, eta, noise)
                elif case == 'better_max_regularization':
                    tr, bc = simulate(game, steps, eta, noise, tau=np.ones(n))
                elif case == 'better_shrink_075':
                    tr, bc = simulate(game, steps, eta, noise, keep=.75)
                elif case == 'weaker_no_gradient':
                    tr, bc = simulate(game, steps, eta, noise,
                                      tau=degree_adaptive_regularizer(game), no_gradient=True)
                elif case == 'simple_baseline':
                    tr, bc = run_no_payoff(n, steps, eta, noise)
                elif case == 'comparator_baseline':
                    tr, bc = run_structure_free(game, steps, budget_matched_step(game, steps, eta), noise)
                elif case == 'shortcut_oracle_selection':
                    tr, bc, choice = shortcut_oracle(game, steps, eta, noise)
                    choices.append(dict(topology=topology, n=n, seed=seed, **choice))
                elif case == 'shortcut_global_private_optimizer':
                    tr, bc = shortcut_global_private_optimizer(game, steps, eta, noise)
                else:
                    raise ValueError(case)
                runs.append(save_run(folder, 'our_method', topology, n, seed, tr, bc))
                records.append((topology, n, seed, hashlib.sha256(tr.tobytes() + bc.tobytes()).hexdigest()))
            print(f'{case}: {topology} N={n} done', flush=True)
        write_manifest(folder, 'our_method', runs)
    result_folder = ROOT / 'redteam' / f'audit_v2_{case}'
    (result_folder / 'artifact_hashes.json').write_text(json.dumps(records, indent=1))
    if choices:
        (result_folder / 'shortcut_choices.json').write_text(json.dumps(choices, indent=1))
    if case in ('reference_seed0', 'better_shrink_075', 'shortcut_oracle_selection'):
        cached = RUNTIME / case
        shutil.rmtree(cached, ignore_errors=True)
        shutil.copytree(DEST, cached)


def transform_shortcut(case):
    cache_name = 'shortcut_oracle_selection' if case == 'shortcut_failure_compensation' else 'reference_seed0'
    cache = RUNTIME / cache_name
    if not cache.exists():
        generate(cache_name)
        shutil.rmtree(DEST)
    shutil.copytree(cache, DEST)
    for manifest_path in sorted(DEST.rglob('manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        revised = []
        baseline_runs = []
        for run in manifest['runs']:
            n, seed, topology = run['n_players'], run['seed'], run['topology']
            if case == 'shortcut_drop_small_games' and n == min(SIZES):
                continue
            if case == 'shortcut_failure_compensation' and topology == 'sparse_knn' and n == min(SIZES) and seed in (0, 1):
                continue
            path = Path(run['trajectory_path'])
            if case in ('shortcut_constant_uniform', 'shortcut_zero_outputs', 'shortcut_nonfinite',
                        'shortcut_empty', 'shortcut_remove_noise'):
                with np.load(path, allow_pickle=False) as arrays:
                    tr, bc = arrays['trajectory'], arrays['broadcasts']
                if case == 'shortcut_constant_uniform':
                    tr = np.full_like(tr, 1.0 / 5)
                    bc = tr + broadcast_noise(topology, n, seed)
                elif case == 'shortcut_zero_outputs':
                    tr = np.zeros_like(tr)
                    before = np.concatenate([np.full((1, n, 5), .2), tr[:-1]])
                    bc = before + broadcast_noise(topology, n, seed)
                elif case == 'shortcut_nonfinite':
                    tr[:] = np.nan if seed % 2 else np.inf
                elif case == 'shortcut_empty':
                    tr, bc = tr[:0], bc[:0]
                elif case == 'shortcut_remove_noise':
                    game = benchmark_game(topology, n, seed)
                    steps, eta, _ = benchmark_hyperparameters(n, topology)
                    tr, bc = run_algorithm1(game, steps, eta, np.zeros_like(bc))
                np.savez_compressed(path, trajectory=tr, broadcasts=bc)
            if case == 'shortcut_forged_claims':
                run.update(eta=0, alpha=-2, sigma=1e30, privacy_bound=-1e30,
                           success=True, elapsed_seconds=-1, score=1e30,
                           cce_exploitability=0, privacy_protocol_compliance=1,
                           test_size=0, game_path='does-not-exist.npz')
            if case == 'shortcut_zero_game_artifact':
                zero = RUNTIME / f'zero_{topology}_{n}_{seed}.npz'
                game = benchmark_game(topology, n, seed)
                np.savez_compressed(zero, n_players=n, n_actions=5, edges=game.edges,
                                    utilities=np.zeros_like(game.utilities))
                run['game_path'] = str(zero)
            if case == 'shortcut_move_baseline':
                bad = ROOT / 'outputs/structure_free_baseline' / topology
                with np.load(path, allow_pickle=False) as arrays:
                    tr, bc = arrays['trajectory'], arrays['broadcasts']
                baseline_runs.append(save_run(bad, 'structure_free_baseline', topology, n, seed,
                                              np.zeros_like(tr), bc))
            revised.append(run)
        manifest['runs'] = revised
        manifest_path.write_text(json.dumps(manifest, indent=1))
        if baseline_runs:
            write_manifest(ROOT / 'outputs/structure_free_baseline' / topology,
                           'structure_free_baseline', baseline_runs)
    if case == 'shortcut_forged_claims':
        # Persisted scores are overwritten by the unchanged hidden test.
        fake = {'experiments': {}}
        for exp in ('dense_er_scaling', 'dense_cluster_scaling', 'sparse_knn_scaling'):
            fake['experiments'][exp] = {'results': {'our_method': {
                'comparator_exploitability_ratio': 0, 'relative_exploitability': 0,
                'relative_final_nash_gap': 0, 'privacy_protocol_compliance': 1}}}
        (ROOT / 'scoring/scores_test.json').write_text(json.dumps(fake))


def main(case):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if case.startswith('shortcut_') and case not in ('shortcut_oracle_selection', 'shortcut_global_private_optimizer'):
        transform_shortcut(case)
    else:
        generate(case)


if __name__ == '__main__':
    main(sys.argv[1])
