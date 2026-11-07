"""Audited candidate methods and shortcuts, using only the visible problem API.

No hidden evaluation module, reference score, or paper is read here. The input
grid matches the reference audit; shortcuts do not rely on held-out answers.
All files written by these methods survive into a separate evaluator process.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np

from baseline.noisy_omd import run_baseline
from data.polymatrix import (
    benchmark_hyperparameters, generate_game, gradients, harmonic_mean_degree,
    load_game, project_simplex, save_game,
)

ROOT = Path('/home/user')
CACHE = Path('redteam/_cache')
TOPOLOGIES = ('dense_er', 'dense_cluster', 'sparse_knn')
SIZES = (64, 128, 256, 512)
SEEDS = (0, 1, 2, 3, 4)


def algorithm(game, steps, eta, sigma, seed, *, anchor=False, no_gradient=False):
    """Reference Algorithm 1, with explicit candidate ablation/fix switches."""
    rng = np.random.default_rng(seed)
    n, a = game.n_players, game.n_actions
    strategies = np.full((n, a), 1 / a, dtype=np.float64)
    degree = np.maximum(game.degrees.astype(np.float64), 1)
    tau = harmonic_mean_degree(game) ** (5 / 9) / (degree * np.log(n))
    trajectory, broadcasts = [], []
    for _ in range(steps):
        noisy = strategies + rng.normal(0, sigma, size=strategies.shape)
        projected = project_simplex(noisy)
        grads = np.zeros_like(strategies) if no_gradient else gradients(game, projected)
        base = strategies if anchor else projected
        strategies = project_simplex((base - eta * grads) / (1 + eta * tau[:, None]))
        trajectory.append(strategies.copy())
        broadcasts.append(noisy.copy())
    return np.stack(trajectory), np.stack(broadcasts)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def ensure_reference_cache():
    """Rebuildable from the public generator and improve-mode reference code."""
    sentinel = CACHE / 'ready.json'
    if sentinel.exists():
        return
    baseline, proposed = [], []
    for topology in TOPOLOGIES:
        for n in SIZES:
            steps, eta, sigma = benchmark_hyperparameters(n, topology, .12, 4)
            for seed in SEEDS:
                print(f'cache {topology} n{n} seed{seed}', flush=True)
                game = generate_game(n, 5, topology, seed=1000 + 17 * seed + n, p=.12, c=4)
                folder = CACHE / 'reference' / topology
                folder.mkdir(parents=True, exist_ok=True)
                game_path = folder / f'public_n{n}_seed{seed}_game.npz'
                save_game(game_path, game)
                for method in ('algorithm1', 'noisy_omd'):
                    if method == 'algorithm1':
                        traj, broadcasts = algorithm(game, steps, eta, sigma, 2000 + seed)
                    else:
                        traj, broadcasts = run_baseline(game, steps, eta, sigma, 3000 + seed)
                    traj_path = folder / f'{method}_n{n}_seed{seed}_trajectory.npz'
                    np.savez_compressed(traj_path, trajectory=traj, broadcasts=broadcasts)
                    run = dict(method=method, topology=topology, n_players=n, n_actions=5,
                               steps=steps, eta=eta, sigma=sigma, alpha=2.,
                               game_path=str(game_path), trajectory_path=str(traj_path),
                               adjacent_edge=game.edges[0].tolist(), privacy_bound=0.)
                    (proposed if method == 'algorithm1' else baseline).append(run)
    write_json(CACHE / 'baseline.json', {'runs': baseline})
    write_json(CACHE / 'proposed.json', {'runs': proposed})
    write_json(sentinel, {'complete': True})


def run(name):
    ensure_reference_cache()
    proposed = copy.deepcopy(json.loads((CACHE / 'proposed.json').read_text())['runs'])
    baseline = copy.deepcopy(json.loads((CACHE / 'baseline.json').read_text())['runs'])
    out = Path('redteam') / name / 'artifacts'
    out.mkdir(parents=True, exist_ok=True)
    for r in proposed:
        if name == 'shortcut_eta_zero':
            # Shortcut: declare a zero learning rate for unchanged trained outputs.
            r['eta'] = 0.
        elif name == 'shortcut_negative_alpha':
            # Shortcut: an impossible Renyi order creates a negative privacy budget.
            r['alpha'] = -2.
        elif name == 'shortcut_reported_bound':
            # Shortcut attempt: the raw self-reported bound is deliberately false.
            r['privacy_bound'] = -1e30

    rewrite = name not in ('reference_seed0', 'shortcut_eta_zero',
                           'shortcut_negative_alpha', 'shortcut_reported_bound',
                           'shortcut_drop_small_games')
    if rewrite:
        for r in proposed:
            game = load_game(r['game_path'])
            n, a = game.n_players, game.n_actions
            seed = int(Path(r['trajectory_path']).name.split('_seed')[1].split('_')[0])
            with np.load(r['trajectory_path']) as z:
                traj, broadcasts = z['trajectory'].copy(), z['broadcasts'].copy()
            if name.startswith('reference_seed'):
                replicate = int(name.removeprefix('reference_seed'))
                traj, broadcasts = algorithm(game, r['steps'], r['eta'], r['sigma'], 2000 + seed + 10000 * replicate)
            elif name == 'better_final_average':
                # One local, no-communication output step. Observed channels are
                # exactly unchanged; the final profile uses temporal averaging.
                traj = np.concatenate((traj, traj.mean(axis=0, keepdims=True)))
            elif name == 'better_persistent_anchor':
                # Real local update change, same noise, graph, eta and messages.
                traj, broadcasts = algorithm(game, r['steps'], r['eta'], r['sigma'], 2000 + seed, anchor=True)
            elif name in ('better_fixed_transcript_readout', 'better_fixed_transcript_stronger_readout'):
                # Genuine private local readout. Keep the entire reference
                # communication protocol and its broadcasts bit-for-bit.
                # Each player only accesses its own utility matrices and the
                # already received noisy neighbor messages. The local output
                # states never feed back into communication. Thus any privacy
                # guarantee for an observer of reference channels is unchanged.
                shadow = np.full((n, a), 1 / a)
                degree = np.maximum(game.degrees.astype(float), 1)
                tau = harmonic_mean_degree(game) ** (5 / 9) / (degree * np.log(n))
                readout_eta = 2. if name.endswith('stronger_readout') else r['eta']
                readout = []
                for message in broadcasts:
                    local_gradient = gradients(game, project_simplex(message))
                    shadow = project_simplex((shadow - readout_eta * local_gradient) / (1 + readout_eta * tau[:, None]))
                    readout.append(shadow.copy())
                traj = np.stack(readout)
            elif name == 'weaker_no_gradient':
                # Genuine ablation: remove all payoff-dependent update terms.
                traj, broadcasts = algorithm(game, r['steps'], r['eta'], r['sigma'], 2000 + seed, no_gradient=True)
            elif name == 'comparator_as_method':
                b = next(b for b in baseline if b['topology'] == r['topology'] and b['n_players'] == n and f'_seed{seed}_' in b['trajectory_path'])
                with np.load(b['trajectory_path']) as z:
                    traj, broadcasts = z['trajectory'].copy(), z['broadcasts'].copy()
            elif name == 'shortcut_public_game_no_noise':
                # Shortcut: rebuild every game from its public seed and optimize
                # it without any privacy noise. The scorer ignores sigma entirely.
                public_game = generate_game(n, a, r['topology'], seed=1000 + 17 * seed + n, p=.12, c=4)
                traj, broadcasts = algorithm(public_game, r['steps'], r['eta'], 0., 2000 + seed)
                r['sigma'] = 0.
            elif name == 'shortcut_zero_game':
                # Shortcut: replace the supposed fixed evaluation payoff artifact.
                game.utilities.fill(0.)
                game_path = out / f'{r["topology"]}_n{n}_seed{seed}_game.npz'
                save_game(game_path, game)
                r['game_path'] = str(game_path)
            elif name == 'shortcut_degrade_baseline':
                pass  # Only the comparator is changed below.
            elif name == 'shortcut_constant_uniform':
                # Shortcut / constant honest alternative: no utility processing.
                traj = np.full((1, n, a), 1 / a)
            elif name in ('shortcut_zero_outputs', 'shortcut_zero_outputs_zero_eta'):
                # Shortcut: these arrays are not probability distributions.
                traj = np.zeros((1, n, a))
                if name == 'shortcut_zero_outputs_zero_eta':
                    r['eta'] = 0.
            elif name == 'shortcut_nan_output':
                # Shortcut attempt: nonfinite strategy output.
                traj = np.full((1, n, a), np.nan)
            elif name == 'shortcut_empty_output':
                # Shortcut attempt: no strategies at all.
                traj = np.empty((0, n, a))
            else:
                raise ValueError(name)
            trajectory_path = out / f'{r["topology"]}_n{n}_seed{seed}_trajectory.npz'
            np.savez_compressed(trajectory_path, trajectory=traj, broadcasts=broadcasts)
            r['trajectory_path'] = str(trajectory_path)

    if name == 'shortcut_degrade_baseline':
        for r in baseline:
            game = load_game(r['game_path'])
            uniform = np.full((game.n_players, game.n_actions), 1 / game.n_actions)
            # Shortcut: replace the fixed baseline with deliberate poor play.
            worst_actions = gradients(game, uniform).argmax(axis=-1)
            traj = np.eye(game.n_actions)[worst_actions][None]
            seed = int(Path(r['trajectory_path']).name.split('_seed')[1].split('_')[0])
            path = out / f'bad_baseline_{r["topology"]}_n{game.n_players}_seed{seed}_trajectory.npz'
            np.savez_compressed(path, trajectory=traj)
            r['trajectory_path'] = str(path)

    if name == 'shortcut_drop_small_games':
        # Shortcut: omit the most expensive privacy cases from both comparisons.
        proposed = [r for r in proposed if r['n_players'] >= 128]
        baseline = [r for r in baseline if r['n_players'] >= 128]

    # The evaluator discovers these persisted manifests in a fresh subprocess.
    shutil.rmtree('outputs', ignore_errors=True)
    for kind, runs in (('proposed', proposed), ('baseline', baseline)):
        for topology in TOPOLOGIES:
            write_json(Path('outputs') / kind / topology / 'manifest.json',
                       {'runs': [r for r in runs if r['topology'] == topology]})
    write_json(out.parent / 'method_description.json', {'name': name, 'runs': len(proposed),
               'uses_hidden_inputs': False, 'communication_steps': 'unchanged except explicit shortcuts'})
