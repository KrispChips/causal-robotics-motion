"""
Model-misspecification ablation: how much does the causal agent's advantage
depend on its physics model being exactly right?

push   The simulator gets real-world effects that the causal agent's textbook
       Coulomb model lacks (realistic_causal_env.WORLDS): stiction (static
       friction 1.3x kinetic), friction rising with speed, and an actuator
       with a deadband and 85% gain. This mirrors sim-to-real: the learners
       (PPO and DR-PPO) are retrained IN each altered world, while the causal
       agent keeps its textbook model. It only re-estimates its noise level by
       empirical Bayes on unlabelled training data from that world, the same
       train-only protocol as before. Planner settings stay as tuned in the
       nominal world.

stack  The causal stacking agent misjudges its own gripper: it believes a
       grip force or pad friction 30% above or below the truth.

Usage
  python train.py --preset P --variant ppo|ppo_dr --world W   (PPO per world)
  python eval_misspecification.py push  --preset pdf
  python eval_misspecification.py stack --preset pdf
"""

import json
import os
import time
from multiprocessing import Pool

import numpy as np

from experiment_config import (parse_preset, labels, paths, figure, load_agent_config,
                               make_causal_policy, PPO_VARIANTS, ppo_path)
from realistic_causal_env import WORLDS

AGENTS = ('PPO', 'PPO (DR)', 'Causal (textbook model)')
VARIANT = {'PPO': 'ppo', 'PPO (DR)': 'ppo_dr'}


def world_sigma(preset, world):
    """Noise level by empirical Bayes on unlabelled train data of this world
    (cached; nominal uses the main tuned config)."""
    if world == 'nominal':
        return load_agent_config(preset)['velocity_noise_std']
    path = os.path.join(paths(preset)['models'], 'world_sigma.json')
    cache = json.load(open(path)) if os.path.exists(path) else {}
    if world not in cache:
        from tune_on_train import tune_noise
        print(f"  empirical-Bayes noise for world '{world}':")
        cache[world] = tune_noise(preset, world)
        with open(path, 'w') as f:
            json.dump(cache, f, indent=2)
    return cache[world]


def _job(args):
    preset, world, agent, mode, sigma, n = args
    from eval_causal_agent import evaluate, PPOPolicy
    if agent in VARIANT:
        v = VARIANT[agent]
        policy = PPOPolicy(ppo_path(preset, v, world), PPO_VARIANTS[v]['history'])
    else:
        cfg = dict(load_agent_config(preset), velocity_noise_std=sigma)
        policy = make_causal_policy(preset, 'none', config=cfg)
    r = evaluate(policy, mode, n, preset=preset, world=world)
    return (world, agent, mode), {'success': float(r['success_rate']),
                                  'ci': [float(c) for c in r['success_ci']],
                                  'final_cm': float(r['mean_final_distance'] * 100),
                                  'friction_mae': float(r.get('friction_mae', np.nan)),
                                  'mass_mae': float(r.get('mass_mae', np.nan))}


def run_push(preset, n=50):
    t0 = time.time()
    worlds = list(WORLDS)
    sigmas = {w: world_sigma(preset, w) for w in worlds}
    jobs = []
    for w in worlds:
        for a in AGENTS:
            if a in VARIANT and not os.path.exists(ppo_path(preset, VARIANT[a], w)):
                print(f"  (skipping {a} in world '{w}': not trained)")
                continue
            jobs += [(preset, w, a, m, sigmas[w], n) for m in labels(preset)]
    with Pool(15) as pool:
        res = dict(pool.map(_job, jobs))

    labs = labels(preset)
    print(f"\nSUCCESS RATE (%), preset '{preset}', {n} paired episodes per cell")
    print(f"{'world':15s}{'agent':26s}" + "".join(f"{l:>11s}" for l in labs.values()))
    for w in worlds:
        for a in AGENTS:
            if (w, a, next(iter(labs))) in res:
                print(f"{w:15s}{a:26s}" + "".join(
                    f"{res[(w, a, m)]['success']*100:11.1f}" for m in labs))
        print(f"{'':15s}causal noise σ = {sigmas[w]}")
    out = os.path.join(paths(preset)['models'], 'misspecification_push.json')
    with open(out, 'w') as f:
        json.dump({'sigmas': sigmas,
                   'results': {'|'.join(k): v for k, v in res.items()}}, f, indent=1)
    plot_push(res, worlds, labs, preset)
    print(f"Saved {out} ({time.time()-t0:.0f}s)")


def plot_push(res, worlds, labs, preset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'PPO': '#eb6834', 'PPO (DR)': '#7b5ea7', 'Causal (textbook model)': '#1baf7a'}
    fig, axes = plt.subplots(1, len(worlds), figsize=(4.2 * len(worlds), 4.2), sharey=True)
    modes = list(labs)
    x = np.arange(len(modes))
    w = 0.8 / len(AGENTS)
    for ax, world in zip(axes, worlds):
        for i, a in enumerate(AGENTS):
            if (world, a, modes[0]) not in res:
                continue
            rate = np.array([res[(world, a, m)]['success'] for m in modes]) * 100
            ci = np.array([res[(world, a, m)]['ci'] for m in modes]) * 100
            ax.bar(x + (i - 1) * w, rate, w * 0.92, color=colors[a], label=a,
                   yerr=np.vstack([rate - ci[:, 0], ci[:, 1] - rate]), capsize=2)
        ax.set_title(world.replace('_', ' '), loc='left')
        ax.set_xticks(x); ax.set_xticklabels([labs[m] for m in modes], fontsize=8)
        ax.set_ylim(0, 105); ax.grid(True, axis='y', alpha=0.25)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('Success rate (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(AGENTS), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = figure('misspecification_push.png', preset)
    fig.savefig(path, dpi=150)
    print(f"✓ Plot saved as '{path}'")


def run_stack(preset):
    import stack_env as S
    from stack_transfer import CausalStackPolicy, load_bank, wilson
    sigma = load_agent_config(preset)['velocity_noise_std']
    beliefs = [('correct', 1.0, 1.0),
               ('grip force +30%', 1.3, 1.0), ('grip force −30%', 0.7, 1.0),
               ('pad friction +30%', 1.0, 1.3), ('pad friction −30%', 1.0, 0.7)]
    force_max = S.PRESETS[preset]['force_max']
    print(f"Causal stacking agent with a wrong belief about its own gripper ('{preset}')")
    print(f"{'belief':20s}{'set':12s}{'success':>9s}{'drops':>8s}{'reward':>9s}")
    out = {}
    for name, ff, fm in beliefs:
        agent = CausalStackPolicy(force_max, sigma, finger_force=S.FINGER_FORCE * ff,
                                  finger_mu=S.FINGER_MU * fm)
        for mode, label in labels(preset).items():
            bank = load_bank(preset, mode, 'eval')
            env = S.PandaStackEnv(preset=preset, distribution_mode=mode)
            succ, drop, rew = [], [], []
            for i in range(len(bank['logs'])):
                obs, _ = env.reset(seed=i, options={'confounders': tuple(bank['confounders'][i]),
                                                    'push_log': bank['logs'][i]})
                action, _ = agent.predict(obs)
                _, r, _, _, info = env.step(action)
                succ.append(info['is_success']); drop.append(info['slipped']); rew.append(r)
            env.close()
            k, n = int(np.sum(succ)), len(succ)
            out[f'{name}|{mode}'] = {'success': k / n, 'ci': wilson(k, n),
                                     'drops': float(np.mean(drop)), 'reward': float(np.mean(rew))}
            print(f"{name:20s}{label:12s}{k/n*100:8.1f}%{np.mean(drop)*100:7.1f}%"
                  f"{np.mean(rew):9.2f}")
    path = os.path.join(paths(preset)['models'], 'misspecification_stack.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"Saved {path}")


if __name__ == '__main__':
    args = parse_preset('Model-misspecification ablation', extra_args=[
        (('command',), dict(choices=['push', 'stack'])),
        (('--episodes',), dict(type=int, default=50))])
    if args.command == 'push':
        run_push(args.preset, args.episodes)
    else:
        run_stack(args.preset)
