"""
Compare all agents across the train / test distributions.

Agents
  PPO                - Baseline 1: model-free RL, 3-step history (plan of record)
  PPO variants       - full-episode history and/or domain randomization over
                       every train+test range (experiment_config.PPO_VARIANTS)
  Model-based RL     - Baseline 2: learned average dynamics + shared planner
  Model-based RL (history) - dynamics conditioned on the episode so far
  Causal (passive)   - Bayesian confounder inference, exploit only
  Causal (active IG) - + safe information-gain probing first (Stage 3)

Every agent sees the same seeded episodes (paired comparison). Success
rates are reported with 95% Wilson intervals.

Run train.py and mbrl_baseline.py first (missing baselines are skipped).
"""

import os
import time
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

from realistic_causal_env import RealisticCausalPushEnv, PRESETS
from bayesian_causal_agent import CausalDiscoveryPolicy
from experiment_config import (parse_preset, paths, labels, figure, load_agent_config,
                               make_causal_policy, make_mbrl_policy, make_mbrl_ctx_policy,
                               PPO_VARIANTS, ppo_path)


class PPOPolicy:
    """Rebuilds the observation the policy was trained with (its own history
    length) from the shared evaluation env's stream: obs[:5] is the current
    state and obs[-1] the force applied on the previous step."""

    def __init__(self, path, history_length=3):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")
        self.H = history_length
        self.reset()

    def reset(self):
        self.states, self.forces = None, None

    def own_obs(self, obs):
        base = np.asarray(obs[:5], dtype=np.float32)
        if self.states is None:                       # first step of the episode
            self.states, self.forces = [base] * self.H, [0.0] * self.H
        else:
            self.states = self.states[1:] + [base]
            self.forces = self.forces[1:] + [float(obs[-1])]
        return np.concatenate([base, *self.states,
                               np.array(self.forces, dtype=np.float32)])

    def predict(self, obs, deterministic=True):
        return self.model.predict(self.own_obs(obs), deterministic=True)


def build_agents(preset='narrow'):
    """name -> zero-arg factory. Baselines are skipped if not trained.

    The only preset-specific information any agent gets is the actuator
    range (PPO gets it through its action space, the planners as
    force_range) -- never the train/test confounder ranges."""
    p = paths(preset)
    cfg = load_agent_config(preset)   # hyperparameters tuned on train only
    agents = {}
    for variant, spec in PPO_VARIANTS.items():
        path = ppo_path(preset, variant)
        if os.path.exists(path):
            agents[spec['label']] = (lambda path=path, h=spec['history']:
                                     PPOPolicy(path, h))
        else:
            print(f"  (skipping {spec['label']}: {path} not found -- "
                  f"run train.py --preset {preset} --variant {variant})")
    if os.path.exists(p['mbrl']):
        agents['Model-based RL'] = lambda: make_mbrl_policy(preset, cfg)
    else:
        print(f"  (skipping model-based RL: {p['mbrl']} not found -- "
              f"run mbrl_baseline.py --preset {preset})")
    if os.path.exists(p['mbrl_ctx']) and 'mbrl_ctx_plan' in cfg:
        agents['Model-based RL (history)'] = lambda: make_mbrl_ctx_policy(preset, cfg)
    else:
        print(f"  (skipping model-based RL (history): run mbrl_context_baseline.py and "
              f"tune_on_train.py --only mbrl_ctx for preset {preset})")
    agents['Causal (passive)'] = lambda: make_causal_policy(preset, 'none', config=cfg)
    agents['Causal (active IG)'] = lambda: make_causal_policy(preset, 'active', config=cfg)
    return agents


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def evaluate(policy, env_mode, n_episodes=50, seed_offset=1000, preset='narrow',
             world='nominal'):
    env = RealisticCausalPushEnv(render_mode=None, distribution_mode=env_mode,
                                 preset=preset, world=world)
    is_causal = isinstance(policy, CausalDiscoveryPolicy)
    r = {k: [] for k in ('success', 'final_distance', 'true_friction', 'true_mass',
                         'friction_est', 'mass_est', 'friction_std', 'mass_std',
                         'n_probes')}
    t0 = time.time()
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        policy.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        r['success'].append(bool(info['is_success']))
        r['final_distance'].append(info['final_distance'])
        r['true_friction'].append(info['true_base_friction'])
        r['true_mass'].append(info['true_mass'])
        if is_causal:
            b = policy.agent.get_beliefs()
            r['friction_est'].append(b['friction_mean'])
            r['mass_est'].append(b['mass_mean'])
            r['friction_std'].append(b['friction_std'])
            r['mass_std'].append(b['mass_std'])
            r['n_probes'].append(policy.n_explore_steps)
    env.close()

    out = {k: np.array(v) for k, v in r.items()}
    k = int(out['success'].sum())
    out['success_rate'] = k / n_episodes
    out['success_ci'] = wilson_interval(k, n_episodes)
    out['mean_final_distance'] = out['final_distance'].mean()
    out['sec_per_episode'] = (time.time() - t0) / n_episodes
    if is_causal:
        out['friction_mae'] = np.abs(out['friction_est'] - out['true_friction']).mean()
        out['mass_mae'] = np.abs(out['mass_est'] - out['true_mass']).mean()
    return out


def _eval_job(job):
    name, mode, n_episodes, preset = job
    policy = build_agents(preset)[name]()
    return evaluate(policy, mode, n_episodes, preset=preset)


def run_comparison(n_episodes=50, preset='narrow'):
    print("=" * 78)
    print(f"AGENT COMPARISON ({n_episodes} paired episodes per distribution, "
          f"preset '{preset}')")
    print("=" * 78)
    distributions = labels(preset)
    agents = build_agents(preset)
    # Each (agent, distribution) pair is an independent job (fresh policy,
    # same seeded episodes in the same order), so run them in parallel.
    jobs = [(name, mode, n_episodes, preset) for name in agents for mode in distributions]
    with Pool(min(len(jobs), 14)) as pool:
        done = dict(zip([(j[0], j[1]) for j in jobs], pool.map(_eval_job, jobs)))
    results = {}
    for name in agents:
        print(f"\n--- {name} ---")
        results[name] = {}
        for mode, label in distributions.items():
            r = done[(name, mode)]
            results[name][mode] = r
            lo, hi = r['success_ci']
            line = (f"  {label:7s} success {r['success_rate']*100:5.1f}% "
                    f"[{lo*100:3.0f}-{hi*100:3.0f}]  final dist "
                    f"{r['mean_final_distance']*100:5.2f}cm")
            if 'friction_mae' in r:
                line += (f"  μ MAE {r['friction_mae']:.3f}  m MAE "
                         f"{r['mass_mae']:.2f}kg  probes {r['n_probes'].mean():.1f}")
            print(line + f"  ({r['sec_per_episode']*1000:.0f} ms/ep)")

    print_summary_table(results, distributions)
    plot_comparison(results, distributions,
                    PRESETS[preset]['success_threshold'] * 100,
                    figure('causal_vs_baseline_comparison.png', preset))
    return results


def print_summary_table(results, distributions):
    names = list(results)
    print("\n" + "=" * 78)
    print("SUCCESS RATE (%)")
    print("=" * 78)
    print(f"{'':26s}" + "".join(f"{l:>11s}" for l in distributions.values()))
    for n in names:
        print(f"{n:26s}" + "".join(
            f"{results[n][mode]['success_rate']*100:11.1f}" for mode in distributions))


def plot_comparison(results, distributions, threshold_cm, path):
    names = list(results)
    modes = list(distributions)
    labels = [distributions[m] for m in modes]
    x = np.arange(len(modes))
    width = 0.8 / len(names)
    colors = {'PPO': '#d95f02', 'PPO (full history)': '#e6550d',
              'PPO (DR)': '#fd8d3c', 'PPO (DR, full hist)': '#a63603',
              'Model-based RL': '#7570b3', 'Model-based RL (history)': '#54278f',
              'Causal (passive)': '#1b9e77', 'Causal (active IG)': '#66a61e'}

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # (a) success rate with 95% CI
    ax = axes[0, 0]
    for i, n in enumerate(names):
        rates = np.array([results[n][m]['success_rate'] for m in modes]) * 100
        ci = np.array([results[n][m]['success_ci'] for m in modes]) * 100
        err = np.vstack([rates - ci[:, 0], ci[:, 1] - rates])
        ax.bar(x + (i - (len(names) - 1) / 2) * width, rates, width, yerr=err,
               capsize=2, label=n, color=colors.get(n))
    ax.set_ylabel('Success rate (%)'); ax.set_ylim(0, 105)
    ax.set_title('Task success (95% Wilson CI)')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    # (b) final distance
    ax = axes[0, 1]
    for i, n in enumerate(names):
        d = [results[n][m]['mean_final_distance'] * 100 for m in modes]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, d, width, label=n,
               color=colors.get(n))
    ax.axhline(threshold_cm, color='gray', ls='--', lw=1,
               label=f'±{threshold_cm:.0f}cm success zone')
    ax.set_yscale('log'); ax.set_ylabel('Mean final distance to target (cm)')
    ax.set_title('Final placement error')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    causal = [n for n in names if n.startswith('Causal')]

    # (c) inference MAE
    ax = axes[1, 0]
    w = 0.8 / (2 * len(causal))
    for i, n in enumerate(causal):
        fm = [results[n][m]['friction_mae'] for m in modes]
        mm = [results[n][m]['mass_mae'] for m in modes]
        ax.bar(x + (2 * i - len(causal) + 0.5) * w, fm, w, label=f'{n}: μ MAE',
               color=colors.get(n))
        ax.bar(x + (2 * i - len(causal) + 1.5) * w, mm, w, label=f'{n}: m MAE (kg)',
               color=colors.get(n), alpha=0.5, hatch='//')
    ax.set_ylabel('Mean absolute error'); ax.set_title('Confounder inference accuracy')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=0.3)

    # (d) inferred vs true friction
    ax = axes[1, 1]
    markers = ['o', '^']
    for i, n in enumerate(causal):
        tf = np.concatenate([results[n][m]['true_friction'] for m in modes])
        ef = np.concatenate([results[n][m]['friction_est'] for m in modes])
        ax.scatter(tf, ef, s=18, alpha=0.5, marker=markers[i % 2], label=n,
                   color=colors.get(n))
    all_f = np.concatenate([results[n][m]['true_friction'] for n in causal for m in modes])
    lims = [max(0.0, all_f.min() - 0.05), all_f.max() + 0.05]
    ax.plot(lims, lims, 'k--', lw=1, label='perfect inference')
    ax.set_xlabel('True friction μ'); ax.set_ylabel('Inferred μ (posterior mean)')
    ax.set_title('Inferred vs true friction (all distributions)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Comparison plot saved as '{path}'")


if __name__ == "__main__":
    args = parse_preset("Compare all agents",
                        extra_args=[(('--episodes',), dict(type=int, default=50))])
    run_comparison(args.episodes, args.preset)
