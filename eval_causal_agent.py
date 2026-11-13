"""
Compare all agents across the train / test distributions.

Agents
  PPO                - Baseline 1: model-free RL (models_realistic/best_model.zip)
  Model-based RL     - Baseline 2: learned average dynamics + shared planner
  Causal (passive)   - Bayesian confounder inference, exploit only
  Causal (active IG) - + safe information-gain probing first (Stage 3)

Every agent sees the same seeded episodes (paired comparison). Success
rates are reported with 95% Wilson intervals.

Run train.py and mbrl_baseline.py first (missing baselines are skipped).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from realistic_causal_env import RealisticCausalPushEnv, PRESETS
from bayesian_causal_agent import CausalDiscoveryPolicy
from mbrl_baseline import ModelBasedPolicy
from experiment_config import parse_preset, paths, labels, figure


class PPOPolicy:
    def __init__(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def reset(self):
        pass

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=True)


def build_agents(preset='narrow'):
    """name -> zero-arg factory. Baselines are skipped if not trained.

    The only preset-specific information any agent gets is the actuator
    range (PPO gets it through its action space, the planners as
    force_range) -- never the train/test confounder ranges."""
    p = paths(preset)
    force_range = (0.0, PRESETS[preset]['force_max'])
    agents = {}
    if os.path.exists(p['ppo']):
        agents['PPO'] = lambda: PPOPolicy(p['ppo'])
    else:
        print(f"  (skipping PPO: {p['ppo']} not found -- run train.py --preset {preset})")
    if os.path.exists(p['mbrl']):
        agents['Model-based RL'] = lambda: ModelBasedPolicy(p['mbrl'], force_range=force_range)
    else:
        print(f"  (skipping model-based RL: {p['mbrl']} not found -- "
              f"run mbrl_baseline.py --preset {preset})")
    agents['Causal (passive)'] = lambda: CausalDiscoveryPolicy(
        exploration='none', rng=np.random.default_rng(0), force_range=force_range)
    agents['Causal (active IG)'] = lambda: CausalDiscoveryPolicy(
        exploration='active', rng=np.random.default_rng(0), force_range=force_range)
    return agents


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def evaluate(policy, env_mode, n_episodes=50, seed_offset=1000, preset='narrow'):
    env = RealisticCausalPushEnv(render_mode=None, distribution_mode=env_mode,
                                 preset=preset)
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


def run_comparison(n_episodes=50, preset='narrow'):
    print("=" * 78)
    print(f"AGENT COMPARISON ({n_episodes} paired episodes per distribution, "
          f"preset '{preset}')")
    print("=" * 78)
    distributions = labels(preset)
    agents = build_agents(preset)
    results = {}
    for name, factory in agents.items():
        print(f"\n--- {name} ---")
        policy = factory()
        results[name] = {}
        for mode, label in distributions.items():
            r = evaluate(policy, mode, n_episodes, preset=preset)
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
    print(f"{'':10s}" + "".join(f"{n:>18s}" for n in names))
    for mode, label in distributions.items():
        print(f"{label:10s}" + "".join(
            f"{results[n][mode]['success_rate']*100:18.1f}" for n in names))


def plot_comparison(results, distributions, threshold_cm, path):
    names = list(results)
    modes = list(distributions)
    labels = [distributions[m] for m in modes]
    x = np.arange(len(modes))
    width = 0.8 / len(names)
    colors = {'PPO': '#d95f02', 'Model-based RL': '#7570b3',
              'Causal (passive)': '#1b9e77', 'Causal (active IG)': '#66a61e'}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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
