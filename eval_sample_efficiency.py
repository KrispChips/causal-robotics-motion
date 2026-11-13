"""
Sample efficiency (PDF H3 / Metric 3)

Two questions:

1. Within an episode: how many interactions does each exploration strategy
   need to pin down the hidden confounders?
     passive - exploit only (information is a by-product of goal pushes)
     random  - safe probes drawn uniformly from the safe force set
     active  - safe probes chosen by expected information gain
   Random and active share the same safety rule and stopping rule, so their
   difference isolates *how* the probe is chosen.

2. Across episodes: how much training experience does each agent need
   before it can do the task at all?
     PPO        - episodes until EvalCallback success >= 80%
     Model-based - episodes of random-force data used to fit the model
     Causal     - none (prior + physics structure; learns within episode)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from realistic_causal_env import RealisticCausalPushEnv, PRESETS
from bayesian_causal_agent import CausalDiscoveryPolicy
from experiment_config import parse_preset, paths, labels, figure

STRATEGIES = {'none': 'Passive (exploit only)', 'random': 'Random probes',
              'active': 'Active (info gain)'}
FRICTION_TOL = 0.02
MAX_STEPS = 20


def run_strategy(strategy, mode, n_episodes=30, seed_offset=1000, preset='narrow'):
    """Per-step absolute errors, shape (n_episodes, MAX_STEPS). Column t is
    the error after t+1 environment interactions."""
    env = RealisticCausalPushEnv(distribution_mode=mode, preset=preset)
    policy = CausalDiscoveryPolicy(exploration=strategy,
                                   rng=np.random.default_rng(0),
                                   force_range=(env.force_min, env.force_max))
    f_err = np.zeros((n_episodes, MAX_STEPS))
    m_err = np.zeros((n_episodes, MAX_STEPS))
    success = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        policy.reset()
        for t in range(MAX_STEPS):
            action, _ = policy.predict(obs)
            # predict() has now absorbed interactions 1..t; record the
            # belief *after* t interactions (t=0 -> prior)
            if t > 0:
                b = policy.agent.get_beliefs()
                f_err[ep, t - 1] = abs(b['friction_mean'] - env.base_friction)
                m_err[ep, t - 1] = abs(b['mass_mean'] - env.mass)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        # fold in the final interaction too
        policy.agent.update_beliefs(policy.last_x, policy.last_v,
                                    float(obs[-1]), float(obs[0]), float(obs[2]))
        b = policy.agent.get_beliefs()
        f_err[ep, -1] = abs(b['friction_mean'] - env.base_friction)
        m_err[ep, -1] = abs(b['mass_mean'] - env.mass)
        success.append(info['is_success'])
    env.close()
    return f_err, m_err, np.mean(success)


def steps_to_tolerance(curve, tol):
    hit = np.nonzero(curve <= tol)[0]
    return int(hit[0]) + 1 if len(hit) else None


def training_cost_summary(preset='narrow'):
    print("\n" + "=" * 74)
    print("TRAINING EXPERIENCE NEEDED BEFORE DEPLOYMENT")
    print("=" * 74)
    path = paths(preset)['evaluations']
    if os.path.exists(path):
        ev = np.load(path)
        succ = ev['successes'].mean(axis=1)
        hit = np.nonzero(succ >= 0.8)[0]
        if len(hit):
            steps = int(ev['timesteps'][hit[0]])
            print(f"  PPO:            ~{steps // MAX_STEPS:,} episodes ({steps:,} steps) "
                  f"to reach 80% eval success")
        else:
            print("  PPO:            never reached 80% eval success")
    else:
        print(f"  PPO:            no logs found (run train.py --preset {preset})")
    print(f"  Model-based RL: 1,000 episodes of random-force data "
          f"(mbrl_baseline.py --preset {preset})")
    print("  Causal agent:   0 episodes (physics structure + prior; infers μ, m "
          "within each episode)")


def main(n_episodes=30, preset='narrow'):
    print("=" * 74)
    print(f"WITHIN-EPISODE IDENTIFICATION ({n_episodes} paired episodes, preset '{preset}')")
    print("=" * 74)
    MODES = labels(preset)
    curves = {}
    for mode, mlabel in MODES.items():
        print(f"\n{mlabel}:")
        for s, slabel in STRATEGIES.items():
            f_err, m_err, succ = run_strategy(s, mode, n_episodes, preset=preset)
            curves[(mode, s)] = (f_err.mean(0), m_err.mean(0))
            k = steps_to_tolerance(f_err.mean(0), FRICTION_TOL)
            print(f"  {slabel:24s} μ err after 3/5/20 steps: "
                  f"{f_err[:, 2].mean():.3f} / {f_err[:, 4].mean():.3f} / "
                  f"{f_err[:, -1].mean():.3f}   steps to μ err≤{FRICTION_TOL}: "
                  f"{k if k else '>20'}   success {succ*100:.0f}%")

    fig, axes = plt.subplots(2, len(MODES), figsize=(4 * len(MODES), 7),
                             sharex=True)
    steps = np.arange(1, MAX_STEPS + 1)
    for j, (mode, mlabel) in enumerate(MODES.items()):
        for s, slabel in STRATEGIES.items():
            fc, mc = curves[(mode, s)]
            axes[0, j].plot(steps, fc, marker='.', label=slabel)
            axes[1, j].plot(steps, mc, marker='.', label=slabel)
        axes[0, j].axhline(FRICTION_TOL, color='gray', ls='--', lw=1)
        axes[0, j].set_title(mlabel)
        for i in range(2):
            axes[i, j].set_yscale('log'); axes[i, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel('Interactions (control steps)')
    axes[0, 0].set_ylabel('Friction |μ̂ − μ|')
    axes[1, 0].set_ylabel('Mass |m̂ − m| (kg)')
    axes[0, 0].legend(fontsize=8)
    plt.suptitle('Confounder identification vs. number of interactions')
    plt.tight_layout()
    out_png = figure('sample_efficiency.png', preset)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Plot saved as '{out_png}'")

    training_cost_summary(preset)


if __name__ == "__main__":
    args = parse_preset("Within-episode identification / sample efficiency",
                        extra_args=[(('--episodes',), dict(type=int, default=30))])
    main(args.episodes, args.preset)
