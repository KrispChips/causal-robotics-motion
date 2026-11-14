"""Score the domain-randomized PPO baselines over their whole training box
and at its four (mass, friction) corners, where uniform sampling is thinnest.

Usage: python check_dr_corners.py
"""

import numpy as np
from multiprocessing import Pool
from realistic_causal_env import RealisticCausalPushEnv, mode_spec
from eval_causal_agent import PPOPolicy
from experiment_config import ppo_path, PPO_VARIANTS


def run(args):
    preset, variant, conf, n = args
    pol = PPOPolicy(ppo_path(preset, variant), PPO_VARIANTS[variant]['history'])
    env = RealisticCausalPushEnv(preset=preset, distribution_mode='train_dr')
    ok, d = 0, []
    for s in range(n):
        obs, _ = env.reset(seed=5000 + s,
                           options=None if conf is None else {'confounders': conf})
        pol.reset()
        done = False
        while not done:
            a, _ = pol.predict(obs)
            obs, _, te, tr, info = env.step(a)
            done = te or tr
        ok += info['is_success']
        d.append(info['final_distance'])
    return (preset, variant, conf), ok / n, np.mean(d) * 100


if __name__ == '__main__':
    jobs = []
    for preset in ('narrow', 'pdf'):
        b = mode_spec(preset, 'train_dr')
        confs = [None] + [(m, f, 0.4) for m in b['mass'] for f in b['friction']]
        for v in ('ppo_dr', 'ppo_dr_hist'):
            jobs += [(preset, v, c, 50 if c is None else 20) for c in confs]
    with Pool(15) as p:
        for (preset, v, c), sr, d in p.map(run, jobs):
            where = 'whole DR box' if c is None else 'm=%.1f mu=%.2f' % c[:2]
            print(f"{preset:6s} {v:12s} {where:16s} success {sr*100:5.1f}%  dist {d:.2f}cm")
