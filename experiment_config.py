"""
Shared experiment settings: which distribution preset a script runs on and
where that preset's models, logs and figures live.

Every script accepts `--preset narrow|pdf` (default: narrow, the original
setup, whose artifacts keep their original paths).
"""

import argparse
import os

from realistic_causal_env import PRESETS

MODE_LABELS = {
    'train': 'Train',
    'test_interp': 'Interp',
    'test_extrap_high': 'High μ',
    'test_extrap_low': 'Low μ',
    'test_adversarial': 'Adv',
    'test_full_range': 'Full range',
}


def parse_preset(description=None, argv=None, extra_args=None):
    """Parse --preset (plus any extra (flags, kwargs) pairs) from argv."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--preset', default='narrow', choices=list(PRESETS))
    for flags, kwargs in (extra_args or []):
        ap.add_argument(*flags, **kwargs)
    return ap.parse_args(argv)


def modes(preset):
    return list(PRESETS[preset]['modes'])


def labels(preset):
    return {m: MODE_LABELS.get(m, m) for m in modes(preset)}


def paths(preset):
    """Model/log directories and a filename suffix for figures."""
    if preset == 'narrow':
        d = {'models': 'models_realistic', 'logs': 'logs_realistic', 'suffix': ''}
    else:
        d = {'models': f'models_{preset}', 'logs': f'logs_{preset}', 'suffix': f'_{preset}'}
    d['ppo'] = os.path.join(d['models'], 'best_model.zip')
    d['mbrl'] = os.path.join(d['models'], 'mbrl_ensemble.npz')
    d['mbrl_ctx'] = os.path.join(d['models'], 'mbrl_context_ensemble.npz')
    d['evaluations'] = os.path.join(d['logs'], 'evaluations.npz')
    return d


# PPO variants. 'ppo' is the original baseline (plan of record); the others
# test whether its failures come from too little memory (full-episode
# history, the same information the causal agent conditions on) or too
# little training coverage (domain randomization over every train and test
# range -- privileged information, so an upper bound for data coverage).
PPO_VARIANTS = {
    'ppo':         dict(label='PPO',                 history=3,  dist='train'),
    'ppo_hist':    dict(label='PPO (full history)',  history=20, dist='train'),
    'ppo_dr':      dict(label='PPO (DR)',            history=3,  dist='train_dr'),
    'ppo_dr_hist': dict(label='PPO (DR, full hist)', history=20, dist='train_dr'),
}


def ppo_dir(preset, variant, world='nominal'):
    """Original PPO keeps its original location; variants get a subfolder;
    PPO trained in a non-nominal world (misspecification ablation) lives
    under world_<name>/."""
    p = paths(preset)
    base = p['models'] if world == 'nominal' else os.path.join(p['models'], f'world_{world}')
    return base if variant == 'ppo' else os.path.join(base, variant)


def ppo_path(preset, variant, world='nominal'):
    return os.path.join(ppo_dir(preset, variant, world), 'best_model.zip')


def agent_config_path(preset):
    return os.path.join(paths(preset)['models'], 'agent_config.json')


def load_agent_config(preset):
    """Hyperparameters of the planning agents, chosen on the TRAINING
    distribution only by tune_on_train.py. Keys: velocity_noise_std,
    causal_plan, causal_explore, mbrl_plan."""
    import json
    path = agent_config_path(preset)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python tune_on_train.py --preset {preset}` first: "
            f"agent hyperparameters must come from training-distribution data only.")
    with open(path) as f:
        return json.load(f)


def make_causal_policy(preset, exploration='none', seed=0, config=None):
    """The causal push agent with train-tuned hyperparameters."""
    import numpy as np
    from bayesian_causal_agent import CausalDiscoveryPolicy
    cfg = config or load_agent_config(preset)
    return CausalDiscoveryPolicy(
        exploration=exploration, rng=np.random.default_rng(seed),
        force_range=(0.0, PRESETS[preset]['force_max']),
        velocity_noise_std=cfg['velocity_noise_std'],
        plan_kwargs=cfg['causal_plan'],
        explore_kwargs={k: v for k, v in cfg['causal_explore'].items()
                        if k in ('min_gain', 'margin', 'max_risk')},
        **{k: v for k, v in cfg['causal_explore'].items()
           if k in ('friction_tol', 'mass_tol')})


def make_mbrl_policy(preset, config=None):
    from mbrl_baseline import ModelBasedPolicy
    cfg = config or load_agent_config(preset)
    return ModelBasedPolicy(paths(preset)['mbrl'],
                            force_range=(0.0, PRESETS[preset]['force_max']),
                            plan_kwargs=cfg['mbrl_plan'])


def make_mbrl_ctx_policy(preset, config=None):
    from mbrl_context_baseline import ContextModelPolicy
    cfg = config or load_agent_config(preset)
    if 'mbrl_ctx_plan' not in cfg:
        raise KeyError(f"no mbrl_ctx_plan in {agent_config_path(preset)}: run "
                       f"`python tune_on_train.py --preset {preset} --only mbrl_ctx`")
    return ContextModelPolicy(paths(preset)['mbrl_ctx'],
                              force_range=(0.0, PRESETS[preset]['force_max']),
                              plan_kwargs=cfg['mbrl_ctx_plan'])


def figure(name, preset):
    """'foo.png' -> 'foo.png' (narrow) or 'foo_pdf.png'."""
    stem, ext = os.path.splitext(name)
    return f"{stem}{paths(preset)['suffix']}{ext}"
