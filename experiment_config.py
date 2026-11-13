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
    d['evaluations'] = os.path.join(d['logs'], 'evaluations.npz')
    return d


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


def figure(name, preset):
    """'foo.png' -> 'foo.png' (narrow) or 'foo_pdf.png'."""
    stem, ext = os.path.splitext(name)
    return f"{stem}{paths(preset)['suffix']}{ext}"
