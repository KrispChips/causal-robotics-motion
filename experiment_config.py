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


def figure(name, preset):
    """'foo.png' -> 'foo.png' (narrow) or 'foo_pdf.png'."""
    stem, ext = os.path.splitext(name)
    return f"{stem}{paths(preset)['suffix']}{ext}"
