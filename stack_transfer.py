"""
Transfer test (PDF Task 2 / H2): push -> stack.

Each episode: a block with hidden (mu, m) is first pushed (shared, fixed
pusher), then a Panda must stack it (stack_env.PandaStackEnv). Every stacking
agent receives the identical push log as its only information.

Agents
  PPO (stack)      - model-free RL, trained on stacking episodes from the
                     training distribution only (raw push log -> carry speed)
  PPO (stack, DR)  - the same, trained on the domain-randomized box covering
                     every train and test range (privileged coverage)
  Causal           - push log -> Bayesian posterior over (mu, m) (the same
                     inference as the push agent) -> friction-cone grasp
                     condition (textbook: no slip allowed) over the robot's
                     own motion plan -> carry speed maximising expected
                     reward. No stacking training at all.
  Fixed fastest    - reference: always carry at the Panda's limit
  Causal, true μ,m - reference: the same decision rule given the true
                     (mu, m); the gap to 'Causal' is the cost of inference

What the causal agent is given that PPO is not: the physics *structure*
(Coulomb friction in pushing and in the grip) and the robot's own constants
(grip force, pad friction, motion plan). It is never given the train/test
confounder ranges, nor the env's drop criterion: it assumes any slip is a
failure, which is conservative. PPO gets the same observation and learns
everything else.

Usage
  python stack_transfer.py bank  --preset pdf   # record push logs (train + eval)
  python stack_transfer.py train --preset pdf   # train PPO stacking baseline
  python stack_transfer.py train --preset pdf --dr   # domain-randomized PPO
  python stack_transfer.py eval  --preset pdf   # paired comparison + figure
"""

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from experiment_config import parse_preset, paths, labels, figure, load_agent_config
from realistic_causal_env import PRESETS, TRAIN_DR, mode_spec
import stack_env as S

N_TRAIN_BANK = 2000
N_EVAL = {'test_full_range': 200}   # where the slippery/heavy corner lives
N_EVAL_DEFAULT = 50
PPO_STEPS = 20000                   # = stacking episodes (one decision each)


# ------------------------------------------------------------- push logs
def _sample(preset, mode, rng):
    spec = mode_spec(preset, mode)
    out = []
    for key in ('mass', 'friction', 'roughness'):
        lo, hi = spec[key]
        out.append(lo if lo == hi else float(rng.uniform(lo, hi)))
    return tuple(out)


def _record(args):
    preset, mode, seed = args
    rng = np.random.default_rng(seed)
    conf = _sample(preset, mode, rng)
    return conf, S.record_push_log(conf, preset, seed)


def bank_path(preset, mode, split):
    return os.path.join(paths(preset)['models'], f'stack_push_bank_{split}_{mode}.npz')


def record_bank(preset, mode, n, seed0, split, workers=12):
    path = bank_path(preset, mode, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    with Pool(workers) as pool:
        out = pool.map(_record, [(preset, mode, seed0 + i) for i in range(n)])
    conf = np.array([o[0] for o in out])
    logs = np.array([o[1] for o in out])
    np.savez(path, confounders=conf, logs=logs)
    print(f"  {split:5s} {mode:17s} {n:5d} push logs -> {path} ({time.time()-t0:.0f}s)")


def load_bank(preset, mode, split):
    with np.load(bank_path(preset, mode, split)) as f:
        return {'confounders': f['confounders'], 'logs': f['logs']}


# ----------------------------------------------------------------- agents
class CausalStackPolicy:
    name = 'Causal'

    def __init__(self, force_max, velocity_noise_std, n_samples=512, n_candidates=18,
                 success_reward=10.0, fail_reward=-5.0, rng=None):
        from bayesian_causal_agent import BayesianCausalAgent
        self.force_max = force_max
        self.agent = BayesianCausalAgent(rng=rng or np.random.default_rng(0),
                                         force_range=(0.0, force_max),
                                         velocity_noise_std=velocity_noise_std)
        self.n_samples = n_samples
        self.cands = np.linspace(S.SPEED_MIN_FRAC * S.A_PEAK_MAX, S.A_PEAK_MAX, n_candidates)
        self.times = np.array([S.task_time(a) for a in self.cands])
        self.r_succ, self.r_fail = success_reward, fail_reward
        self.last = {}

    def infer(self, obs):
        log = S.decode_push_log(obs, self.force_max)
        self.agent.reset_beliefs()
        x0 = v0 = 0.0
        for x, v, F in log:
            self.agent.update_beliefs(x0, v0, F, x, v)
            x0, v0 = x, v
        return self.agent.sample_parameters(self.n_samples)

    def predict(self, obs, deterministic=True):
        mu_s, m_s = self.infer(obs)
        p_drop = np.array([1.0 - S.holds(mu_s, m_s, a).mean() for a in self.cands])
        exp_r = (1 - p_drop) * (self.r_succ - self.times) + p_drop * self.r_fail
        i = int(np.argmax(exp_r))
        b = self.agent.get_beliefs()
        self.last = {'mu_est': b['friction_mean'], 'm_est': b['mass_mean'],
                     'p_drop': p_drop[i]}
        return S.peak_accel_to_action(self.cands[i]), None


class OracleStackPolicy:
    """The causal agent's decision rule given the true (mu, m): fastest
    candidate inside the friction cone."""
    name = 'Causal, true μ,m'

    def __init__(self, n_candidates=18):
        self.cands = np.linspace(S.SPEED_MIN_FRAC * S.A_PEAK_MAX, S.A_PEAK_MAX, n_candidates)
        self.env = None

    def predict(self, obs, deterministic=True):
        mu, m = self.env.base_friction, self.env.mass
        safe = [a for a in self.cands if S.holds(mu, m, a)]
        return S.peak_accel_to_action(max(safe) if safe else self.cands[0]), None


class FixedSpeedPolicy:
    name = 'Fixed fastest'

    def predict(self, obs, deterministic=True):
        return S.peak_accel_to_action(S.A_PEAK_MAX), None


class PPOStackPolicy:
    def __init__(self, path, name='PPO (stack)'):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device='cpu')
        self.name = name

    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=True)


# ------------------------------------------------------------- PPO train
def ppo_stack_path(preset, dr=False):
    return os.path.join(paths(preset)['models'], 'ppo_stack_dr.zip' if dr else 'ppo_stack.zip')


def train_ppo(preset, timesteps=PPO_STEPS, dr=False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    dist = TRAIN_DR if dr else 'train'
    bank = load_bank(preset, dist, 'train')
    log_dir = os.path.join(paths(preset)['logs'], 'stack_dr' if dr else 'stack')
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(S.PandaStackEnv(preset=preset, distribution_mode=dist, push_bank=bank),
                  os.path.join(log_dir, 'monitor'), info_keywords=('is_success', 'slipped'))
    env.reset(seed=0)
    model = PPO('MlpPolicy', env, n_steps=512, batch_size=64, n_epochs=10,
                learning_rate=3e-4, gamma=0.0,   # one decision per episode
                ent_coef=0.0, seed=0, verbose=1, device='cpu')
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    out = ppo_stack_path(preset, dr)
    model.save(out)
    print(f"Saved {out} ({time.time()-t0:.0f}s, {timesteps} stacking episodes)")


# -------------------------------------------------------------- evaluate
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def evaluate(preset):
    force_max = PRESETS[preset]['force_max']
    agents = []
    for dr, name in ((False, 'PPO (stack)'), (True, 'PPO (stack, DR)')):
        path = ppo_stack_path(preset, dr)
        if os.path.exists(path):
            agents.append(PPOStackPolicy(path, name))
        else:
            print(f"  (skipping {name}: {path} missing -- run: stack_transfer.py train"
                  f"{' --dr' if dr else ''})")
    sigma = load_agent_config(preset)['velocity_noise_std']   # tuned on train only
    agents += [CausalStackPolicy(force_max, sigma), FixedSpeedPolicy(), OracleStackPolicy()]

    results = {}
    for mode, label in labels(preset).items():
        bank = load_bank(preset, mode, 'eval')
        n = len(bank['logs'])
        env = S.PandaStackEnv(preset=preset, distribution_mode=mode)
        results[mode] = {}
        print(f"\n{label} ({n} paired episodes)")
        for agent in agents:
            if isinstance(agent, OracleStackPolicy):
                agent.env = env
            r = {k: [] for k in ('success', 'dropped', 'reward', 'time', 'mu', 'm',
                                 'mu_est', 'm_est', 'accel')}
            t0 = time.time()
            for i in range(n):
                conf = tuple(bank['confounders'][i])
                obs, _ = env.reset(seed=i, options={'confounders': conf,
                                                    'push_log': bank['logs'][i]})
                action, _ = agent.predict(obs)
                _, rew, _, _, info = env.step(action)
                r['success'].append(info['is_success']); r['dropped'].append(info['slipped'])
                r['reward'].append(rew); r['time'].append(info['task_time'])
                r['mu'].append(conf[1]); r['m'].append(conf[0])
                r['accel'].append(info['peak_accel'])
                if isinstance(agent, CausalStackPolicy):
                    r['mu_est'].append(agent.last['mu_est']); r['m_est'].append(agent.last['m_est'])
            r = {k: np.array(v) for k, v in r.items()}
            k = int(r['success'].sum())
            r['ci'] = wilson(k, n)
            results[mode][agent.name] = r
            extra = ''
            if len(r['mu_est']):
                extra = (f"  μ MAE {np.abs(r['mu_est']-r['mu']).mean():.3f}"
                         f"  m MAE {np.abs(r['m_est']-r['m']).mean():.2f}kg")
            print(f"  {agent.name:20s} success {k/n*100:5.1f}% [{r['ci'][0]*100:3.0f}-{r['ci'][1]*100:3.0f}]"
                  f"  drops {r['dropped'].mean()*100:4.1f}%  reward {r['reward'].mean():5.2f}"
                  f"  carry accel {r['accel'].mean():4.1f} m/s²  time {r['time'].mean():.2f}s{extra}"
                  f"  ({(time.time()-t0)/n*1000:.0f} ms/ep)")
        env.close()
    plot(results, preset)
    return results


def plot(results, preset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes = list(results)
    names = list(results[modes[0]])
    lab = labels(preset)
    colors = ['#2a78d6', '#7b5ea7', '#eb6834', '#1baf7a', '#eda100']  # fixed categorical order
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    x = np.arange(len(modes))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        rate = np.array([results[m][n]['success'].mean() for m in modes]) * 100
        ci = np.array([results[m][n]['ci'] for m in modes]) * 100
        axes[0].bar(x + (i - (len(names) - 1) / 2) * w, rate, w * 0.92, color=colors[i],
                    yerr=np.vstack([rate - ci[:, 0], ci[:, 1] - rate]), capsize=2, label=n)
        rew = [results[m][n]['reward'].mean() for m in modes]
        axes[1].bar(x + (i - (len(names) - 1) / 2) * w, rew, w * 0.92, color=colors[i], label=n)
    for ax, t, yl in ((axes[0], 'Stack success (95% Wilson CI)', 'Success rate (%)'),
                      (axes[1], 'Mean reward (10 − task time, −5 on failure)', 'Reward')):
        ax.set_xticks(x); ax.set_xticklabels([lab[m] for m in modes])
        ax.set_title(t, loc='left'); ax.set_ylabel(yl)
        ax.grid(True, axis='y', alpha=0.25)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylim(0, 105)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='upper center', ncol=len(names), frameon=False,
               fontsize=9)

    # where in (mu, m) space each agent drops the block (full-range set)
    fr = 'test_full_range' if 'test_full_range' in results else modes[-1]
    ax = axes[2]
    ref = results[fr][names[0]]
    ax.scatter(ref['mu'], ref['m'], s=10, color='#c3c2b7', label='episode')
    markers = ['x', 'D', 'o', '+', 's']
    for i, n in enumerate(names):
        d = results[fr][n]['dropped'] | ~results[fr][n]['success']
        if d.any():
            style = (dict(facecolors='none', edgecolors=colors[i]) if markers[i] in 'os'
                     else dict(color=colors[i]))
            ax.scatter(results[fr][n]['mu'][d], results[fr][n]['m'][d], s=40,
                       marker=markers[i], label=f'{n}: failed', **style)
    ax.set_xlabel('True friction μ'); ax.set_ylabel('True mass m (kg)')
    ax.set_title(f'Failures on {lab[fr]} set', loc='left')
    ax.legend(frameon=False, fontsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = figure('stack_transfer.png', preset)
    fig.savefig(out, dpi=150)
    print(f"\n✓ Plot saved as '{out}'")


# ------------------------------------------------------------------ main
if __name__ == '__main__':
    args = parse_preset('Push -> stack transfer experiment',
                        extra_args=[(('command',), dict(choices=['bank', 'train', 'eval'])),
                                    (('--dr',), dict(action='store_true',
                                                     help='domain-randomized training'))])
    preset = args.preset
    if args.command == 'bank' and args.dr:
        record_bank(preset, TRAIN_DR, N_TRAIN_BANK, seed0=5_000_000, split='train')
    elif args.command == 'bank':
        record_bank(preset, 'train', N_TRAIN_BANK, seed0=0, split='train')
        for i, mode in enumerate(labels(preset)):
            record_bank(preset, mode, N_EVAL.get(mode, N_EVAL_DEFAULT),
                        seed0=10_000_000 * (i + 1), split='eval')
    elif args.command == 'train':
        train_ppo(preset, dr=args.dr)
    else:
        evaluate(preset)
