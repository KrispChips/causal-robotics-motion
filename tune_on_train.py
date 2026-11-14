"""
Choose every hyperparameter of the planning agents from the TRAINING
distribution only, so no test-set information reaches them.

1. velocity_noise_std (causal agent's observation noise): empirical Bayes.
   Pick the sigma that maximises the marginal likelihood of unlabelled
   training-distribution push data (random-force episodes, the same kind of
   data the model-based baseline trains on). No true mu/m labels are used.
2. Planner weights (speed_weight, overshoot_weight, recourse_steps): grid
   search, maximising mean episode return on training-distribution episodes.
   Done separately for the causal agent and the model-based baseline, which
   share the planner code but not its settings.
3. Active-exploration settings (max_risk, min_gain, stopping tolerances):
   grid search the same way, on top of the tuned planner.

Seeds used here (2_000_000+) never overlap the evaluation seeds (1000+).
The planner's RNG is seeded per episode, so each setting is scored over
different random draws, not one lucky stream.
The test distributions are never instantiated.

Usage:  python tune_on_train.py --preset pdf
Writes: models_<preset>/agent_config.json
"""

import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from experiment_config import parse_preset, paths, agent_config_path
from realistic_causal_env import RealisticCausalPushEnv, PRESETS
from bayesian_causal_agent import BayesianCausalAgent, simulate_step

TUNE_SEED = 2_000_000
N_NOISE_EPISODES = 300
N_TUNE_EPISODES = 100

SIGMAS = [0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
PLAN_GRID = {'overshoot_weight': [1.0, 3.0, 10.0, 30.0],
             'recourse_steps': [1, 3, 5],
             'speed_weight': [0.03, 0.1, 0.3]}
EXPLORE_GRID = {'max_risk': [0.005, 0.01, 0.05],
                'min_gain': [0.02, 0.05, 0.1],
                'tol_scale': [0.5, 1.0, 2.0]}   # x (friction_tol 0.02, mass_tol 0.05)


# ------------------------------------------------------------ 1. noise
def tune_noise(preset):
    from mbrl_baseline import collect_transitions
    data = collect_transitions('train', n_episodes=N_NOISE_EPISODES, seed=TUNE_SEED // 100000,
                               preset=preset)
    steps = RealisticCausalPushEnv(preset=preset).max_episode_steps
    data = data.reshape(N_NOISE_EPISODES, steps, 4)            # v, F, dx, v_next
    grid = BayesianCausalAgent(force_range=(0.0, PRESETS[preset]['force_max']))
    n_grid = grid.mu_flat.size
    sq = np.zeros((N_NOISE_EPISODES, n_grid))                  # sum_t residual^2
    for e in range(N_NOISE_EPISODES):
        for v, F, _, v_next in data[e]:
            _, v_pred = simulate_step(0.0, v, F, grid.mu_flat, grid.m_flat)
            sq[e] += (v_next - v_pred) ** 2
    log_ev = []
    for s in SIGMAS:
        # log p(data | sigma) = sum_episodes logmeanexp_theta log N(...)  (uniform prior)
        ll = -sq / (2 * s * s) - steps * np.log(np.sqrt(2 * np.pi) * s)
        m = ll.max(axis=1, keepdims=True)
        log_ev.append(float(np.sum(m[:, 0] + np.log(np.mean(np.exp(ll - m), axis=1)))))
    best = SIGMAS[int(np.argmax(log_ev))]
    for s, le in zip(SIGMAS, log_ev):
        print(f"    sigma {s:.4f}  log evidence {le:12.1f}{'  <- best' if s == best else ''}")
    return best


# ------------------------------------------------- 2./3. episode returns
def _episode_return(args):
    preset, kind, cfg, seed = args
    from experiment_config import make_causal_policy, make_mbrl_policy, make_mbrl_ctx_policy
    if kind == 'mbrl':
        policy = make_mbrl_policy(preset, config=cfg)
    elif kind == 'mbrl_ctx':
        policy = make_mbrl_ctx_policy(preset, config=cfg)
    else:
        # Planner RNG varies per episode (as in evaluation). With one fixed
        # stream every setting is scored on the same random draws, and the
        # search partly selects for luck on that stream.
        policy = make_causal_policy(preset, exploration=kind, config=cfg, seed=seed)
    env = RealisticCausalPushEnv(distribution_mode='train', preset=preset)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    total, done = 0.0, False
    while not done:
        action, _ = policy.predict(obs)
        obs, r, term, trunc, info = env.step(action)
        total += r
        done = term or trunc
    env.close()
    return total, bool(info['is_success'])


def grid_search(pool, preset, kind, base_cfg, key, grid, expand):
    names = list(grid)
    combos = list(itertools.product(*[grid[n] for n in names]))
    jobs, cfgs = [], []
    for combo in combos:
        cfg = json.loads(json.dumps(base_cfg))
        cfg[key] = expand(dict(zip(names, combo)))
        cfgs.append(cfg)
        jobs += [(preset, kind, cfg, TUNE_SEED + i) for i in range(N_TUNE_EPISODES)]
    out = pool.map(_episode_return, jobs, chunksize=4)
    scores = []
    for c in range(len(combos)):
        chunk = out[c * N_TUNE_EPISODES:(c + 1) * N_TUNE_EPISODES]
        scores.append((float(np.mean([r for r, _ in chunk])),
                       float(np.mean([s for _, s in chunk]))))
    best = int(np.argmax([s[0] for s in scores]))
    print(f"    best {dict(zip(names, combos[best]))}: train return {scores[best][0]:.3f}, "
          f"success {scores[best][1]*100:.0f}%  (worst combo {min(s[0] for s in scores):.3f})")
    return cfgs[best][key], [dict(zip(names, c), ret=s[0], success=s[1])
                             for c, s in zip(combos, scores)]


def tune_only_mbrl_ctx(preset):
    """Add the history-conditioned MBRL planner settings to an existing
    config (same grid, seeds and episodes as every other agent)."""
    t0 = time.time()
    path = agent_config_path(preset)
    with open(path) as f:
        cfg = json.load(f)
    cfg['mbrl_ctx_plan'] = dict(cfg['mbrl_plan'])        # overwritten per grid point
    print(f"Tuning history-conditioned MBRL planner on '{preset}' train only")
    with Pool(15) as pool:
        cfg['mbrl_ctx_plan'], log = grid_search(pool, preset, 'mbrl_ctx', cfg,
                                                'mbrl_ctx_plan', PLAN_GRID, lambda d: d)
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(path.replace('.json', '_search_log_mbrl_ctx.json'), 'w') as f:
        json.dump(log, f, indent=1)
    print(f"Saved {path} ({time.time()-t0:.0f}s): mbrl_ctx_plan = {cfg['mbrl_ctx_plan']}")


def main(preset):
    t0 = time.time()
    print(f"Tuning on the TRAINING distribution of preset '{preset}' only")
    cfg = {'velocity_noise_std': None,
           'causal_plan': {'speed_weight': 0.1, 'overshoot_weight': 10.0, 'recourse_steps': 3},
           'causal_explore': {'max_risk': 0.01, 'min_gain': 0.05, 'margin': 0.02,
                              'friction_tol': 0.02, 'mass_tol': 0.05},
           'mbrl_plan': {'speed_weight': 0.1, 'overshoot_weight': 10.0, 'recourse_steps': 3}}

    print("\n1. Observation noise (empirical Bayes, unlabelled train data)")
    cfg['velocity_noise_std'] = tune_noise(preset)

    log = {}
    with Pool(14) as pool:
        print("\n2a. Causal agent planner (passive episodes)")
        cfg['causal_plan'], log['causal_plan'] = grid_search(
            pool, preset, 'none', cfg, 'causal_plan', PLAN_GRID, lambda d: d)
        if os.path.exists(paths(preset)['mbrl']):
            print("\n2b. Model-based baseline planner")
            cfg['mbrl_plan'], log['mbrl_plan'] = grid_search(
                pool, preset, 'mbrl', cfg, 'mbrl_plan', PLAN_GRID, lambda d: d)
        else:
            print(f"\n2b. (skipped: {paths(preset)['mbrl']} missing)")
        if os.path.exists(paths(preset)['mbrl_ctx']):
            print("\n2c. History-conditioned model-based baseline planner")
            cfg['mbrl_ctx_plan'] = dict(cfg['mbrl_plan'])
            cfg['mbrl_ctx_plan'], log['mbrl_ctx_plan'] = grid_search(
                pool, preset, 'mbrl_ctx', cfg, 'mbrl_ctx_plan', PLAN_GRID, lambda d: d)
        print("\n3. Active exploration settings")
        cfg['causal_explore'], log['causal_explore'] = grid_search(
            pool, preset, 'active', cfg, 'causal_explore', EXPLORE_GRID,
            lambda d: {'max_risk': d['max_risk'], 'min_gain': d['min_gain'], 'margin': 0.02,
                       'friction_tol': 0.02 * d['tol_scale'], 'mass_tol': 0.05 * d['tol_scale']})

    cfg['_protocol'] = (f"tuned on '{preset}' train distribution only, seeds {TUNE_SEED}+, "
                        f"{N_TUNE_EPISODES} episodes per setting; noise by empirical Bayes on "
                        f"{N_NOISE_EPISODES} unlabelled random-force episodes")
    path = agent_config_path(preset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(path.replace('.json', '_search_log.json'), 'w') as f:
        json.dump(log, f, indent=1)
    print(f"\nSaved {path} ({time.time()-t0:.0f}s)")
    print(json.dumps({k: v for k, v in cfg.items() if not k.startswith('_')}, indent=2))


if __name__ == '__main__':
    args = parse_preset('Tune planning agents on the training distribution only',
                        extra_args=[(('--only',), dict(choices=['mbrl_ctx'], default=None))])
    if args.only == 'mbrl_ctx':
        tune_only_mbrl_ctx(args.preset)
    else:
        main(args.preset)
