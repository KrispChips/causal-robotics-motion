"""
Baseline 2b: history-conditioned model-based RL.

The plain model-based baseline (mbrl_baseline.py) maps (v, F) -> next state,
so it can only learn the *average* block and cannot adapt to the one in
front of it. This version conditions its dynamics on everything that has
happened in the episode so far, in the spirit of context-aware dynamics
models (e.g. Lee et al. 2020, "Context-aware Dynamics Model"):

    context c = mean_i phi(v_i, F_i, dx_i, v_next_i)    over past transitions
    (dx, v_next) = f(v, F, c, n_past)

It starts from the same unlabelled random-force training-distribution data
as the plain baseline, predicts state deltas, and then adds on-policy data
in a few aggregation rounds (standard practice, e.g. PETS / MBPO): random
pushes mostly drive the block far faster than the task ever does, so a
model fit to them alone is inaccurate near rest, where the task is decided.
It never sees mu or m. If it can infer the block's
properties implicitly from its history, it has the same information as the
causal agent, minus the physics structure. It plans with the same shared
planner (plan_force) as every other planning agent.

Usage:  python mbrl_context_baseline.py --preset pdf
"""

import time
import numpy as np
import torch

from realistic_causal_env import force_to_action
from bayesian_causal_agent import plan_force
from mbrl_baseline import collect_transitions
from experiment_config import parse_preset, paths, modes

T_STEPS = 20


def collect_episodes(mode, n_episodes, seed, preset):
    """(n_episodes, T, 4) array of (v, F, dx, v_next) transitions."""
    data = collect_transitions(mode, n_episodes=n_episodes, seed=seed, preset=preset)
    return data.reshape(n_episodes, T_STEPS, 4)


TARGET_SCALE = np.array([1e-3, 1e-2])   # (dx m, dv m/s): linear below ~1 mm, 1 cm/s


def _to_target(d):
    return np.arcsinh(d / TARGET_SCALE)


def _from_target(y):
    return np.sinh(y) * TARGET_SCALE


def _n_feat(n_past):
    """Scalar 'how much history' input, 0 with no history."""
    return np.log1p(n_past) / np.log1p(T_STEPS)


class ContextDynamicsEnsemble:
    def __init__(self, params):
        self.p = params

    @property
    def n_members(self):
        return self.p['W1'].shape[0]

    # ---------------------------------------------------------- training
    @classmethod
    def train(cls, episodes, n_members=5, hidden=64, n_iters=8000, batch_size=256,
              lr=1e-3, seed=0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        flat = episodes.reshape(-1, 4)
        t_mu, t_sd = flat.mean(0), flat.std(0) + 1e-8          # transition features
        D = torch.tensor((episodes - t_mu) / t_sd, dtype=torch.float32)   # (N, T, 4)
        # Targets are state deltas (dx, dv = v_next - v), as in PETS/MBPO:
        # near rest a small error in dv stays small instead of a fraction of
        # the whole velocity scale.
        # Targets span 0 to tens of cm per step, so they go through an asinh
        # ("symlog", as in DreamerV3) transform: linear near zero, log for
        # large values. Otherwise a small normalised error is ~1 cm at rest.
        tgt = _to_target(np.stack([episodes[..., 2], episodes[..., 3] - episodes[..., 0]], -1))
        o_mu = tgt.reshape(-1, 2).mean(0); o_sd = tgt.reshape(-1, 2).std(0) + 1e-8
        Y = torch.tensor((tgt - o_mu) / o_sd, dtype=torch.float32)          # (N, T, 2)
        N, T, _ = D.shape
        E, H = n_members, hidden

        def P(*shape, fan_in):
            return torch.nn.Parameter(torch.randn(*shape) / np.sqrt(fan_in))
        def Z(*shape):
            return torch.nn.Parameter(torch.zeros(*shape))
        # encoder phi: 4 -> H -> H ; head: (2 + H + 1) -> H -> H -> 2
        prm = dict(We1=P(E, 4, H, fan_in=4), be1=Z(E, 1, H),
                   We2=P(E, H, H, fan_in=H), be2=Z(E, 1, H),
                   W1=P(E, 3 + H, H, fan_in=3 + H), b1=Z(E, 1, H),
                   W2=P(E, H, H, fan_in=H), b2=Z(E, 1, H),
                   W3=P(E, H, 2, fan_in=H), b3=Z(E, 1, 2))
        opt = torch.optim.Adam(prm.values(), lr=lr)
        boot = [rng.integers(0, N, N) for _ in range(E)]     # bootstrap episodes
        steps = torch.arange(T)

        for it in range(n_iters):
            ep = torch.tensor(np.stack([b[rng.integers(0, N, batch_size)] for b in boot]))
            t = torch.tensor(rng.integers(0, T, (E, batch_size)))
            ctx = D[ep]                                          # (E, B, T, 4)
            mask = (steps[None, None, :] < t[..., None]).float()  # past only
            cur = D[ep, t]                                       # (E, B, 4)
            h = torch.tanh(torch.einsum('ebtf,efh->ebth', ctx, prm['We1']) + prm['be1'][:, :, None])
            h = torch.tanh(torch.einsum('ebth,ehk->ebtk', h, prm['We2']) + prm['be2'][:, :, None])
            n = mask.sum(-1, keepdim=True)
            c = (h * mask[..., None]).sum(2) / n.clamp(min=1)    # (E, B, H)
            nf = torch.log1p(n) / np.log1p(T)
            x = torch.cat([cur[..., :2], c, nf], -1)
            z = torch.tanh(torch.baddbmm(prm['b1'], x, prm['W1']))
            z = torch.tanh(torch.baddbmm(prm['b2'], z, prm['W2']))
            out = torch.baddbmm(prm['b3'], z, prm['W3'])
            loss = ((out - Y[ep, t]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 2000 == 0 or it == n_iters - 1:
                print(f"    iter {it:5d}  loss {loss.item():.5f}")

        p = {k: v.detach().numpy().astype(np.float64) for k, v in prm.items()}
        p.update(t_mu=t_mu, t_sd=t_sd, o_mu=o_mu, o_sd=o_sd)
        return cls(p)

    # --------------------------------------------------------- inference
    def encode(self, history):
        """history: (k, 4) past transitions -> (context (E, H), n feature)."""
        p = self.p
        E, H = self.n_members, p['W2'].shape[1]
        if len(history) == 0:
            return np.zeros((E, H)), 0.0
        x = (np.asarray(history, dtype=np.float64) - p['t_mu']) / p['t_sd']   # (k, 4)
        h = np.tanh(np.einsum('tf,efh->eth', x, p['We1']) + p['be1'])
        h = np.tanh(h @ p['We2'] + p['be2'])
        return h.mean(1), _n_feat(len(history))

    def step(self, v, F, context):
        """v, F broadcastable to (N, E); context from encode(). Returns
        (dx, v_next) each (N, E); column e uses ensemble member e."""
        p = self.p
        c, nf = context
        v, F, _ = np.broadcast_arrays(np.asarray(v, float), np.asarray(F, float),
                                      np.empty((1, self.n_members)))
        vn = (v - p['t_mu'][0]) / p['t_sd'][0]
        Fn = (F - p['t_mu'][1]) / p['t_sd'][1]
        inp = np.stack([vn, Fn], -1).transpose(1, 0, 2)                 # (E, N, 2)
        Nn = inp.shape[1]
        x = np.concatenate([inp, np.repeat(c[:, None, :], Nn, 1),
                            np.full((self.n_members, Nn, 1), nf)], -1)
        z = np.tanh(x @ p['W1'] + p['b1'])
        z = np.tanh(z @ p['W2'] + p['b2'])
        out = (z @ p['W3'] + p['b3']).transpose(1, 0, 2)                 # (N, E, 2)
        d = _from_target(out * p['o_sd'] + p['o_mu'])
        dx, v_next = d[..., 0], v + d[..., 1]
        return dx, np.maximum(v_next, 0.0)       # no backward motion (as plain MBRL)

    def rollout_fn(self, history):
        """plan_force rollout with the context fixed to the real history."""
        context = self.encode(history)

        def rollout(x, v, F, n_steps):
            shape = (F.shape[0], self.n_members)
            x = np.full(shape, x, dtype=np.float64)
            dx, v = self.step(np.full(shape, v), F, context)
            x = x + dx
            for _ in range(n_steps - 1):
                dx, v = self.step(v, 0.0, context)
                x = x + dx
            return x, v
        return rollout

    def save(self, path):
        np.savez(path, **self.p)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls({k: f[k] for k in f.files})


class ContextModelPolicy:
    """Same interface as the other push agents. Builds its context from its
    own observations: (v, F, dx, v_next) for every step taken so far."""

    X, VX, TARGET = 0, 2, 4

    def __init__(self, model_path, max_episode_steps=T_STEPS, force_range=(0.0, 8.0),
                 force_resolution=0.1, plan_kwargs=None):
        self.model = ContextDynamicsEnsemble.load(model_path)
        self.plan_kwargs = dict(plan_kwargs or {})
        self.max_episode_steps = max_episode_steps
        self.force_range = force_range
        f_lo, f_hi = force_range
        self.candidate_forces = np.linspace(
            f_lo, f_hi, int(round((f_hi - f_lo) / force_resolution)) + 1)
        self.reset()

    def reset(self):
        self.episode_step = 0
        self.history = []
        self.prev = None

    def predict(self, observation, deterministic=True):
        x, v = float(observation[self.X]), float(observation[self.VX])
        target = float(observation[self.TARGET])
        if self.prev is not None:
            x0, v0 = self.prev
            self.history.append((v0, float(observation[-1]), x - x0, v))  # obs[-1] = last force
        force = plan_force(self.model.rollout_fn(self.history), x, v, target,
                           self.max_episode_steps - self.episode_step,
                           self.candidate_forces, **self.plan_kwargs)
        self.prev = (x, v)
        self.episode_step += 1
        return np.array([force_to_action(force, *self.force_range)], dtype=np.float32), None


N_RANDOM_EPISODES = 1000     # same random-force data as the plain baseline
N_ROUNDS = 4                 # then on-policy aggregation rounds (PETS/MBPO-style)
N_ROUND_EPISODES = 300
P_RANDOM_ACTION = 0.3        # exploration noise during on-policy collection


def _on_policy_episode(args):
    model_path, preset, seed = args
    from realistic_causal_env import RealisticCausalPushEnv, PRESETS
    f_max = PRESETS[preset]['force_max']
    policy = ContextModelPolicy(model_path, force_range=(0.0, f_max))   # default planner
    env = RealisticCausalPushEnv(distribution_mode='train', preset=preset)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    rows = []
    for _ in range(env.max_episode_steps):
        action, _ = policy.predict(obs)
        if rng.random() < P_RANDOM_ACTION:
            action = np.array([env.force_to_action(rng.uniform(0.0, f_max))], dtype=np.float32)
        x, v = float(obs[0]), float(obs[2])
        obs, _, term, trunc, info = env.step(action)
        rows.append((v, info['applied_force'], float(obs[0]) - x, float(obs[2])))
    env.close()
    return np.array(rows)


def train_with_aggregation(preset, out_path, seed=0):
    """Random-force data first, then rounds of: plan with the current model
    (plus random actions) on fresh training-distribution episodes, add the
    data, retrain. No labels, no physics; training distribution only."""
    from multiprocessing import Pool
    eps = collect_episodes('train', N_RANDOM_EPISODES, seed=seed, preset=preset)
    print(f"  round 0: {len(eps)} random-force episodes")
    model = ContextDynamicsEnsemble.train(eps, seed=seed)
    for r in range(1, N_ROUNDS + 1):
        model.save(out_path)
        seeds = [3_000_000 + r * 10_000 + i for i in range(N_ROUND_EPISODES)]
        with Pool(12) as pool:
            new = pool.map(_on_policy_episode, [(out_path, preset, s) for s in seeds])
        eps = np.concatenate([eps, np.stack(new)])
        print(f"  round {r}: +{len(new)} on-policy episodes ({len(eps)} total)")
        model = ContextDynamicsEnsemble.train(eps, seed=seed + r)
    model.save(out_path)
    return model


def one_step_error(model, episodes):
    """Mean abs one-step error with the context = the episode so far."""
    ex, ev = [], []
    for e in episodes:
        for t in range(len(e)):
            dx, vn = model.step(e[t, 0:1][:, None], e[t, 1:2][:, None], model.encode(e[:t]))
            ex.append(abs(dx.mean() - e[t, 2])); ev.append(abs(vn.mean() - e[t, 3]))
    return np.mean(ex), np.mean(ev)


if __name__ == "__main__":
    preset = parse_preset("Train the history-conditioned model-based baseline").preset
    out = paths(preset)['mbrl_ctx']
    t0 = time.time()
    torch.set_num_threads(4)
    model = train_with_aggregation(preset, out)
    print(f"Saved {out} ({time.time()-t0:.0f}s)")

    from mbrl_baseline import DynamicsEnsemble
    from mbrl_baseline import one_step_error as plain_error
    plain = DynamicsEnsemble.load(paths(preset)['mbrl'])
    print("\nOne-step error, mean abs (plain MBRL -> history-conditioned):")
    for mode in modes(preset):
        held = collect_episodes(mode, 50, seed=1, preset=preset)
        px, pv = plain_error(plain, held.reshape(-1, 4))
        cx, cv = one_step_error(model, held)
        print(f"  {mode:17s} dx {px*1000:6.2f} -> {cx*1000:6.2f} mm   "
              f"v_next {pv*1000:6.1f} -> {cv*1000:6.1f} mm/s")
