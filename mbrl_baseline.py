"""
Baseline 2: Model-Based RL (PDF Phase 4)

Learns a forward model of the block's dynamics from training-distribution
interaction data and plans with it -- but has no notion of hidden
confounders. The model maps (velocity, force) -> (dx, next velocity), so it
learns the *average* dynamics of the training blocks.

For a controlled comparison it uses exactly the same planner as the causal
agent (bayesian_causal_agent.plan_force); the ensemble members play the role
the posterior samples play for the causal agent. The only difference
between the two agents is therefore:
    learned average dynamics  vs  physics model + inferred (mu, m).

Usage:
    python mbrl_baseline.py          # collect data, train, report accuracy
"""

import os
import time
import numpy as np
import torch

from realistic_causal_env import (RealisticCausalPushEnv, FORCE_MIN, FORCE_MAX,
                                  force_to_action)
from bayesian_causal_agent import plan_force
from experiment_config import parse_preset, paths, modes

MODEL_PATH = "models_realistic/mbrl_ensemble.npz"


# ---------------------------------------------------------------- data
def collect_transitions(mode='train', n_episodes=1000, p_zero=0.4, seed=0,
                        preset='narrow'):
    """Random-force interaction data. Force is 0 with prob p_zero (so the
    model sees coasting/braking), otherwise uniform over the force range."""
    env = RealisticCausalPushEnv(distribution_mode=mode, preset=preset)
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 100000 + ep)
        for _ in range(env.max_episode_steps):
            F = 0.0 if rng.random() < p_zero else rng.uniform(env.force_min, env.force_max)
            x, v = obs[0], obs[2]
            obs, _, terminated, truncated, info = env.step(
                np.array([env.force_to_action(F)]))
            rows.append((v, info['applied_force'], obs[0] - x, obs[2]))
            if terminated or truncated:
                break
    env.close()
    return np.array(rows, dtype=np.float64)  # columns: v, F, dx, v_next


# --------------------------------------------------------------- model
class DynamicsEnsemble:
    """Ensemble of small MLPs, (v, F) -> (dx, v_next). Inference in numpy
    (planning calls it thousands of times; torch overhead would dominate)."""

    def __init__(self, params):
        self.p = params  # dict of numpy arrays

    @classmethod
    def train(cls, data, n_members=5, hidden=64, n_iters=4000,
              batch_size=512, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        X, Y = data[:, :2], data[:, 2:]
        x_mu, x_sd = X.mean(0), X.std(0) + 1e-8
        y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-8
        Xn = torch.tensor((X - x_mu) / x_sd, dtype=torch.float32)
        Yn = torch.tensor((Y - y_mu) / y_sd, dtype=torch.float32)

        E = n_members
        def init(*shape, fan_in):
            return torch.nn.Parameter(torch.randn(*shape) / np.sqrt(fan_in))
        W1, W2, W3 = init(E, 2, hidden, fan_in=2), init(E, hidden, hidden, fan_in=hidden), \
            init(E, hidden, 2, fan_in=hidden)
        b1, b2, b3 = (torch.nn.Parameter(torch.zeros(E, 1, n))
                      for n in (hidden, hidden, 2))
        params = [W1, b1, W2, b2, W3, b3]
        opt = torch.optim.Adam(params, lr=lr)
        # Bootstrap: each member trains on its own resample of the data
        boot = [rng.integers(0, len(Xn), len(Xn)) for _ in range(E)]

        for it in range(n_iters):
            idx = torch.tensor(np.stack(
                [b[rng.integers(0, len(b), batch_size)] for b in boot]))
            xb, yb = Xn[idx], Yn[idx]                       # (E, B, 2)
            h = torch.tanh(torch.baddbmm(b1, xb, W1))
            h = torch.tanh(torch.baddbmm(b2, h, W2))
            loss = ((torch.baddbmm(b3, h, W3) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 1000 == 0 or it == n_iters - 1:
                print(f"    iter {it:5d}  loss {loss.item():.5f}")

        to_np = lambda t: t.detach().numpy().astype(np.float64)
        return cls(dict(W1=to_np(W1), b1=to_np(b1), W2=to_np(W2), b2=to_np(b2),
                        W3=to_np(W3), b3=to_np(b3), x_mu=x_mu, x_sd=x_sd,
                        y_mu=y_mu, y_sd=y_sd))

    @property
    def n_members(self):
        return self.p['W1'].shape[0]

    def step(self, v, F):
        """v, F broadcastable to (N, E) -> (dx, v_next) each (N, E).
        Column e is propagated by ensemble member e."""
        p = self.p
        v, F = np.broadcast_arrays(v, F)
        inp = np.stack([(v - p['x_mu'][0]) / p['x_sd'][0],
                        (F - p['x_mu'][1]) / p['x_sd'][1]], axis=-1)  # (N, E, 2)
        inp = inp.transpose(1, 0, 2)                                  # (E, N, 2)
        h = np.tanh(inp @ p['W1'] + p['b1'])
        h = np.tanh(h @ p['W2'] + p['b2'])
        out = (h @ p['W3'] + p['b3']) * p['y_sd'] + p['y_mu']        # (E, N, 2)
        out = out.transpose(1, 0, 2)
        # Same physical constraint the causal model has: no backward motion
        return out[..., 0], np.maximum(out[..., 1], 0.0)

    def rollout(self, x, v, F, n_steps):
        """Plan rollout for plan_force: F (n_forces, 1) for one step, then
        coast. Returns (x_T, v_T) shaped (n_forces, n_members)."""
        shape = (F.shape[0], self.n_members)
        x = np.full(shape, x, dtype=np.float64)
        v = np.full(shape, v, dtype=np.float64)
        dx, v = self.step(v, F)
        x = x + dx
        for _ in range(n_steps - 1):
            dx, v = self.step(v, 0.0)
            x = x + dx
        return x, v

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **self.p)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls({k: f[k] for k in f.files})


# -------------------------------------------------------------- policy
class ModelBasedPolicy:
    """Gym-style wrapper with the same interface as CausalDiscoveryPolicy."""

    X, VX, TARGET = 0, 2, 4

    def __init__(self, model_path=MODEL_PATH, max_episode_steps=20,
                 force_range=(FORCE_MIN, FORCE_MAX), force_resolution=0.1,
                 plan_kwargs=None):
        self.model = DynamicsEnsemble.load(model_path)
        self.plan_kwargs = dict(plan_kwargs or {})
        self.max_episode_steps = max_episode_steps
        self.force_range = force_range
        f_lo, f_hi = force_range
        self.candidate_forces = np.linspace(
            f_lo, f_hi, int(round((f_hi - f_lo) / force_resolution)) + 1)
        self.reset()

    def reset(self):
        self.episode_step = 0

    def predict(self, observation, deterministic=True):
        x, v = float(observation[self.X]), float(observation[self.VX])
        target = float(observation[self.TARGET])
        force = plan_force(self.model.rollout, x, v, target,
                           self.max_episode_steps - self.episode_step,
                           self.candidate_forces, **self.plan_kwargs)
        self.episode_step += 1
        return np.array([force_to_action(force, *self.force_range)],
                        dtype=np.float32), None


# ---------------------------------------------------------------- main
def one_step_error(model, data):
    dx, v_next = model.step(data[:, 0:1], data[:, 1:2])
    return (np.abs(dx.mean(1) - data[:, 2]).mean(),
            np.abs(v_next.mean(1) - data[:, 3]).mean())


if __name__ == "__main__":
    preset = parse_preset("Train the model-based RL baseline").preset
    model_path = paths(preset)['mbrl']
    print("=" * 70)
    print(f"MODEL-BASED RL BASELINE: learn dynamics on training distribution ({preset})")
    print("=" * 70)
    t0 = time.time()
    data = collect_transitions('train', n_episodes=1000, seed=0, preset=preset)
    print(f"Collected {len(data)} transitions from 1000 training episodes "
          f"({time.time() - t0:.0f}s)")

    print("Training 5-member ensemble...")
    model = DynamicsEnsemble.train(data)
    model.save(model_path)
    print(f"Saved: {model_path}")

    print("\nOne-step prediction error (mean abs) by distribution:")
    for mode in modes(preset):
        held_out = collect_transitions(mode, n_episodes=50, seed=1, preset=preset)
        ex, ev = one_step_error(model, held_out)
        print(f"  {mode:17s} dx {ex*1000:6.2f} mm   v_next {ev*1000:6.1f} mm/s")
