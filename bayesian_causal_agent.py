"""
Bayesian Causal Agent

Treats the block's friction (mu) and mass (m) as hidden confounders of the
force -> motion relationship and infers them from interaction:

    mu ──┐
         ├──> block motion (x, v)
    m  ──┘        ^
                  │
            applied force F   (the agent's intervention)

Method
------
* Structural model: Coulomb friction, integrated at the simulator's 240 Hz
  substep rate,  dv/dt = (F - mu*m*g)/m  with v clamped at 0 (friction can
  stop the block but never push it backwards). Validated against PyBullet:
  ~0.15 mm position / ~3 mm/s velocity error per 0.1 s control step.
* Inference: exact Bayesian update on a 2-D grid over (mu, m). Parameters
  are constant within an episode, so a grid avoids the degeneracy and
  resampling/jitter artefacts a particle filter has for static parameters.
  Each control step contributes the likelihood of the observed velocity
  given the *observed* starting state (x, v) -- not a from-rest rollout.
  "Block did not move under force F" is an observation like any other:
  every (mu, m) with mu*m*g < F predicts motion and is penalised.
* Identifiability: pushing gives  a = F/m - mu*g,  coasting gives  a = -mu*g.
  Both phases happen in every episode, so mu and m are separable.
* Control (exploit): at each step, choose the force for this step (then
  coast with F = 0) that minimises the posterior-expected cost of the final
  position. Re-planned every step as the posterior sharpens.
* Active discovery (PDF Stage 3): before exploiting, choose *safe* probe
  forces that maximise expected information gain
      IG(F) = H[P(mu,m | h)] - E_y[ H[P(mu,m | h, F, y)] ]
  where "safe" means the posterior probability that the probe makes the
  block overshoot the target is small (overshoot is irreversible because
  the agent can only push).
"""

import numpy as np
import matplotlib.pyplot as plt

from realistic_causal_env import FORCE_MIN, FORCE_MAX, force_to_action

G = 9.8
SIM_DT = 1.0 / 240.0
SUBSTEPS = 24  # 240 Hz physics / 10 Hz control


def simulate_step(x, v, force, mu, mass, substeps=SUBSTEPS, dt=SIM_DT):
    """Advance one control step under constant force. Vectorised over any
    broadcastable combination of x, v, force, mu, mass (numpy arrays)."""
    decel_friction = mu * G
    for _ in range(substeps):
        v = np.maximum(v + (force / mass - decel_friction) * dt, 0.0)
        x = x + v * dt
    return x, v


def plan_force(rollout, x, v, target, n_steps_remaining, candidate_forces,
               speed_weight=0.1, overshoot_weight=10.0, recourse_steps=3):
    """Shared planner (used by the causal agent and the model-based baseline,
    so they differ only in their dynamics model).

    rollout(x, v, F, n_steps) -> (x_T, v_T), each shaped (n_forces, n_samples):
    final state after applying F[i] for one step and coasting afterwards,
    under each of n_samples plausible dynamics (posterior samples or
    ensemble members).

    Cost is asymmetric while there is time left to correct: undershoot is
    recoverable on a later step but overshoot never is, so early pushes are
    cautious while the dynamics are still uncertain.
    """
    if n_steps_remaining <= 0:
        return 0.0
    x_T, v_T = rollout(x, v, candidate_forces[:, None], n_steps_remaining)
    err = x_T - target
    w_over = overshoot_weight if n_steps_remaining > recourse_steps else 1.0
    pos_cost = np.where(err > 0, w_over, 1.0) * err ** 2
    cost = np.mean(pos_cost + speed_weight * v_T ** 2, axis=1)
    return float(candidate_forces[np.argmin(cost)])


class BayesianCausalAgent:
    """Grid-based Bayesian posterior over (friction, mass) + planner."""

    def __init__(self,
                 friction_range=(0.05, 0.80),
                 mass_range=(0.3, 2.5),
                 grid_size=(120, 120),
                 velocity_noise_std=0.015,
                 n_plan_samples=256,
                 n_candidate_forces=81,
                 rng=None):
        """
        Prior: uniform over a box that covers every train/test setting with
        margin (the PDF's full physical range is mu in [0.2,0.8], m in
        [0.5,2.0]). The agent is NOT told the training distribution.

        velocity_noise_std: per-step observation noise on velocity (m/s).
        Covers model error (~0.003) plus the env's per-step friction
        resampling, sigma_mu = 0.02*roughness -> sigma_v ~ 0.014 at r = 0.7.
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.friction_values = np.linspace(*friction_range, grid_size[0])
        self.mass_values = np.linspace(*mass_range, grid_size[1])
        mu_grid, m_grid = np.meshgrid(self.friction_values, self.mass_values,
                                      indexing='ij')
        self.mu_flat = mu_grid.ravel()
        self.m_flat = m_grid.ravel()
        self.velocity_noise_std = velocity_noise_std
        self.n_plan_samples = n_plan_samples
        self.candidate_forces = np.linspace(FORCE_MIN, FORCE_MAX, n_candidate_forces)
        self.reset_beliefs()

    # ------------------------------------------------------------------ beliefs
    def reset_beliefs(self):
        self.log_post = np.zeros_like(self.mu_flat)  # uniform prior
        self.interaction_history = []

    @property
    def weights(self):
        w = np.exp(self.log_post - self.log_post.max())
        return w / w.sum()

    def _log_likelihood(self, v_observed, v_predicted):
        resid = (v_observed - v_predicted) / self.velocity_noise_std
        return -0.5 * resid ** 2

    def update_beliefs(self, x_before, v_before, force_applied, x_after, v_after):
        """Bayes rule for one control step:
        log P(mu,m | data) += log N(v_after; v_pred(mu,m), sigma_v^2)."""
        _, v_pred = simulate_step(x_before, v_before, force_applied,
                                  self.mu_flat, self.m_flat)
        self.log_post += self._log_likelihood(v_after, v_pred)
        self.log_post -= self.log_post.max()  # keep numerically bounded
        self.interaction_history.append({
            'x_before': x_before, 'v_before': v_before, 'force': force_applied,
            'x_after': x_after, 'v_after': v_after,
        })

    def get_beliefs(self):
        w = self.weights
        mu_mean = np.sum(w * self.mu_flat)
        m_mean = np.sum(w * self.m_flat)
        mu_std = np.sqrt(np.sum(w * (self.mu_flat - mu_mean) ** 2))
        m_std = np.sqrt(np.sum(w * (self.m_flat - m_mean) ** 2))
        # Friction force mu*m*g is often pinned down before mu and m separate
        fric_force = self.mu_flat * self.m_flat * G
        ff_mean = np.sum(w * fric_force)
        ff_std = np.sqrt(np.sum(w * (fric_force - ff_mean) ** 2))
        return {
            'friction_mean': mu_mean, 'friction_std': mu_std,
            'mass_mean': m_mean, 'mass_std': m_std,
            'friction_force_mean': ff_mean, 'friction_force_std': ff_std,
            'entropy': _entropy(w),
        }

    def sample_parameters(self, n):
        idx = self.rng.choice(self.mu_flat.size, size=n, p=self.weights)
        return self.mu_flat[idx], self.m_flat[idx]

    # ----------------------------------------------------------------- planning
    def predict_final_state(self, x, v, first_force, mu, mass, n_steps):
        """Apply first_force for one step, then coast (F=0) for the rest."""
        x, v = simulate_step(x, v, first_force, mu, mass)
        for _ in range(n_steps - 1):
            x, v = simulate_step(x, v, 0.0, mu, mass)
        return x, v

    def _posterior_rollout(self):
        mu_s, m_s = self.sample_parameters(self.n_plan_samples)
        return lambda x, v, F, n: self.predict_final_state(
            x, v, F, mu_s[None, :], m_s[None, :], n)

    def choose_force(self, x, v, target, n_steps_remaining, **plan_kwargs):
        """Exploit: posterior-expected best force (see plan_force)."""
        return plan_force(self._posterior_rollout(), x, v, target,
                          n_steps_remaining, self.candidate_forces, **plan_kwargs)

    # ---------------------------------------------------------- active discovery
    def safe_forces(self, x, v, target, n_steps_remaining,
                    margin=0.02, max_risk=0.01, n_samples=1024):
        """Forces whose posterior probability of carrying the block (after
        coasting) beyond target - margin is at most max_risk.

        max_risk is small on purpose: an overshoot ends the episode, and with
        a wide prior the true block can easily sit in a 5% tail (e.g. the
        light, slippery test block)."""
        mu_s, m_s = self.sample_parameters(n_samples)
        x_T, _ = self.predict_final_state(
            x, v, self.candidate_forces[:, None], mu_s[None, :], m_s[None, :],
            n_steps_remaining)
        risk = np.mean(x_T > target - margin, axis=1)
        return self.candidate_forces[risk <= max_risk]

    def expected_information_gain(self, x, v, forces, n_outcomes=32,
                                  prune=1e-6):
        """Monte-Carlo estimate of IG(F) for each F in `forces` (nats).

        For each F: draw (mu, m) from the posterior, simulate the next
        velocity, add observation noise -> hypothetical outcome y; update
        the posterior with y and measure its entropy. IG = H_now - E[H_after].
        Grid points with negligible posterior mass are pruned for speed.
        """
        w = self.weights
        keep = w > w.max() * prune
        w_k = w[keep] / w[keep].sum()
        log_w_k = np.log(w_k)
        mu_k, m_k = self.mu_flat[keep], self.m_flat[keep]
        h_now = _entropy(w_k)

        ig = np.empty(len(forces))
        for i, F in enumerate(forces):
            _, v_pred = simulate_step(x, v, F, mu_k, m_k)            # (K,)
            truth = self.rng.choice(len(w_k), size=n_outcomes, p=w_k)
            y = v_pred[truth] + self.rng.normal(0, self.velocity_noise_std,
                                                n_outcomes)
            log_post = log_w_k[None, :] + self._log_likelihood(
                y[:, None], v_pred[None, :])                          # (n_out, K)
            log_post -= log_post.max(axis=1, keepdims=True)
            post = np.exp(log_post)
            post /= post.sum(axis=1, keepdims=True)
            ig[i] = h_now - np.mean([_entropy(p) for p in post])
        return ig

    def choose_exploratory_force(self, x, v, target, n_steps_remaining,
                                 strategy='active', max_candidates=17,
                                 min_gain=0.05):
        """Return a safe probe force, or None if exploration should stop
        (no safe informative force left)."""
        safe = self.safe_forces(x, v, target, n_steps_remaining)
        if len(safe) == 0:
            return None
        if len(safe) > max_candidates:
            safe = safe[np.linspace(0, len(safe) - 1, max_candidates).astype(int)]
        if strategy == 'random':
            return float(self.rng.choice(safe))
        ig = self.expected_information_gain(x, v, safe)
        if ig.max() < min_gain:
            return None
        return float(safe[np.argmax(ig)])

    # -------------------------------------------------------------- inspection
    def visualize_beliefs(self, true_friction=None, true_mass=None):
        """Joint posterior plus marginals."""
        w = self.weights.reshape(len(self.friction_values), len(self.mass_values))
        b = self.get_beliefs()
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        axes[0].imshow(w.T, origin='lower', aspect='auto', cmap='viridis',
                       extent=[self.friction_values[0], self.friction_values[-1],
                               self.mass_values[0], self.mass_values[-1]])
        if true_friction is not None and true_mass is not None:
            axes[0].plot(true_friction, true_mass, 'r+', ms=14, mew=2, label='truth')
            axes[0].legend()
        axes[0].set_xlabel('Friction μ'); axes[0].set_ylabel('Mass (kg)')
        axes[0].set_title('Joint posterior P(μ, m | history)')

        axes[1].plot(self.friction_values, w.sum(axis=1))
        axes[1].axvline(b['friction_mean'], color='red', ls='--',
                        label=f"mean {b['friction_mean']:.3f}")
        axes[1].set_xlabel('Friction μ'); axes[1].set_title('Marginal: friction')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.mass_values, w.sum(axis=0))
        axes[2].axvline(b['mass_mean'], color='red', ls='--',
                        label=f"mean {b['mass_mean']:.2f}kg")
        axes[2].set_xlabel('Mass (kg)'); axes[2].set_title('Marginal: mass')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def _entropy(p):
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


class CausalDiscoveryPolicy:
    """Gym-style wrapper: infer confounders online, act on the posterior.

    exploration:
      'none'   - passive: always exploit; information comes only as a
                 by-product of goal-directed pushes.
      'active' - Stage 3: safe probes chosen by expected information gain
                 until the posterior is sharp (or no informative safe probe
                 remains, or max_explore_steps is hit), then exploit.
      'random' - control for 'active': same safety rule and stopping rule,
                 but the probe force is drawn uniformly from the safe set.
    """

    # Observation layout from RealisticCausalPushEnv:
    # [x, y, vx, vy, target, history(5*H), forces(H)] -- most recent force last
    X, VX, TARGET = 0, 2, 4

    def __init__(self, exploration='none', max_explore_steps=8,
                 friction_tol=0.02, mass_tol=0.05,
                 max_episode_steps=20, rng=None, **agent_kwargs):
        assert exploration in ('none', 'active', 'random')
        self.agent = BayesianCausalAgent(rng=rng, **agent_kwargs)
        self.exploration = exploration
        self.max_explore_steps = max_explore_steps
        self.friction_tol = friction_tol
        self.mass_tol = mass_tol
        self.max_episode_steps = max_episode_steps
        self.reset()

    def reset(self):
        self.agent.reset_beliefs()
        self.episode_step = 0
        self.last_x = None
        self.last_v = None
        self.exploring = self.exploration != 'none'
        self.n_explore_steps = 0

    def _still_uncertain(self):
        b = self.agent.get_beliefs()
        return b['friction_std'] > self.friction_tol or b['mass_std'] > self.mass_tol

    def predict(self, observation, deterministic=True):
        x = float(observation[self.X])
        v = float(observation[self.VX])
        target = float(observation[self.TARGET])
        steps_left = self.max_episode_steps - self.episode_step

        # Learn from the step we just took (force as actually applied by env)
        if self.episode_step > 0:
            last_force = float(observation[-1])
            self.agent.update_beliefs(self.last_x, self.last_v, last_force, x, v)

        force = None
        if self.exploring:
            if (self.n_explore_steps < self.max_explore_steps
                    and self._still_uncertain()):
                force = self.agent.choose_exploratory_force(
                    x, v, target, steps_left, strategy=self.exploration)
            if force is None:
                self.exploring = False  # switch to exploitation for good
            else:
                self.n_explore_steps += 1
        if force is None:
            force = self.agent.choose_force(x, v, target, steps_left)

        self.last_x, self.last_v = x, v
        self.episode_step += 1
        return np.array([force_to_action(force)], dtype=np.float32), None
