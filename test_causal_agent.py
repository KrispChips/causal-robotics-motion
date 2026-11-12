"""
Quick checks of the Bayesian causal agent before full evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from realistic_causal_env import RealisticCausalPushEnv, force_to_action
from bayesian_causal_agent import BayesianCausalAgent, CausalDiscoveryPolicy, simulate_step


def test_physics_model():
    """Agent's dynamics model vs PyBullet, using the friction the env actually
    applied on each step (isolates model error from roughness noise)."""
    print("=" * 70)
    print("TEST 1: Dynamics model matches simulator")
    print("=" * 70)
    env = RealisticCausalPushEnv(distribution_mode='test_interp')
    x_err, v_err = [], []
    for seq in ([3, 3, 0, 0, 0], [6, 2, 0, 0, 0], [8, 0, 0, 0, 0], [1, 2, 5, 0, 0]):
        obs, _ = env.reset(seed=0)
        for F in seq:
            x, v = obs[0], obs[2]
            obs, _, _, _, info = env.step(np.array([force_to_action(F)]))
            px, pv = simulate_step(x, v, F, info['current_friction'], env.mass)
            x_err.append(abs(obs[0] - px)); v_err.append(abs(obs[2] - pv))
    env.close()
    print(f"  max |x error| per step: {max(x_err)*1000:.2f} mm")
    print(f"  max |v error| per step: {max(v_err)*1000:.1f} mm/s")
    assert max(x_err) < 0.002 and max(v_err) < 0.02, "model/sim mismatch"
    print("✓ Model matches simulator")


def test_belief_updating():
    """Posterior should concentrate near the truth after one episode."""
    print("\n" + "=" * 70)
    print("TEST 2: Belief updating")
    print("=" * 70)
    env = RealisticCausalPushEnv(distribution_mode='test_extrap_high')
    policy = CausalDiscoveryPolicy(rng=np.random.default_rng(0))
    obs, _ = env.reset(seed=0); policy.reset()
    prior = policy.agent.get_beliefs()
    done = False
    while not done:
        action, _ = policy.predict(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    b = policy.agent.get_beliefs()
    print(f"  True:      μ={env.base_friction:.3f}, m={env.mass:.2f}kg")
    print(f"  Prior:     μ={prior['friction_mean']:.3f}±{prior['friction_std']:.3f}, "
          f"m={prior['mass_mean']:.2f}±{prior['mass_std']:.2f}kg")
    print(f"  Posterior: μ={b['friction_mean']:.3f}±{b['friction_std']:.3f}, "
          f"m={b['mass_mean']:.2f}±{b['mass_std']:.2f}kg")
    assert abs(b['friction_mean'] - env.base_friction) < 0.05
    assert b['friction_std'] < prior['friction_std'] / 3
    fig = policy.agent.visualize_beliefs(env.base_friction, env.mass)
    plt.savefig('belief_visualization_test.png', dpi=150)
    plt.close(fig)
    env.close()
    print("✓ Posterior concentrates near truth (plot: belief_visualization_test.png)")


def test_force_selection():
    """With beliefs pinned to the truth, the planned push should land on target."""
    print("\n" + "=" * 70)
    print("TEST 3: Force selection with known parameters")
    print("=" * 70)
    agent = BayesianCausalAgent(rng=np.random.default_rng(0))
    mu, m = 0.25, 1.0
    agent.log_post[:] = -np.inf
    i = np.argmin(abs(agent.friction_values - mu)) * len(agent.mass_values) \
        + np.argmin(abs(agent.mass_values - m))
    agent.log_post[i] = 0.0
    mu, m = agent.mu_flat[i], agent.m_flat[i]
    # Closed loop in the agent's own model (one push can't cover 10cm at
    # F_max, so the planner must chain pushes and then coast)
    x, v, forces = 0.0, 0.0, []
    for t in range(20):
        F = agent.choose_force(x, v, 0.1, n_steps_remaining=20 - t)
        forces.append(F)
        x, v = simulate_step(x, v, F, mu, m)
    print(f"  μ={mu:.3f}, m={m:.2f}kg -> forces {np.round(forces[:6], 2)}..., "
          f"rest at {x*100:.2f}cm")
    assert abs(x - 0.1) < 0.01
    print("✓ Planned forces reach target")


def test_single_episode():
    print("\n" + "=" * 70)
    print("TEST 4: Complete episode")
    print("=" * 70)
    env = RealisticCausalPushEnv(distribution_mode='train')
    policy = CausalDiscoveryPolicy(rng=np.random.default_rng(1))
    obs, _ = env.reset(seed=1); policy.reset()
    print(f"  True: μ={env.base_friction:.3f}, m={env.mass:.2f}kg, r={env.roughness:.2f}")
    done, step = False, 0
    while not done:
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if step < 6:
            b = policy.agent.get_beliefs()
            print(f"  step {step+1}: F={info['applied_force']:.2f}N x={obs[0]*100:.2f}cm "
                  f"v={obs[2]:.3f}  μ̂={b['friction_mean']:.3f}±{b['friction_std']:.3f}")
        step += 1
    print(f"  Final distance: {info['final_distance']*100:.2f}cm, success={info['is_success']}")
    env.close()


if __name__ == "__main__":
    test_physics_model()
    test_belief_updating()
    test_force_selection()
    test_single_episode()
    print("\nALL TESTS PASSED ✓")
