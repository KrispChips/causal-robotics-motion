"""
Physics validation for RealisticCausalPushEnv.

Checks the PyBullet simulation against closed-form Coulomb friction (the same
model the Bayesian agent uses in `simulate_step`), plus the env's plumbing:
force mapping, friction noise, seeding, observations, termination and reward.

Replaces the old env_valid.py / test_env.py, which used a stale force mapping
(3.0 + 2.5*a) and held a constant force for 10-15 steps.

Usage:
    python test_sim_physics.py           # run all checks, exit 1 on failure
    python test_sim_physics.py --plot    # also save force_distance_calibration.png
    pytest test_sim_physics.py           # works too (plain asserts)
"""

import sys

import numpy as np
import pybullet as p

from realistic_causal_env import (RealisticCausalPushEnv, action_to_force,
                                  force_to_action, FORCE_MIN, FORCE_MAX, PRESETS)
from bayesian_causal_agent import simulate_step, G


def fixed_modes(preset):
    """(mu, mass) of a preset's deterministic test distributions."""
    out = {}
    for mode, spec in PRESETS[preset]['modes'].items():
        (m_lo, m_hi), (f_lo, f_hi) = spec['mass'], spec['friction']
        if m_lo == m_hi and f_lo == f_hi:
            out[mode] = (f_lo, m_lo)
    return out


FIXED_MODES = fixed_modes("narrow")
REFERENCE_SEED5 = (1.122001, 0.268476)  # narrow/train, recorded before the preset refactor


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_env(mode='test_interp', seed=0, preset='narrow'):
    env = RealisticCausalPushEnv(render_mode=None, distribution_mode=mode, preset=preset)
    env.reset(seed=seed)
    return env


def pin(env, mu, mass):
    """Fix the hidden confounders and switch off per-step friction noise
    (roughness 0 -> step() resamples friction with std 0)."""
    env.mass, env.base_friction, env.roughness = mass, mu, 0.0
    p.changeDynamics(env.block_id, -1, mass=mass, lateralFriction=mu,
                     physicsClientId=env.client)


def state(env):
    pos, orn = p.getBasePositionAndOrientation(env.block_id, physicsClientId=env.client)
    lin, ang = p.getBaseVelocity(env.block_id, physicsClientId=env.client)
    return np.array(pos), np.array(p.getEulerFromQuaternion(orn)), np.array(lin), np.array(ang)


def run_sim(forces, mu, mass, mode='test_interp', preset='narrow'):
    """Step the env with a force sequence (N). Returns x, vx, and the max
    off-axis drift |y|, |z - 0.05| and |roll/pitch/yaw| seen."""
    env = make_env(mode, preset=preset)
    pin(env, mu, mass)
    xs, vs, off = [0.0], [0.0], np.zeros(3)
    for F in forces:
        env.step(np.array([env.force_to_action(F)]))
        pos, rpy, lin, _ = state(env)
        xs.append(pos[0]); vs.append(lin[0])
        off = np.maximum(off, [abs(pos[1]), abs(pos[2] - 0.05), np.abs(rpy).max()])
    env.close()
    return np.array(xs), np.array(vs), off


def run_analytic(forces, mu, mass):
    x, v = np.array(0.0), np.array(0.0)
    xs, vs = [0.0], [0.0]
    for F in forces:
        x, v = simulate_step(x, v, F, mu, mass)
        xs.append(float(x)); vs.append(float(v))
    return np.array(xs), np.array(vs)


def coast_decel(mu, mass, v0, n_substeps=12, plane_mu=None):
    """Launch the block at v0 with no push; return measured deceleration."""
    env = make_env()
    pin(env, mu, mass)
    if plane_mu is not None:
        p.changeDynamics(env.plane_id, -1, lateralFriction=plane_mu, physicsClientId=env.client)
    p.resetBaseVelocity(env.block_id, [v0, 0, 0], [0, 0, 0], physicsClientId=env.client)
    for _ in range(n_substeps):
        p.stepSimulation(physicsClientId=env.client)
    v1 = state(env)[2][0]
    env.close()
    return (v0 - v1) / (n_substeps / 240.0)


def planned_push(mu, mass, target=0.10, n_steps=20, max_push_steps=3, force_max=FORCE_MAX):
    """Shortest constant push (k control steps of force F, then coast) that
    stops the block on the target under the analytic Coulomb model.
    Returns (k, F); k is the fewest steps for which force_max can reach."""
    def final_x(F, k):
        return run_analytic([F] * k + [0.0] * (n_steps - k), mu, mass)[0][-1]
    for k in range(1, max_push_steps + 1):
        if final_x(force_max, k) >= target:
            lo, hi = mu * mass * G, force_max
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if final_x(mid, k) < target else (lo, mid)
            return k, 0.5 * (lo + hi)
    raise AssertionError(f"mu={mu} m={mass}: target unreachable in {max_push_steps} push steps")


# --------------------------------------------------------------------------
# force mapping / force range
# --------------------------------------------------------------------------
def test_force_mapping():
    assert action_to_force(-1) == FORCE_MIN and action_to_force(1) == FORCE_MAX
    assert action_to_force(5.0) == FORCE_MAX and action_to_force(-5.0) == FORCE_MIN  # clipped
    for F in np.linspace(FORCE_MIN, FORCE_MAX, 17):
        assert abs(action_to_force(force_to_action(F)) - F) < 1e-9


def test_force_range_covers_every_distribution():
    """Max force must beat static friction (with >=25% headroom) for the
    stickiest, heaviest corner of every distribution in every preset."""
    for preset, cfg in PRESETS.items():
        for mode, spec in cfg['modes'].items():
            need = spec['friction'][1] * spec['mass'][1] * G
            assert cfg['force_max'] >= 1.25 * need, \
                f"{preset}/{mode}: needs {need:.2f}N, max {cfg['force_max']}N"


# --------------------------------------------------------------------------
# Coulomb physics
# --------------------------------------------------------------------------
def test_block_at_rest_stays_put():
    xs, vs, off = run_sim([0.0] * 20, 0.25, 1.0)
    assert np.abs(xs).max() < 1e-4 and np.abs(vs).max() < 1e-3
    assert off[0] < 1e-3 and off[1] < 1e-3 and off[2] < 1e-3


def test_static_friction_threshold():
    """Below mu*m*g nothing moves; above it the block slides as predicted."""
    for mu, mass in FIXED_MODES.values():
        thr = mu * mass * G
        x_below = run_sim([0.9 * thr] * 5, mu, mass)[0][-1]
        assert abs(x_below) < 5e-4, f"mu={mu} m={mass}: crept {x_below*100:.3f}cm below threshold"
        x_above = run_sim([1.1 * thr] * 5, mu, mass)[0][-1]
        x_pred = run_analytic([1.1 * thr] * 5, mu, mass)[0][-1]
        assert x_above > 0.01 and abs(x_above - x_pred) < 5e-4, \
            f"mu={mu} m={mass}: above threshold sim {x_above*100:.2f}cm vs {x_pred*100:.2f}cm"


def test_trajectories_match_coulomb_model():
    """Push-then-coast at task speeds: sim within max(0.5mm, 2% of distance
    travelled) and 0.01 m/s of analytic. The relative part covers Bullet's
    ~1-2% speed-dependent friction at the faster PDF-preset speeds."""
    cases = [('narrow', 0.25, 1.0, [4.0]), ('narrow', 0.25, 1.0, [3.0, 3.0]),
             ('narrow', 0.15, 0.6, [2.0]), ('narrow', 0.35, 1.4, [6.0, 6.0]),
             ('narrow', 0.30, 0.7, [5.0]), ('narrow', 0.25, 1.0, [3.5, 0.0, 3.5]),
             # PDF ranges: sticky/light, slippery/heavy, sticky/heavy corner
             ('pdf', 0.75, 0.6, [11.0]), ('pdf', 0.25, 1.9, [7.0, 7.0]),
             ('pdf', 0.80, 2.0, [19.0, 19.0, 19.0])]
    for preset, mu, mass, pushes in cases:
        forces = pushes + [0.0] * (20 - len(pushes))
        xs, vs, off = run_sim(forces, mu, mass, preset=preset)
        ax, av = run_analytic(forces, mu, mass)
        tag = f"mu={mu} m={mass} F={pushes}"
        x_tol = max(5e-4, 0.02 * ax[-1])
        assert np.abs(xs - ax).max() < x_tol, f"{tag}: max |dx| {np.abs(xs-ax).max()*100:.3f}cm"
        assert np.abs(vs - av).max() < 1e-2, f"{tag}: max |dv| {np.abs(vs-av).max():.4f}"
        # A push through the centre of mass with mu < 1 must not tip, lift or veer.
        assert off[0] < 1e-3 and off[1] < 1e-3 and off[2] < 1e-2, f"{tag}: off-axis {off}"


def test_coasting_deceleration_is_mu_g_and_mass_independent():
    for mu in (0.15, 0.25, 0.35):
        decels = [coast_decel(mu, m, v0=0.5) for m in (0.6, 1.0, 1.4)]
        assert np.ptp(decels) < 1e-3, f"mu={mu}: decel depends on mass {decels}"
        assert abs(decels[0] / (mu * G) - 1) < 0.02, f"mu={mu}: decel {decels[0]:.3f} vs {mu*G:.3f}"


def test_friction_speed_dependence_small_in_task_regime():
    """Bullet's effective mu creeps up with speed (+~9% at 2 m/s). Coasting
    onto a 10cm target needs v = sqrt(2*mu*g*0.1) (0.54-0.83 m/s here); at
    that speed the mu error must stay under 3%, i.e. <3mm of a 10cm push,
    well inside the 1cm success band. Beyond that (big overshoots) the
    agent's Coulomb model over-predicts distance."""
    for mode, (mu, mass) in {**FIXED_MODES, **fixed_modes('pdf')}.items():
        v_needed = np.sqrt(2 * mu * G * 0.10)
        eff = coast_decel(mu, mass, v0=v_needed) / G
        assert abs(eff / mu - 1) < 0.03, f"{mode}: v={v_needed:.2f} effective mu {eff:.4f} vs {mu}"


def test_plane_friction_is_unity():
    """PyBullet multiplies the two bodies' friction, so the block's mu is the
    contact mu only while the plane stays at 1.0."""
    env = make_env()
    assert p.getDynamicsInfo(env.plane_id, -1, physicsClientId=env.client)[1] == 1.0
    env.close()
    eff_half = coast_decel(0.25, 1.0, v0=0.5, plane_mu=0.5) / G
    assert abs(eff_half - 0.125) < 0.005  # confirms the product rule


def test_friction_change_takes_effect_mid_slide():
    env = make_env()
    pin(env, 0.15, 1.0)
    p.resetBaseVelocity(env.block_id, [0.5, 0, 0], [0, 0, 0], physicsClientId=env.client)
    for mu in (0.15, 0.45):
        p.changeDynamics(env.block_id, -1, lateralFriction=mu, physicsClientId=env.client)
        v0 = state(env)[2][0]
        for _ in range(12):
            p.stepSimulation(physicsClientId=env.client)
        eff = (v0 - state(env)[2][0]) / 0.05 / G
        assert abs(eff / mu - 1) < 0.03, f"set mu={mu}, measured {eff:.3f}"
    env.close()


# --------------------------------------------------------------------------
# hidden confounders / noise / seeding
# --------------------------------------------------------------------------
def test_confounders_match_distribution_spec():
    for preset, cfg in PRESETS.items():
        for mode, spec in cfg['modes'].items():
            env = make_env(mode, preset=preset)
            assert env.force_max == cfg['force_max']
            assert env.success_threshold == cfg['success_threshold']
            for s in range(20 if spec['mass'][0] < spec['mass'][1] else 1):
                env.reset(seed=s)
                for val, key in ((env.mass, 'mass'), (env.base_friction, 'friction'),
                                 (env.roughness, 'roughness')):
                    lo, hi = spec[key]
                    assert lo <= val <= hi, f"{preset}/{mode}: {key}={val} not in [{lo},{hi}]"
                assert p.getDynamicsInfo(env.block_id, -1,
                                         physicsClientId=env.client)[0] == env.mass
            env.close()


def test_narrow_preset_unchanged():
    """The preset refactor must reproduce the original narrow episodes."""
    # Values recorded from the pre-preset implementation (seed 5, train).
    env = RealisticCausalPushEnv(distribution_mode='train')  # default preset
    env.reset(seed=5)
    assert (round(env.mass, 6), round(env.base_friction, 6)) == REFERENCE_SEED5
    assert env.force_max == 8.0 and env.success_threshold == 0.010
    env.close()


def test_high_friction_not_clipped():
    """Regression: per-step friction used to be clipped to [0.1, 0.5], which
    silently turned the PDF's mu=0.75 block into mu=0.5."""
    env = make_env('test_extrap_high', preset='pdf')
    fr = [env._apply_stochastic_friction() for _ in range(2000)]
    assert abs(np.mean(fr) - 0.75) < 2e-3
    env.close()
    decel = coast_decel(0.75, 0.6, v0=1.0) / G
    assert abs(decel / 0.75 - 1) < 0.03, f"mu=0.75 block decelerates like mu={decel:.3f}"


def test_friction_resampled_once_per_action():
    env = make_env('test_extrap_high')
    calls = []
    orig = env._apply_stochastic_friction
    env._apply_stochastic_friction = lambda: calls.append(1) or orig()
    env.step(np.array([0.0]))
    assert len(calls) == 1
    env.close()


def test_roughness_noise_scale():
    env = make_env('test_extrap_high')  # mu 0.35, roughness 0.7 -> std 0.014
    fr = np.array([env._apply_stochastic_friction() for _ in range(5000)])
    assert abs(fr.mean() - 0.35) < 1e-3
    assert abs(fr.std() / (0.02 * env.roughness) - 1) < 0.05
    env.close()


def test_seeding_is_deterministic():
    def rollout(seed):
        env = make_env('train', seed)
        xs = [env.step(np.array([a]))[0][0] for a in [-0.2] * 2 + [-1.0] * 10]
        out = (np.array(xs), env.mass, env.base_friction)
        env.close()
        return out
    a, b, c = rollout(5), rollout(5), rollout(6)
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:]
    assert a[1:] != c[1:]


# --------------------------------------------------------------------------
# observation / termination / reward
# --------------------------------------------------------------------------
def test_observation_layout():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (23,) and np.all(np.isfinite(obs))
    for F in (4.0, 0.0, 2.0):
        obs, *_ = env.step(np.array([force_to_action(F)]))
    assert np.allclose(obs[15:20], obs[0:5])        # newest history frame == current
    assert np.allclose(obs[-3:], [4.0, 0.0, 2.0])   # force history in newtons, oldest first
    assert obs[4] == env.target_dist
    env.close()


def test_episode_terminates_after_max_steps():
    env = make_env()
    for k in range(1, env.max_episode_steps + 1):
        _, _, term, trunc, _ = env.step(np.array([-1.0]))
        assert not trunc
        assert term == (k == env.max_episode_steps)
    env.close()


def test_reward_is_telescoping_progress():
    """Return = 100*(d0 - d_final) - effort + 10*success; doing nothing = 0."""
    for pushes in ([], [4.0], [2.6, 2.6]):
        env = make_env()
        pin(env, 0.25, 1.0)
        forces = pushes + [0.0] * (env.max_episode_steps - len(pushes))
        R = 0.0
        for F in forces:
            _, r, term, _, info = env.step(np.array([force_to_action(F)]))
            R += r
        effort = sum(0.01 * (F / FORCE_MAX) ** 2 for F in forces)
        expected = 100 * (0.1 - info['final_distance']) - effort + 10.0 * info['is_success']
        assert term and abs(R - expected) < 1e-4, f"pushes={pushes}: R={R} expected {expected}"
        env.close()


# --------------------------------------------------------------------------
# task-level: reachable everywhere, and distributions actually differ
# --------------------------------------------------------------------------
def test_target_reachable_and_distributions_separated():
    """An analytically-planned constant push succeeds in every test
    distribution (with the env's real roughness noise), and the plan for one
    distribution misses badly on another, so no single fixed open-loop
    policy can solve them all. Note: even at FORCE_MAX, the nominal block
    needs >=2 push steps and the high-friction block needs 3."""
    for preset, cfg in PRESETS.items():
        tol = cfg['success_threshold']
        plans = {}
        for mode, (mu, mass) in fixed_modes(preset).items():
            k, F = planned_push(mu, mass, force_max=cfg['force_max'])
            plans[mode] = (k, F)
            wins = 0
            for seed in range(5):
                env = make_env(mode, seed, preset=preset)
                for step in range(env.max_episode_steps):
                    f = F if step < k else 0.0
                    _, _, term, _, info = env.step(np.array([env.force_to_action(f)]))
                wins += info['is_success']
                env.close()
            assert wins >= 4, f"{preset}/{mode}: planned push {k}x{F:.2f}N succeeded {wins}/5"
        pairs = [(a, b) for a in plans for b in plans if a != b and a != 'test_interp'
                 and b != 'test_interp']
        for a, b in pairs:
            k, F = plans[a]
            mu, mass = fixed_modes(preset)[b]
            x = run_analytic([F] * k + [0.0] * (20 - k), mu, mass)[0][-1]
            assert abs(x - 0.10) > tol, f"{preset}: {a}'s plan also works on {b} (x={x*100:.1f}cm)"


# --------------------------------------------------------------------------
# Panda stacking (stack_env.py)
# --------------------------------------------------------------------------
def _stack(mu, mass, a_peak):
    import stack_env as S
    env = S.PandaStackEnv(preset='pdf', distribution_mode='test_interp')
    log = np.zeros((S.PUSH_STEPS, 3))  # push phase irrelevant here; skip recording it
    env.reset(seed=0, options={'confounders': (mass, mu, 0.3), 'push_log': log})
    info = env.step(S.peak_accel_to_action(a_peak))[4]
    env.close()
    return info


def test_stack_secure_grasp_places_block_exactly():
    """With a secure grasp the block must not drift in the hand (needs the
    finger gear coupling) and must land centred on block A."""
    for mu, mass, a in ((0.45, 1.2, 13.0), (0.75, 0.6, 13.0), (0.3, 1.5, 2.0)):
        info = _stack(mu, mass, a)
        assert info['is_success'] and not info['slipped']
        assert info['max_slip'] < 0.01, f"mu={mu} m={mass}: block drifted {info['max_slip']*100:.2f}cm"
        assert info['final_offset'] < 0.005, f"placed {info['final_offset']*100:.2f}cm off"


def test_stack_grasp_physics_matches_coulomb():
    """(a) Inside the friction cone (the causal stacking agent's rule) the
    block is never dropped. (b) While the block stays in the hand, Bullet's
    in-grip slip matches the Coulomb sliding model within 0.6cm (the sim
    adds ~0.3-0.7cm of elastic give). (c) A block predicted to slide well
    past half its height is dropped."""
    import stack_env as S
    cases = [(0.20, 2.0, 4.0), (0.20, 2.0, 7.0), (0.20, 2.0, 10.0), (0.20, 1.7, 10.0),
             (0.20, 1.7, 13.0), (0.25, 2.0, 7.0), (0.25, 2.0, 10.0), (0.25, 2.0, 13.0),
             (0.25, 1.7, 13.0), (0.30, 2.0, 13.0)]
    for mu, mass, a in cases:
        info = _stack(mu, mass, a)
        pred = float(S.predicted_slip(mu, mass, a))
        tag = f"mu={mu} m={mass} a={a}: pred slip {pred*100:.1f}cm, sim {info['max_slip']*100:.1f}cm"
        if S.holds(mu, mass, a):
            assert not info['slipped'] and info['is_success'], tag + " (inside friction cone)"
        if not info['slipped']:
            assert abs(info['max_slip'] - pred) < 0.006, tag
        if pred > 0.045:
            assert info['slipped'], tag


# --------------------------------------------------------------------------
# calibration plot (optional)
# --------------------------------------------------------------------------
def plot_calibration(path='force_distance_calibration.png'):
    """Final position vs force of a 2-step (0.2 s) push, then coast, per
    distribution: sim points over the analytic Coulomb curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # fixed categorical order
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(0.09, 0.11, color='#1baf7a', alpha=0.10, lw=0)
    ax.axhline(0.10, color='#52514e', lw=1, ls='--')
    ax.text(0.15, 0.102, 'target ±1cm', color='#52514e', fontsize=9)
    for (mode, (mu, mass)), c in zip(FIXED_MODES.items(), colors):
        F_dense = np.linspace(FORCE_MIN, FORCE_MAX, 81)
        x_pred = [run_analytic([F] * 2 + [0.0] * 18, mu, mass)[0][-1] for F in F_dense]
        F_pts = np.linspace(FORCE_MIN, FORCE_MAX, 17)
        x_sim = [run_sim([F] * 2 + [0.0] * 18, mu, mass, mode)[0][-1] for F in F_pts]
        ax.plot(F_dense, x_pred, color=c, lw=2, label=f'{mode}  (μ={mu}, m={mass})')
        ax.plot(F_pts, x_sim, 'o', color=c, ms=6, mec='#fcfcfb', mew=1.5)
    ax.set_ylim(-0.005, 0.30)
    ax.set_xlabel('Push force held for 2 steps / 0.2 s (N), then coast')
    ax.set_ylabel('Final block position (m)')
    ax.set_title('Force–distance calibration: lines = Coulomb model, dots = PyBullet',
                 loc='left', fontsize=11)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


# --------------------------------------------------------------------------
def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if '--plot' in sys.argv:
        plot_calibration()
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
