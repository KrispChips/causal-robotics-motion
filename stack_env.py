"""
Task 2 (PDF Phase 3): stack block B on block A with a Franka Panda.

The hidden confounders are block B's friction coefficient and mass, drawn
from the same preset distributions as the push task. Before stacking, block B
is *pushed* (RealisticCausalPushEnv with the same hidden properties and the
same 6cm size); that push trajectory is the stacking agent's observation.
This is the PDF's transfer test: does knowledge gained by pushing help with
stacking?

Why friction and mass matter here (and only here): a parallel-jaw grasp
holds while
        m * |a + g|  <=  2 * mu_pad * F_finger
where a is the hand's acceleration. A heavy, slippery block must therefore be
carried slowly or it slides out of the fingers. The agent picks one number,
the carry speed, trading task time against the risk of dropping the block.

Physical parameters (real hardware where available):
  * Franka Panda arm, pybullet_data URDF; Cartesian acceleration limit
    13 m/s^2 (Franka datasheet)
  * Franka Hand: 70 N continuous grasp force -> 35 N per finger; max opening
    8 cm, so blocks are 6cm cubes (0.5-2.0 kg = aluminium..brass density)
  * Rubber finger pads: lateralFriction 2.0. Bullet multiplies the two
    coefficients, so pad-block friction = 2 * mu_block.

Observation: the push log, 20 steps of [x / 0.1m, v (m/s), F / F_max]
             after each push step (flattened, 60 floats).
Action:      carry speed in [-1, 1] -> peak acceleration 15%..100% of 13 m/s^2.
Reward:      10 - task_time (s) if B stands on A after a 2 s stability
             check, else -5. One decision per episode.
Info:        true confounders, task time, whether the block was dropped
             ('slipped': lost contact with both fingers for 50 ms) and the
             max in-grip slip.
"""

import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_data
from gymnasium import spaces

from realistic_causal_env import RealisticCausalPushEnv, PRESETS

G = 9.8
DT = 1.0 / 240.0

BLOCK_SIZE = 0.06
HALF = BLOCK_SIZE / 2
A_PEAK_MAX = 13.0          # m/s^2, Panda Cartesian acceleration limit
SPEED_MIN_FRAC = 0.15
FINGER_FORCE = 35.0        # N per finger (70 N continuous grasp force)
FINGER_MU = 2.0            # rubber pads; contact mu = FINGER_MU * mu_block
GRASP_SETTLE = 0.4         # s to close the gripper
RELEASE_TIME = 0.3         # s to open the gripper
STABILITY_TIME = 2.0       # s the stack must stand (PDF)
CARRY_Z = 0.20             # m, carry height of the block centre

B_START = np.array([0.50, -0.15, HALF])
A_POS = np.array([0.50, 0.20, HALF])
A_MASS = 1.0

EE_LINK, FINGERS, ARM = 11, (9, 10), tuple(range(7))
HOME_Q = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785]
DOWN = p.getQuaternionFromEuler([np.pi, 0, 0])

PUSH_STEPS = 20


def action_to_peak_accel(action):
    a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
    frac = SPEED_MIN_FRAC + (a + 1.0) * 0.5 * (1.0 - SPEED_MIN_FRAC)
    return frac * A_PEAK_MAX


def peak_accel_to_action(a_peak):
    frac = a_peak / A_PEAK_MAX
    return np.array([2.0 * (frac - SPEED_MIN_FRAC) / (1.0 - SPEED_MIN_FRAC) - 1.0],
                    dtype=np.float32)


def min_jerk_duration(distance, a_peak):
    """Duration of a minimum-jerk move whose peak acceleration is a_peak
    (peak accel of min-jerk = 10/sqrt(3) * D / T^2)."""
    return max(0.15, np.sqrt(10.0 / np.sqrt(3.0) * distance / a_peak))


def min_jerk(p0, p1, tau):
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return p0 + (p1 - p0) * s


def plan(a_peak):
    """The robot's motion plan: list of (name, start, end, duration, fingers
    closed). Shared with agents -- a robot knows its own plan."""
    above_b = B_START + [0, 0, 0.15]
    lift = np.array([B_START[0], B_START[1], CARRY_Z])
    above_a = np.array([A_POS[0], A_POS[1], CARRY_Z])
    place = A_POS + [0, 0, BLOCK_SIZE + 0.003]
    segs = []
    def move(name, a, b, closed):
        segs.append((name, np.asarray(a, float), np.asarray(b, float),
                     min_jerk_duration(np.linalg.norm(np.subtract(b, a)), a_peak), closed))
    move('descend', above_b, B_START, False)
    segs.append(('grasp', B_START, B_START, GRASP_SETTLE, True))
    move('lift', B_START, lift, True)
    move('carry', lift, above_a, True)
    move('lower', above_a, place, True)
    segs.append(('release', place, place, RELEASE_TIME, False))
    move('retract', place, place + [0, 0, 0.12], False)
    return segs


def task_time(a_peak):
    return sum(seg[3] for seg in plan(a_peak))


def peak_grasp_load(a_peak):
    """Largest |a + g| the held block experiences under the plan (per unit
    mass). Vertical moves peak at g + a_peak; horizontal at sqrt(g^2+a^2)."""
    return G + a_peak


def holds(mu_block, mass, a_peak, finger_force=FINGER_FORCE, finger_mu=FINGER_MU):
    """Coulomb grasp condition (vectorised): no slip at all."""
    return 2.0 * finger_mu * mu_block * finger_force >= mass * peak_grasp_load(a_peak)


def predicted_slip(mu_block, mass, a_peak, finger_force=FINGER_FORCE,
                   finger_mu=FINGER_MU, n_points=400):
    """Coulomb model of the block sliding down through the fingers.

    While held, the block needs vertical force m*(g + a_z) from friction,
    which can supply at most 2*finger_mu*mu*F. Beyond that it slides
    (relative acceleration (g + a_z) - cap/m, relative speed >= 0), the same
    one-dimensional Coulomb law as the push model. Integrated over the
    robot's own planned vertical moves with the block in hand (the
    horizontal carry runs along the fingers' closing axis, so it loads the
    pads in compression, not shear). Vectorised over mu_block / mass.
    Returns total slip distance (m). Used to validate the simulator's grasp
    physics; agents do not use it (they would need to know how much slip
    the env tolerates, which is the grader's business)."""
    mu_block, mass = np.broadcast_arrays(np.asarray(mu_block, float),
                                         np.asarray(mass, float))
    cap = 2.0 * finger_mu * mu_block * finger_force / mass      # m/s^2
    slip = np.zeros(mu_block.shape)
    v_rel = np.zeros(mu_block.shape)
    for name, start, end, dur, closed in plan(a_peak):
        if name not in ('lift', 'carry', 'lower'):
            continue
        dz = end[2] - start[2]
        dt = dur / n_points
        for k in range(n_points):
            tau = (k + 0.5) / n_points
            a_z = dz / dur ** 2 * (60 * tau - 180 * tau ** 2 + 120 * tau ** 3)
            load = G + a_z
            v_rel = np.maximum(v_rel + (load - cap) * dt, 0.0)
            slip = slip + v_rel * dt
        # the hand stops at the segment end; any sliding is arrested by then
        v_rel = np.where(cap > G, 0.0, v_rel)
    return slip



# ------------------------------------------------------------------ push log
def make_default_pusher(force_range):
    from bayesian_causal_agent import CausalDiscoveryPolicy
    return CausalDiscoveryPolicy(exploration='none', rng=np.random.default_rng(0),
                                 force_range=force_range)


def record_push_log(confounders, preset, seed, pusher=None):
    """Push the 6cm block with the given hidden properties; return the
    (PUSH_STEPS, 3) log of [x, v, F] after each step (raw units)."""
    env = RealisticCausalPushEnv(preset=preset, block_size=BLOCK_SIZE)
    if pusher is None:
        pusher = make_default_pusher((env.force_min, env.force_max))
    obs, _ = env.reset(seed=seed, options={'confounders': confounders})
    pusher.reset()
    log = np.zeros((PUSH_STEPS, 3))
    for t in range(PUSH_STEPS):
        action, _ = pusher.predict(obs)
        obs, _, term, trunc, info = env.step(action)
        log[t] = (obs[0], obs[2], info['applied_force'])
        if term or trunc:
            break
    env.close()
    return log


def encode_push_log(log, force_max):
    scale = np.array([1 / 0.1, 1.0, 1 / force_max])
    return (log * scale).astype(np.float32).ravel()


def decode_push_log(obs, force_max):
    scale = np.array([0.1, 1.0, force_max])
    return np.asarray(obs, dtype=np.float64).reshape(PUSH_STEPS, 3) * scale


# ---------------------------------------------------------------- the env
class PandaStackEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, preset='pdf', distribution_mode='train', render_mode=None,
                 push_bank=None):
        """push_bank: optional dict with 'confounders' (N,3) and 'logs'
        (N,PUSH_STEPS,3) of pre-recorded push phases to sample from
        (speeds up RL training; identical content to recording on the fly)."""
        super().__init__()
        assert distribution_mode in PRESETS[preset]['modes']
        self.preset = preset
        self.distribution_mode = distribution_mode
        self.force_max = PRESETS[preset]['force_max']
        self.render_mode = render_mode
        self.push_bank = push_bank
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(PUSH_STEPS * 3,),
                                            dtype=np.float32)
        self.client = p.connect(p.GUI if render_mode == 'human' else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

    # -------------------------------------------------------------- helpers
    def _sample_confounders(self):
        spec = PRESETS[self.preset]['modes'][self.distribution_mode]
        vals = []
        for key in ('mass', 'friction', 'roughness'):
            lo, hi = spec[key]
            vals.append(lo if lo == hi else float(self.np_random.uniform(lo, hi)))
        return tuple(vals)

    def _build_scene(self):
        c = self.client
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -G, physicsClientId=c)
        p.setTimeStep(DT, physicsClientId=c)
        plane = p.loadURDF("plane.urdf", physicsClientId=c)
        p.changeDynamics(plane, -1, lateralFriction=1.0, physicsClientId=c)
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True,
                                physicsClientId=c)
        for f in FINGERS:
            p.changeDynamics(self.robot, f, lateralFriction=FINGER_MU, physicsClientId=c)
        for j, q in zip(ARM, HOME_Q):
            p.resetJointState(self.robot, j, q, physicsClientId=c)
        for f in FINGERS:
            p.resetJointState(self.robot, f, 0.04, physicsClientId=c)
        # The real Franka Hand's fingers are mechanically coupled; the URDF
        # models them as independent joints, which lets a held block drift
        # ~1cm along the closing axis with no restoring force. Couple them
        # (as in PyBullet's own Panda example).
        gear = p.createConstraint(self.robot, FINGERS[0], self.robot, FINGERS[1],
                                  jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0],
                                  parentFramePosition=[0, 0, 0],
                                  childFramePosition=[0, 0, 0], physicsClientId=c)
        p.changeConstraint(gear, gearRatio=-1, erp=0.1, maxForce=50, physicsClientId=c)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[HALF] * 3, physicsClientId=c)
        vis_a = p.createVisualShape(p.GEOM_BOX, halfExtents=[HALF] * 3,
                                    rgbaColor=[0.6, 0.6, 0.6, 1], physicsClientId=c)
        vis_b = p.createVisualShape(p.GEOM_BOX, halfExtents=[HALF] * 3,
                                    rgbaColor=[0.2, 0.6, 0.8, 1], physicsClientId=c)
        self.block_a = p.createMultiBody(A_MASS, col, vis_a, A_POS, physicsClientId=c)
        p.changeDynamics(self.block_a, -1, lateralFriction=1.0, physicsClientId=c)
        self.block_b = p.createMultiBody(self.mass, col, vis_b, B_START, physicsClientId=c)
        p.changeDynamics(self.block_b, -1, lateralFriction=self.base_friction,
                         physicsClientId=c)
        # Start the hand above B so the first move is short.
        q = p.calculateInverseKinematics(self.robot, EE_LINK, B_START + [0, 0, 0.15], DOWN,
                                         maxNumIterations=100, residualThreshold=1e-5,
                                         physicsClientId=c)
        for j in ARM:
            p.resetJointState(self.robot, j, q[j], physicsClientId=c)

    def _drive(self, pos, closed):
        c = self.client
        q = p.calculateInverseKinematics(self.robot, EE_LINK, pos, DOWN, maxNumIterations=50,
                                         residualThreshold=1e-5, physicsClientId=c)
        for j in ARM:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, q[j],
                                    force=87.0 if j < 4 else 12.0, positionGain=0.3,
                                    velocityGain=1.0, physicsClientId=c)
        for f in FINGERS:
            p.setJointMotorControl2(self.robot, f, p.POSITION_CONTROL,
                                    0.0 if closed else 0.04, force=FINGER_FORCE,
                                    physicsClientId=c)
        p.stepSimulation(physicsClientId=c)
        if self.render_mode == 'human':
            import time
            time.sleep(DT)

    def _b_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.block_b, physicsClientId=self.client)
        vel, _ = p.getBaseVelocity(self.block_b, physicsClientId=self.client)
        return np.array(pos), orn, np.array(vel)

    # ------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.push_bank is not None:
            i = int(self.np_random.integers(len(self.push_bank['logs'])))
            self.mass, self.base_friction, self.roughness = \
                (float(v) for v in self.push_bank['confounders'][i])
            self.push_log = np.asarray(self.push_bank['logs'][i])
        else:
            if options and 'confounders' in options:
                self.mass, self.base_friction, self.roughness = options['confounders']
            else:
                self.mass, self.base_friction, self.roughness = self._sample_confounders()
            if options and 'push_log' in options:
                # pre-recorded push of this same block (paired evaluation)
                self.push_log = np.asarray(options['push_log'], dtype=float)
            else:
                push_seed = int(self.np_random.integers(2 ** 31))
                self.push_log = record_push_log(
                    (self.mass, self.base_friction, self.roughness), self.preset, push_seed)
        self._build_scene()
        return encode_push_log(self.push_log, self.force_max), {}

    def _grip_offset(self):
        ee = np.array(p.getLinkState(self.robot, EE_LINK, physicsClientId=self.client)[4])
        return self._b_state()[0] - ee

    def step(self, action):
        a_peak = action_to_peak_accel(action)
        slipped = False
        max_slip = 0.0
        offset0 = None
        for name, start, end, dur, closed in plan(a_peak):
            n = max(1, int(round(dur / DT)))
            carrying = name in ('lift', 'carry', 'lower')
            if carrying and offset0 is None:
                offset0 = self._grip_offset()   # where the block sits once gripped
            no_contact = 0
            for k in range(1, n + 1):
                self._drive(min_jerk(start, end, k / n), closed)
                if carrying:
                    if k % 4 == 0:
                        max_slip = max(max_slip,
                                       float(np.linalg.norm(self._grip_offset() - offset0)))
                    # Dropped = the block has physically left both fingers
                    # (no contact for 50 ms), not an arbitrary slip distance.
                    touching = any(p.getContactPoints(self.block_b, self.robot, -1, f,
                                                      physicsClientId=self.client)
                                   for f in FINGERS)
                    no_contact = 0 if touching else no_contact + 1
                    if no_contact >= 12:
                        slipped = True
                        break
            if slipped:
                break
        if not slipped:
            for _ in range(int(STABILITY_TIME / DT)):
                p.stepSimulation(physicsClientId=self.client)
        pos, orn, vel = self._b_state()
        top_of_a = A_POS + [0, 0, BLOCK_SIZE]
        tilt = np.degrees(np.arccos(np.clip(
            np.dot(p.getMatrixFromQuaternion(orn)[6:9], [0, 0, 1]), -1, 1)))
        success = bool(not slipped
                       and np.linalg.norm(pos[:2] - top_of_a[:2]) < 0.015
                       and abs(pos[2] - top_of_a[2]) < 0.01
                       and tilt < 10.0 and np.linalg.norm(vel) < 0.02)
        t_task = task_time(a_peak)
        reward = 10.0 - t_task if success else -5.0
        info = {'is_success': success, 'slipped': slipped, 'max_slip': max_slip,
                'task_time': t_task,
                'peak_accel': a_peak, 'true_mass': self.mass,
                'true_base_friction': self.base_friction, 'true_roughness': self.roughness,
                'final_offset': float(np.linalg.norm(pos - top_of_a))}
        return encode_push_log(self.push_log, self.force_max), reward, True, False, info

    def close(self):
        try:
            p.disconnect(self.client)
        except p.error:
            pass
