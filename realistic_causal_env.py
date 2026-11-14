import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time

# Default ('narrow' preset) action -> force mapping. Kept as module constants
# for backwards compatibility; envs carry their own limits (env.force_min /
# env.force_max) and agents must read them from the env they act in.
FORCE_MIN = 0.0   # N  (0 lets the block coast/brake on friction alone)
FORCE_MAX = 8.0   # N  (high-friction test block needs > 0.35*1.4*9.8 = 4.8N)


def action_to_force(action, force_min=FORCE_MIN, force_max=FORCE_MAX):
    a = float(np.clip(action, -1.0, 1.0))
    return force_min + (a + 1.0) * 0.5 * (force_max - force_min)


def force_to_action(force, force_min=FORCE_MIN, force_max=FORCE_MAX):
    a = 2.0 * (force - force_min) / (force_max - force_min) - 1.0
    return float(np.clip(a, -1.0, 1.0))


# Distribution presets. Each mode gives (lo, hi) ranges for mass (kg), base
# friction and roughness; lo == hi means a fixed value (no RNG draw, so seeds
# reproduce exactly the episodes of the original implementation).
PRESETS = {
    # Original narrow setup: small friction range, 8N actuator, +-1cm.
    'narrow': {
        'force_max': 8.0,
        'success_threshold': 0.010,
        'modes': {
            'train':            {'mass': (0.8, 1.2), 'friction': (0.22, 0.28), 'roughness': (0.2, 0.4)},
            'test_interp':      {'mass': (1.0, 1.0), 'friction': (0.25, 0.25), 'roughness': (0.30, 0.30)},
            'test_extrap_high': {'mass': (1.4, 1.4), 'friction': (0.35, 0.35), 'roughness': (0.7, 0.7)},
            'test_extrap_low':  {'mass': (0.6, 0.6), 'friction': (0.15, 0.15), 'roughness': (0.1, 0.1)},
            # high friction + LOW mass: breaks the "heavy = sticky" intuition
            'test_adversarial': {'mass': (0.7, 0.7), 'friction': (0.30, 0.30), 'roughness': (0.7, 0.7)},
        },
    },
    # Project plan (PDF Phase 3): mu in [0.2, 0.8], m in [0.5, 2.0] overall,
    # F in [0, 20N], success zone +-2cm. Roughness is not in the PDF; it keeps
    # the same role (per-step friction noise) as in the narrow preset.
    'pdf': {
        'force_max': 20.0,
        'success_threshold': 0.020,
        'modes': {
            'train':            {'mass': (0.8, 1.5), 'friction': (0.30, 0.60), 'roughness': (0.2, 0.4)},
            'test_interp':      {'mass': (1.2, 1.2), 'friction': (0.45, 0.45), 'roughness': (0.30, 0.30)},
            # "very sticky, very light"
            'test_extrap_high': {'mass': (0.6, 0.6), 'friction': (0.75, 0.75), 'roughness': (0.7, 0.7)},
            # "very slippery, very heavy"
            'test_extrap_low':  {'mass': (1.9, 1.9), 'friction': (0.25, 0.25), 'roughness': (0.1, 0.1)},
            # whole physical range, for success vs. distance-from-training plots
            'test_full_range':  {'mass': (0.5, 2.0), 'friction': (0.20, 0.80), 'roughness': (0.0, 1.0)},
        },
    },
}


# Domain-randomized training distribution for the coverage baselines: the
# bounding box of every train AND test range of a preset. Training on it
# gives a learner privileged coverage of the test conditions, so it is an
# upper bound on what more data alone can do -- not a fair default.
TRAIN_DR = 'train_dr'


def mode_spec(preset, mode):
    """(lo, hi) ranges of mass/friction/roughness for a mode of a preset."""
    modes = PRESETS[preset]['modes']
    if mode == TRAIN_DR:
        return {k: (min(s[k][0] for s in modes.values()),
                    max(s[k][1] for s in modes.values()))
                for k in ('mass', 'friction', 'roughness')}
    return modes[mode]


class RealisticCausalPushEnv(gym.Env):
    """
    Realistic Causal Robotics Environment
    
    Task: Push a 10cm cube to a target 10cm away
    
    Hidden Confounders (Realistic):
    1. Mass (m): Object density - affects inertia and momentum
    2. Base Friction (μ_base): Surface material - affects sliding resistance  
    3. Surface Roughness (r): Microscopic texture - adds stochasticity
    
    Key Design Principles:
    - Fair to both RL and Causal methods
    - Physically realistic, not artificially hard
    - Observable through dynamics
    - Partial observability (agent sees history, not true properties)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, distribution_mode='train', history_length=3,
                 preset='narrow', block_size=0.10):
        super().__init__()
        assert preset in PRESETS, f"preset must be one of {list(PRESETS)}"
        self.preset = preset
        self._preset = PRESETS[preset]
        assert distribution_mode in self._preset['modes'] or distribution_mode == TRAIN_DR,             f"{preset} preset modes: {list(self._preset['modes'])}"
        self.render_mode = render_mode
        self.distribution_mode = distribution_mode
        self.history_length = history_length
        self.force_min = 0.0
        self.force_max = self._preset['force_max']
        # Cube edge (m). Sliding under Coulomb friction does not depend on
        # size, so a smaller block (e.g. the 6cm stacking block) can be probed
        # with the same task.
        self.block_half = block_size / 2.0
        
        # ACTION SPACE: Normalized force [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # OBSERVATION SPACE: [current_state, history, force_history]
        # Current: [x, y, vx, vy, target_dist] = 5
        # History: 5 * history_length = 15
        # Forces: history_length = 3
        # Total: 5 + 15 + 3 = 23 dimensions
        obs_dim = 5 + (5 * history_length) + history_length
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # PHYSICS CONFIGURATION
        self.target_dist = 0.1  # 10cm target
        self.success_threshold = self._preset['success_threshold']
        self.sim_freq = 240.0
        self.policy_freq = 10.0
        self.sim_steps_per_action = int(self.sim_freq / self.policy_freq)
        self.max_episode_steps = 20  # 2 seconds
        # Block must be nearly at rest in the target zone to count as placed
        self.success_speed = 0.05  # m/s

        # Episode tracking
        self.current_step = 0
        self.obs_history = []
        self.force_history = []
        
        # Connect to PyBullet
        if render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        
        # Object IDs
        self.block_id = None
        self.plane_id = None
        self.target_visual_id = None
        
        # Hidden confounders (will be sampled at reset)
        self.mass = 1.0
        self.base_friction = 0.25
        self.roughness = 0.5
        self.friction_std = 0.01

    def set_distribution_mode(self, mode):
        """Switch between this preset's train/test distributions"""
        valid_modes = list(self._preset['modes']) + [TRAIN_DR]
        assert mode in valid_modes, f"Mode must be one of {valid_modes}"
        self.distribution_mode = mode

    def action_to_force(self, action):
        return action_to_force(action, self.force_min, self.force_max)

    def force_to_action(self, force):
        return force_to_action(force, self.force_min, self.force_max)

    def _sample_confounders(self):
        """
        Sample the hidden physical confounders for this episode.

        Returns:
            mass: Object mass in kg
            base_friction: Mean friction coefficient
            roughness: Surface roughness (0=smooth, 1=rough)
        """
        spec = mode_spec(self.preset, self.distribution_mode)
        values = []
        for key in ('mass', 'friction', 'roughness'):
            lo, hi = spec[key]
            values.append(lo if lo == hi else self.np_random.uniform(lo, hi))
        return tuple(values)

    def _apply_stochastic_friction(self):
        """
        Apply friction with variance based on surface roughness.
        Rougher surfaces have more unpredictable friction.
        """
        # Sample actual friction from distribution
        # Roughness determines variance
        friction_std = self.roughness * 0.02
        actual_friction = self.np_random.normal(self.base_friction, friction_std)
        # Floor keeps friction physical; no artificial ceiling (the old 0.5
        # cap would silently clip the PDF's mu up to 0.8).
        actual_friction = max(actual_friction, 0.02)
        
        p.changeDynamics(
            self.block_id, -1,
            lateralFriction=actual_friction,
            physicsClientId=self.client
        )
        
        return actual_friction

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_step = 0
        
        # Reset physics
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=self.client)
        
        # Create block (10cm cube)
        col_shape = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=[self.block_half] * 3,
            physicsClientId=self.client
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.block_half] * 3,
            rgbaColor=[0.2, 0.6, 0.8, 1],
            physicsClientId=self.client
        )
        
        start_pos = [0, 0, self.block_half]
        self.block_id = p.createMultiBody(
            baseMass=1.0,  # Will be changed by confounders
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=start_pos,
            physicsClientId=self.client
        )
        
        # Create target visualization
        t_vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.001],
            rgbaColor=[0, 1, 0, 0.4],
            physicsClientId=self.client
        )
        self.target_visual_id = p.createMultiBody(
            baseVisualShapeIndex=t_vis_shape,
            basePosition=[self.target_dist, 0, 0.001],
            physicsClientId=self.client
        )
        
        # Sample hidden confounders
        # options={'confounders': (mass, friction, roughness)} pins the hidden
        # properties (used to probe a specific object, e.g. before stacking it).
        if options and 'confounders' in options:
            self.mass, self.base_friction, self.roughness = options['confounders']
        else:
            self.mass, self.base_friction, self.roughness = self._sample_confounders()
        
        # Apply mass
        p.changeDynamics(self.block_id, -1, mass=self.mass, physicsClientId=self.client)
        
        # Apply initial friction (will vary each step due to roughness)
        self.current_friction = self._apply_stochastic_friction()
        
        # Initialize observation history
        base_obs = self._get_base_obs()
        self.obs_history = [base_obs.copy() for _ in range(self.history_length)]
        self.force_history = [0.0] * self.history_length
        self.prev_dist = abs(base_obs[0] - self.target_dist)
        
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5,
                cameraYaw=0,
                cameraPitch=-40,
                cameraTargetPosition=[0.05, 0, 0],
                physicsClientId=self.client
            )
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Parse action
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                act = float(action)
            else:
                act = action[0]
        else:
            act = action
        
        force_mag = self.action_to_force(act)

        # Friction is resampled once per action (roughness = per-step noise)
        self.current_friction = self._apply_stochastic_friction()

        # No artificial damping: the block slows down through Coulomb friction
        # only. (The old per-substep resetBaseVelocity(v*0.98) shrank velocity
        # 0.98**24 = 0.62x per control step and fought the contact solver.)
        for _ in range(self.sim_steps_per_action):
            # In WORLD_FRAME posObj is a world point, so push at the block's
            # current centre of mass (not the world origin).
            block_pos, _ = p.getBasePositionAndOrientation(
                self.block_id, physicsClientId=self.client)
            p.applyExternalForce(
                self.block_id,
                -1,
                forceObj=[force_mag, 0, 0],
                posObj=block_pos,
                flags=p.WORLD_FRAME,
                physicsClientId=self.client
            )
            p.stepSimulation(physicsClientId=self.client)

            if self.render_mode == "human":
                time.sleep(1./self.sim_freq)

        # Update observation history
        base_obs = self._get_base_obs()
        self.obs_history.append(base_obs.copy())
        self.obs_history.pop(0)
        self.force_history.append(force_mag)
        self.force_history.pop(0)

        obs = self._get_obs()
        block_x = base_obs[0]
        dist_to_target = abs(block_x - self.target_dist)
        speed = abs(base_obs[2])
        success = bool(dist_to_target <= self.success_threshold
                       and speed <= self.success_speed)

        # Fixed-horizon task: the episode really ends (and is scored) here,
        # so this is termination, not a time-limit truncation.
        terminated = self.current_step >= self.max_episode_steps
        truncated = False

        # REWARD
        # Potential-based progress shaping: sums to 100*(d_start - d_final)
        # over the episode, so doing nothing scores 0, reaching the target
        # scores ~+10, and overshooting is penalised automatically (distance
        # grows again). The old reward made "never push" (-20) beat any
        # attempt to push, and PPO collapsed to it.
        reward = 100.0 * (self.prev_dist - dist_to_target)
        self.prev_dist = dist_to_target

        # Small effort cost (tie-breaker, can't dominate the task reward)
        reward -= 0.01 * (force_mag / self.force_max) ** 2

        # Terminal bonus for a block resting in the target zone
        if terminated and success:
            reward += 10.0

        # Info for analysis
        info = {
            "true_mass": self.mass,
            "true_base_friction": self.base_friction,
            "true_roughness": self.roughness,
            "current_friction": self.current_friction,
            "is_success": success,
            "final_distance": dist_to_target,
            "applied_force": force_mag,
        }
        
        return obs, reward, terminated, truncated, info

    def _get_base_obs(self):
        """Get basic state: [x, y, vx, vy, target_dist]"""
        pos, _ = p.getBasePositionAndOrientation(self.block_id, physicsClientId=self.client)
        lin_vel, _ = p.getBaseVelocity(self.block_id, physicsClientId=self.client)
        
        return np.array([
            pos[0],
            pos[1],
            lin_vel[0],
            lin_vel[1],
            self.target_dist
        ], dtype=np.float32)

    def _get_obs(self):
        """
        Get full observation with history.
        This allows agent to infer dynamics:
        "I applied 3N, block moved 2cm → must have high friction"
        """
        current = self._get_base_obs()
        history = np.concatenate(self.obs_history)
        forces = np.array(self.force_history, dtype=np.float32)
        
        return np.concatenate([current, history, forces])

    def close(self):
        if hasattr(self, 'client'):
            p.disconnect(self.client)