import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time

# Action -> force mapping. Shared by the env, the causal agent and the eval
# scripts so they can never disagree about what an action means.
FORCE_MIN = 0.0   # N  (0 lets the block coast/brake on friction alone)
FORCE_MAX = 8.0   # N  (high-friction test block needs > 0.35*1.4*9.8 = 4.8N)


def action_to_force(action):
    a = float(np.clip(action, -1.0, 1.0))
    return FORCE_MIN + (a + 1.0) * 0.5 * (FORCE_MAX - FORCE_MIN)


def force_to_action(force):
    a = 2.0 * (force - FORCE_MIN) / (FORCE_MAX - FORCE_MIN) - 1.0
    return float(np.clip(a, -1.0, 1.0))


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

    def __init__(self, render_mode=None, distribution_mode='train', history_length=3):
        super().__init__()
        self.render_mode = render_mode
        self.distribution_mode = distribution_mode
        self.history_length = history_length
        
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
        self.success_threshold = 0.010  # TIGHTENED: ±1cm (was ±2cm)
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
        """Switch between train/test distributions"""
        valid_modes = ['train', 'test_interp', 'test_extrap_high', 
                      'test_extrap_low', 'test_adversarial']
        assert mode in valid_modes, f"Mode must be one of {valid_modes}"
        self.distribution_mode = mode

    def _sample_confounders(self):
        """
        Sample realistic physical confounders based on distribution mode.
        
        Returns:
            mass: Object mass in kg
            base_friction: Mean friction coefficient
            roughness: Surface roughness (0=smooth, 1=rough)
        """
        if self.distribution_mode == 'train':
            # Training: Narrow range of "typical" blocks
            mass = self.np_random.uniform(0.8, 1.2)
            base_friction = self.np_random.uniform(0.22, 0.28)
            roughness = self.np_random.uniform(0.2, 0.4)  # REDUCED variance (was 0.5)
            
        elif self.distribution_mode == 'test_interp':
            # Interpolation: Middle of training range
            mass = 1.0
            base_friction = 0.25
            roughness = 0.30
            
        elif self.distribution_mode == 'test_extrap_high':
            # High friction: Heavy, rough, sticky block
            mass = 1.4  # INCREASED from 1.3
            base_friction = 0.35  # INCREASED from 0.30
            roughness = 0.7
            
        elif self.distribution_mode == 'test_extrap_low':
            # Low friction: Light, smooth, slippery block
            mass = 0.6
            base_friction = 0.15  # DECREASED from 0.18
            roughness = 0.1
            
        elif self.distribution_mode == 'test_adversarial':
            # Adversarial: Breaks training correlations
            # In training: high friction often paired with high mass
            # Here: high friction + LOW mass (misleading)
            mass = 0.7  # Light
            base_friction = 0.30  # High friction
            roughness = 0.7
            
        else:
            # Fallback: Wide range
            mass = self.np_random.uniform(0.6, 1.4)
            base_friction = self.np_random.uniform(0.18, 0.32)
            roughness = self.np_random.uniform(0.0, 1.0)
            
        return mass, base_friction, roughness

    def _apply_stochastic_friction(self):
        """
        Apply friction with variance based on surface roughness.
        Rougher surfaces have more unpredictable friction.
        """
        # Sample actual friction from distribution
        # Roughness determines variance
        friction_std = self.roughness * 0.02
        actual_friction = self.np_random.normal(self.base_friction, friction_std)
        actual_friction = np.clip(actual_friction, 0.1, 0.5)
        
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
            halfExtents=[0.05, 0.05, 0.05],
            physicsClientId=self.client
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.05, 0.05, 0.05],
            rgbaColor=[0.2, 0.6, 0.8, 1],
            physicsClientId=self.client
        )
        
        start_pos = [0, 0, 0.05]
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
        
        force_mag = action_to_force(act)

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
        reward -= 0.01 * (force_mag / FORCE_MAX) ** 2

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