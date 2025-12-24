import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time

class CausalPushEnv(gym.Env):
    """
    A high-fidelity simulation for Causal Robotics.
    Task: Push a 10cm cube to a target 10cm away.
    Hidden Properties: Friction (mu) and Mass (m).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, distribution_mode='train'):
        super().__init__()
        self.render_mode = render_mode
        self.distribution_mode = distribution_mode
        
        # ACTION SPACE
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # OBSERVATION SPACE
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        # PHYSICS CONFIGURATION
        self.target_dist = 0.1
        self.success_threshold = 0.02
        self.sim_freq = 240.0
        self.policy_freq = 10.0
        self.sim_steps_per_action = int(self.sim_freq / self.policy_freq)
        
        # Add maximum episode length to prevent infinite episodes
        self.max_episode_steps = 100  # 10 seconds at 10Hz control
        self.current_step = 0

        # Connect to Physics Server with Explicit ID
        if render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        
        self.block_id = None
        self.plane_id = None
        self.target_visual_id = None
        
        # Hidden Confounders
        self.current_friction = 0.5
        self.current_mass = 1.0

    def set_distribution_mode(self, mode):
        """Switch between 'train', 'test_interp', 'test_extrap_high', 'test_extrap_low'"""
        valid_modes = ['train', 'test_interp', 'test_extrap_high', 'test_extrap_low']
        assert mode in valid_modes, f"Mode must be one of {valid_modes}"
        self.distribution_mode = mode

    def _sample_confounders(self):
        """Samples hidden properties based on the specific research phase"""
        if self.distribution_mode == 'train':
            mu = np.random.uniform(0.3, 0.6)
            m = np.random.uniform(0.8, 1.5)
            
        elif self.distribution_mode == 'test_interp':
            mu = 0.45
            m = 1.2
            
        elif self.distribution_mode == 'test_extrap_high':
            mu = 0.75
            m = 0.6
            
        elif self.distribution_mode == 'test_extrap_low':
            mu = 0.25
            m = 1.9
            
        else:
            mu = np.random.uniform(0.2, 0.8)
            m = np.random.uniform(0.5, 2.0)
            
        return mu, m

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset simulation
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        
        # 1. Load Plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        # 2. Create Block Procedurally
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
            baseMass=1.0, 
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=start_pos,
            physicsClientId=self.client
        )
        
        # 3. Create Target Visualization
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

        # 4. Apply Hidden Confounders
        self.current_mu, self.current_m = self._sample_confounders()
        
        p.changeDynamics(self.block_id, -1, mass=self.current_m, physicsClientId=self.client)
        p.changeDynamics(self.block_id, -1, lateralFriction=self.current_mu, physicsClientId=self.client)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=self.client) 
        
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
        # Increment step counter
        self.current_step += 1
        
        # FIX 1: Handle both array and scalar actions from vectorized envs
        if isinstance(action, np.ndarray):
            if action.ndim == 0:  # scalar array
                act = float(action)
            else:  # 1D array
                act = action[0]
        else:
            act = action
        
        act = np.clip(act, -1.0, 1.0)
        
        # FIX 2: Correct force mapping - higher action should mean higher force
        # Map [-1, 1] to [0N, 20N]
        force_mag = (act + 1.0) * 10.0  # -1 -> 0N, 0 -> 10N, +1 -> 20N
        
        for _ in range(self.sim_steps_per_action):
            p.applyExternalForce(
                self.block_id, 
                -1, 
                forceObj=[force_mag, 0, 0], 
                posObj=[0, 0, 0], 
                flags=p.WORLD_FRAME,
                physicsClientId=self.client
            )
            p.stepSimulation(physicsClientId=self.client)
            if self.render_mode == "human":
                time.sleep(1./self.sim_freq)
                
        obs = self._get_obs()
        block_x = obs[0]
        
        dist_to_target = abs(block_x - self.target_dist)
        success = dist_to_target < self.success_threshold
        
        # Reward: negative distance (want to minimize) + bonus for success
        reward = -dist_to_target 
        if success:
            reward += 10.0
            
        # Terminate on success OR when max steps reached
        terminated = bool(success)
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            "true_friction": self.current_mu,
            "true_mass": self.current_m,
            "is_success": success,
            "TimeLimit.truncated": truncated
        }
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.block_id, physicsClientId=self.client)
        lin_vel, _ = p.getBaseVelocity(self.block_id, physicsClientId=self.client)
        
        return np.array([
            pos[0], 
            pos[1], 
            lin_vel[0], 
            lin_vel[1], 
            self.target_dist
        ], dtype=np.float32)

    def close(self):
        if hasattr(self, 'client'):
            p.disconnect(self.client)