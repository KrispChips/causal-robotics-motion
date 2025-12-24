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
        
        # --- ACTION SPACE ---
        # 1D Control: Horizontal Force
        # Normalized [-1, 1] mapped to [0N, 20N] per PDF [cite: 52]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # --- OBSERVATION SPACE ---
        # Current: [x, y, vx, vy, target_dist]
        # Future proofing: We intentionally EXCLUDE mass/friction to force inference.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        # --- PHYSICS CONFIGURATION ---
        self.target_dist = 0.1  # 10cm target distance [cite: 51]
        self.success_threshold = 0.02 # +/- 2cm tolerance [cite: 53]
        self.sim_freq = 240.0   # PyBullet default
        self.policy_freq = 10.0 # Control frequency (how often agent acts)
        self.sim_steps_per_action = int(self.sim_freq / self.policy_freq)

        # Connect to Physics Server
        if render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Clean UI
        else:
            self.client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
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
        print(f"Environment switched to mode: {mode}")

    def _sample_confounders(self):
        """
        Samples hidden properties based on the specific research phase [cite: 63-73].
        """
        if self.distribution_mode == 'train':
            # Narrow range: u=[0.3, 0.6], m=[0.8, 1.5] [cite: 64, 65]
            mu = np.random.uniform(0.3, 0.6)
            m = np.random.uniform(0.8, 1.5)
            
        elif self.distribution_mode == 'test_interp':
            # Middle of range [cite: 67]
            mu = 0.45
            m = 1.2
            
        elif self.distribution_mode == 'test_extrap_high':
            # High friction, Light mass [cite: 70, 71]
            mu = 0.75
            m = 0.6
            
        elif self.distribution_mode == 'test_extrap_low':
            # Low friction, Heavy mass [cite: 72, 73]
            mu = 0.25
            m = 1.9
            
        else:
            # Fallback (Uniform full range) [cite: 45]
            mu = np.random.uniform(0.2, 0.8)
            m = np.random.uniform(0.5, 2.0)
            
        return mu, m

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        
        # 1. Load Plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 2. Create Block Procedurally (Exact 10cm cube)
        # halfExtents is half the side length, so 0.05 -> 10cm size
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[0.2, 0.6, 0.8, 1])
        
        start_pos = [0, 0, 0.05] # Sitting on ground
        self.block_id = p.createMultiBody(
            baseMass=1.0, # Placeholder, updated below
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=start_pos
        )
        
        # 3. Create Target Visualization (Phantom, no collision)
        t_vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.001], rgbaColor=[0, 1, 0, 0.4])
        self.target_visual_id = p.createMultiBody(
            baseVisualShapeIndex=t_vis_shape,
            basePosition=[self.target_dist, 0, 0.001]
        )

        # 4. Apply Hidden Confounders
        self.current_mu, self.current_m = self._sample_confounders()
        
        # Apply to Block
        p.changeDynamics(self.block_id, -1, mass=self.current_m)
        p.changeDynamics(self.block_id, -1, lateralFriction=self.current_mu)
        # Apply to Floor (friction interaction is product of both)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0) 
        
        # Reset Camera if rendering
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.05, 0, 0])

        return self._get_obs(), {}

    def step(self, action):
        # 1. Convert Action to Force
        # Action is [-1, 1], Force is [0, 20N]
        # We clip to ensure safety, though PPO usually respects bounds
        act = np.clip(action[0], -1.0, 1.0)
        force_mag = (act + 1.0) * 10.0 # Map -1->0, 1->20 [cite: 52]
        
        # 2. Apply Force and Step Simulation
        # We apply the force continuously over the duration of the 'step'
        for _ in range(self.sim_steps_per_action):
            p.applyExternalForce(
                self.block_id, 
                -1, 
                forceObj=[force_mag, 0, 0], 
                posObj=[0, 0, 0], # Center of mass
                flags=p.WORLD_FRAME
            )
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./self.sim_freq)
                
        # 3. Get Observation
        obs = self._get_obs()
        block_x = obs[0]
        
        # 4. Calculate Metrics
        dist_to_target = abs(block_x - self.target_dist)
        success = dist_to_target < self.success_threshold
        
        # 5. Reward Function
        # Dense reward for distance, sparse bonus for success
        reward = -dist_to_target 
        if success:
            reward += 10.0
            
        # 6. Termination
        terminated = bool(success)
        truncated = False # Handled by TimeLimit wrapper in main script
        
        # Info includes Ground Truth for evaluation/debugging
        info = {
            "true_friction": self.current_mu,
            "true_mass": self.current_m,
            "is_success": success
        }
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.block_id)
        lin_vel, _ = p.getBaseVelocity(self.block_id)
        
        # State: [x, y, vx, vy, target_dist]
        return np.array([
            pos[0], 
            pos[1], 
            lin_vel[0], 
            lin_vel[1], 
            self.target_dist
        ], dtype=np.float32)

    def close(self):
        p.disconnect()