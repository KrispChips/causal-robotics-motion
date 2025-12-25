"""
Bayesian Causal Discovery Agent

Key Idea: Infer hidden confounders (friction, mass) from interaction dynamics,
then use physics model to choose optimal force.

This beats standard RL by explicitly reasoning about causation.
"""

import numpy as np
from scipy.stats import beta, norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class BayesianCausalAgent:
    """
    Agent that infers friction and mass through Bayesian updating,
    then uses physics model to choose actions.
    """
    
    def __init__(self, 
                 friction_prior=(2, 2),      # Beta distribution parameters
                 mass_prior=(1.0, 0.3),      # Normal distribution (mean, std)
                 n_particles=1000):          # Particle filter resolution
        
        # Prior beliefs about confounders
        self.friction_prior = friction_prior
        self.mass_prior = mass_prior
        self.n_particles = n_particles
        
        # Initialize particle filter for beliefs
        self.reset_beliefs()
        
        # Physics constants
        self.g = 9.8  # gravity
        self.dt = 0.1  # timestep (10Hz control)
        
        # Interaction history
        self.interaction_history = []
    
    def reset_beliefs(self):
        """Reset beliefs to prior for new episode"""
        # Sample particles from prior
        # Friction: Beta(2,2) scaled to [0.15, 0.35]
        friction_samples = beta.rvs(self.friction_prior[0], self.friction_prior[1], 
                                    size=self.n_particles)
        self.friction_particles = 0.15 + friction_samples * 0.20  # Scale to [0.15, 0.35]
        
        # Mass: Normal(1.0, 0.3) clipped to [0.6, 1.4]
        self.mass_particles = norm.rvs(self.mass_prior[0], self.mass_prior[1],
                                      size=self.n_particles)
        self.mass_particles = np.clip(self.mass_particles, 0.6, 1.4)
        
        # Equal weights initially
        self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Clear history
        self.interaction_history = []
    
    def predict_distance(self, force, friction, mass, n_steps=20):
        """
        Simplified physics model: predict how far block moves
        
        CALIBRATED to match environment dynamics
        """
        position = 0.0
        velocity = 0.0
        
        # Physics parameters (tuned to match environment)
        damping = 0.98  # Matches environment
        dt = 1.0 / 10.0  # 10Hz control frequency
        
        for step in range(n_steps):
            # Friction force (static + kinetic combined)
            f_friction = friction * mass * self.g
            
            # Net force (can't pull backward)
            if force > f_friction:
                f_net = force - f_friction
            else:
                f_net = 0.0
                velocity *= 0.5  # Rapid decay if not enough force
            
            # Acceleration
            accel = f_net / mass if mass > 0 else 0
            
            # Update velocity (with damping)
            velocity += accel * dt
            velocity *= damping
            
            # Update position
            position += velocity * dt
            
            # Stop if velocity is negligible
            if abs(velocity) < 0.001 and step > 5:
                break
        
        return position
    
    def likelihood(self, observed_distance, force, friction, mass, n_steps):
        """
        Compute likelihood: P(observed_distance | force, friction, mass)
        
        Uses Gaussian likelihood around predicted distance
        """
        predicted_distance = self.predict_distance(force, friction, mass, n_steps)
        
        # Likelihood is Gaussian with some noise (accounts for roughness)
        noise_std = 0.02  # 2cm standard deviation
        likelihood = norm.pdf(observed_distance, predicted_distance, noise_std)
        
        return likelihood
    
    def update_beliefs(self, force_applied, observed_distance, n_steps=20):
        """
        Bayesian update: P(μ,m | observation) ∝ P(observation | μ,m,force) × P(μ,m)
        
        Uses particle filter for tractability
        """
        # Compute likelihood for each particle
        likelihoods = np.array([
            self.likelihood(observed_distance, force_applied, 
                          self.friction_particles[i], self.mass_particles[i], n_steps)
            for i in range(self.n_particles)
        ])
        
        # Update weights: posterior ∝ prior × likelihood
        self.weights *= likelihoods
        
        # Normalize
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        else:
            # If all weights are 0, reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        # Resample if effective sample size is low
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < self.n_particles / 2:
            self.resample_particles()
        
        # Store interaction
        self.interaction_history.append({
            'force': force_applied,
            'distance': observed_distance,
            'n_steps': n_steps
        })
    
    def resample_particles(self):
        """Resample particles according to weights (particle filter)"""
        indices = np.random.choice(self.n_particles, size=self.n_particles, 
                                  p=self.weights)
        self.friction_particles = self.friction_particles[indices]
        self.mass_particles = self.mass_particles[indices]
        
        # Add small noise to avoid particle depletion
        self.friction_particles += np.random.normal(0, 0.01, self.n_particles)
        self.friction_particles = np.clip(self.friction_particles, 0.15, 0.35)
        
        self.mass_particles += np.random.normal(0, 0.05, self.n_particles)
        self.mass_particles = np.clip(self.mass_particles, 0.6, 1.4)
        
        # Reset weights to uniform
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def get_beliefs(self):
        """Get current belief estimates (weighted mean)"""
        friction_estimate = np.average(self.friction_particles, weights=self.weights)
        mass_estimate = np.average(self.mass_particles, weights=self.weights)
        
        # Also compute uncertainty (weighted std)
        friction_std = np.sqrt(np.average(
            (self.friction_particles - friction_estimate)**2, weights=self.weights))
        mass_std = np.sqrt(np.average(
            (self.mass_particles - mass_estimate)**2, weights=self.weights))
        
        return {
            'friction_mean': friction_estimate,
            'friction_std': friction_std,
            'mass_mean': mass_estimate,
            'mass_std': mass_std
        }
    
    def choose_force(self, target_distance=0.1, n_steps=20, mode='exploit'):
        """
        Choose force to apply
        
        IMPROVED: Better optimization with bounds checking
        """
        beliefs = self.get_beliefs()
        friction_est = beliefs['friction_mean']
        mass_est = beliefs['mass_mean']
        
        if mode == 'explore':
            # Exploration: systematic force variation
            if len(self.interaction_history) == 0:
                return 2.5  # Start with moderate force
            elif len(self.interaction_history) == 1:
                return 4.0  # Try higher force
            elif len(self.interaction_history) == 2:
                return 1.5  # Try lower force
            else:
                mode = 'exploit'  # Switch to exploitation
        
        if mode == 'exploit':
            # Exploitation: Use grid search over reasonable force range
            # This is more robust than gradient-based optimization
            
            forces_to_try = np.linspace(0.5, 5.5, 50)
            best_force = forces_to_try[0]
            best_error = float('inf')
            
            for f in forces_to_try:
                predicted = self.predict_distance(f, friction_est, mass_est, n_steps)
                error = abs(predicted - target_distance)
                
                if error < best_error:
                    best_error = error
                    best_force = f
            
            return best_force
    
    def visualize_beliefs(self):
        """Plot current beliefs about friction and mass"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Friction distribution
        axes[0].hist(self.friction_particles, weights=self.weights, bins=50, 
                    alpha=0.7, edgecolor='black')
        beliefs = self.get_beliefs()
        axes[0].axvline(beliefs['friction_mean'], color='red', linestyle='--', 
                       label=f"Mean: {beliefs['friction_mean']:.3f}")
        axes[0].set_xlabel('Friction Coefficient (μ)')
        axes[0].set_ylabel('Probability Density')
        axes[0].set_title('Belief Distribution: Friction')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Mass distribution
        axes[1].hist(self.mass_particles, weights=self.weights, bins=50,
                    alpha=0.7, edgecolor='black')
        axes[1].axvline(beliefs['mass_mean'], color='red', linestyle='--',
                       label=f"Mean: {beliefs['mass_mean']:.2f}kg")
        axes[1].set_xlabel('Mass (kg)')
        axes[1].set_ylabel('Probability Density')
        axes[1].set_title('Belief Distribution: Mass')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class CausalDiscoveryPolicy:
    """
    Policy wrapper that uses Bayesian causal agent for decision making
    Compatible with Gym interface
    """
    
    def __init__(self, exploration_steps=3):
        self.agent = BayesianCausalAgent()
        self.exploration_steps = exploration_steps
        self.episode_step = 0
    
    def reset(self):
        """Reset for new episode"""
        self.agent.reset_beliefs()
        self.episode_step = 0
        self.last_position = 0.0
    
    def predict(self, observation, deterministic=True):
        """
        Choose action given observation
        
        FIXED: Properly extract information from observation
        """
        # Extract current position from observation
        # observation structure: [x, y, vx, vy, target, history(15), forces(3)]
        current_position = observation[0]
        current_velocity = observation[2]
        
        # Update beliefs if we've taken an action
        if self.episode_step > 0:
            # Calculate distance moved since last step
            distance_moved = abs(current_position - self.last_position)
            
            # Get last applied force from observation history
            # Last 3 values are force history, most recent is [-1]
            if len(observation) >= 23:  # Ensure we have full observation
                last_force = observation[-1]
            else:
                last_force = 3.0  # Default fallback
            
            # Only update if we have meaningful movement or lack thereof
            if self.episode_step <= 10:  # Update during first half of episode
                self.agent.update_beliefs(
                    force_applied=last_force, 
                    observed_distance=distance_moved,
                    n_steps=1  # One control step
                )
        
        # Choose force based on current mode
        remaining_distance = 0.1 - current_position  # Distance to target
        remaining_steps = 20 - self.episode_step  # Steps left
        
        # Exploration phase: gather information
        if self.episode_step < self.exploration_steps:
            mode = 'explore'
            force = self.agent.choose_force(
                target_distance=remaining_distance, 
                n_steps=remaining_steps, 
                mode=mode
            )
        else:
            # Exploitation: use inferred beliefs to reach target
            mode = 'exploit'
            force = self.agent.choose_force(
                target_distance=remaining_distance,
                n_steps=remaining_steps,
                mode=mode
            )
        
        # Convert force to normalized action
        action = (force - 3.0) / 2.5
        action = np.clip(action, -1.0, 1.0)
        
        # Update state tracking
        self.last_position = current_position
        self.episode_step += 1
        
        return np.array([action]), None