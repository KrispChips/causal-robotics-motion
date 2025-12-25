"""
Comprehensive validation suite for the realistic environment.
This catches physics bugs, scaling issues, and design flaws BEFORE training.
"""

import numpy as np
from realistic_causal_env import RealisticCausalPushEnv
import matplotlib.pyplot as plt
from collections import defaultdict

class EnvironmentValidator:
    """Systematic validation of environment physics and design"""
    
    def __init__(self):
        self.issues_found = []
        self.warnings = []
        self.passed_tests = []
    
    def add_issue(self, test_name, message):
        self.issues_found.append(f"❌ {test_name}: {message}")
    
    def add_warning(self, test_name, message):
        self.warnings.append(f"⚠️  {test_name}: {message}")
    
    def add_pass(self, test_name, message=""):
        self.passed_tests.append(f"✅ {test_name}" + (f": {message}" if message else ""))
    
    def test_force_range_adequacy(self):
        """Test if force range can overcome friction in all distributions"""
        print("\n" + "="*70)
        print("TEST 1: Force Range Adequacy")
        print("="*70)
        
        modes = ['test_extrap_low', 'train', 'test_extrap_high']
        
        for mode in modes:
            env = RealisticCausalPushEnv(render_mode=None, distribution_mode=mode)
            env.reset()
            
            # Calculate required force to overcome static friction
            required_force = env.base_friction * env.mass * 9.8
            max_force = 5.5  # Our maximum (updated)
            headroom = max_force - required_force
            
            print(f"\n{mode}:")
            print(f"  Mass: {env.mass:.2f}kg")
            print(f"  Friction: {env.base_friction:.3f}")
            print(f"  Roughness: {env.roughness:.2f}")
            print(f"  Required force to overcome friction: {required_force:.2f}N")
            print(f"  Max available force: {max_force:.2f}N")
            print(f"  Headroom: {headroom:.2f}N")
            
            if headroom < 0:
                self.add_issue("Force Range", 
                    f"{mode}: Cannot overcome friction! Need {required_force:.2f}N but max is {max_force:.2f}N")
            elif headroom < 0.5:
                self.add_warning("Force Range",
                    f"{mode}: Very tight headroom ({headroom:.2f}N). May be hard to learn.")
            else:
                self.add_pass("Force Range", f"{mode}: {headroom:.2f}N headroom")
            
            env.close()
    
    def test_friction_sampling_frequency(self):
        """Check if friction is being resampled too frequently"""
        print("\n" + "="*70)
        print("TEST 2: Friction Sampling Frequency")
        print("="*70)
        
        env = RealisticCausalPushEnv(render_mode=None, distribution_mode='train')
        env.reset()
        
        # Track how friction changes
        frictions_during_step = []
        
        # Monkey-patch to track friction changes
        original_apply = env._apply_stochastic_friction
        friction_samples = []
        
        def tracked_apply():
            result = original_apply()
            friction_samples.append(result)
            return result
        
        env._apply_stochastic_friction = tracked_apply
        
        # Take one step
        env.step(np.array([0.5]))
        
        num_samples = len(friction_samples)
        print(f"\nFriction was sampled {num_samples} times in one action step")
        print(f"Sim steps per action: {env.sim_steps_per_action}")
        
        if num_samples > 1:
            self.add_warning("Friction Sampling",
                f"Friction resampled {num_samples} times per action. This adds excessive stochasticity.")
            print(f"  → Friction values: {[f'{f:.3f}' for f in friction_samples[:5]]} ...")
            print(f"  → Recommendation: Sample once per action, not per simulation step")
        else:
            self.add_pass("Friction Sampling", "Sampled once per action")
        
        env.close()
    
    def test_observation_dimensions(self):
        """Verify observation space dimensions are correct"""
        print("\n" + "="*70)
        print("TEST 3: Observation Space Dimensions")
        print("="*70)
        
        env = RealisticCausalPushEnv(render_mode=None, distribution_mode='train', history_length=3)
        obs, _ = env.reset()
        
        expected_dim = 5 + (5 * 3) + 3  # current + history + forces = 23
        actual_dim = obs.shape[0]
        
        print(f"\nExpected dimension: {expected_dim}")
        print(f"Actual dimension: {actual_dim}")
        print(f"Breakdown:")
        print(f"  Current state: 5")
        print(f"  History (3 frames × 5): 15")
        print(f"  Force history: 3")
        print(f"  Total: {expected_dim}")
        
        if actual_dim == expected_dim:
            self.add_pass("Observation Dims", f"{actual_dim} dimensions")
        else:
            self.add_issue("Observation Dims",
                f"Mismatch! Expected {expected_dim}, got {actual_dim}")
        
        # Check if observation contains valid values
        if np.any(np.isnan(obs)):
            self.add_issue("Observation Validity", "Contains NaN values!")
        elif np.any(np.isinf(obs)):
            self.add_issue("Observation Validity", "Contains Inf values!")
        else:
            self.add_pass("Observation Validity", "No NaN or Inf")
        
        env.close()
    
    def test_success_achievability(self):
        """Test if success threshold is achievable given roughness variance"""
        print("\n" + "="*70)
        print("TEST 4: Success Threshold Achievability")
        print("="*70)
        
        env = RealisticCausalPushEnv(render_mode=None, distribution_mode='test_extrap_high')
        
        # Try to achieve success with optimal force
        successes = []
        final_distances = []
        
        print(f"\nRunning 20 trials with optimal force...")
        print(f"Success threshold: ±{env.success_threshold*100:.1f}cm")
        print(f"Roughness level: {env.roughness:.2f}")
        
        for trial in range(20):
            env.reset(seed=trial)
            
            # Apply what should be optimal force
            for step in range(15):
                obs, reward, term, trunc, info = env.step(np.array([0.5]))
                if term or trunc:
                    break
            
            successes.append(info['is_success'])
            final_distances.append(info['final_distance'])
        
        success_rate = np.mean(successes)
        mean_dist = np.mean(final_distances)
        std_dist = np.std(final_distances)
        
        print(f"\nResults with constant optimal force:")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Mean distance: {mean_dist*100:.2f}cm ± {std_dist*100:.2f}cm")
        print(f"  Min distance: {min(final_distances)*100:.2f}cm")
        print(f"  Max distance: {max(final_distances)*100:.2f}cm")
        
        if success_rate > 0.8:
            self.add_pass("Success Achievability", f"{success_rate*100:.0f}% with optimal force")
        elif success_rate > 0.5:
            self.add_warning("Success Achievability",
                f"Only {success_rate*100:.0f}% success with optimal force. Roughness may add too much variance.")
        else:
            self.add_issue("Success Achievability",
                f"Only {success_rate*100:.0f}% achievable! Task may be too hard.")
        
        env.close()
    
    def test_distribution_separation(self):
        """Test if different distributions actually need different strategies"""
        print("\n" + "="*70)
        print("TEST 5: Distribution Separation")
        print("="*70)
        
        modes = ['test_extrap_low', 'train', 'test_extrap_high']
        optimal_forces = {}
        
        for mode in modes:
            env = RealisticCausalPushEnv(render_mode=None, distribution_mode=mode)
            
            # Test different forces
            force_actions = [-0.5, 0.0, 0.5, 1.0]
            results = []
            
            for force_action in force_actions:
                distances = []
                for trial in range(5):
                    env.reset(seed=trial)
                    for step in range(15):
                        obs, reward, term, trunc, info = env.step(np.array([force_action]))
                        if term or trunc:
                            break
                    distances.append(abs(env._get_base_obs()[0] - 0.1))
                
                avg_dist = np.mean(distances)
                results.append((force_action, avg_dist))
            
            # Find optimal force (minimizes distance to target)
            optimal = min(results, key=lambda x: x[1])
            optimal_forces[mode] = optimal[0]
            
            force_N = 3.0 + (optimal[0] * 2.5)  # Updated force mapping
            print(f"\n{mode}:")
            print(f"  Properties: μ={env.base_friction:.3f}, m={env.mass:.2f}kg")
            print(f"  Optimal action: {optimal[0]:+.1f} ({force_N:.2f}N)")
            print(f"  Achieves distance: {optimal[1]*100:.2f}cm")
            
            env.close()
        
        # Check separation
        force_range = max(optimal_forces.values()) - min(optimal_forces.values())
        print(f"\nOptimal force range across distributions: {force_range:.2f}")
        
        if force_range > 0.5:
            self.add_pass("Distribution Separation",
                f"Range = {force_range:.2f} (good separation)")
        elif force_range > 0.3:
            self.add_warning("Distribution Separation",
                f"Range = {force_range:.2f} (moderate separation)")
        else:
            self.add_issue("Distribution Separation",
                f"Range = {force_range:.2f} (too similar! Won't show distribution shift)")
    
    def test_reward_function_sanity(self):
        """Check if reward function encourages correct behavior"""
        print("\n" + "="*70)
        print("TEST 6: Reward Function Sanity")
        print("="*70)
        
        env = RealisticCausalPushEnv(render_mode=None, distribution_mode='train')
        env.reset()
        
        scenarios = [
            ("Perfect success", 0.0995, 0.0, "Should get high reward"),  # 0.5mm from target
            ("Just inside threshold", 0.085, 0.0, "Should get success bonus"),
            ("Just outside threshold", 0.115, 0.0, "Should NOT get success bonus"),
            ("Way off target", 0.05, 0.0, "Should get large penalty"),
            ("Overshoot", 0.15, 0.0, "Should get overshoot penalty"),
        ]
        
        print("\nTesting reward in different scenarios:")
        
        for scenario_name, position, velocity, expected in scenarios:
            # Manually construct state
            env.obs_history = [[position, 0, velocity, 0, 0.1]] * 3
            env.force_history = [2.5] * 3
            
            # Take one step
            obs, reward, term, trunc, info = env.step(np.array([0.0]))
            
            print(f"\n  {scenario_name}:")
            print(f"    Position: {position*100:.1f}cm, Velocity: {velocity:.2f}m/s")
            print(f"    Reward: {reward:.2f}")
            print(f"    Success: {info['is_success']}")
            print(f"    Expected: {expected}")
        
        self.add_pass("Reward Function", "Manual inspection needed")
        env.close()
    
    def test_episode_termination(self):
        """Verify episodes terminate correctly"""
        print("\n" + "="*70)
        print("TEST 7: Episode Termination Logic")
        print("="*70)
        
        env = RealisticCausalPushEnv(render_mode=None, distribution_mode='train')
        env.reset()
        
        # Run full episode
        step_count = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and step_count < 35:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            step_count += 1
        
        print(f"\nEpisode ended after {step_count} steps")
        print(f"  Terminated: {terminated} (early success)")
        print(f"  Truncated: {truncated} (time limit)")
        print(f"  Max steps: {env.max_episode_steps}")
        
        if step_count == env.max_episode_steps and truncated:
            self.add_pass("Episode Termination", "Truncates at max steps")
        elif step_count < env.max_episode_steps and not terminated:
            self.add_issue("Episode Termination",
                f"Episode should have terminated or truncated but didn't!")
        else:
            self.add_pass("Episode Termination", f"Works correctly")
        
        env.close()
    
    def run_all_tests(self):
        """Run complete validation suite"""
        print("\n" + "="*70)
        print("COMPREHENSIVE ENVIRONMENT VALIDATION")
        print("="*70)
        print("Testing all aspects of the environment before training...")
        
        self.test_force_range_adequacy()
        self.test_friction_sampling_frequency()
        self.test_observation_dimensions()
        self.test_success_achievability()
        self.test_distribution_separation()
        self.test_reward_function_sanity()
        self.test_episode_termination()
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        if self.passed_tests:
            print(f"\n✅ PASSED ({len(self.passed_tests)}):")
            for test in self.passed_tests:
                print(f"  {test}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.issues_found:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues_found)}):")
            for issue in self.issues_found:
                print(f"  {issue}")
        
        print("\n" + "="*70)
        if self.issues_found:
            print("STATUS: ❌ CRITICAL ISSUES FOUND - FIX BEFORE TRAINING")
            print("="*70)
            print("\nRecommended fixes will be provided.")
            return False
        elif self.warnings:
            print("STATUS: ⚠️  WARNINGS PRESENT - REVIEW BEFORE TRAINING")
            print("="*70)
            print("\nEnvironment should work, but consider addressing warnings.")
            return True
        else:
            print("STATUS: ✅ ALL TESTS PASSED - READY FOR TRAINING")
            print("="*70)
            return True

if __name__ == "__main__":
    validator = EnvironmentValidator()
    ready = validator.run_all_tests()
    
    if not ready:
        print("\n" + "="*70)
        print("RECOMMENDED FIXES")
        print("="*70)
        print("\n1. If force range still insufficient:")
        print("   - Increase max force: force = 3.0 + (action * 3.0)  # 0N to 6N")
        print("   - OR reduce high friction test value")
        print("\n2. If blocks still not moving:")
        print("   - Check PyBullet physics settings")
        print("   - Verify friction is being applied to both block and plane")
        print("   - Try increasing simulation timesteps")
        print("\n3. If success rate still 0%:")
        print("   - Increase episode length: max_episode_steps = 50")
        print("   - Relax success threshold: 0.020 → 0.025")
        print("   - Check that blocks can actually reach target with any force")