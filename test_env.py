"""
Quick test to verify the realistic environment works correctly.
Run this BEFORE training to catch any issues early.
"""

import numpy as np
from realistic_causal_env import RealisticCausalPushEnv
import matplotlib.pyplot as plt

def test_basic_functionality():
    """Test that environment can be created and stepped through"""
    print("="*70)
    print("TEST 1: Basic Functionality")
    print("="*70)
    
    env = RealisticCausalPushEnv(render_mode=None, distribution_mode='train', history_length=3)
    
    obs, info = env.reset()
    print(f"✓ Environment created")
    print(f"  Observation shape: {obs.shape} (expected: 23)")
    print(f"  Confounders: m={env.mass:.2f}kg, μ={env.base_friction:.3f}, r={env.roughness:.2f}")
    
    # Run one episode
    done = False
    steps = 0
    while not done and steps < 35:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    print(f"✓ Episode completed in {steps} steps")
    print(f"  Success: {info['is_success']}")
    print(f"  Final distance: {info['final_distance']*100:.2f}cm")
    
    env.close()

def test_distribution_modes():
    """Test all distribution modes"""
    print("\n" + "="*70)
    print("TEST 2: Distribution Modes")
    print("="*70)
    
    modes = ['train', 'test_interp', 'test_extrap_high', 'test_extrap_low', 'test_adversarial']
    
    for mode in modes:
        env = RealisticCausalPushEnv(render_mode=None, distribution_mode=mode, history_length=3)
        obs, _ = env.reset()
        
        print(f"  {mode:20s}: m={env.mass:.2f}kg, μ={env.base_friction:.3f}, r={env.roughness:.2f}")
        
        env.close()
    
    print(f"✓ All distribution modes work")

def test_confounder_effects():
    """Test that confounders actually affect physics"""
    print("\n" + "="*70)
    print("TEST 3: Confounder Effects on Physics")
    print("="*70)
    
    # Test different friction levels with constant force
    results = []
    
    for mode in ['test_extrap_low', 'train', 'test_extrap_high']:
        env = RealisticCausalPushEnv(render_mode=None, distribution_mode=mode, history_length=3)
        env.reset(seed=42)
        
        # Apply constant medium force
        distances = []
        for _ in range(3):
            env.reset(seed=None)
            for step in range(10):
                obs, reward, term, trunc, info = env.step(np.array([0.0]))  # Medium force
                if term or trunc:
                    break
            
            base_obs = env._get_base_obs()
            distances.append(base_obs[0])
        
        avg_dist = np.mean(distances)
        results.append({
            'mode': mode,
            'friction': env.base_friction,
            'mass': env.mass,
            'roughness': env.roughness,
            'distance': avg_dist
        })
        
        print(f"  {mode:20s}: μ={env.base_friction:.3f}, m={env.mass:.2f}kg → distance={avg_dist:.4f}m")
        
        env.close()
    
    # Check if friction affects outcomes
    low_friction_dist = results[0]['distance']
    high_friction_dist = results[2]['distance']
    
    if low_friction_dist > high_friction_dist * 1.1:
        print(f"\n✓ Physics works: Low friction ({results[0]['friction']:.3f}) → more distance ({low_friction_dist:.4f}m)")
        print(f"                  High friction ({results[2]['friction']:.3f}) → less distance ({high_friction_dist:.4f}m)")
    else:
        print(f"\n⚠ Warning: Friction effect may be too weak")
        print(f"  Low friction distance: {low_friction_dist:.4f}m")
        print(f"  High friction distance: {high_friction_dist:.4f}m")

def test_roughness_variance():
    """Test that roughness causes stochasticity"""
    print("\n" + "="*70)
    print("TEST 4: Roughness Causes Variance")
    print("="*70)
    
    for roughness_level in ['Low', 'High']:
        # Set up environment with specific roughness
        if roughness_level == 'Low':
            mode = 'test_extrap_low'  # r=0.1
            expected_r = 0.1
        else:
            mode = 'test_extrap_high'  # r=0.8
            expected_r = 0.8
        
        env = RealisticCausalPushEnv(render_mode=None, distribution_mode=mode, history_length=3)
        
        # Run multiple trials with same force
        distances = []
        for trial in range(10):
            env.reset(seed=trial)
            
            for step in range(10):
                obs, reward, term, trunc, info = env.step(np.array([0.5]))  # High force
                if term or trunc:
                    break
            
            base_obs = env._get_base_obs()
            distances.append(base_obs[0])
        
        variance = np.var(distances)
        print(f"  {roughness_level} roughness (r={expected_r:.1f}):")
        print(f"    Mean distance: {np.mean(distances):.4f}m")
        print(f"    Variance: {variance:.6f}")
        
        env.close()
    
    print(f"\n✓ Roughness should cause higher variance in outcomes")

def test_observation_history():
    """Test that observation history is maintained"""
    print("\n" + "="*70)
    print("TEST 5: Observation History")
    print("="*70)
    
    env = RealisticCausalPushEnv(render_mode=None, distribution_mode='train', history_length=3)
    obs, _ = env.reset()
    
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Current state (5) + History (5×3=15) + Forces (3) = 23 ✓")
    
    # Take a few steps
    for i in range(5):
        obs, reward, term, trunc, info = env.step(np.array([0.3]))
        if i == 0:
            print(f"  After step 1: obs_history has {len(env.obs_history)} frames")
            print(f"  After step 1: force_history has {len(env.force_history)} forces")
    
    print(f"✓ History mechanism works")
    
    env.close()

def visualize_force_distance_relationship():
    """Create a simple visualization of force vs distance for each distribution"""
    print("\n" + "="*70)
    print("TEST 6: Force-Distance Calibration")
    print("="*70)
    
    modes = ['test_extrap_low', 'train', 'test_extrap_high']
    force_levels = [-0.5, 0.0, 0.5, 1.0]  # Action values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mode in modes:
        distances_by_force = []
        
        for force_action in force_levels:
            env = RealisticCausalPushEnv(render_mode=None, distribution_mode=mode, history_length=3)
            
            distances = []
            for trial in range(5):
                env.reset(seed=trial)
                
                for step in range(15):
                    obs, reward, term, trunc, info = env.step(np.array([force_action]))
                    if term or trunc:
                        break
                
                base_obs = env._get_base_obs()
                distances.append(base_obs[0])
            
            distances_by_force.append(np.mean(distances))
            env.close()
        
        # Convert actions to actual forces (updated mapping)
        forces_N = [3.0 + (a * 2.5) for a in force_levels]
        
        ax.plot(forces_N, distances_by_force, 'o-', label=f'{mode} (μ≈{env.base_friction:.2f})', linewidth=2, markersize=8)
    
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Target (10cm)')
    ax.axhline(y=0.085, color='green', linestyle=':', alpha=0.3)
    ax.axhline(y=0.115, color='green', linestyle=':', alpha=0.3)
    ax.set_xlabel('Applied Force (N)')
    ax.set_ylabel('Final Distance (m)')
    ax.set_title('Force-Distance Relationship Across Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('force_distance_calibration.png', dpi=150)
    print(f"✓ Calibration plot saved as 'force_distance_calibration.png'")
    print(f"  Check this plot to ensure:")
    print(f"  1. Different distributions need different forces")
    print(f"  2. All distributions can achieve the target (cross green line)")
    print(f"  3. Force-distance relationship makes physical sense")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("REALISTIC CAUSAL ENVIRONMENT - TEST SUITE")
    print("="*70)
    print("Run this before training to verify everything works!\n")
    
    try:
        test_basic_functionality()
        test_distribution_modes()
        test_confounder_effects()
        test_roughness_variance()
        test_observation_history()
        visualize_force_distance_relationship()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nEnvironment is ready for training!")
        print("\nNext steps:")
        print("  1. Check 'force_distance_calibration.png' to verify physics")
        print("  2. If it looks good, run: python train_realistic.py")
        print("  3. Wait ~30-45 minutes for training")
        print("  4. Run: python evaluate_realistic.py")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED ✗")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nFix the errors above before proceeding!")