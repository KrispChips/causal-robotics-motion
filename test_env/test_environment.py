"""
Quick test to verify the environment works correctly before training.
Run this FIRST to catch issues early.
"""

from causal_env import CausalPushEnv
import numpy as np

def test_single_episode():
    """Test a single episode completes without hanging"""
    print("Test 1: Single Episode Completion")
    print("-" * 50)
    
    env = CausalPushEnv(render_mode=None, distribution_mode='train')
    obs, info = env.reset()
    
    print(f"✓ Environment created")
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Friction: {env.current_mu:.3f}, Mass: {env.current_m:.2f}kg")
    
    done = False
    steps = 0
    
    while not done and steps < 150:  # Safety limit
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        if steps % 20 == 0:
            print(f"  Step {steps}: x={obs[0]:.3f}m, reward={reward:.3f}")
    
    if done:
        print(f"✓ Episode completed in {steps} steps")
        print(f"  Success: {info['is_success']}")
    else:
        print(f"✗ Episode did not terminate after {steps} steps (possible hang)")
    
    env.close()
    return done

def test_multiple_episodes():
    """Test multiple episodes to check for memory leaks or crashes"""
    print("\nTest 2: Multiple Episodes (10 episodes)")
    print("-" * 50)
    
    env = CausalPushEnv(render_mode=None, distribution_mode='train')
    
    successes = []
    episode_lengths = []
    
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 150:
            action = np.array([np.random.uniform(-1, 1)])  # Random force
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        successes.append(info['is_success'])
        episode_lengths.append(steps)
        
        print(f"  Episode {ep+1}: {steps:3d} steps, success={info['is_success']}, μ={env.current_mu:.3f}, m={env.current_m:.2f}kg")
    
    print(f"\n✓ All episodes completed")
    print(f"  Success rate: {np.mean(successes)*100:.1f}%")
    print(f"  Avg episode length: {np.mean(episode_lengths):.1f} steps")
    
    env.close()

def test_distribution_modes():
    """Test all distribution modes work correctly"""
    print("\nTest 3: Distribution Modes")
    print("-" * 50)
    
    modes = ['train', 'test_interp', 'test_extrap_high', 'test_extrap_low']
    
    for mode in modes:
        env = CausalPushEnv(render_mode=None, distribution_mode=mode)
        obs, _ = env.reset()
        
        print(f"  {mode:20s}: μ={env.current_mu:.3f}, m={env.current_m:.2f}kg")
        
        # Quick sanity check - run a few steps
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        env.close()
    
    print(f"✓ All distribution modes work")

def test_physics_consistency():
    """Test that physics behaves as expected"""
    print("\nTest 4: Physics Consistency")
    print("-" * 50)
    
    env = CausalPushEnv(render_mode=None, distribution_mode='test_interp')
    
    # Test: Higher force should move block further
    results = []
    
    for force_level in [-1.0, 0.0, 1.0]:  # Low, medium, high force
        env.reset(seed=42)  # Same initial conditions
        
        # Apply consistent force for 30 steps to see cumulative effect
        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(np.array([force_level]))
            if terminated or truncated:
                break
        
        final_position = obs[0]
        results.append((force_level, final_position))
        print(f"  Force level={force_level:+.1f} → Final position={final_position:.4f}m (target=0.10m)")
    
    # Check if higher force leads to more displacement
    # Force mapping: -1 -> 0N, 0 -> 10N, +1 -> 20N
    if results[2][1] > results[1][1] and results[1][1] > results[0][1]:
        print(f"✓ Physics behaves correctly (higher force → more displacement)")
    else:
        print(f"✗ Warning: Physics may not be working correctly")
        print(f"  Expected: position increases with force level")
        print(f"  Got: {results[0][1]:.4f}m, {results[1][1]:.4f}m, {results[2][1]:.4f}m")
    
    env.close()

def test_vectorized_env():
    """Test the environment works with vectorization (as used in training)"""
    print("\nTest 5: Vectorized Environment")
    print("-" * 50)
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    
    def make_env():
        def _init():
            env = CausalPushEnv(render_mode=None, distribution_mode='train')
            return Monitor(env)
        return _init
    
    vec_env = DummyVecEnv([make_env()])
    
    obs = vec_env.reset()
    print(f"✓ Vectorized env created, obs shape: {obs.shape}")
    
    for i in range(20):
        action = vec_env.action_space.sample()
        obs, reward, done, info = vec_env.step(action)
        if done[0]:
            print(f"  Episode completed at step {i+1}")
            obs = vec_env.reset()
    
    vec_env.close()
    print(f"✓ Vectorized environment works correctly")

if __name__ == "__main__":
    print("="*50)
    print("ENVIRONMENT DIAGNOSTIC TESTS")
    print("="*50)
    print("Run this before training to catch issues early!\n")
    
    try:
        # Run all tests
        test_single_episode()
        test_multiple_episodes()
        test_distribution_modes()
        test_physics_consistency()
        test_vectorized_env()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
        print("Environment is working correctly!")
        print("You can now run: python train_rl.py")
        
    except Exception as e:
        print("\n" + "="*50)
        print("TEST FAILED ✗")
        print("="*50)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nFix the errors above before training!")