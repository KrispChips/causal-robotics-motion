# test_physics.py
from causal_env import CausalPushEnv
import numpy as np
import time

def run_verification():
    print("=== BEGINNING PHYSICS VERIFICATION ===")
    env = CausalPushEnv(render_mode="human")
    
    # Test Case 1: High Friction (Sticky), Heavy
    print("\nTest 1: Sticky & Heavy (Expect minimal movement)")
    env.set_distribution_mode('test_extrap_low') # Forces specific parameters
    obs, _ = env.reset()
    # Manually OVERRIDE for visual clarity of verification
    p = env.client
    # Let's force extreme values to be absolutely sure
    env.current_mu = 0.8 
    env.current_m = 2.0
    import pybullet as pb
    pb.changeDynamics(env.block_id, -1, mass=2.0, lateralFriction=0.8)
    
    print(f"Parameters -> Friction: {env.current_mu}, Mass: {env.current_m}")
    
    # Apply constant medium force (Action 0.0 -> 10N)
    for i in range(20):
        obs, reward, term, trunc, info = env.step([0.0]) 
        print(f"Step {i}: Pos X = {obs[0]:.4f}")
    
    final_pos_1 = obs[0]
    time.sleep(1)
    
    # Test Case 2: Low Friction (Slippery), Light
    print("\nTest 2: Slippery & Light (Expect large movement)")
    env.reset()
    # Manually OVERRIDE
    env.current_mu = 0.2
    env.current_m = 0.5
    pb.changeDynamics(env.block_id, -1, mass=0.5, lateralFriction=0.2)
    
    print(f"Parameters -> Friction: {env.current_mu}, Mass: {env.current_m}")
    
    for i in range(20):
        obs, reward, term, trunc, info = env.step([0.0])
        print(f"Step {i}: Pos X = {obs[0]:.4f}")
        
    final_pos_2 = obs[0]
    
    print("\n=== VERIFICATION RESULTS ===")
    print(f"Sticky/Heavy Distance: {final_pos_1:.4f}m")
    print(f"Slippery/Light Distance: {final_pos_2:.4f}m")
    
    if final_pos_2 > final_pos_1 * 1.5:
        print("SUCCESS: Physics engine is correctly simulating confounders!")
    else:
        print("FAILURE: Physics differences are negligible. Check simulation params.")
        
    env.close()

if __name__ == "__main__":
    run_verification()