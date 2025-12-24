import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from causal_env import CausalPushEnv
import numpy as np

def evaluate_robust(mode_name, models_dir="models_robust/"):
    # 1. Recreate the environment structure
    env = DummyVecEnv([lambda: CausalPushEnv(render_mode=None, distribution_mode=mode_name)])
    
    # 2. Load the SAVED normalization stats
    # This ensures the test environment scales inputs exactly like the training environment did
    try:
        env = VecNormalize.load(f"{models_dir}/vec_normalize.pkl", env)
    except FileNotFoundError:
        print("Error: Normalization stats not found. Run training first.")
        return 0
        
    # We do NOT update normalization stats during testing (training=False)
    env.training = False
    env.norm_reward = False 
    
    # 3. Load the BEST model
    model = PPO.load(f"{models_dir}/best_model", env=env)
    
    print(f"\n--- Robust Evaluation on {mode_name.upper()} ---")
    
    obs = env.reset()
    success_count = 0
    episodes = 50
    
    for i in range(episodes):
        done = False
        while not done:
            # Predict action (deterministic=True suppresses exploration noise)
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # VecEnv automatically resets on done, so we check the info dict or 'dones'
            if dones[0]:
                done = True
                # Access the 'is_success' info from the underlying env
                if infos[0].get('is_success'):
                    success_count += 1
                    
    success_rate = (success_count / episodes) * 100
    print(f"Success Rate: {success_rate:.1f}%")
    return success_rate

if __name__ == "__main__":
    evaluate_robust('train')            # Should be very high (>90%)
    evaluate_robust('test_extrap_high') # The moment of truth
    evaluate_robust('test_extrap_low')  # The moment of truth