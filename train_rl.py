import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from causal_env import CausalPushEnv

def make_env(rank, mode='train'):
    """Utility function for multiprocessed env."""
    def _init():
        env = CausalPushEnv(render_mode=None, distribution_mode=mode)
        env = Monitor(env) 
        return env
    return _init

def train_strong_baseline():
    # Paths
    log_dir = "logs_rl/"
    models_dir = "models_rl/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Create Vectorized Environment
    env = DummyVecEnv([make_env(0, mode='train')])
    
    # 2. Add Normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Create Evaluation Environment
    eval_env = DummyVecEnv([make_env(0, mode='train')])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # FIX: Sync normalization stats (they'll update during training)
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    
    # 4. Evaluation Callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=models_dir,
        log_path=log_dir, 
        eval_freq=5000, 
        n_eval_episodes=10,
        deterministic=True, 
        render=False,
        verbose=1  # FIX: Add verbose to see progress
    )
    
    # FIX: Add checkpoint callback to save periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="ppo_checkpoint",
        verbose=1
    )

    # 5. Define the PPO Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log=log_dir,
        device="cuda"  # FIX: Explicitly set device
    )
    
    print("--- Starting Robust Training ---")
    print("Goal: Maximize performance on the Training Distribution")
    print(f"Training for 200k timesteps with evaluations every 5k steps")
    
    try:
        # Train with both callbacks
        model.learn(
            total_timesteps=200000, 
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True  # FIX: Show progress bar
        )
        
        # Save final model
        model.save(f"{models_dir}/final_model")
        env.save(f"{models_dir}/vec_normalize.pkl")
        
        print(f"--- Training Complete! ---")
        print(f"Best model saved in {models_dir}/best_model.zip")
        print(f"Checkpoints saved every 10k steps")
        
    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
        model.save(f"{models_dir}/interrupted_model")
        env.save(f"{models_dir}/vec_normalize.pkl")
        print("Progress saved!")
        
    except Exception as e:
        print(f"\n--- Training failed with error: {e} ---")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always clean up
        env.close()
        eval_env.close()

if __name__ == "__main__":
    train_strong_baseline()