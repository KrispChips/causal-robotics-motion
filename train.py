import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Import the new realistic environment
# Make sure this file is saved as 'realistic_causal_env.py'
from realistic_causal_env import RealisticCausalPushEnv

def make_env(rank, mode='train'):
    """Create environment factory"""
    def _init():
        env = RealisticCausalPushEnv(
            render_mode=None,
            distribution_mode=mode,
            history_length=3
        )
        return Monitor(env)
    return _init

def train_baseline():
    """
    Train baseline RL agent (PPO) on training distribution.
    This will establish what standard RL can achieve.
    """
    
    log_dir = "logs_realistic/"
    models_dir = "models_realistic/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("="*70)
    print("TRAINING BASELINE RL AGENT")
    print("="*70)
    print("Environment: Realistic Causal Push")
    print("Confounders: Mass, Base Friction, Surface Roughness")
    print("Distribution: Training (narrow range)")
    print("="*70)
    
    # Create training environment
    env = DummyVecEnv([make_env(0, mode='train')])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(0, mode='train')])
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="ppo_realistic",
        verbose=1
    )
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device="cpu"
    )
    
    print("\nStarting training...")
    print("Total timesteps: 300,000")
    print("Expected time: ~30-45 minutes on CPU\n")
    
    try:
        model.learn(
            total_timesteps=300000,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        model.save(f"{models_dir}/final_model")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Best model saved: {models_dir}/best_model.zip")
        print(f"Checkpoints saved: {models_dir}/ppo_realistic_*_steps.zip")
        print("\nNext steps:")
        print("  1. Run: python evaluate_realistic.py")
        print("  2. This will test on all distributions and show distribution shift")
        
    except KeyboardInterrupt:
        print("\n--- Training interrupted ---")
        model.save(f"{models_dir}/interrupted_model")
        print("Progress saved!")
        
    except Exception as e:
        print(f"\n--- Training failed: {e} ---")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    train_baseline()