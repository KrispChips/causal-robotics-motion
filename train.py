import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from realistic_causal_env import RealisticCausalPushEnv
from experiment_config import parse_preset, paths, PPO_VARIANTS, ppo_dir

def make_env(rank, mode='train', preset='narrow', history=3):
    """Create environment factory"""
    def _init():
        env = RealisticCausalPushEnv(
            render_mode=None,
            distribution_mode=mode,
            history_length=history,
            preset=preset,
        )
        return Monitor(env)
    return _init

class DetailedLoggingCallback(BaseCallback):
    """
    Custom callback to log detailed training progress
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
    def _on_step(self) -> bool:
        # Check if episode ended
        for idx, done in enumerate(self.locals['dones']):
            if done:
                # Get episode info
                info = self.locals['infos'][idx]
                
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                
                # Track success if available
                if 'is_success' in info:
                    self.episode_successes.append(info['is_success'])
                
                # Print progress every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    recent_successes = self.episode_successes[-10:] if self.episode_successes else []
                    
                    print(f"\n[Episode {len(self.episode_rewards)}]")
                    print(f"  Mean reward (last 10): {np.mean(recent_rewards):.2f}")
                    if recent_successes:
                        print(f"  Success rate (last 10): {np.mean(recent_successes)*100:.1f}%")
        
        return True

def train_baseline(preset='narrow', variant='ppo', timesteps=500000):
    """
    Train a PPO baseline. variant picks history length and training
    distribution (experiment_config.PPO_VARIANTS).
    """
    import torch
    torch.set_num_threads(1)   # several variants train side by side
    spec = PPO_VARIANTS[variant]
    dist, history = spec['dist'], spec['history']
    sub = '' if variant == 'ppo' else variant + '/'
    log_dir = paths(preset)['logs'] + "/" + sub
    models_dir = ppo_dir(preset, variant) + "/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("="*70)
    print("TRAINING BASELINE RL AGENT (IMPROVED)")
    print("="*70)
    print("Environment: Realistic Causal Push (FIXED)")
    probe = RealisticCausalPushEnv(preset=preset)
    print(f"Preset: {preset}   variant: {variant} ({spec['label']}: "
          f"history {history}, distribution '{dist}')")
    print(f"Success threshold: ±{probe.success_threshold*100:.1f}cm")
    print(f"Force range: [{probe.force_min}, {probe.force_max}]N")
    probe.close()
    print("="*70)
    
    # Create training environment
    env = DummyVecEnv([make_env(0, mode=dist, preset=preset, history=history)])

    # Model selection on the agent's own training distribution
    eval_env = DummyVecEnv([make_env(0, mode=dist, preset=preset, history=history)])
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=20,  # INCREASED from 10 for better estimates
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
    
    detailed_callback = DetailedLoggingCallback(verbose=1)
    
    # Create PPO agent with IMPROVED hyperparameters
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
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device="cpu",
        seed=0,
    )
    
    print("\nStarting training...")
    print(f"Total timesteps: {timesteps:,}")
    print("\nMonitoring:")
    print("  - Episode rewards and success rate will be logged")
    print("  - Best model saved based on mean reward")
    print("  - Checkpoints saved every 10K steps")
    print()
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[eval_callback, checkpoint_callback, detailed_callback],
            progress_bar=True
        )
        
        model.save(f"{models_dir}/final_model")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Best model saved: {models_dir}/best_model.zip")
        print(f"Final model saved: {models_dir}/final_model.zip")
        print(f"Checkpoints: {models_dir}/ppo_realistic_*_steps.zip")
        
        # Quick evaluation on final model
        print("\n" + "="*70)
        print("FINAL MODEL EVALUATION (Training Distribution)")
        print("="*70)
        
        test_env = RealisticCausalPushEnv(render_mode=None, distribution_mode=dist,
                                          preset=preset, history_length=history)
        successes = []
        distances = []

        for ep in range(20):
            obs, _ = test_env.reset(seed=ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
            
            successes.append(info['is_success'])
            distances.append(info['final_distance'] * 100)
        
        test_env.close()
        
        print(f"\nResults (20 episodes):")
        print(f"  Success rate: {np.mean(successes)*100:.1f}%")
        print(f"  Mean distance: {np.mean(distances):.2f}cm")
        print(f"  Std distance: {np.std(distances):.2f}cm")
        
        if np.mean(successes) < 0.5:
            print("\n⚠️  WARNING: Success rate < 50% on training distribution!")
            print("   Consider:")
            print("   - Training longer (increase timesteps)")
            print("   - Adjusting reward function")
            print("   - Checking if task is achievable")
        else:
            print("\n✓ Model learned successfully!")
            print(f"  Next: python eval_causal_agent.py --preset {preset}")
        
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
    args = parse_preset("Train a PPO baseline", extra_args=[
        (('--variant',), dict(default='ppo', choices=list(PPO_VARIANTS))),
        (('--timesteps',), dict(type=int, default=500000))])
    train_baseline(args.preset, args.variant, args.timesteps)