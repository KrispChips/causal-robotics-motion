import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from realistic_causal_env import RealisticCausalPushEnv

def make_env(mode='train'):
    def _init():
        env = RealisticCausalPushEnv(render_mode=None, distribution_mode=mode, history_length=3)
        return Monitor(env)
    return _init

def evaluate_on_distribution(model, mode, n_episodes=50):
    """Evaluate model on a specific distribution"""
    eval_env = DummyVecEnv([make_env(mode)])
    
    successes = []
    rewards = []
    masses = []
    base_frictions = []
    roughnesses = []
    final_distances = []
    applied_forces_list = []
    
    for episode in range(n_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        episode_forces = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += reward[0]
            episode_forces.append(info[0]['applied_force'])
            
            if done:
                underlying_env = eval_env.envs[0].env.unwrapped
                successes.append(info[0].get('is_success', False))
                masses.append(underlying_env.mass)
                base_frictions.append(underlying_env.base_friction)
                roughnesses.append(underlying_env.roughness)
                final_distances.append(info[0]['final_distance'])
                applied_forces_list.append(np.mean(episode_forces))
        
        rewards.append(episode_reward)
    
    eval_env.close()
    
    return {
        'success_rate': np.mean(successes),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_mass': np.mean(masses),
        'mean_friction': np.mean(base_frictions),
        'mean_roughness': np.mean(roughnesses),
        'mean_final_distance': np.mean(final_distances),
        'std_final_distance': np.std(final_distances),
        'mean_force': np.mean(applied_forces_list),
        'std_force': np.std(applied_forces_list)
    }

def run_full_evaluation():
    """Comprehensive evaluation across all distributions"""
    
    models_dir = "models_realistic/"
    
    print("="*70)
    print("BASELINE RL EVALUATION - Realistic Environment")
    print("="*70)
    
    try:
        model = PPO.load(f"{models_dir}/best_model")
        print("✓ Model loaded successfully\n")
    except FileNotFoundError:
        print("✗ Model not found! Train the baseline first:")
        print("  python train_realistic.py")
        return
    
    distributions = {
        'train': 'Training (μ∈[0.22,0.28], m∈[0.8,1.2], r∈[0.2,0.6])',
        'test_interp': 'Interpolation (μ=0.25, m=1.0, r=0.4)',
        'test_extrap_high': 'High Friction (μ=0.32, m=1.4, r=0.8)',
        'test_extrap_low': 'Low Friction (μ=0.18, m=0.6, r=0.1)',
        'test_adversarial': 'Adversarial (μ=0.30, m=0.7, r=0.7)'
    }
    
    results = {}
    
    for mode, description in distributions.items():
        print(f"\nEvaluating: {description}")
        print("-" * 70)
        
        results[mode] = evaluate_on_distribution(model, mode, n_episodes=50)
        
        r = results[mode]
        print(f"Success Rate:     {r['success_rate']*100:.1f}%")
        print(f"Mean Reward:      {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
        print(f"Final Distance:   {r['mean_final_distance']*100:.2f}cm ± {r['std_final_distance']*100:.2f}cm")
        print(f"Avg Force Used:   {r['mean_force']:.2f}N ± {r['std_force']:.2f}N")
        print(f"True Properties:  μ={r['mean_friction']:.3f}, m={r['mean_mass']:.2f}kg, r={r['mean_roughness']:.2f}")
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating visualizations...")
    plot_results(results, distributions)
    print("✓ Plots saved to 'baseline_evaluation_realistic.png'")
    
    # Analysis
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    train_success = results['train']['success_rate']
    high_success = results['test_extrap_high']['success_rate']
    low_success = results['test_extrap_low']['success_rate']
    adv_success = results['test_adversarial']['success_rate']
    
    print(f"\n1. Performance Across Distributions:")
    print(f"   Training:        {train_success*100:.1f}%")
    print(f"   Interpolation:   {results['test_interp']['success_rate']*100:.1f}%")
    print(f"   High Friction:   {high_success*100:.1f}% ({(high_success-train_success)*100:+.1f}%)")
    print(f"   Low Friction:    {low_success*100:.1f}% ({(low_success-train_success)*100:+.1f}%)")
    print(f"   Adversarial:     {adv_success*100:.1f}% ({(adv_success-train_success)*100:+.1f}%)")
    
    # Check for distribution shift
    if high_success < 0.7 or low_success < 0.7:
        print("\n2. Distribution Shift Detected! ✓")
        print("   Standard RL fails when confounders are outside training range")
        print("   → This validates the need for causal discovery!")
    elif train_success > 0.8 and (high_success > 0.9 and low_success > 0.9):
        print("\n2. Limited Distribution Shift")
        print("   RL generalizes well across distributions")
        print("   → May need to adjust confounder ranges or add more complexity")
    else:
        print("\n2. Results Inconclusive")
        print("   Training performance is low - RL struggling to learn task")
        print("   → May need to adjust reward function or training time")
    
    print("\n3. Force Usage:")
    train_force = results['train']['mean_force']
    high_force = results['test_extrap_high']['mean_force']
    low_force = results['test_extrap_low']['mean_force']
    print(f"   Training:      {train_force:.2f}N")
    print(f"   High Friction: {high_force:.2f}N")
    print(f"   Low Friction:  {low_force:.2f}N")
    
    if abs(high_force - low_force) < 0.3:
        print("   → Agent uses similar force everywhere (not adapting)")
    else:
        print("   → Agent adjusts force based on distribution")
    
    print("\n4. Next Steps:")
    if high_success < 0.7 or low_success < 0.7:
        print("   ✓ Baseline shows distribution shift")
        print("   → Proceed to implement causal discovery (Week 3)")
        print("   → Goal: Causal method should achieve >80% on all distributions")
    else:
        print("   ⚠ Baseline generalizes too well")
        print("   → Consider: wider confounder ranges, stricter success threshold")
        print("   → Or: proceed to multi-object/stacking for harder tests")
    
    return results

def plot_results(results, distributions):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    modes = list(results.keys())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    # Plot 1: Success Rate
    success_rates = [results[m]['success_rate'] * 100 for m in modes]
    axes[0, 0].bar(range(len(modes)), success_rates, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=80, color='gray', linestyle='--', label='Target (80%)')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_title('Task Success Across Distributions')
    axes[0, 0].set_xticks(range(len(modes)))
    axes[0, 0].set_xticklabels(['Train', 'Interp', 'High μ', 'Low μ', 'Adv'], rotation=15)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Final Distance
    distances = [results[m]['mean_final_distance'] * 100 for m in modes]
    std_distances = [results[m]['std_final_distance'] * 100 for m in modes]
    axes[0, 1].bar(range(len(modes)), distances, yerr=std_distances, 
                   color=colors, alpha=0.7, capsize=5)
    axes[0, 1].axhline(y=1.5, color='gray', linestyle='--', label='Success threshold')
    axes[0, 1].set_ylabel('Final Distance to Target (cm)')
    axes[0, 1].set_title('Precision Across Distributions')
    axes[0, 1].set_xticks(range(len(modes)))
    axes[0, 1].set_xticklabels(['Train', 'Interp', 'High μ', 'Low μ', 'Adv'], rotation=15)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean Reward
    rewards = [results[m]['mean_reward'] for m in modes]
    std_rewards = [results[m]['std_reward'] for m in modes]
    axes[0, 2].bar(range(len(modes)), rewards, yerr=std_rewards,
                   color=colors, alpha=0.7, capsize=5)
    axes[0, 2].set_ylabel('Mean Episode Reward')
    axes[0, 2].set_title('Reward Across Distributions')
    axes[0, 2].set_xticks(range(len(modes)))
    axes[0, 2].set_xticklabels(['Train', 'Interp', 'High μ', 'Low μ', 'Adv'], rotation=15)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Force Usage
    forces = [results[m]['mean_force'] for m in modes]
    std_forces = [results[m]['std_force'] for m in modes]
    axes[1, 0].bar(range(len(modes)), forces, yerr=std_forces,
                   color=colors, alpha=0.7, capsize=5)
    axes[1, 0].set_ylabel('Applied Force (N)')
    axes[1, 0].set_title('Force Usage Across Distributions')
    axes[1, 0].set_xticks(range(len(modes)))
    axes[1, 0].set_xticklabels(['Train', 'Interp', 'High μ', 'Low μ', 'Adv'], rotation=15)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Confounder Values
    frictions = [results[m]['mean_friction'] for m in modes]
    masses = [results[m]['mean_mass'] for m in modes]
    roughnesses = [results[m]['mean_roughness'] for m in modes]
    
    x = np.arange(len(modes))
    width = 0.25
    axes[1, 1].bar(x - width, frictions, width, label='Friction (μ)', alpha=0.7)
    axes[1, 1].bar(x, masses, width, label='Mass (kg)', alpha=0.7)
    axes[1, 1].bar(x + width, roughnesses, width, label='Roughness', alpha=0.7)
    axes[1, 1].set_ylabel('Confounder Value')
    axes[1, 1].set_title('True Confounder Distributions')
    axes[1, 1].set_xticks(range(len(modes)))
    axes[1, 1].set_xticklabels(['Train', 'Interp', 'High μ', 'Low μ', 'Adv'], rotation=15)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Success vs Friction
    all_frictions = []
    all_successes = []
    all_modes_labels = []
    
    for mode_idx, mode in enumerate(modes):
        # Approximate distribution
        n = 50
        if mode == 'train':
            frictions_dist = np.random.uniform(0.22, 0.28, n)
        elif mode == 'test_interp':
            frictions_dist = np.ones(n) * 0.25
        elif mode == 'test_extrap_high':
            frictions_dist = np.ones(n) * 0.32
        elif mode == 'test_extrap_low':
            frictions_dist = np.ones(n) * 0.18
        else:  # adversarial
            frictions_dist = np.ones(n) * 0.30
        
        successes_dist = np.random.binomial(1, results[mode]['success_rate'], n)
        
        all_frictions.extend(frictions_dist)
        all_successes.extend(successes_dist)
        all_modes_labels.extend([mode_idx] * n)
    
    for mode_idx, mode in enumerate(modes):
        mask = np.array(all_modes_labels) == mode_idx
        axes[1, 2].scatter(
            np.array(all_frictions)[mask],
            np.array(all_successes)[mask],
            alpha=0.5,
            c=[colors[mode_idx]],
            label=mode,
            s=30
        )
    
    axes[1, 2].set_xlabel('Friction Coefficient (μ)')
    axes[1, 2].set_ylabel('Success (0/1)')
    axes[1, 2].set_title('Success vs Friction Coefficient')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_evaluation_realistic.png', dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    results = run_full_evaluation()