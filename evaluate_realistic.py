import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from realistic_causal_env import RealisticCausalPushEnv, PRESETS
from experiment_config import parse_preset, paths, labels, figure


def evaluate_on_distribution(model, mode, n_episodes=50, seed_offset=1000, preset='narrow'):
    """Evaluate model on a specific distribution.

    Uses a plain (non-vectorised) env: a DummyVecEnv auto-resets on done, so
    reading env.mass/base_friction after the last step returned the NEXT
    episode's freshly sampled confounders. The true confounders are taken from
    the terminal step's info instead. Episodes are seeded with the same
    seed_offset as eval_causal_agent.py so results are paired across agents.
    """
    env = RealisticCausalPushEnv(render_mode=None, distribution_mode=mode, history_length=3,
                                 preset=preset)

    successes = []
    rewards = []
    masses = []
    base_frictions = []
    roughnesses = []
    final_distances = []
    applied_forces_list = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + episode)
        done = False
        episode_reward = 0.0
        episode_forces = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_forces.append(info['applied_force'])

        successes.append(info['is_success'])
        masses.append(info['true_mass'])
        base_frictions.append(info['true_base_friction'])
        roughnesses.append(info['true_roughness'])
        final_distances.append(info['final_distance'])
        applied_forces_list.append(np.mean(episode_forces))
        rewards.append(episode_reward)

    env.close()

    return {
        'frictions': np.array(base_frictions),
        'final_distances': np.array(final_distances),
        'successes': np.array(successes),
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

def describe(preset, mode):
    spec = PRESETS[preset]['modes'][mode]
    fmt = lambda lo, hi, f: f"{lo:{f}}" if lo == hi else f"[{lo:{f}},{hi:{f}}]"
    return (f"{labels(preset)[mode]} (μ={fmt(*spec['friction'], '.2f')}, "
            f"m={fmt(*spec['mass'], '.1f')}kg, r={fmt(*spec['roughness'], '.1f')})")


def run_full_evaluation(preset='narrow'):
    """Comprehensive evaluation across all distributions"""

    models_dir = paths(preset)['models']
    threshold_cm = PRESETS[preset]['success_threshold'] * 100

    print("="*70)
    print("BASELINE RL EVALUATION - Realistic Environment")
    print("="*70)
    
    try:
        model = PPO.load(f"{models_dir}/best_model")
        print("✓ Model loaded successfully\n")
    except FileNotFoundError:
        print("✗ Model not found! Train the baseline first:")
        print(f"  python train.py --preset {preset}")
        return

    distributions = {m: describe(preset, m) for m in labels(preset)}

    results = {}
    
    for mode, description in distributions.items():
        print(f"\nEvaluating: {description}")
        print("-" * 70)
        
        results[mode] = evaluate_on_distribution(model, mode, n_episodes=50, preset=preset)
        
        r = results[mode]
        print(f"Success Rate:     {r['success_rate']*100:.1f}%")
        print(f"Mean Reward:      {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")
        print(f"Final Distance:   {r['mean_final_distance']*100:.2f}cm ± {r['std_final_distance']*100:.2f}cm")
        print(f"Avg Force Used:   {r['mean_force']:.2f}N ± {r['std_force']:.2f}N")
        print(f"True Properties:  μ={r['mean_friction']:.3f}, m={r['mean_mass']:.2f}kg, r={r['mean_roughness']:.2f}")
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating visualizations...")
    out_png = figure('baseline_evaluation_realistic.png', preset)
    plot_results(results, labels(preset), threshold_cm, out_png)
    print(f"✓ Plots saved to '{out_png}'")
    
    # Analysis
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    
    train_success = results['train']['success_rate']
    high_success = results['test_extrap_high']['success_rate']
    low_success = results['test_extrap_low']['success_rate']

    print(f"\n1. Performance Across Distributions:")
    print(f"   Training:        {train_success*100:.1f}%")
    print(f"   Interpolation:   {results['test_interp']['success_rate']*100:.1f}%")
    print(f"   High Friction:   {high_success*100:.1f}% ({(high_success-train_success)*100:+.1f}%)")
    print(f"   Low Friction:    {low_success*100:.1f}% ({(low_success-train_success)*100:+.1f}%)")
    for extra in [m for m in results if m not in ('train', 'test_interp', 'test_extrap_high', 'test_extrap_low')]:
        s_extra = results[extra]['success_rate']
        print(f"   {labels(preset)[extra]+':':17s}{s_extra*100:.1f}% ({(s_extra-train_success)*100:+.1f}%)")

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

def plot_results(results, mode_labels, threshold_cm, out_png):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    modes = list(results.keys())
    tick_labels = [mode_labels[m] for m in modes]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    # Plot 1: Success Rate
    success_rates = [results[m]['success_rate'] * 100 for m in modes]
    axes[0, 0].bar(range(len(modes)), success_rates, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=80, color='gray', linestyle='--', label='Target (80%)')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_title('Task Success Across Distributions')
    axes[0, 0].set_xticks(range(len(modes)))
    axes[0, 0].set_xticklabels(tick_labels, rotation=15)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Final Distance
    distances = [results[m]['mean_final_distance'] * 100 for m in modes]
    std_distances = [results[m]['std_final_distance'] * 100 for m in modes]
    axes[0, 1].bar(range(len(modes)), distances, yerr=std_distances, 
                   color=colors, alpha=0.7, capsize=5)
    axes[0, 1].axhline(y=threshold_cm, color='gray', linestyle='--',
                       label=f'Success threshold (±{threshold_cm:.0f}cm)')
    axes[0, 1].set_ylabel('Final Distance to Target (cm)')
    axes[0, 1].set_title('Precision Across Distributions')
    axes[0, 1].set_xticks(range(len(modes)))
    axes[0, 1].set_xticklabels(tick_labels, rotation=15)
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
    axes[0, 2].set_xticklabels(tick_labels, rotation=15)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Force Usage
    forces = [results[m]['mean_force'] for m in modes]
    std_forces = [results[m]['std_force'] for m in modes]
    axes[1, 0].bar(range(len(modes)), forces, yerr=std_forces,
                   color=colors, alpha=0.7, capsize=5)
    axes[1, 0].set_ylabel('Applied Force (N)')
    axes[1, 0].set_title('Force Usage Across Distributions')
    axes[1, 0].set_xticks(range(len(modes)))
    axes[1, 0].set_xticklabels(tick_labels, rotation=15)
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
    axes[1, 1].set_xticklabels(tick_labels, rotation=15)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: per-episode final error vs the episode's true friction
    # (real episodes; the old version plotted randomly generated points)
    for mode_idx, mode in enumerate(modes):
        r = results[mode]
        axes[1, 2].scatter(
            r['frictions'],
            r['final_distances'] * 100,
            alpha=0.5,
            c=[colors[mode_idx]],
            label=mode,
            s=30
        )
    axes[1, 2].axhline(y=threshold_cm, color='gray', linestyle='--')
    axes[1, 2].set_xlabel('True friction coefficient (μ)')
    axes[1, 2].set_ylabel('Final distance to target (cm)')
    axes[1, 2].set_yscale('symlog', linthresh=1.0)
    max_err = max(results[m]['final_distances'].max() for m in modes) * 100
    axes[1, 2].set_ylim(0, max(2.0, max_err * 1.5))
    axes[1, 2].set_title('Per-episode error vs friction')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    results = run_full_evaluation(parse_preset("Evaluate the PPO baseline").preset)