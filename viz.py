"""
Visual Comparison: Baseline RL vs Causal Discovery

Watch both agents attempt the same task in PyBullet GUI.
Shows real-time inference, force application, and success/failure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from realistic_causal_env import RealisticCausalPushEnv, action_to_force
from bayesian_causal_agent import CausalDiscoveryPolicy
from stable_baselines3 import PPO
import time
import pybullet as p

class VisualComparison:
    """
    Side-by-side visual comparison of agents
    """
    
    def __init__(self, baseline_path="models_realistic/best_model.zip"):
        self.baseline_model = PPO.load(baseline_path)
        self.causal_policy = CausalDiscoveryPolicy()
        
    def run_episode_with_visualization(self, distribution_mode='test_extrap_high', 
                                       agent_type='causal', show_gui=True):
        """
        Run single episode with PyBullet visualization
        
        Args:
            distribution_mode: Which test distribution
            agent_type: 'baseline' or 'causal'
            show_gui: Show PyBullet GUI window
        """
        # Create environment with GUI
        render_mode = 'human' if show_gui else None
        env = RealisticCausalPushEnv(render_mode=render_mode, distribution_mode=distribution_mode)
        
        obs, _ = env.reset()
        
        # Get ground truth
        true_friction = env.base_friction
        true_mass = env.mass
        true_roughness = env.roughness
        
        print("="*70)
        print(f"VISUAL DEMONSTRATION: {agent_type.upper()} AGENT")
        print("="*70)
        print(f"Distribution: {distribution_mode}")
        print(f"True properties: μ={true_friction:.3f}, m={true_mass:.2f}kg, r={true_roughness:.2f}")
        print("-"*70)
        
        # Setup policy
        if agent_type == 'causal':
            policy = self.causal_policy
            policy.reset()
        else:
            policy = self.baseline_model
        
        # Track trajectory
        positions = []
        velocities = []
        forces_applied = []
        beliefs_friction = []
        beliefs_mass = []
        
        done = False
        step = 0
        
        while not done and step < 20:
            # Get action
            action, _ = policy.predict(obs)
            
            # Convert to force
            force = action_to_force(action[0])
            forces_applied.append(force)
            
            # Get beliefs (causal only)
            if agent_type == 'causal':
                beliefs = policy.agent.get_beliefs()
                beliefs_friction.append(beliefs['friction_mean'])
                beliefs_mass.append(beliefs['mass_mean'])
                
                print(f"\nStep {step+1}:")
                print(f"  Position: {obs[0]*100:.2f}cm, Velocity: {obs[2]*100:.2f}cm/s")
                print(f"  Applied force: {force:.2f}N")
                print(f"  Inferred: μ={beliefs['friction_mean']:.3f}±{beliefs['friction_std']:.3f}, "
                      f"m={beliefs['mass_mean']:.2f}±{beliefs['mass_std']:.2f}kg")
            else:
                print(f"\nStep {step+1}:")
                print(f"  Position: {obs[0]*100:.2f}cm, Velocity: {obs[2]*100:.2f}cm/s")
                print(f"  Applied force: {force:.2f}N")
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track
            positions.append(obs[0])
            velocities.append(obs[2])
            
            done = terminated or truncated
            step += 1
            
            if show_gui:
                time.sleep(0.1)  # Slow down for visualization
        
        # Final results
        print("\n" + "="*70)
        print("EPISODE COMPLETE")
        print("="*70)
        print(f"Final position: {obs[0]*100:.2f}cm (target: 10.0cm)")
        print(f"Final distance: {info['final_distance']*100:.2f}cm")
        print(f"Success: {info['is_success']}")
        
        if agent_type == 'causal' and len(beliefs_friction) > 0:
            print(f"\nFinal inference:")
            print(f"  Friction: {beliefs_friction[-1]:.3f} (true: {true_friction:.3f})")
            print(f"  Mass: {beliefs_mass[-1]:.2f}kg (true: {true_mass:.2f}kg)")
            print(f"  Friction error: {abs(beliefs_friction[-1] - true_friction):.3f}")
            print(f"  Mass error: {abs(beliefs_mass[-1] - true_mass):.2f}kg")
        
        env.close()
        
        return {
            'positions': positions,
            'velocities': velocities,
            'forces': forces_applied,
            'beliefs_friction': beliefs_friction if agent_type == 'causal' else None,
            'beliefs_mass': beliefs_mass if agent_type == 'causal' else None,
            'success': info['is_success'],
            'final_distance': info['final_distance'],
            'true_friction': true_friction,
            'true_mass': true_mass
        }
    
    def compare_side_by_side(self, distribution_mode='test_extrap_high', n_runs=3):
        """
        Run both agents on same distribution and compare trajectories
        """
        print("\n" + "="*70)
        print("SIDE-BY-SIDE COMPARISON")
        print("="*70)
        print(f"Distribution: {distribution_mode}")
        print(f"Running {n_runs} episodes per agent...")
        
        baseline_results = []
        causal_results = []
        
        # Run baseline
        print("\n--- BASELINE RL AGENT ---")
        for i in range(n_runs):
            print(f"\nRun {i+1}/{n_runs}:")
            result = self.run_episode_with_visualization(
                distribution_mode, agent_type='baseline', show_gui=False)
            baseline_results.append(result)
        
        # Run causal
        print("\n--- CAUSAL DISCOVERY AGENT ---")
        for i in range(n_runs):
            print(f"\nRun {i+1}/{n_runs}:")
            result = self.run_episode_with_visualization(
                distribution_mode, agent_type='causal', show_gui=False)
            causal_results.append(result)
        
        # Visualize comparison
        self.plot_comparison(baseline_results, causal_results, distribution_mode)
        
        # Summary statistics
        baseline_success_rate = np.mean([r['success'] for r in baseline_results])
        causal_success_rate = np.mean([r['success'] for r in causal_results])
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"Baseline RL:       {baseline_success_rate*100:.0f}% success")
        print(f"Causal Discovery:  {causal_success_rate*100:.0f}% success")
        print(f"Improvement:       {(causal_success_rate - baseline_success_rate)*100:+.0f}%")
    
    def plot_comparison(self, baseline_results, causal_results, distribution_mode):
        """Create detailed comparison plots"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Position trajectories
        ax1 = fig.add_subplot(gs[0, :])
        for i, result in enumerate(baseline_results):
            ax1.plot(result['positions'], 'o-', alpha=0.5, color='orange', 
                    label='Baseline' if i == 0 else '')
        for i, result in enumerate(causal_results):
            ax1.plot(result['positions'], 's-', alpha=0.5, color='green',
                    label='Causal' if i == 0 else '')
        
        ax1.axhline(y=0.1, color='green', linestyle='--', linewidth=2, label='Target')
        ax1.axhline(y=0.09, color='green', linestyle=':', alpha=0.5)
        ax1.axhline(y=0.11, color='green', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Position (m)')
        ax1.set_title(f'Block Position Over Time ({distribution_mode})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Velocity trajectories
        ax2 = fig.add_subplot(gs[1, 0])
        for result in baseline_results:
            ax2.plot(result['velocities'], alpha=0.5, color='orange')
        for result in causal_results:
            ax2.plot(result['velocities'], alpha=0.5, color='green')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Force application
        ax3 = fig.add_subplot(gs[1, 1])
        for result in baseline_results:
            ax3.plot(result['forces'], alpha=0.5, color='orange')
        for result in causal_results:
            ax3.plot(result['forces'], alpha=0.5, color='green')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Applied Force (N)')
        ax3.set_title('Force Application')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final distances
        ax4 = fig.add_subplot(gs[1, 2])
        baseline_dists = [r['final_distance']*100 for r in baseline_results]
        causal_dists = [r['final_distance']*100 for r in causal_results]
        
        positions = [1, 2]
        ax4.boxplot([baseline_dists, causal_dists], positions=positions,
                   labels=['Baseline', 'Causal'])
        ax4.axhline(y=1.0, color='green', linestyle='--', label='Success threshold')
        ax4.set_ylabel('Final Distance to Target (cm)')
        ax4.set_title('Final Distance Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Belief evolution (causal only)
        ax5 = fig.add_subplot(gs[2, 0])
        if causal_results[0]['beliefs_friction'] is not None:
            for result in causal_results:
                if result['beliefs_friction']:
                    ax5.plot(result['beliefs_friction'], alpha=0.6)
            
            true_friction = causal_results[0]['true_friction']
            ax5.axhline(y=true_friction, color='red', linestyle='--', 
                       label=f'True μ={true_friction:.3f}')
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Inferred Friction (μ)')
            ax5.set_title('Causal: Friction Belief Evolution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Mass belief evolution
        ax6 = fig.add_subplot(gs[2, 1])
        if causal_results[0]['beliefs_mass'] is not None:
            for result in causal_results:
                if result['beliefs_mass']:
                    ax6.plot(result['beliefs_mass'], alpha=0.6)
            
            true_mass = causal_results[0]['true_mass']
            ax6.axhline(y=true_mass, color='red', linestyle='--',
                       label=f'True m={true_mass:.2f}kg')
            ax6.set_xlabel('Step')
            ax6.set_ylabel('Inferred Mass (kg)')
            ax6.set_title('Causal: Mass Belief Evolution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Plot 7: Success comparison
        ax7 = fig.add_subplot(gs[2, 2])
        baseline_successes = [r['success'] for r in baseline_results]
        causal_successes = [r['success'] for r in causal_results]
        
        x = np.arange(len(baseline_results))
        width = 0.35
        
        ax7.bar(x - width/2, baseline_successes, width, label='Baseline', 
               alpha=0.7, color='orange')
        ax7.bar(x + width/2, causal_successes, width, label='Causal',
               alpha=0.7, color='green')
        ax7.set_xlabel('Run')
        ax7.set_ylabel('Success (0/1)')
        ax7.set_title('Success Rate by Run')
        ax7.set_xticks(x)
        ax7.set_xticklabels([f'Run {i+1}' for i in range(len(baseline_results))])
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(-0.1, 1.1)
        
        plt.suptitle(f'Baseline vs Causal Comparison: {distribution_mode}', 
                    fontsize=16, fontweight='bold')
        plt.savefig(f'comparison_{distribution_mode}.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved as 'comparison_{distribution_mode}.png'")
        plt.close()


def interactive_demo():
    """
    Interactive demonstration menu
    """
    print("\n" + "="*70)
    print("VISUAL COMPARISON TOOL")
    print("="*70)
    print("\nOptions:")
    print("1. Watch single episode in PyBullet GUI (Baseline)")
    print("2. Watch single episode in PyBullet GUI (Causal)")
    print("3. Compare multiple runs (no GUI, generate plots)")
    print("4. Compare ALL distributions")
    print("5. Exit")
    
    comparator = VisualComparison()
    
    while True:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            print("\nDistributions:")
            print("  a) Training")
            print("  b) High Friction")
            print("  c) Low Friction")
            print("  d) Adversarial")
            
            dist_choice = input("Choose distribution (a-d): ").strip().lower()
            dist_map = {
                'a': 'train',
                'b': 'test_extrap_high',
                'c': 'test_extrap_low',
                'd': 'test_adversarial'
            }
            
            if dist_choice in dist_map:
                comparator.run_episode_with_visualization(
                    distribution_mode=dist_map[dist_choice],
                    agent_type='baseline',
                    show_gui=True
                )
            
        elif choice == '2':
            print("\nDistributions:")
            print("  a) Training")
            print("  b) High Friction")
            print("  c) Low Friction")
            print("  d) Adversarial")
            
            dist_choice = input("Choose distribution (a-d): ").strip().lower()
            dist_map = {
                'a': 'train',
                'b': 'test_extrap_high',
                'c': 'test_extrap_low',
                'd': 'test_adversarial'
            }
            
            if dist_choice in dist_map:
                comparator.run_episode_with_visualization(
                    distribution_mode=dist_map[dist_choice],
                    agent_type='causal',
                    show_gui=True
                )
        
        elif choice == '3':
            print("\nDistributions:")
            print("  a) Training")
            print("  b) High Friction")
            print("  c) Low Friction")
            print("  d) Adversarial")
            
            dist_choice = input("Choose distribution (a-d): ").strip().lower()
            dist_map = {
                'a': 'train',
                'b': 'test_extrap_high',
                'c': 'test_extrap_low',
                'd': 'test_adversarial'
            }
            
            if dist_choice in dist_map:
                n_runs = int(input("Number of runs per agent (default 3): ") or "3")
                comparator.compare_side_by_side(
                    distribution_mode=dist_map[dist_choice],
                    n_runs=n_runs
                )
        
        elif choice == '4':
            print("\nComparing all distributions (3 runs each)...")
            distributions = ['train', 'test_extrap_high', 'test_extrap_low', 'test_adversarial']
            
            for dist in distributions:
                print(f"\n{'='*70}")
                print(f"Distribution: {dist}")
                print('='*70)
                comparator.compare_side_by_side(distribution_mode=dist, n_runs=3)
        
        elif choice == '5':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")


if __name__ == "__main__":
    interactive_demo()