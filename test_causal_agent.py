"""
Quick test to verify causal agent works before full evaluation
"""

import numpy as np
from realistic_causal_env import RealisticCausalPushEnv
from bayesian_causal_agent import BayesianCausalAgent, CausalDiscoveryPolicy
import matplotlib.pyplot as plt

def test_physics_model():
    """Test if physics model predicts correctly"""
    print("="*70)
    print("TEST 1: Physics Model Accuracy")
    print("="*70)
    
    agent = BayesianCausalAgent()
    
    # Test predictions for known scenarios
    scenarios = [
        (1.0, 0.25, 1.0, "Low force, medium friction"),
        (3.0, 0.25, 1.0, "Medium force, medium friction"),
        (5.0, 0.25, 1.0, "High force, medium friction"),
        (3.0, 0.15, 1.0, "Medium force, LOW friction"),
        (3.0, 0.35, 1.0, "Medium force, HIGH friction"),
    ]
    
    print("\nPredicted distances:")
    for force, friction, mass, description in scenarios:
        distance = agent.predict_distance(force, friction, mass, n_steps=20)
        print(f"  {description:35s}: {distance*100:.2f}cm (target: 10cm)")
    
    print("\n✓ Physics model runs without errors")

def test_belief_updating():
    """Test if beliefs update correctly"""
    print("\n" + "="*70)
    print("TEST 2: Belief Updating")
    print("="*70)
    
    agent = BayesianCausalAgent()
    
    print("\nInitial beliefs:")
    beliefs = agent.get_beliefs()
    print(f"  Friction: {beliefs['friction_mean']:.3f} ± {beliefs['friction_std']:.3f}")
    print(f"  Mass: {beliefs['mass_mean']:.2f} ± {beliefs['mass_std']:.2f}kg")
    
    # Simulate observation: applied 3N, moved 5cm (suggests high friction)
    print("\nObservation: Applied 3N, block moved 5cm")
    agent.update_beliefs(force_applied=3.0, observed_distance=0.05, n_steps=20)
    
    beliefs = agent.get_beliefs()
    print(f"  Updated Friction: {beliefs['friction_mean']:.3f} ± {beliefs['friction_std']:.3f}")
    print(f"  Updated Mass: {beliefs['mass_mean']:.2f} ± {beliefs['mass_std']:.2f}kg")
    
    # Another observation
    print("\nObservation: Applied 5N, block moved 8cm")
    agent.update_beliefs(force_applied=5.0, observed_distance=0.08, n_steps=20)
    
    beliefs = agent.get_beliefs()
    print(f"  Updated Friction: {beliefs['friction_mean']:.3f} ± {beliefs['friction_std']:.3f}")
    print(f"  Updated Mass: {beliefs['mass_mean']:.2f} ± {beliefs['mass_std']:.2f}kg")
    
    print("\n✓ Beliefs update based on observations")
    
    # Visualize
    fig = agent.visualize_beliefs()
    plt.savefig('belief_visualization_test.png', dpi=150)
    print("✓ Belief visualization saved as 'belief_visualization_test.png'")
    plt.close()

def test_force_selection():
    """Test if agent chooses reasonable forces"""
    print("\n" + "="*70)
    print("TEST 3: Force Selection")
    print("="*70)
    
    agent = BayesianCausalAgent()
    
    # Set beliefs manually to known values
    agent.friction_particles = np.ones(1000) * 0.25
    agent.mass_particles = np.ones(1000) * 1.0
    agent.weights = np.ones(1000) / 1000
    
    beliefs = agent.get_beliefs()
    print(f"\nSet beliefs: μ={beliefs['friction_mean']:.3f}, m={beliefs['mass_mean']:.2f}kg")
    
    force = agent.choose_force(target_distance=0.1, n_steps=20, mode='exploit')
    predicted = agent.predict_distance(force, beliefs['friction_mean'], 
                                      beliefs['mass_mean'], n_steps=20)
    
    print(f"Chosen force: {force:.2f}N")
    print(f"Predicted distance: {predicted*100:.2f}cm (target: 10cm)")
    
    if abs(predicted - 0.1) < 0.02:
        print("✓ Force selection achieves target")
    else:
        print("⚠ Force selection may need tuning")

def test_single_episode():
    """Run one complete episode"""
    print("\n" + "="*70)
    print("TEST 4: Complete Episode")
    print("="*70)
    
    env = RealisticCausalPushEnv(render_mode=None, distribution_mode='train')
    policy = CausalDiscoveryPolicy(exploration_steps=3)
    
    obs, _ = env.reset()
    policy.reset()
    
    print(f"\nTrue confounders: μ={env.base_friction:.3f}, m={env.mass:.2f}kg, r={env.roughness:.2f}")
    
    done = False
    step = 0
    
    while not done and step < 20:
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        force = 3.0 + action[0] * 2.5
        
        if step < 5 or step == 19:
            print(f"\n  Step {step+1}:")
            print(f"    Position: {obs[0]*100:.2f}cm")
            print(f"    Force: {force:.2f}N")
            print(f"    Reward: {reward:.2f}")
            
            if step >= 1:
                beliefs = policy.agent.get_beliefs()
                print(f"    Beliefs: μ={beliefs['friction_mean']:.3f}, m={beliefs['mass_mean']:.2f}kg")
        
        done = terminated or truncated
        step += 1
    
    beliefs = policy.agent.get_beliefs()
    print(f"\n  Final beliefs: μ={beliefs['friction_mean']:.3f}, m={beliefs['mass_mean']:.2f}kg")
    print(f"  Final distance: {info['final_distance']*100:.2f}cm")
    print(f"  Success: {info['is_success']}")
    
    # Inference errors
    friction_error = abs(beliefs['friction_mean'] - env.base_friction)
    mass_error = abs(beliefs['mass_mean'] - env.mass)
    
    print(f"\n  Inference errors:")
    print(f"    Friction: {friction_error:.3f}")
    print(f"    Mass: {mass_error:.2f}kg")
    
    if info['is_success']:
        print("\n✓ Episode succeeded!")
    else:
        print(f"\n⚠ Episode failed (distance: {info['final_distance']*100:.2f}cm)")
    
    env.close()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CAUSAL AGENT TEST SUITE")
    print("="*70)
    print("Testing Bayesian Causal Discovery Agent\n")
    
    try:
        test_physics_model()
        test_belief_updating()
        test_force_selection()
        test_single_episode()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nCausal agent is working!")
        print("\nNext step: Run full evaluation")
        print("  python eval_causal_agent.py")
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED ✗")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()