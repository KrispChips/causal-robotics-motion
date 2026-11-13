# Experiment: structured confounder inference vs. structure-free learning

## What the results can and cannot claim

**The causal agent is given the causal structure and infers parameters within it.**
It knows which variables cause what (friction and mass → block motion, given
the applied force) and the functional form (Coulomb friction). It infers the
*values* of friction μ and mass m from interaction. In control terms this is
Bayesian system identification with a known model class.

| Claim | Supported? |
|---|---|
| Knowing the physical structure and inferring hidden parameters gives robustness to distribution shift that a structure-free learner trained on a narrow distribution lacks | Yes, within this simulator (see results) |
| Causal *discovery* (learning the graph or equations from data) beats RL | **Not tested.** The structure is given, not discovered. The PC-algorithm and VAE stages of the plan would test this |
| The advantage would hold on a real robot | **Not tested.** The agent's physics model was validated against the same simulator it is scored in, so these are best-case numbers for a structured agent |

## What each agent gets

| | Observation | Prior knowledge | Data / tuning |
|---|---|---|---|
| **PPO** | Current state + last 3 steps of history | None | 500k steps on the training distribution |
| **Model-based RL** | (velocity, force) | None | 1,000 random-force training episodes; planner settings tuned on train |
| **Causal (passive / active)** | Same as PPO; uses full episode history | Coulomb model; a fixed prior μ∈[0.02,1.2], m∈[0.1,4] kg that is wider than any train or test range | Noise and planner settings tuned on train only |
| **Stacking: PPO** | The block's push log | None | 20k stacking episodes on the training distribution |
| **Stacking: Causal** | The same push log | Coulomb grasp friction (friction cone) plus the robot's own grip force, pad friction and motion plan | No stacking training; noise level from push tuning |

Every agent knows the actuator limits (PPO through its action space). None is
told the train or test confounder ranges.

### Hyperparameter protocol (`tune_on_train.py`)

- **Observation noise σ:** empirical Bayes. Choose the σ that maximises the marginal
  likelihood of 300 unlabelled training-distribution episodes. No true μ/m
  labels are used. Result: narrow 0.0075, pdf 0.03 m/s.
- **Planner and exploration settings:** grid search on 100 training-distribution
  episodes per setting, with the planner RNG varied per episode. Done separately
  for the causal agent and MBRL.
- **Seeds:** tuning uses seeds 2,000,000+; evaluation uses 1000+. The test
  distributions are never instantiated during tuning.

### Leaks found and removed (2026-11-13)

1. **Noise σ:** was hand-set from the high-friction *test* set's roughness.
2. **Planner and exploration settings:** were chosen while test results were visible.
3. **Stacking drop rule:** the agent planned with the env's own "dropped" threshold.
   It now uses the textbook friction-cone condition (any slip counts as failure).
   The env's drop check became physical: the block has lost contact with both
   fingers.
4. **Tuning procedure:** the first tuning run scored every setting on one fixed
   planner-RNG stream, so it partly selected for luck. It now varies the stream
   per episode and uses 100 episodes per setting.

## Results (50 paired episodes per set; 200 for full-range stacking)

### Push, `narrow` preset (8 N actuator, ±1 cm)

| Success % | Train | Interp | High μ | Low μ | Adv |
|---|---|---|---|---|---|
| PPO | 100 | 100 | 96 | **0** | 100 |
| Model-based RL | 60 | 100 | 0 | 0 | 0 |
| Causal (passive) | 100 | 100 | 100 | 100 | 96 |
| Causal (active) | 100 | 100 | 100 | 100 | 96 |

### Push, `pdf` preset (project-plan ranges, 20 N, ±2 cm)

| Success % | Train | Interp | High μ | Low μ | Full range |
|---|---|---|---|---|---|
| PPO | 100 | 100 | 100 | 100 | **78** |
| Model-based RL | 68 | 100 | 0 | 100 | 44 |
| Causal (passive) | 100 | 100 | 100 | 100 | 100 |
| Causal (active) | 100 | 100 | 100 | 100 | 100 |

**PPO does not fail on the plan's own test sets.** The plan pairs high
friction with low mass (and low with high), so the force needed to start
sliding (μ·m·g ≈ 4.4–4.7 N) stays inside the training range of 2.4–8.8 N.
Distribution shift only bites on the full-range set.

### Push → stack transfer (`pdf`)

| | Train | Interp | High μ | Low μ | Full range |
|---|---|---|---|---|---|
| PPO success / reward | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | **97 / 7.54** (3% dropped) |
| Causal success / reward | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | 100 / **6.73** | 100 / 7.79 |
| Causal rule with true μ, m | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | 100 / 7.59 | 100 / 7.90 |

- **With real Panda limits, stacking barely depends on friction and mass.**
  Only heavy blocks near μ≈0.2 slip out at full speed.
- **PPO never saw a drop in 20,000 training episodes,** so it cannot learn to
  slow down for those blocks.
- **The causal agent is over-cautious.** Its no-slip rule plus posterior
  uncertainty (from the large tuned σ) make it carry the Low-μ block at
  3.8 m/s² when full speed was safe. That costs reward (6.73 against PPO's 7.92).
- **Verdict on H2 (transfer):** weakly supported. The benefit is real but confined
  to the slippery-heavy corner, and it is paid for with slowness elsewhere.

### Effect of removing the leaks

Push results barely moved: passive narrow-adversarial went from 90 to 96, and
active narrow-adversarial from 100 to 96. Stacking became more conservative
(Low-μ reward fell from 7.62 to 6.73), because the agent no longer knows how
much slip the env tolerates.

## Known remaining weaknesses (not yet addressed)

- **PPO's memory:** PPO sees only 3 steps of history while the causal agent uses
  all of them. A recurrent PPO is the fair model-free comparison.
- **Training distribution:** PPO trains only on the narrow distribution, as the
  plan specifies. A domain-randomized PPO trained on the full range would show
  whether the advantage comes from structure or merely from data coverage.
- **MBRL inputs:** MBRL's model sees only (velocity, force), so it cannot infer μ or
  m even in principle. A history-conditioned model is the fair version.
- **Model correctness:** there is no test of the causal agent with a *wrong*
  model, for example ignoring speed-dependent friction or misjudging grip force.

## Reproducing

```
python test_sim_physics.py                      # 21 simulator checks
python train.py --preset pdf                    # PPO
python mbrl_baseline.py --preset pdf            # model-based RL
python tune_on_train.py --preset pdf            # all planner hyperparameters (train only)
python eval_causal_agent.py --preset pdf        # push comparison
python stack_transfer.py bank|train|eval --preset pdf
```
