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
| **PPO (full history)** | Current state + the whole episode (20 steps), the same information the causal agent uses | None | 1M steps on the training distribution |
| **PPO (DR)** / **(DR, full hist)** | 3-step / whole-episode history | None, but trained on every test condition (see below) | 1M steps on the domain-randomized box |
| **Model-based RL** | (velocity, force) | None | 1,000 random-force training episodes; planner settings tuned on train |
| **Model-based RL (history)** | (velocity, force) + an encoding of every past transition in the episode | None | Same 1,000 random-force episodes + 4 rounds × 300 on-policy episodes; planner tuned on train |
| **Causal (passive / active)** | Same as PPO; uses full episode history | Coulomb model; a fixed prior μ∈[0.02,1.2], m∈[0.1,4] kg that is wider than any train or test range | Noise and planner settings tuned on train only |
| **Stacking: PPO** | The block's push log | None | 20k stacking episodes on the training distribution |
| **Stacking: Causal** | The same push log | Coulomb grasp friction (friction cone) plus the robot's own grip force, pad friction and motion plan | No stacking training; noise level from push tuning |

Every agent knows the actuator limits (PPO through its action space). None is
told the train or test confounder ranges, except the DR baselines.

### Stronger baselines (added 2026-11-13)

These test the two obvious objections to "structure wins":

- **Memory.** Plain PPO sees 3 steps of history, while the causal agent conditions on the whole
  episode. The *full history* variants see all of it. The history-conditioned
  model-based agent encodes every past transition with a permutation-invariant
  (deep-sets) encoder, in the style of context-aware dynamics models (Lee et al. 2020).
- **Coverage.** The *DR* variants train on the **domain-randomized box**: the
  bounding box of every train and test range of the preset (narrow: μ 0.15–0.35,
  m 0.6–1.4 kg; pdf: μ 0.2–0.8, m 0.5–2.0 kg; roughness over its full range).
  This gives them privileged knowledge of the test conditions, so they are an
  upper bound on what more data alone buys, not a fair default.

Every new baseline got 1M PPO steps, twice the original's budget. The
history-conditioned model needed the standard model-based-RL training recipe
to work at all: state-delta targets, a symlog (asinh) target transform, and
on-policy data aggregation. Trained only on random pushes, which mostly
drive the block at 1–3 m/s, it predicted motion at rest and its planner
learned to do nothing.

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
| PPO (full history) | 100 | 100 | 0 | 0 | 96 |
| PPO (DR) | 100 | 100 | 100 | **0** (1.2 cm, just outside ±1) | 100 |
| PPO (DR, full hist) | 100 | 100 | 0 | 0 | 100 |
| Model-based RL | 60 | 100 | 0 | 0 | 0 |
| Model-based RL (history) | 98 | 100 | 100 | 0 | 34 |
| Causal (passive) | 100 | 100 | 100 | 100 | 96 |
| Causal (active) | 100 | 100 | 100 | 100 | 96 |

### Push, `pdf` preset (project-plan ranges, 20 N, ±2 cm)

| Success % | Train | Interp | High μ | Low μ | Full range |
|---|---|---|---|---|---|
| PPO | 100 | 100 | 100 | 100 | **78** |
| PPO (full history) | 98 | 100 | 100 | 100 | 72 |
| PPO (DR) | 100 | 100 | 98 | 100 | 98 |
| PPO (DR, full hist) | 100 | 100 | 100 | 100 | 98 |
| Model-based RL | 68 | 100 | 0 | 100 | 44 |
| Model-based RL (history) | 60 | 100 | 0 | 100 | 56 |
| Causal (passive) | 100 | 100 | 100 | 100 | 100 |
| Causal (active) | 100 | 100 | 100 | 100 | 100 |

### What the stronger baselines show

- **More memory does not help PPO; it hurts out of distribution.** Full-history
  PPO matches plain PPO in distribution but drops to 0% on narrow High μ, where
  plain PPO gets 96%. More inputs give more ways for a test episode to look
  unfamiliar. Lack of memory is therefore not why PPO failed.
- **Coverage closes most of the gap, but not at the edges.** Trained on every test
  condition, DR-PPO reaches 98% on the pdf full range, up from 78%, against
  causal 100%. Across its own training box it scores 100%. At the box's four
  corners it does not (20 episodes each, `check_dr_corners.py`):

  | DR corner (m, μ) | narrow DR | narrow DR+hist | pdf DR | pdf DR+hist |
  |---|---|---|---|---|
  | light, slippery | 40% | 0% | **0%** | **0%** |
  | light, sticky | 100 | 100 | 95 | 100 |
  | heavy, slippery | 100 | 100 | 100 | 100 |
  | heavy, sticky | 100 | 0 | 100 | 0 |

  The light, slippery corner needs the smallest and most precise pushes, and
  uniform sampling rarely reaches corners. The narrow Low-μ test set sits exactly
  at that corner. The causal agent has no "corner": its behaviour follows from
  the model wherever the posterior lands.
- **History lets the model-based agent adapt only where its data reached.**
  It fixes narrow High μ (0 to 100%) but not the low-friction sets, where blocks
  slide farther and faster than in any training data.
- **Revised claim.** Structure is not the only route to robustness here. Enough
  coverage gets a model-free learner most of the way. What structure buys is
  (1) robustness without being shown the test conditions and (2) no failure
  pockets in rarely sampled regions.

**PPO does not fail on the plan's own test sets.** The plan pairs high
friction with low mass (and low with high), so the force needed to start
sliding (μ·m·g ≈ 4.4–4.7 N) stays inside the training range of 2.4–8.8 N.
Distribution shift only bites on the full-range set.

### Push → stack transfer (`pdf`)

| | Train | Interp | High μ | Low μ | Full range |
|---|---|---|---|---|---|
| PPO success / reward | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | **97 / 7.54** (3% dropped) |
| PPO (DR) success / reward | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | **97 / 7.54** (3% dropped) |
| Causal success / reward | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | 100 / **6.73** | 100 / 7.79 |
| Causal rule with true μ, m | 100 / 7.92 | 100 / 7.92 | 100 / 7.92 | 100 / 7.59 | 100 / 7.90 |

- **With real Panda limits, stacking barely depends on friction and mass.**
  Only heavy blocks near μ≈0.2 slip out at full speed.
- **PPO never saw a drop in 20,000 training episodes,** so it cannot learn to
  slow down for those blocks.
- **Domain randomization does not fix this.** DR-PPO trained on the full-range box
  (20k episodes, the same budget) saw 261 drops (1.3%). It still converged to
  "always carry at full speed", which is identical to the fixed-fastest policy.
  The signal is too rare for a model-free learner to separate the slippery-heavy
  corner from the push log. The causal agent gets this right with zero stacking
  episodes, because the friction cone tells it where the corner is.
- **The causal agent is over-cautious.** Its no-slip rule plus posterior
  uncertainty (from the large tuned σ) make it carry the Low-μ block at
  3.8 m/s² when full speed was safe. That costs reward (6.73 against PPO's 7.92).
- **Verdict on H2 (transfer):** weakly supported. The benefit is real but confined
  to the slippery-heavy corner, and it is paid for with slowness elsewhere.
  It survives the domain-randomized baseline.

### Effect of removing the leaks

Push results barely moved: passive narrow-adversarial went from 90 to 96, and
active narrow-adversarial from 100 to 96. Stacking became more conservative
(Low-μ reward fell from 7.62 to 6.73), because the agent no longer knows how
much slip the env tolerates.

## Known remaining weaknesses (not yet addressed)

- **Model correctness:** there is no test of the causal agent with a *wrong*
  model, for example ignoring speed-dependent friction or misjudging grip force.
- **Baseline tuning:** the PPO variants use the original PPO hyperparameters and
  uniform domain randomization. Corner oversampling, curriculum, or a recurrent
  (LSTM) policy might remove DR-PPO's corner failures. The full-history MLP gives
  the same information as an LSTM but is not the same architecture.

## Reproducing

```
python test_sim_physics.py                      # 21 simulator checks
python train.py --preset pdf                    # PPO
python train.py --preset pdf --variant ppo_hist|ppo_dr|ppo_dr_hist --timesteps 1000000
python mbrl_baseline.py --preset pdf            # model-based RL
python mbrl_context_baseline.py --preset pdf    # history-conditioned model-based RL
python tune_on_train.py --preset pdf            # all planner hyperparameters (train only)
python eval_causal_agent.py --preset pdf        # push comparison
python stack_transfer.py bank|train|eval --preset pdf
python stack_transfer.py bank --preset pdf --dr && python stack_transfer.py train --preset pdf --dr
```
