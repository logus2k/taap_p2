# PPO Hyperparameter Tuning Report — LunarLander-v3

## Starting Point

The initial approach was to reuse the PPO hyperparameters from the course notebook (MountainCarContinuous-v0) directly on LunarLander-v3. This failed because MountainCarContinuous is a continuous-action environment with very different reward dynamics, while LunarLander-v3 uses a discrete action space (4 actions: nothing, left engine, main engine, right engine) with shaped rewards including crash penalties, fuel costs, and landing bonuses.

---

## Phase 1 — Baseline (MountainCarContinuous defaults)

| Parameter | Value |
|-----------|-------|
| `total_timesteps` | 750,000 |
| `learning_rate` | 7.7e-5 |
| `n_steps` | 2,048 |
| `batch_size` | 256 |
| `n_epochs` | 10 |
| `gamma` | 0.9999 |
| `gae_lambda` | 0.90 |
| `ent_coef` | 0.005 |
| `clip_range` | 0.1 |
| `device` | GPU (with SB3 warning) |

**Result:** Agent learns to hover but never lands. Episode lengths trend toward 1000 (environment timeout), confirming the agent burns fuel staying airborne until time runs out. Reward curve is noisy with a slight upward trend but mostly negative. The GIF shows the lander in the upper-left corner, far from the landing zone.

**Diagnosis:**

- `gamma=0.9999` was the main culprit — the agent barely discounted future rewards, so there was no urgency to land quickly.
- `learning_rate=7.7e-5` was too slow for the policy to adapt.
- `clip_range=0.1` was too conservative, limiting how much the policy could change per update.
- `batch_size=256` resulted in only 8 mini-batches per epoch, reducing update granularity.

**Key lesson:** Hyperparameters tuned for a continuous-control task with sparse rewards (MountainCar) do not transfer to a discrete-control task with shaped rewards (LunarLander).

---

## Phase 2 — Tuned for LunarLander

| Parameter | Value | Change |
|-----------|-------|--------|
| `total_timesteps` | 750,000 | — |
| `learning_rate` | 2.5e-4 | 3.2x higher |
| `n_steps` | 2,048 | — |
| `batch_size` | 64 | 4x smaller (32 mini-batches/epoch) |
| `n_epochs` | 10 | — |
| `gamma` | 0.999 | lower — encourages faster landing |
| `gae_lambda` | 0.95 | higher — better advantage estimation |
| `ent_coef` | 0.01 | 2x more exploration |
| `clip_range` | 0.2 | 2x larger — allows bigger policy steps |
| `device` | CPU | explicit — avoids SB3 GPU warning |

**Result:** Clean, centered landing between the flags. Evaluation over 10 episodes: **mean reward 265.50 ± 20.55**, min 244.50, max 311.47. The agent never fails to land.

**Training curve analysis:**

- **Episode Reward:** Negative rewards for the first ~300 episodes, sharp upward transition around episode 300–400, stabilizing at 250–300 by episode 500. Clear and clean convergence.
- **Episode Length:** Settles around 250–400 steps after training. Occasional 1000-step timeout spikes become rare after episode 500, confirming the agent learned to land efficiently rather than hover.
- **Value Loss:** Sharp initial spike followed by steady decline to near 0. Periodic bumps during the transition phase (rollouts 75–125) correlate with the agent discovering landing behavior. Healthy overall.
- **Entropy:** Smooth decline from 1.4 to 0.6. No premature collapse. The policy becomes gradually more deterministic as confidence increases.

---

## Summary of Progression

| Phase | Timesteps | Mean Reward | Outcome |
|-------|-----------|-------------|---------|
| 1 — Baseline | 750k | Negative (hovering) | Learns to survive, not to land |
| 2 — Tuned | 750k | 265.50 ± 20.55 | Clean, consistent landings |

## Key Takeaways

1. **`gamma` was the most impactful parameter.** Reducing from 0.9999 to 0.999 gave the agent urgency to land rather than hover indefinitely.
2. **Higher learning rate enabled faster convergence.** 2.5e-4 vs 7.7e-5 allowed the policy to adapt within the training budget.
3. **Larger `clip_range` (0.2) allowed meaningful policy updates.** The conservative 0.1 clipping was restricting learning progress.
4. **Smaller batch size improved update quality.** 64 instead of 256 gave 32 mini-batches per epoch, providing more frequent gradient steps.
5. **PPO required only 2 tuning phases** to achieve strong performance, compared to 4 for DQN on the same environment.
