# DQN Hyperparameter Tuning Report — LunarLander-v3

## Starting Point

The initial approach was to reuse the DQN hyperparameters from the course notebook (CartPole-v1) directly on LunarLander-v3. This failed because LunarLander has a larger state space (8 vs 4 dimensions), shaped rewards (crash penalties, fuel costs, landing bonuses), and requires the agent to learn a much more complex control strategy.

---

## Phase 1 — Baseline (CartPole defaults)

| Parameter | Value |
|-----------|-------|
| `total_timesteps` | 100,000 |
| `learning_rate` | 1e-3 |
| `exploration_fraction` | 0.2 |
| `exploration_final_eps` | 0.05 (default) |
| `buffer_size` | 50,000 |
| `batch_size` | 64 |
| `learning_starts` | 500 |
| `target_update_interval` | 10,000 |
| `train_freq` | 4 |

**Result:** Agent flies off screen to the right. Too few timesteps, too aggressive learning rate, insufficient exploration time for the complexity of the environment.

---

## Phase 2 — Conservative tuning

| Parameter | Value | Change |
|-----------|-------|--------|
| `total_timesteps` | 500,000 | 5x more |
| `learning_rate` | 1e-4 | 10x lower |
| `exploration_fraction` | 0.4 | 2x longer |
| `buffer_size` | 100,000 | 2x larger |
| `batch_size` | 64 | — |
| `learning_starts` | 2,500 | 5x more |
| `target_update_interval` | 2,500 | 4x more frequent |

**Result:** Agent flies off screen to the left. Reward curve is completely flat across 1300+ episodes. The learning rate was too conservative and the exploration phase too long — the agent spent 200k steps acting randomly, filling the buffer with low-quality data, then learned too slowly to recover.

**Key lesson:** More timesteps and lower learning rate do not help if the core balance between exploration and exploitation is wrong.

---

## Phase 3 — Aggressive exploitation

| Parameter | Value | Change |
|-----------|-------|--------|
| `total_timesteps` | 500,000 | — |
| `learning_rate` | 6.3e-4 | 6.3x higher |
| `exploration_fraction` | 0.12 | 3.3x shorter |
| `exploration_final_eps` | 0.1 | higher floor |
| `buffer_size` | 50,000 | halved |
| `batch_size` | 128 | 2x larger |
| `learning_starts` | 0 | immediate |
| `target_update_interval` | 250 | 10x more frequent |

**Result:** Agent lands successfully. Reward curve shows clear phase transition around episode 400. Rewards reach 200+ regularly but with notable variance and occasional crashes. Evaluation showed ~80% success rate.

**Key lesson:** LunarLander needs a higher learning rate, short exploration decay, and frequent target updates — the opposite of the conservative approach.

---

## Phase 4 — Final refinement

| Parameter | Value | Change |
|-----------|-------|--------|
| `total_timesteps` | 750,000 | 50% more |
| `learning_rate` | 6.3e-4 | — |
| `exploration_fraction` | 0.12 | — |
| `exploration_final_eps` | 0.02 | lower floor |
| `buffer_size` | 50,000 | — |
| `batch_size` | 128 | — |
| `learning_starts` | 0 | — |
| `target_update_interval` | 250 | — |

**Result:** Clean, centered landing. Reward distribution over 50 evaluation episodes shows ~40/50 episodes scoring 200–320. Training curve stabilizes around episode 500 with consistent high rewards. The lower `exploration_final_eps` reduced noise during exploitation, and the extra 250k timesteps allowed the policy to consolidate.

---

## Summary of Progression

| Phase | Timesteps | Reward Range | Outcome |
|-------|-----------|-------------|---------|
| 1 — Baseline | 100k | N/A (crash) | Flies off screen |
| 2 — Conservative | 500k | -200 to +100 (flat) | No learning |
| 3 — Aggressive | 500k | 200+ with variance | Lands, rough |
| 4 — Refined | 750k | 200–320 (80%+ success) | Clean landing |

## Key Takeaways

1. **Hyperparameters don't transfer across environments.** CartPole defaults fail on LunarLander.
2. **Learning rate was the most impactful parameter.** 1e-4 was too slow; 6.3e-4 worked.
3. **Short exploration is better for LunarLander.** 12% of training spent exploring, not 40%.
4. **Frequent target updates (250 steps) helped** stabilize learning compared to 2,500 or 10,000.
5. **More timesteps only help once the core configuration is right.** 500k with bad params produced nothing; 500k with good params solved the task.
