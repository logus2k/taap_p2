"""
Validate models against LunarLander-v3 requirements:

  1. SOLVED criterion: average reward >= 200 over 100 episodes
     Scans ALL checkpoints (+ best_model + final model) per run.
     Pre-screens with 20 episodes, then validates with 100.

  2. EVALUATION criterion: 20 deterministic episodes per seed
     Runs 20 deterministic episodes on the best passing checkpoint
     and reports mean +/- std (as required by the assignment).

Usage:
    python validate_best_models.py                   # all sessions
    python validate_best_models.py --session lab008   # specific session
"""

import argparse
import glob
import os
import time

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ALGORITHMS = {"dqn": DQN, "ppo": PPO}
ENV_ID = "LunarLander-v3"
WIND_ENABLED = False
DEVICE = "cuda"

PRESCREEN_EPISODES = 20
PRESCREEN_THRESHOLD = 180
VALIDATION_EPISODES = 100
SOLVED_THRESHOLD = 200
EVAL_EPISODES = 20  # deterministic evaluation as required by assignment


def find_runs(session_filter=None):
    """Find all completed run folders, optionally filtered by session."""
    runs = []
    for algo_name in ALGORITHMS:
        algo_dir = os.path.join(MODELS_ROOT, algo_name)
        if not os.path.isdir(algo_dir):
            continue

        for run_folder in sorted(glob.glob(os.path.join(algo_dir, "????-??-??_??_??_??"))):
            best_model_path = os.path.join(run_folder, "best_model.zip")
            if not os.path.isfile(best_model_path):
                continue

            session = "unknown"
            seed_str = "?"
            for f in os.listdir(run_folder):
                if f.startswith("lab") and f.endswith(".zip") and f != "best_model.zip":
                    parts = f.replace(".zip", "").split("_")
                    session = parts[0] if len(parts) >= 1 else "unknown"
                    seed_str = parts[-1] if len(parts) >= 3 else "?"
                    break

            if session_filter and session != session_filter:
                continue

            # Collect all model files in this run
            model_files = []
            model_files.append(("best_model", best_model_path))

            for f in os.listdir(run_folder):
                if f.startswith("lab") and f.endswith(".zip") and f != "best_model.zip":
                    model_files.append(("final", os.path.join(run_folder, f)))

            ckpt_dir = os.path.join(run_folder, "checkpoints")
            if os.path.isdir(ckpt_dir):
                for f in sorted(os.listdir(ckpt_dir)):
                    if f.endswith(".zip"):
                        label = f.replace(".zip", "")
                        model_files.append((label, os.path.join(ckpt_dir, f)))

            runs.append({
                "session": session,
                "algo": algo_name,
                "seed": seed_str,
                "run_folder": run_folder,
                "folder_short": f"models/{algo_name}/{os.path.basename(run_folder)}/",
                "model_files": model_files,
            })

    return runs


def evaluate_model(algo_name, model_path, seed_int, n_episodes):
    """Run n_episodes deterministic episodes, return rewards array."""
    def make_env(s=seed_int):
        env = gym.make(ENV_ID, render_mode=None, enable_wind=WIND_ENABLED)
        env.reset(seed=s)
        return env

    model = ALGORITHMS[algo_name].load(model_path, env=DummyVecEnv([make_env]), device=DEVICE)

    eval_env = gym.make(ENV_ID, enable_wind=WIND_ENABLED)
    eval_env.reset(seed=seed_int)

    rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )

    eval_env.close()
    return np.array(rewards)


def main():
    parser = argparse.ArgumentParser(description="Validate RL checkpoints")
    parser.add_argument("--session", type=str, default=None,
                        help="Filter by session (e.g. lab008). Default: all.")
    args = parser.parse_args()

    runs = find_runs(session_filter=args.session)
    if not runs:
        print("No completed training runs found.")
        return

    sessions = sorted(set(r["session"] for r in runs))

    for session in sessions:
        session_runs = [r for r in runs if r["session"] == session]

        # ==================================================================
        # PART 1: Solved criterion (100 episodes, mean >= 200)
        # ==================================================================
        print(f"\n{'='*70}")
        print(f"SESSION: {session}")
        print(f"{'='*70}")
        print(f"\n--- PART 1: Solved Criterion ---")
        print(f"Pre-screen: {PRESCREEN_EPISODES} episodes (mean >= {PRESCREEN_THRESHOLD})")
        print(f"Validation: mean reward >= {SOLVED_THRESHOLD} over {VALIDATION_EPISODES} episodes")

        solved_results = []  # per run: best passing checkpoint info
        best_model_paths = {}  # (algo, seed) -> model_path of best passing ckpt

        for r in sorted(session_runs, key=lambda x: (x["algo"], x["seed"])):
            algo = r["algo"]
            seed_int = int(r["seed"]) if r["seed"].isdigit() else 42
            n_models = len(r["model_files"])
            label = f"{algo.upper()} seed {r['seed']}"

            print(f"\n  {label} — {n_models} models to check ({r['folder_short']})")

            passed_models = []
            screened = 0
            prescreen_passed = 0

            t_start = time.time()

            for model_label, model_path in r["model_files"]:
                screened += 1
                progress = f"[{screened}/{n_models}]"

                # Stage 1: Pre-screen
                rewards_pre = evaluate_model(algo, model_path, seed_int, PRESCREEN_EPISODES)
                pre_mean = float(np.mean(rewards_pre))

                if pre_mean < PRESCREEN_THRESHOLD:
                    print(f"    {progress} {model_label}: pre-screen mean {pre_mean:.1f} — skip")
                    continue

                prescreen_passed += 1

                # Stage 2: Full validation
                rewards_full = evaluate_model(algo, model_path, seed_int, VALIDATION_EPISODES)
                mean_r = float(np.mean(rewards_full))
                std_r = float(np.std(rewards_full))
                min_r = float(np.min(rewards_full))
                score = mean_r - std_r
                success_count = int(np.sum(rewards_full >= SOLVED_THRESHOLD))
                solved = mean_r >= SOLVED_THRESHOLD
                status = "SOLVED" if solved else "FAIL"

                print(
                    f"    {progress} {model_label}: "
                    f"mean {mean_r:.1f} +/- {std_r:.1f} | "
                    f"min {min_r:.1f} | "
                    f"{success_count}/{VALIDATION_EPISODES} >= {SOLVED_THRESHOLD} | "
                    f"{status}"
                )

                if solved:
                    passed_models.append({
                        "model_label": model_label,
                        "model_path": model_path,
                        "mean": mean_r,
                        "std": std_r,
                        "min": min_r,
                        "score": score,
                        "success_count": success_count,
                    })

            elapsed = time.time() - t_start
            print(
                f"  {label} done in {elapsed:.0f}s — "
                f"{screened} checked, {prescreen_passed} passed pre-screen, "
                f"{len(passed_models)} solved"
            )

            if passed_models:
                best = max(passed_models, key=lambda x: x["score"])
                best_model_paths[(algo, r["seed"])] = best["model_path"]
                solved_results.append({
                    "Algorithm": algo.upper(),
                    "Seed": r["seed"],
                    "Best Checkpoint": best["model_label"],
                    "Mean Reward": f"{best['mean']:.2f}",
                    "Std Reward": f"{best['std']:.2f}",
                    "Min Reward": f"{best['min']:.2f}",
                    "Score (mean-std)": f"{best['score']:.2f}",
                    f">={SOLVED_THRESHOLD} episodes": f"{best['success_count']}/{VALIDATION_EPISODES}",
                    "Solved": "YES",
                    "Total Passing": len(passed_models),
                    "Run Folder": r["folder_short"],
                })
            else:
                solved_results.append({
                    "Algorithm": algo.upper(),
                    "Seed": r["seed"],
                    "Best Checkpoint": "—",
                    "Mean Reward": "—",
                    "Std Reward": "—",
                    "Min Reward": "—",
                    "Score (mean-std)": "—",
                    f">={SOLVED_THRESHOLD} episodes": "—",
                    "Solved": "NO",
                    "Total Passing": 0,
                    "Run Folder": r["folder_short"],
                })

        print(f"\n{'-'*70}")
        print(f"PART 1 RESULTS: Solved criterion (mean >= {SOLVED_THRESHOLD} over {VALIDATION_EPISODES} ep)")
        print(f"{'-'*70}\n")
        print(pd.DataFrame(solved_results).to_string(index=False))

        total = len(solved_results)
        solved_total = sum(1 for r in solved_results if r["Solved"] == "YES")
        print(f"\n  {solved_total}/{total} runs solved.\n")

        # ==================================================================
        # PART 2: 20-episode deterministic evaluation (on best passing ckpt)
        # ==================================================================
        if not best_model_paths:
            print("  No solved models — skipping 20-episode evaluation.\n")
            continue

        print(f"--- PART 2: {EVAL_EPISODES}-Episode Deterministic Evaluation ---")
        print(f"Running {EVAL_EPISODES} deterministic episodes on each best checkpoint.\n")

        eval_rows = []
        for r in sorted(session_runs, key=lambda x: (x["algo"], x["seed"])):
            algo = r["algo"]
            key = (algo, r["seed"])
            if key not in best_model_paths:
                continue

            seed_int = int(r["seed"]) if r["seed"].isdigit() else 42
            label = f"{algo.upper()} seed {r['seed']}"

            rewards = evaluate_model(algo, best_model_paths[key], seed_int, EVAL_EPISODES)
            mean_r = float(np.mean(rewards))
            std_r = float(np.std(rewards))
            min_r = float(np.min(rewards))
            max_r = float(np.max(rewards))
            success = int(np.sum(rewards >= SOLVED_THRESHOLD))

            print(
                f"  {label}: {mean_r:.2f} +/- {std_r:.2f} | "
                f"min {min_r:.1f} | max {max_r:.1f} | "
                f"{success}/{EVAL_EPISODES} >= {SOLVED_THRESHOLD}"
            )

            eval_rows.append({
                "Algorithm": algo.upper(),
                "Seed": r["seed"],
                "Mean Reward": f"{mean_r:.2f}",
                "Std Reward": f"{std_r:.2f}",
                "Min Reward": f"{min_r:.2f}",
                "Max Reward": f"{max_r:.2f}",
                "Success Rate": f"{success}/{EVAL_EPISODES}",
            })

        # Per-algorithm aggregate (mean +/- std across seeds)
        algo_agg_rows = []
        for algo_name in ALGORITHMS:
            algo_rows = [r for r in eval_rows if r["Algorithm"] == algo_name.upper()]
            if not algo_rows:
                continue
            means = [float(r["Mean Reward"]) for r in algo_rows]
            algo_agg_rows.append({
                "Algorithm": algo_name.upper(),
                "Seeds": len(algo_rows),
                "Mean across seeds": f"{np.mean(means):.2f}",
                "Std across seeds": f"{np.std(means):.2f}",
            })

        print(f"\n{'-'*70}")
        print(f"PART 2 RESULTS: {EVAL_EPISODES}-episode deterministic evaluation")
        print(f"{'-'*70}\n")
        print(pd.DataFrame(eval_rows).to_string(index=False))

        if algo_agg_rows:
            print(f"\nAggregated (mean +/- std of per-seed means):")
            print(pd.DataFrame(algo_agg_rows).to_string(index=False))

        print()


if __name__ == "__main__":
    main()