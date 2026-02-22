"""
Reconstruct Best-Model Summary from saved eval logs.

Scans models/ folders for completed training runs, loads evaluations.npz
files, and displays the best-model summary tables grouped by session (lab prefix).

Usage:
    python best_model_summary.py
"""

import glob
import os

import numpy as np
import pandas as pd

MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ALGORITHMS = ["dqn", "ppo"]
SUCCESS_THRESHOLD = 200


def collect_runs():
    all_runs = []

    for algo_name in ALGORITHMS:
        algo_dir = os.path.join(MODELS_ROOT, algo_name)
        if not os.path.isdir(algo_dir):
            continue

        for run_folder in sorted(glob.glob(os.path.join(algo_dir, "????-??-??_??_??_??"))):
            best_model_path = os.path.join(run_folder, "best_model.zip")
            eval_log_path = os.path.join(run_folder, "eval_log", "evaluations.npz")

            if not os.path.isfile(best_model_path):
                continue

            # Extract session (lab prefix) and seed from final model filename
            session = "unknown"
            seed_str = "?"
            for f in os.listdir(run_folder):
                if f.startswith("lab") and f.endswith(".zip") and f != "best_model.zip":
                    parts = f.replace(".zip", "").split("_")
                    session = parts[0] if len(parts) >= 1 else "unknown"
                    seed_str = parts[-1] if len(parts) >= 3 else "?"
                    break

            timestamp = os.path.basename(run_folder)

            run_entry = {
                "session": session,
                "algo": algo_name.upper(),
                "seed": seed_str,
                "timestamp": timestamp,
                "folder": f"models/{algo_name}/{timestamp}/",
            }

            if os.path.isfile(eval_log_path):
                data = np.load(eval_log_path, allow_pickle=True)
                timesteps = data["timesteps"]
                results = data["results"]

                best_score = -np.inf
                best_idx = 0
                for i in range(len(timesteps)):
                    ep_rewards = results[i]
                    score = np.mean(ep_rewards) - np.std(ep_rewards)
                    if score > best_score:
                        best_score = score
                        best_idx = i

                ep = results[best_idx]
                run_entry.update({
                    "mean_reward": np.mean(ep),
                    "std_reward": np.std(ep),
                    "success": np.sum(ep >= SUCCESS_THRESHOLD) / len(ep) * 100,
                    "score": best_score,
                    "timestep": int(timesteps[best_idx]),
                    "has_eval": True,
                })
            else:
                run_entry["has_eval"] = False

            all_runs.append(run_entry)

    return all_runs


def print_summary(all_runs):
    if not all_runs:
        print("No completed training runs found in models/.")
        return

    sessions = sorted(set(r["session"] for r in all_runs))

    for session in sessions:
        session_runs = [r for r in all_runs if r["session"] == session]

        print(f"{'='*70}")
        print(f"SESSION: {session}")
        print(f"{'='*70}")

        rows = []
        for r in sorted(session_runs, key=lambda x: (x["algo"], x["seed"])):
            if r["has_eval"]:
                rows.append({
                    "Algorithm": r["algo"],
                    "Seed": r["seed"],
                    "Mean Reward": f"{r['mean_reward']:.2f}",
                    "Std Reward": f"{r['std_reward']:.2f}",
                    "Success": f"{r['success']:.0f}%",
                    "Score (mean-std)": f"{r['score']:.2f}",
                    "@ Timestep": f"{r['timestep']:,}",
                    "Run Folder": r["folder"],
                })
            else:
                rows.append({
                    "Algorithm": r["algo"],
                    "Seed": r["seed"],
                    "Mean Reward": "N/A",
                    "Std Reward": "N/A",
                    "Success": "N/A",
                    "Score (mean-std)": "N/A",
                    "@ Timestep": "N/A",
                    "Run Folder": r["folder"] + " (no eval log)",
                })

        print(pd.DataFrame(rows).to_string(index=False))
        print(f"\nRuns in this session: {len(session_runs)}")
        print(f"Each 'Run Folder' contains: best_model.zip, checkpoints/, eval_log/")
        print()


if __name__ == "__main__":
    runs = collect_runs()
    print_summary(runs)