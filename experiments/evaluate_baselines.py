"""
experiments/evaluate_baselines.py

Evaluate baseline policies on CascadingMitigationEnv.

Compared policies:
    1. No-action
    2. Random
    3. Degree-based protection
    4. Load-ratio-based protection
    5. One-step greedy
    6. Optional Vector-DQN checkpoint

Run from project root:

    python experiments/evaluate_baselines.py

Recommended first run:

    python experiments/evaluate_baselines.py --episodes 200 --N 20 --max_steps 5 --budget 3 --no_disconnect --initial_failure_strategy highest_load --protect_strength 2.0 --protect_duration 5

If you already trained DQN:

    python experiments/evaluate_baselines.py --episodes 200 --include_dqn --dqn_path outputs/checkpoints/vector_dqn_best.pt --N 20 --max_steps 5 --budget 3 --no_disconnect --initial_failure_strategy highest_load --protect_strength 2.0 --protect_duration 5
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------
# Make imports work when running:
#   python experiments/evaluate_baselines.py
# ---------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from envs.cascading_mitigation_env import CascadingMitigationEnv

from baselines.no_action import no_action_policy
from baselines.random_policy import random_policy
from baselines.degree_policy import degree_policy
from baselines.load_policy import load_policy
from baselines.greedy_policy import greedy_policy

from agents.vector_dqn_agent import VectorDQNAgent, infer_state_dim


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(device_arg)


def make_env(args: argparse.Namespace, seed: int) -> CascadingMitigationEnv:
    env = CascadingMitigationEnv(
        N=args.N,
        network_type=args.network_type,
        m=args.m,
        p=args.p,
        k=args.k,
        rewiring_p=args.rewiring_p,
        alpha=args.alpha,
        max_steps=args.max_steps,
        budget=args.budget,
        protect_cost=args.protect_cost,
        disconnect_cost=args.disconnect_cost,
        do_nothing_cost=args.do_nothing_cost,
        protect_strength=args.protect_strength,
        protect_duration=args.protect_duration,
        failure_model=args.failure_model,
        redistribution_mode=args.redistribution_mode,
        failure_gamma=args.failure_gamma,
        failure_sharpness=args.failure_sharpness,
        load_type=args.load_type,
        initial_failures=args.initial_failures,
        initial_failure_strategy=args.initial_failure_strategy,
        allow_disconnect=not args.no_disconnect,
        seed=seed,
    )
    return env


def evaluate_policy(
    policy_name: str,
    policy_fn: Callable,
    args: argparse.Namespace,
    episodes: int,
    seed_offset: int = 0,
) -> tuple[dict, list[dict]]:
    """
    Evaluate one policy.

    policy_fn signature:
        action = policy_fn(obs, env)
    """
    episode_rows = []

    returns = []
    failed_fractions = []
    lcc_ratios = []
    lost_load_ratios = []
    served_load_ratios = []
    budget_used_list = []
    damage_list = []
    steps_list = []

    iterator = tqdm(range(episodes), desc=f"Evaluating {policy_name}")

    for ep in iterator:
        seed = args.seed + seed_offset + ep
        env = make_env(args, seed=seed)
        obs, info = env.reset(seed=seed)

        done = False
        episode_return = 0.0
        final_info = info

        while not done:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += float(reward)
            done = terminated or truncated
            final_info = info

        row = {
            "policy": policy_name,
            "episode": ep,
            "return": float(episode_return),
            "failed_fraction": float(final_info["failed_fraction"]),
            "lcc_ratio": float(final_info["lcc_ratio"]),
            "lost_load_ratio": float(final_info["lost_load_ratio"]),
            "served_load_ratio": float(final_info["served_load_ratio"]),
            "budget_used": float(final_info["budget_used"]),
            "damage": float(final_info["damage"]),
            "steps": int(final_info["step"]),
            "initial_failed_nodes": str(final_info["initial_failed_nodes"]),
        }

        episode_rows.append(row)

        returns.append(row["return"])
        failed_fractions.append(row["failed_fraction"])
        lcc_ratios.append(row["lcc_ratio"])
        lost_load_ratios.append(row["lost_load_ratio"])
        served_load_ratios.append(row["served_load_ratio"])
        budget_used_list.append(row["budget_used"])
        damage_list.append(row["damage"])
        steps_list.append(row["steps"])

    summary = {
        "policy": policy_name,

        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),

        "failed_fraction_mean": float(np.mean(failed_fractions)),
        "failed_fraction_std": float(np.std(failed_fractions)),

        "lcc_ratio_mean": float(np.mean(lcc_ratios)),
        "lcc_ratio_std": float(np.std(lcc_ratios)),

        "lost_load_ratio_mean": float(np.mean(lost_load_ratios)),
        "lost_load_ratio_std": float(np.std(lost_load_ratios)),

        "served_load_ratio_mean": float(np.mean(served_load_ratios)),
        "served_load_ratio_std": float(np.std(served_load_ratios)),

        "budget_used_mean": float(np.mean(budget_used_list)),
        "budget_used_std": float(np.std(budget_used_list)),

        "damage_mean": float(np.mean(damage_list)),
        "damage_std": float(np.std(damage_list)),

        "steps_mean": float(np.mean(steps_list)),
        "steps_std": float(np.std(steps_list)),
    }

    return summary, episode_rows


def build_dqn_policy(args: argparse.Namespace):
    """
    Load a trained Vector-DQN checkpoint and return a policy function.
    """
    if not os.path.exists(args.dqn_path):
        raise FileNotFoundError(
            f"DQN checkpoint not found: {args.dqn_path}"
        )

    device = select_device(args.device)

    temp_env = make_env(args, seed=args.seed + 999999)
    obs, _ = temp_env.reset(seed=args.seed + 999999)

    state_dim = infer_state_dim(obs)
    action_dim = temp_env.action_space.n

    agent = VectorDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        target_update=args.target_update,
        device=device,
        double_dqn=True,
    )

    agent.load(args.dqn_path, map_location=device)
    agent.epsilon = 0.0

    def dqn_policy(obs: dict, env) -> int:
        return int(agent.take_action(obs, evaluate=True))

    return dqn_policy


def save_csv(path: str, rows: list[dict]) -> None:
    if len(rows) == 0:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(summary_rows: list[dict]) -> None:
    """
    Print a compact summary table.
    """
    headers = [
        "Policy",
        "Return ↑",
        "Failed ↓",
        "LCC ↑",
        "Lost Load ↓",
        "Budget Used ↓",
        "Damage ↓",
    ]

    print("\n" + "=" * 110)
    print("Baseline Comparison Summary")
    print("=" * 110)

    print(
        f"{headers[0]:<16}"
        f"{headers[1]:>14}"
        f"{headers[2]:>14}"
        f"{headers[3]:>14}"
        f"{headers[4]:>16}"
        f"{headers[5]:>18}"
        f"{headers[6]:>14}"
    )

    print("-" * 110)

    for row in summary_rows:
        print(
            f"{row['policy']:<16}"
            f"{row['return_mean']:>14.4f}"
            f"{row['failed_fraction_mean']:>14.4f}"
            f"{row['lcc_ratio_mean']:>14.4f}"
            f"{row['lost_load_ratio_mean']:>16.4f}"
            f"{row['budget_used_mean']:>18.4f}"
            f"{row['damage_mean']:>14.4f}"
        )

    print("=" * 110)


def plot_summary(summary_rows: list[dict]) -> None:
    """
    Save comparison bar charts.
    """
    os.makedirs("outputs/figures", exist_ok=True)

    policies = [row["policy"] for row in summary_rows]

    metrics = [
        ("return_mean", "Return", "baseline_return.png"),
        ("failed_fraction_mean", "Final Failed Fraction", "baseline_failed_fraction.png"),
        ("lcc_ratio_mean", "LCC Ratio", "baseline_lcc_ratio.png"),
        ("lost_load_ratio_mean", "Lost Load Ratio", "baseline_lost_load_ratio.png"),
        ("damage_mean", "Final Damage", "baseline_damage.png"),
    ]

    for key, ylabel, filename in metrics:
        values = [row[key] for row in summary_rows]

        plt.figure(figsize=(10, 5))
        plt.bar(policies, values)
        plt.ylabel(ylabel)
        plt.title(f"Baseline Comparison: {ylabel}")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join("outputs", "figures", filename)
        plt.savefig(save_path, dpi=300)
        plt.close()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    policy_dict = {
        "No-action": no_action_policy,
        "Random": random_policy,
        "Degree": degree_policy,
        "Load-ratio": load_policy,
    }

    if not args.no_greedy:
        policy_dict["Greedy"] = lambda obs, env: greedy_policy(
            obs,
            env,
            max_candidates=args.greedy_max_candidates,
        )

    if args.include_dqn:
        policy_dict["Vector-DQN"] = build_dqn_policy(args)

    summary_rows = []
    all_episode_rows = []

    for i, (policy_name, policy_fn) in enumerate(policy_dict.items()):
        summary, episode_rows = evaluate_policy(
            policy_name=policy_name,
            policy_fn=policy_fn,
            args=args,
            episodes=args.episodes,
            seed_offset=0,
        )

        summary_rows.append(summary)
        all_episode_rows.extend(episode_rows)

    # Sort by failed_fraction ascending, then return descending
    summary_rows = sorted(
        summary_rows,
        key=lambda row: (row["failed_fraction_mean"], -row["return_mean"]),
    )

    print_summary_table(summary_rows)

    save_csv(
        path="outputs/results/baseline_summary.csv",
        rows=summary_rows,
    )

    save_csv(
        path="outputs/results/baseline_episode_details.csv",
        rows=all_episode_rows,
    )

    plot_summary(summary_rows)

    print("\nSaved results:")
    print("  outputs/results/baseline_summary.csv")
    print("  outputs/results/baseline_episode_details.csv")
    print("\nSaved figures:")
    print("  outputs/figures/baseline_return.png")
    print("  outputs/figures/baseline_failed_fraction.png")
    print("  outputs/figures/baseline_lcc_ratio.png")
    print("  outputs/figures/baseline_lost_load_ratio.png")
    print("  outputs/figures/baseline_damage.png")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--network_type", type=str, default="BA", choices=["BA", "ER", "WS"])
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--rewiring_p", type=float, default=0.1)

    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--budget", type=float, default=3.0)

    parser.add_argument("--protect_cost", type=float, default=1.0)
    parser.add_argument("--disconnect_cost", type=float, default=1.0)
    parser.add_argument("--do_nothing_cost", type=float, default=0.0)
    parser.add_argument("--protect_strength", type=float, default=2.0)
    parser.add_argument("--protect_duration", type=int, default=5)

    parser.add_argument(
        "--failure_model",
        type=str,
        default="deterministic",
        choices=["deterministic", "logistic"],
    )
    parser.add_argument(
        "--redistribution_mode",
        type=str,
        default="uniform",
        choices=["uniform", "stochastic", "degree_weighted", "load_weighted"],
    )
    parser.add_argument("--failure_gamma", type=float, default=1.5)
    parser.add_argument("--failure_sharpness", type=float, default=10.0)

    parser.add_argument(
        "--load_type",
        type=str,
        default="degree",
        choices=["degree", "betweenness"],
    )
    parser.add_argument("--initial_failures", type=int, default=1)
    parser.add_argument(
        "--initial_failure_strategy",
        type=str,
        default="highest_load",
        choices=["random", "highest_load"],
    )

    parser.add_argument(
        "--no_disconnect",
        action="store_true",
        help="Disable edge-disconnection actions.",
    )

    # Evaluation
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--no_greedy",
        action="store_true",
        help="Skip greedy baseline because it may be slow.",
    )
    parser.add_argument(
        "--greedy_max_candidates",
        type=int,
        default=None,
        help="Maximum number of valid actions evaluated by greedy baseline. Default: all.",
    )

    # Optional DQN
    parser.add_argument(
        "--include_dqn",
        action="store_true",
        help="Include trained Vector-DQN checkpoint in comparison.",
    )
    parser.add_argument(
        "--dqn_path",
        type=str,
        default="outputs/checkpoints/vector_dqn_best.pt",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
