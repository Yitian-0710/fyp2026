"""
experiments/inspect_policy_behavior.py

Diagnose what a policy is doing inside CascadingMitigationEnv.

Main purpose:
    Training curves and baseline bar charts only tell us whether a policy performs well.
    They do not tell us why. This script records the step-by-step behavior of a policy:

        - Which action does it select?
        - Does it often choose do-nothing?
        - Does it protect high-risk nodes or low-risk nodes?
        - Is the selected action better than the do-nothing counterfactual?
        - What is the reward/benefit/damage change after each action?

Supported policies:
    - no_action
    - random
    - degree
    - load
    - greedy
    - dqn

Run from project root:

    python experiments/inspect_policy_behavior.py --policy dqn --dqn_path outputs/checkpoints/vector_dqn_best.pt

Recommended first run:

    python experiments/inspect_policy_behavior.py \
        --policy dqn \
        --episodes 5 \
        --dqn_path outputs/checkpoints/vector_dqn_best.pt \
        --N 20 \
        --max_steps 5 \
        --budget 3 \
        --no_disconnect \
        --initial_failure_strategy highest_load \
        --protect_strength 2.0 \
        --protect_duration 5

Compare DQN with a heuristic on the same type of setting:

    python experiments/inspect_policy_behavior.py --policy degree --episodes 5 --N 20 --max_steps 5 --budget 3 --no_disconnect --initial_failure_strategy highest_load --protect_strength 2.0 --protect_duration 5

Outputs:
    outputs/results/behavior_<policy>_steps.csv
    outputs/results/behavior_<policy>_episodes.csv
    outputs/figures/behavior_<policy>_action_types.png
    outputs/figures/behavior_<policy>_benefit_by_step.png
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------
# Make imports work when running:
#   python experiments/inspect_policy_behavior.py
# ---------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envs.cascading_mitigation_env import CascadingMitigationEnv
from envs.actions import action_to_string

from baselines.no_action import no_action_policy
from baselines.random_policy import random_policy
from baselines.degree_policy import degree_policy
from baselines.load_policy import load_policy
from baselines.greedy_policy import greedy_policy

from agents.vector_dqn_agent import VectorDQNAgent, flatten_obs, infer_state_dim


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------
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
    return CascadingMitigationEnv(
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


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def stringify(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return str(value.tolist())
    return str(value)


# ---------------------------------------------------------------------
# DQN loading and diagnostics
# ---------------------------------------------------------------------
def build_dqn_agent(args: argparse.Namespace) -> VectorDQNAgent:
    if not os.path.exists(args.dqn_path):
        raise FileNotFoundError(f"DQN checkpoint not found: {args.dqn_path}")

    device = select_device(args.device)

    temp_env = make_env(args, seed=args.seed + 999999)
    temp_obs, _ = temp_env.reset(seed=args.seed + 999999)

    state_dim = infer_state_dim(temp_obs)
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

    try:
        agent.load(args.dqn_path, map_location=device)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load the DQN checkpoint. This usually means the current "
            "observation flattening/state_dim or action_dim is different from the one "
            "used during training. Retrain Vector-DQN with the current code and the "
            "same environment parameters.\n"
            f"Original error:\n{exc}"
        ) from exc

    agent.epsilon = 0.0
    return agent


def get_dqn_top_actions(
    agent: VectorDQNAgent,
    obs: dict,
    env: CascadingMitigationEnv,
    top_k: int = 5,
) -> tuple[str, float, float]:
    """
    Return readable top-k valid Q actions and simple Q statistics.

    Returns
    -------
    tuple
        top_action_string, max_valid_q, mean_valid_q
    """
    state = flatten_obs(obs)
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
    valid_mask = action_mask == 1

    with torch.no_grad():
        q_values = agent.q_net(state_tensor).squeeze(0).detach().cpu().numpy()

    q_valid = np.where(valid_mask, q_values, -np.inf)
    valid_q_values = q_values[valid_mask]

    if len(valid_q_values) == 0:
        return "", np.nan, np.nan

    top_ids = np.argsort(q_valid)[::-1][:top_k]
    parts = []
    for action_id in top_ids:
        if not np.isfinite(q_valid[action_id]):
            continue
        action_name = action_to_string(int(action_id), env.N, env.edge_list)
        parts.append(f"{int(action_id)}:{action_name}:Q={q_valid[action_id]:.4f}")

    return " | ".join(parts), float(np.max(valid_q_values)), float(np.mean(valid_q_values))


# ---------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------
def build_policy(args: argparse.Namespace) -> tuple[str, Callable, VectorDQNAgent | None]:
    policy = args.policy.lower()

    if policy == "no_action":
        return "No-action", no_action_policy, None

    if policy == "random":
        return "Random", random_policy, None

    if policy == "degree":
        return "Degree", degree_policy, None

    if policy == "load":
        return "Load-ratio", load_policy, None

    if policy == "greedy":
        def policy_fn(obs: dict, env: CascadingMitigationEnv) -> int:
            return greedy_policy(
                obs,
                env,
                max_candidates=args.greedy_max_candidates,
            )
        return "Greedy", policy_fn, None

    if policy == "dqn":
        agent = build_dqn_agent(args)

        def policy_fn(obs: dict, env: CascadingMitigationEnv) -> int:
            return int(agent.take_action(obs, evaluate=True))

        return "Vector-DQN", policy_fn, agent

    raise ValueError(
        f"Unknown policy: {args.policy}. "
        "Expected one of: no_action, random, degree, load, greedy, dqn."
    )


# ---------------------------------------------------------------------
# Main inspection logic
# ---------------------------------------------------------------------
def inspect_policy(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    policy_name, policy_fn, dqn_agent = build_policy(args)

    step_rows: list[dict] = []
    episode_rows: list[dict] = []

    for ep in range(args.episodes):
        seed = args.seed + ep
        env = make_env(args, seed=seed)
        obs, info = env.reset(seed=seed)

        done = False
        episode_return = 0.0
        step_idx = 0
        action_type_counter = Counter()
        positive_benefit_steps = 0
        final_info = info

        while not done:
            action_mask = np.asarray(obs["action_mask"], dtype=np.int32)
            valid_action_count = int(np.sum(action_mask == 1))

            dqn_top_actions = ""
            dqn_max_valid_q = np.nan
            dqn_mean_valid_q = np.nan

            if dqn_agent is not None:
                dqn_top_actions, dqn_max_valid_q, dqn_mean_valid_q = get_dqn_top_actions(
                    agent=dqn_agent,
                    obs=obs,
                    env=env,
                    top_k=args.top_k,
                )

            action = int(policy_fn(obs, env))
            action_readable_before = action_to_string(action, env.N, env.edge_list)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = bool(terminated or truncated)

            benefit = safe_float(next_info.get("benefit"))
            if np.isfinite(benefit) and benefit > 0:
                positive_benefit_steps += 1

            action_type = str(next_info.get("action_type", "unknown"))
            action_type_counter[action_type] += 1

            step_row = {
                "policy": policy_name,
                "episode": ep,
                "seed": seed,
                "step": step_idx,
                "initial_failed_nodes": stringify(next_info.get("initial_failed_nodes")),
                "valid_action_count": valid_action_count,
                "selected_action_id": action,
                "selected_action_name_before_step": action_readable_before,
                "action_name": str(next_info.get("action_name", action_readable_before)),
                "action_type": action_type,
                "target": stringify(next_info.get("target")),
                "edge": stringify(next_info.get("edge")),
                "reward": safe_float(reward),
                "benefit": benefit,
                "damage_do_nothing": safe_float(next_info.get("damage_do_nothing")),
                "damage_after_action": safe_float(next_info.get("damage_after_action")),
                "damage": safe_float(next_info.get("damage")),
                "new_failed_nodes": stringify(next_info.get("new_failed_nodes")),
                "failed_fraction": safe_float(next_info.get("failed_fraction")),
                "lcc_ratio": safe_float(next_info.get("lcc_ratio")),
                "lost_load_ratio": safe_float(next_info.get("lost_load_ratio")),
                "served_load_ratio": safe_float(next_info.get("served_load_ratio")),
                "budget_left": safe_float(next_info.get("budget_left")),
                "budget_used": safe_float(next_info.get("budget_used")),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "dqn_top_valid_actions": dqn_top_actions,
                "dqn_max_valid_q": dqn_max_valid_q,
                "dqn_mean_valid_q": dqn_mean_valid_q,
            }

            step_rows.append(step_row)

            episode_return += float(reward)
            final_info = next_info
            obs = next_obs
            step_idx += 1

        total_steps = max(step_idx, 1)

        episode_row = {
            "policy": policy_name,
            "episode": ep,
            "seed": seed,
            "episode_return": float(episode_return),
            "steps": int(step_idx),
            "positive_benefit_steps": int(positive_benefit_steps),
            "positive_benefit_ratio": float(positive_benefit_steps / total_steps),
            "protect_node_count": int(action_type_counter.get("protect_node", 0)),
            "disconnect_edge_count": int(action_type_counter.get("disconnect_edge", 0)),
            "do_nothing_count": int(action_type_counter.get("do_nothing", 0)),
            "invalid_count": int(action_type_counter.get("invalid", 0)),
            "final_failed_fraction": safe_float(final_info.get("failed_fraction")),
            "final_lcc_ratio": safe_float(final_info.get("lcc_ratio")),
            "final_lost_load_ratio": safe_float(final_info.get("lost_load_ratio")),
            "final_served_load_ratio": safe_float(final_info.get("served_load_ratio")),
            "final_damage": safe_float(final_info.get("damage")),
            "budget_used": safe_float(final_info.get("budget_used")),
            "initial_failed_nodes": stringify(final_info.get("initial_failed_nodes")),
        }

        episode_rows.append(episode_row)

    return step_rows, episode_rows


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------
def save_csv(path: str, rows: list[dict]) -> None:
    if len(rows) == 0:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_episode_summary(episode_rows: list[dict]) -> None:
    if len(episode_rows) == 0:
        return

    keys = [
        "episode_return",
        "positive_benefit_ratio",
        "protect_node_count",
        "disconnect_edge_count",
        "do_nothing_count",
        "final_failed_fraction",
        "final_lcc_ratio",
        "final_lost_load_ratio",
        "final_damage",
        "budget_used",
    ]

    print("\n" + "=" * 90)
    print("Policy Behavior Summary")
    print("=" * 90)

    for key in keys:
        values = np.asarray([safe_float(row.get(key)) for row in episode_rows], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        print(f"{key:<28} mean={np.mean(values):>10.4f}  std={np.std(values):>10.4f}")

    print("=" * 90)


def print_first_steps(step_rows: list[dict], max_rows: int = 20) -> None:
    print("\nFirst inspected steps:")
    print("-" * 120)
    header = (
        f"{'ep':>3} {'st':>3} {'action':<32} {'reward':>9} {'benefit':>9} "
        f"{'D_none':>9} {'D_act':>9} {'failed':>8} {'lcc':>8} {'budget':>8}"
    )
    print(header)
    print("-" * 120)

    for row in step_rows[:max_rows]:
        print(
            f"{row['episode']:>3} "
            f"{row['step']:>3} "
            f"{row['action_name']:<32} "
            f"{row['reward']:>9.4f} "
            f"{row['benefit']:>9.4f} "
            f"{row['damage_do_nothing']:>9.4f} "
            f"{row['damage_after_action']:>9.4f} "
            f"{row['failed_fraction']:>8.4f} "
            f"{row['lcc_ratio']:>8.4f} "
            f"{row['budget_left']:>8.4f}"
        )

    print("-" * 120)


def plot_behavior(policy_slug: str, step_rows: list[dict], episode_rows: list[dict]) -> None:
    os.makedirs("outputs/figures", exist_ok=True)

    # 1. Action type counts
    action_types = [row["action_type"] for row in step_rows]
    counts = Counter(action_types)

    if len(counts) > 0:
        labels = list(counts.keys())
        values = [counts[label] for label in labels]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values)
        plt.xlabel("Action type")
        plt.ylabel("Count")
        plt.title(f"Policy Behavior: Action Type Counts ({policy_slug})")
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"outputs/figures/behavior_{policy_slug}_action_types.png", dpi=300)
        plt.close()

    # 2. Benefit over step index
    benefits_by_step: dict[int, list[float]] = defaultdict(list)
    for row in step_rows:
        benefit = safe_float(row.get("benefit"))
        if np.isfinite(benefit):
            benefits_by_step[int(row["step"])].append(benefit)

    if len(benefits_by_step) > 0:
        steps = sorted(benefits_by_step.keys())
        means = [float(np.mean(benefits_by_step[s])) for s in steps]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, means, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Mean benefit")
        plt.title(f"Policy Behavior: Mean Counterfactual Benefit ({policy_slug})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"outputs/figures/behavior_{policy_slug}_benefit_by_step.png", dpi=300)
        plt.close()

    # 3. Episode return distribution
    returns = [safe_float(row.get("episode_return")) for row in episode_rows]
    returns = [x for x in returns if np.isfinite(x)]

    if len(returns) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(returns, bins=20)
        plt.xlabel("Episode return")
        plt.ylabel("Frequency")
        plt.title(f"Policy Behavior: Episode Return Distribution ({policy_slug})")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"outputs/figures/behavior_{policy_slug}_return_hist.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Which policy to inspect
    parser.add_argument(
        "--policy",
        type=str,
        default="dqn",
        choices=["no_action", "random", "degree", "load", "greedy", "dqn"],
        help="Policy to inspect.",
    )

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

    # DQN
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

    # Inspection
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--greedy_max_candidates",
        type=int,
        default=None,
        help="Maximum actions evaluated by greedy policy. Default: all valid actions.",
    )
    parser.add_argument(
        "--print_rows",
        type=int,
        default=20,
        help="Number of step rows printed in terminal.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    policy_slug = args.policy.lower()

    step_rows, episode_rows = inspect_policy(args)

    os.makedirs("outputs/results", exist_ok=True)

    step_path = f"outputs/results/behavior_{policy_slug}_steps.csv"
    episode_path = f"outputs/results/behavior_{policy_slug}_episodes.csv"

    save_csv(step_path, step_rows)
    save_csv(episode_path, episode_rows)

    print_episode_summary(episode_rows)
    print_first_steps(step_rows, max_rows=args.print_rows)

    plot_behavior(policy_slug, step_rows, episode_rows)

    print("\nSaved files:")
    print(f"  {step_path}")
    print(f"  {episode_path}")
    print("\nSaved figures:")
    print(f"  outputs/figures/behavior_{policy_slug}_action_types.png")
    print(f"  outputs/figures/behavior_{policy_slug}_benefit_by_step.png")
    print(f"  outputs/figures/behavior_{policy_slug}_return_hist.png")

    if args.policy.lower() == "dqn":
        print("\nDQN-specific note:")
        print("  Check the 'dqn_top_valid_actions' column in the step CSV.")
        print("  It shows the top valid actions ranked by Q-value before each selected action.")


if __name__ == "__main__":
    main()
