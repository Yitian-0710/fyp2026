"""
experiments/train_graph_dqn.py

Train Graph-DQN on CascadingMitigationEnv.

Graph-DQN V1 replaces Vector-DQN's flattened state representation with:
    graph observation -> dense GCN encoder -> Q-value heads

Recommended first run from project root:

    python experiments/train_graph_dqn.py --episodes 1000 --N 20 --max_steps 5 --budget 3 --no_disconnect --initial_failure_strategy highest_load --protect_strength 2.0 --protect_duration 5

Recommended N=50 run:

    python experiments/train_graph_dqn.py --episodes 3000 --N 50 --max_steps 5 --budget 3 --no_disconnect --initial_failure_strategy highest_load --protect_strength 5.0 --protect_duration 5 --hidden_dim 128 --num_gnn_layers 2 --batch_size 128 --minimal_size 1000 --epsilon_decay 0.997

Important:
    For Graph-DQN V1, use --no_disconnect first.
    This version learns protect-node and do-nothing actions.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import deque
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------
# Make imports work when running:
#   python experiments/train_graph_dqn.py
# ---------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.graph_dqn_agent import GraphDQNAgent
from agents.graph_replay_buffer import GraphReplayBuffer
from envs.cascading_mitigation_env import CascadingMitigationEnv
from utils.graph_obs_utils import infer_graph_obs_dims, summarize_graph_obs


# ---------------------------------------------------------------------
# Utilities
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


def moving_average(values: list[float], window: int = 50) -> np.ndarray:
    if len(values) == 0:
        return np.array([])

    if len(values) < window:
        return np.asarray(values, dtype=float)

    values_arr = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values_arr, kernel, mode="valid")


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


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_agent(
    agent: GraphDQNAgent,
    args: argparse.Namespace,
    eval_episodes: int = 20,
    seed_offset: int = 10000,
) -> dict[str, float]:
    """
    Evaluate current Graph-DQN with greedy actions.
    """
    returns = []
    failed_fractions = []
    lcc_ratios = []
    lost_load_ratios = []
    served_load_ratios = []
    budget_used_list = []
    damage_list = []
    steps_list = []
    protect_counts = []
    do_nothing_counts = []
    positive_benefit_ratios = []

    for ep in range(eval_episodes):
        seed = args.seed + seed_offset + ep
        env = make_env(args, seed=seed)
        obs, info = env.reset(seed=seed)

        done = False
        episode_return = 0.0
        final_info = info
        protect_count = 0
        do_nothing_count = 0
        positive_benefit_count = 0
        step_count = 0

        while not done:
            action = agent.take_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += float(reward)
            done = terminated or truncated
            final_info = info

            action_type = info.get("action_type", "")
            if action_type == "protect_node":
                protect_count += 1
            elif action_type == "do_nothing":
                do_nothing_count += 1

            benefit = info.get("benefit", None)
            if benefit is not None and float(benefit) > 0:
                positive_benefit_count += 1

            step_count += 1

        returns.append(float(episode_return))
        failed_fractions.append(float(final_info["failed_fraction"]))
        lcc_ratios.append(float(final_info["lcc_ratio"]))
        lost_load_ratios.append(float(final_info["lost_load_ratio"]))
        served_load_ratios.append(float(final_info["served_load_ratio"]))
        budget_used_list.append(float(final_info["budget_used"]))
        damage_list.append(float(final_info["damage"]))
        steps_list.append(float(final_info["step"]))
        protect_counts.append(float(protect_count))
        do_nothing_counts.append(float(do_nothing_count))
        positive_benefit_ratios.append(
            float(positive_benefit_count / max(step_count, 1))
        )

    return {
        "eval_return": float(np.mean(returns)),
        "eval_failed_fraction": float(np.mean(failed_fractions)),
        "eval_lcc_ratio": float(np.mean(lcc_ratios)),
        "eval_lost_load_ratio": float(np.mean(lost_load_ratios)),
        "eval_served_load_ratio": float(np.mean(served_load_ratios)),
        "eval_budget_used": float(np.mean(budget_used_list)),
        "eval_damage": float(np.mean(damage_list)),
        "eval_steps": float(np.mean(steps_list)),
        "eval_protect_count": float(np.mean(protect_counts)),
        "eval_do_nothing_count": float(np.mean(do_nothing_counts)),
        "eval_positive_benefit_ratio": float(np.mean(positive_benefit_ratios)),
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    print("=" * 80)
    print("Graph-DQN Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"N: {args.N}")
    print(f"Network type: {args.network_type}")
    print(f"Episodes: {args.episodes}")
    print(f"Failure model: {args.failure_model}")
    print(f"Redistribution: {args.redistribution_mode}")
    print(f"Allow disconnect: {not args.no_disconnect}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"GNN layers: {args.num_gnn_layers}")
    print(f"Pooling: {args.pooling}")
    print("=" * 80)

    if not args.no_disconnect:
        print(
            "WARNING: Graph-DQN V1 is mainly designed for --no_disconnect. "
            "Edge-disconnection actions are not modeled with an edge Q-head yet. "
            "They may be effectively ignored by the placeholder edge Q-values."
        )

    env = make_env(args, seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    dims = infer_graph_obs_dims(obs)

    print("Observation summary:")
    print("  " + summarize_graph_obs(obs))
    print(f"Initial info: {info}")
    print(f"Graph obs dims: {dims}")
    print("=" * 80)

    agent = GraphDQNAgent.from_obs(
        obs=obs,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        device=device,
        double_dqn=not args.no_double_dqn,
        grad_clip_norm=args.grad_clip_norm,
        dropout=args.dropout,
        pooling=args.pooling,
        use_global_features=not args.no_global_features,
    )

    replay_buffer = GraphReplayBuffer(args.buffer_size)

    train_returns: list[float] = []
    train_failed_fraction: list[float] = []
    train_lcc_ratio: list[float] = []
    train_lost_load_ratio: list[float] = []
    train_damage: list[float] = []
    loss_window = deque(maxlen=100)

    log_rows: list[dict[str, Any]] = []
    best_eval_return = -float("inf")
    best_eval_damage = float("inf")

    progress = tqdm(range(1, args.episodes + 1), desc="Training Graph-DQN")

    for episode in progress:
        # Use different seeds so the agent sees a distribution of graphs/failures.
        seed = args.seed + episode
        env = make_env(args, seed=seed)
        obs, info = env.reset(seed=seed)

        done = False
        episode_return = 0.0
        final_info = info

        while not done:
            action = agent.take_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            replay_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            obs = next_obs
            episode_return += float(reward)
            final_info = info

            if replay_buffer.size() >= args.minimal_size:
                batch = replay_buffer.sample(args.batch_size)
                update_info = agent.update(batch)
                loss_window.append(float(update_info["loss"]))

        agent.decay_epsilon()

        train_returns.append(float(episode_return))
        train_failed_fraction.append(float(final_info["failed_fraction"]))
        train_lcc_ratio.append(float(final_info["lcc_ratio"]))
        train_lost_load_ratio.append(float(final_info["lost_load_ratio"]))
        train_damage.append(float(final_info["damage"]))

        if episode % args.eval_interval == 0:
            eval_info = evaluate_agent(
                agent=agent,
                args=args,
                eval_episodes=args.eval_episodes,
                seed_offset=10000 + episode * 10,
            )

            # Main checkpoint criterion: larger eval return.
            if eval_info["eval_return"] > best_eval_return:
                best_eval_return = eval_info["eval_return"]
                agent.save("outputs/checkpoints/graph_dqn_best.pt")

            # Additional checkpoint based on final damage.
            if eval_info["eval_damage"] < best_eval_damage:
                best_eval_damage = eval_info["eval_damage"]
                agent.save("outputs/checkpoints/graph_dqn_best_damage.pt")

            row = {
                "episode": episode,
                "train_return_mean_50": float(np.mean(train_returns[-50:])),
                "train_failed_fraction_mean_50": float(np.mean(train_failed_fraction[-50:])),
                "train_lcc_ratio_mean_50": float(np.mean(train_lcc_ratio[-50:])),
                "train_lost_load_ratio_mean_50": float(np.mean(train_lost_load_ratio[-50:])),
                "train_damage_mean_50": float(np.mean(train_damage[-50:])),
                "epsilon": float(agent.epsilon),
                "loss_mean_100": float(np.mean(loss_window)) if len(loss_window) > 0 else np.nan,
                "buffer_size": replay_buffer.size(),
                **eval_info,
            }
            log_rows.append(row)

            progress.set_postfix(
                {
                    "ret50": f"{row['train_return_mean_50']:.3f}",
                    "eval_ret": f"{eval_info['eval_return']:.3f}",
                    "eval_fail": f"{eval_info['eval_failed_fraction']:.3f}",
                    "eval_dmg": f"{eval_info['eval_damage']:.3f}",
                    "eps": f"{agent.epsilon:.3f}",
                }
            )

        elif episode % args.log_interval == 0:
            progress.set_postfix(
                {
                    "ret50": f"{np.mean(train_returns[-50:]):.3f}",
                    "fail50": f"{np.mean(train_failed_fraction[-50:]):.3f}",
                    "dmg50": f"{np.mean(train_damage[-50:]):.3f}",
                    "eps": f"{agent.epsilon:.3f}",
                    "buffer": replay_buffer.size(),
                }
            )

    agent.save("outputs/checkpoints/graph_dqn_last.pt")

    # ------------------------------------------------------------------
    # Save logs
    # ------------------------------------------------------------------
    log_path = "outputs/results/graph_dqn_training_log.csv"
    if len(log_rows) > 0:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    raw_path = "outputs/results/graph_dqn_raw_returns.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "return",
                "failed_fraction",
                "lcc_ratio",
                "lost_load_ratio",
                "damage",
            ]
        )
        for i, (ret, failed, lcc, lost, damage) in enumerate(
            zip(
                train_returns,
                train_failed_fraction,
                train_lcc_ratio,
                train_lost_load_ratio,
                train_damage,
            ),
            start=1,
        ):
            writer.writerow([i, ret, failed, lcc, lost, damage])

    # ------------------------------------------------------------------
    # Plot curves
    # ------------------------------------------------------------------
    ma_return = moving_average(train_returns, window=args.plot_window)

    plt.figure(figsize=(10, 5))
    plt.plot(train_returns, alpha=0.25, label="Episode return")
    if len(ma_return) > 0:
        start_x = args.plot_window if len(train_returns) >= args.plot_window else 1
        plt.plot(
            np.arange(len(ma_return)) + start_x,
            ma_return,
            linewidth=2,
            label=f"Moving average ({args.plot_window})",
        )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Graph-DQN Training Return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/graph_dqn_training_return.png", dpi=300)
    plt.close()

    ma_failed = moving_average(train_failed_fraction, window=args.plot_window)

    plt.figure(figsize=(10, 5))
    plt.plot(train_failed_fraction, alpha=0.25, label="Episode failed fraction")
    if len(ma_failed) > 0:
        start_x = args.plot_window if len(train_failed_fraction) >= args.plot_window else 1
        plt.plot(
            np.arange(len(ma_failed)) + start_x,
            ma_failed,
            linewidth=2,
            label=f"Moving average ({args.plot_window})",
        )
    plt.xlabel("Episode")
    plt.ylabel("Final failed fraction")
    plt.title("Graph-DQN Final Failed Fraction")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/graph_dqn_failed_fraction.png", dpi=300)
    plt.close()

    ma_damage = moving_average(train_damage, window=args.plot_window)

    plt.figure(figsize=(10, 5))
    plt.plot(train_damage, alpha=0.25, label="Episode damage")
    if len(ma_damage) > 0:
        start_x = args.plot_window if len(train_damage) >= args.plot_window else 1
        plt.plot(
            np.arange(len(ma_damage)) + start_x,
            ma_damage,
            linewidth=2,
            label=f"Moving average ({args.plot_window})",
        )
    plt.xlabel("Episode")
    plt.ylabel("Final damage")
    plt.title("Graph-DQN Final Damage")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/graph_dqn_damage.png", dpi=300)
    plt.close()

    print("\nGraph-DQN training finished.")
    print(f"Best eval return: {best_eval_return:.4f}")
    print(f"Best eval damage: {best_eval_damage:.4f}")
    print("Saved checkpoints:")
    print("  outputs/checkpoints/graph_dqn_best.pt")
    print("  outputs/checkpoints/graph_dqn_best_damage.pt")
    print("  outputs/checkpoints/graph_dqn_last.pt")
    print("Saved figures:")
    print("  outputs/figures/graph_dqn_training_return.png")
    print("  outputs/figures/graph_dqn_failed_fraction.png")
    print("  outputs/figures/graph_dqn_damage.png")
    print("Saved logs:")
    print(f"  {log_path}")
    print(f"  {raw_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
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
        help="Disable edge-disconnection actions. Recommended for Graph-DQN V1.",
    )

    # Graph-DQN model
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "sum", "max", "mean_max"],
    )
    parser.add_argument(
        "--no_global_features",
        action="store_true",
        help="Do not concatenate global features into the graph embedding.",
    )

    # DQN training
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)

    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)

    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--minimal_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--no_double_dqn",
        action="store_true",
        help="Disable Double-DQN target calculation.",
    )
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    # Logging and evaluation
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--plot_window", type=int, default=50)

    # System
    parser.add_argument("--seed", type=int, default=42)
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
    train(args)
