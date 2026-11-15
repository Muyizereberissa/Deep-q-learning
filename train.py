import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dqn.agent import DQNAgent
from dqn.wrappers import make_env


ENV_ID = "ALE/Pong-v5"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on Atari Pong")
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--member", type=str, default="shared")
    parser.add_argument("--experiment", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=1e-6)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--save-best-as-root", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def plot_training(history, plot_path):
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(history["episode_rewards"])
    axes[0].set_title("Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")

    axes[1].plot(history["episode_lengths"])
    axes[1].set_title("Episode Length")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")

    axes[2].plot(history["episode_end_steps"], history["episode_rewards"])
    axes[2].set_title("Reward vs Steps")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("Episode Reward")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def save_history_csv(history, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "length", "end_step"])
        for i, (r, length, step) in enumerate(
            zip(
                history["episode_rewards"],
                history["episode_lengths"],
                history["episode_end_steps"],
            ),
            start=1,
        ):
            writer.writerow([i, r, length, step])


def main():
    args = parse_args()

    if args.smoke_test:
        args.total_steps = 2000
        args.learning_starts = 200
        args.log_every = 500
        args.buffer_size = 5000

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Environment:", args.env_id)

    env = make_env(args.env_id, render_mode=None, seed=args.seed)
    num_actions = int(env.action_space.n)

    agent = DQNAgent(
        num_actions=num_actions,
        device=device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
    )

    member_dir = Path("experiments") / args.member / f"exp{args.experiment:02d}"
    member_dir.mkdir(parents=True, exist_ok=True)
    model_path = member_dir / "dqn_model.pth"
    best_model_path = member_dir / "dqn_model_best.pth"
    plot_path = Path("plots") / args.member / f"exp{args.experiment:02d}_training.png"

    config = vars(args).copy()
    config["device"] = str(device)
    config["num_actions"] = num_actions
    with (member_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    state, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode = 0
    best_mean_reward = float("-inf")
    recent_rewards = []

    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_end_steps": [],
        "losses": [],
    }

    start_time = time.time()
    print("Starting training...")

    for step in range(1, args.total_steps + 1):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # clip reward like in the DQN paper
        clipped_reward = float(max(-1.0, min(1.0, reward)))
        agent.store(state, action, clipped_reward, next_state, done)
        agent.decay_epsilon()

        state = next_state
        episode_reward += float(reward)
        episode_length += 1

        if step >= args.learning_starts and step % args.train_freq == 0:
            loss = agent.learn()
            if loss is not None:
                history["losses"].append(loss)

        if done:
            episode += 1
            history["episode_rewards"].append(episode_reward)
            history["episode_lengths"].append(episode_length)
            history["episode_end_steps"].append(step)
            recent_rewards.append(episode_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            mean_reward = sum(recent_rewards) / len(recent_rewards)
            print(
                f"Episode {episode:4d} | step {step:7d} | "
                f"reward {episode_reward:6.1f} | length {episode_length:4d} | "
                f"eps {agent.epsilon:.3f} | mean100 {mean_reward:6.1f}"
            )

            if mean_reward > best_mean_reward and len(recent_rewards) >= 5:
                best_mean_reward = mean_reward
                agent.save(best_model_path)

            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            sps = step / max(elapsed, 1e-8)
            print(
                f"[log] step={step} eps={agent.epsilon:.3f} "
                f"buffer={len(agent.replay_buffer)} sps={sps:.1f}"
            )

    agent.save(model_path)
    if args.save_best_as_root:
        source = best_model_path if best_model_path.exists() else model_path
        Path("dqn_model.pth").write_bytes(source.read_bytes())
        print("Saved root model -> dqn_model.pth")

    save_history_csv(history, member_dir / "training_log.csv")
    plot_training(history, plot_path)

    last_rewards = history["episode_rewards"][-10:]
    summary = {
        "episodes": episode,
        "total_steps": args.total_steps,
        "final_epsilon": agent.epsilon,
        "best_mean100_reward": (
            best_mean_reward if best_mean_reward > float("-inf") else None
        ),
        "last_10_mean_reward": (
            sum(last_rewards) / len(last_rewards) if last_rewards else None
        ),
        "plot": str(plot_path),
        "model": str(model_path),
        "best_model": str(best_model_path) if best_model_path.exists() else None,
    }
    with (member_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    print("Training finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
