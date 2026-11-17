import argparse
import time
from pathlib import Path

import torch

from dqn.agent import DQNAgent
from dqn.wrappers import make_env


ENV_ID = "ALE/Pong-v5"


def parse_args():
    parser = argparse.ArgumentParser(description="Play Pong with a trained DQN")
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--model", type=str, default="dqn_model.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with train.py"
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    render_mode = None if args.no_render else "human"
    env = make_env(args.env_id, render_mode=render_mode)
    num_actions = int(env.action_space.n)

    agent = DQNAgent(num_actions=num_actions, device=device)
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Playing {args.episodes} episode(s) on {args.env_id}...")

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            # greedy actions only (no exploration)
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            steps += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"Episode {ep}: reward={total_reward:.1f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    main()
