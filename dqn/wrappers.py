from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

gym.register_envs(ale_py)


class FrameStack(gym.Wrapper):
    # keep the last N frames so the agent can see motion
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return np.stack(list(self.frames), axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(list(self.frames), axis=0), reward, terminated, truncated, info


def make_env(env_id="ALE/Pong-v5", render_mode=None, frame_skip=4, seed=None):
    # frameskip=1 here because AtariPreprocessing handles skipping
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=frame_skip,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStack(env, num_stack=4)
    if seed is not None:
        env.reset(seed=seed)
    return env
