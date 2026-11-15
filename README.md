# Deep Q-Learning on Atari Pong

## Project Introduction

This project implements a **Deep Q-Network (DQN)** agent from scratch using **PyTorch** and **Gymnasium**. The agent learns to play Atari Pong by interacting with the environment, storing experience in a replay buffer, and updating a convolutional Q-network with Bellman targets from a target network.

We do **not** use Stable-Baselines3, RLlib, CleanRL, TorchRL, or any other library that provides a ready-made DQN.

Environment and dependency management use **uv**.

## Group Members

| Name | Role / experiment results |
|---|---|
| Berissa Muyizere | Shared training code; experiments in `experiments/berissa/`; `Berissa_Hyperparameter_experiment_results.md` |
| Hugues Munezero | Experiments in `experiments/hugues/`; `Hugues_Hyperparameter_experiment_results.md` |
| Jacques Twizeyimana | Experiments in `experiments/jacques/`; `Jacques_Hyperparameter_experiment_results.md` |

Task allocation is documented in `Task_allocation_sheet.pdf`.

## Selected Atari Environment

- **Environment:** `ALE/Pong-v5`
- **Objective:** Control the right paddle and score points against the opponent.
- **Reward:** Typically `+1` when the agent scores and `-1` when the opponent scores.
- **Why Pong:** It provides a clear reward signal and is suitable for formative DQN experiments compared with harder Atari games.

Observations are preprocessed to grayscale 84×84 frames and stacked (4 frames) so the network can see motion.

## DQN Architecture

The Q-network is a CNN implemented with `torch.nn` (no pretrained weights):

| Layer | Details |
|---|---|
| Input | Stacked frames `(4, 84, 84)` |
| Conv1 | 32 filters, kernel 8, stride 4 + ReLU |
| Conv2 | 64 filters, kernel 4, stride 2 + ReLU |
| Conv3 | 64 filters, kernel 3, stride 1 + ReLU |
| Fully connected | 512 units + ReLU |
| Output | Q-value for each discrete action |

### Main algorithm components

- **Online network:** predicts `Q(s, a)` and is trained by gradient descent  
- **Target network:** provides stable Bellman targets; weights are copied from the online network periodically  
- **Replay buffer:** stores `(s, a, r, s', done)` and samples random mini-batches  
- **ε-greedy policy:** random action with probability ε, otherwise `argmax Q(s, ·)`; ε decays during training  
- **Loss:** Smooth L1 between `Q(s, a)` and `r + γ max_{a'} Q_target(s', a')` (target is `r` when the episode ends)

### Network / algorithm diagram

![DQN architecture](plots/dqn_architecture.png)

## Explanation of the Training Process

1. Reset the Pong environment and preprocess frames (grayscale, resize, frame stack).
2. Select an action with ε-greedy using the online network.
3. Step the environment, clip the reward to `[-1, 1]`, and store the transition in the replay buffer.
4. Decay ε toward a minimum exploration rate.
5. After a warm-up period, sample a mini-batch and update the online network with the Bellman loss.
6. Periodically copy online weights into the target network.
7. Log episode reward, episode length, and steps; save models and training plots.

Training is run with `train.py`. Evaluation / live play uses `play.py`, which loads saved weights and acts greedily (no exploration).

### Best performing experiment (Berissa)

| Item | Value |
|---|---|
| Experiment | 1 |
| Hyperparameters | lr=`1e-4`, gamma=`0.99`, batch=`32`, ε: `1.0 → 0.01`, decay=`1e-6` |
| Training steps | 1,000,000 |
| Episodes | 731 |
| Best mean-100 reward | -16.83 |
| Last-10 mean reward | -15.1 |
| Training plot | `plots/berissa/exp01_training.png` |
| Saved model | `dqn_model.pth` (from `experiments/berissa/exp01/dqn_model_best.pth`) |

Full hyperparameter tables for each member are in:

- `Berissa_Hyperparameter_experiment_results.md`
- `Hugues_Hyperparameter_experiment_results.md`
- `Jacques_Hyperparameter_experiment_results.md`

## Gameplay Demonstration

The trained agent can be run live with:

```bash
uv sync
uv run python play.py --model dqn_model.pth
```

`play.py` reconstructs the same CNN, loads `dqn_model.pth`, and selects actions greedily while rendering the game.

### Demo video

A short recording of the best agent playing Pong (same clip in two places):

- [YouTube](https://youtu.be/j0WhIRzJJIk)
- [Local file in the repo](demos/screen-capture.webm)

## How to reproduce (brief)

```bash
uv sync
uv run python train.py --member <name> --experiment <n> [hyperparameters...]
uv run python play.py --model dqn_model.pth
```
