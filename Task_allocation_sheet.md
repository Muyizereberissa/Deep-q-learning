# Task Allocation Sheet

Project: DQN on Atari Pong  
Group size: 3

Export this file to PDF before final submission. Also add the presentation booking PDF.

## Members

| # | Name | GitHub |
|---|---|---|
| 1 | Berissa Muyizere | Muyizereberissa |
| 2 | Hugues Munezero | |
| 3 | Jacques Twizeyimana | |

## Shared code (everyone must understand this)

| Part | Files | Who built first | Who presents |
|---|---|---|---|
| CNN Q-network | `dqn/network.py` | Berissa | Berissa |
| Replay buffer | `dqn/replay_buffer.py` | Hugues | Hugues |
| Agent / Bellman / target net / epsilon | `dqn/agent.py` | Jacques | Jacques |
| Env wrappers | `dqn/wrappers.py` | Berissa | Berissa |
| Training loop | `train.py` | Berissa | Berissa |
| Evaluation | `play.py` | Hugues | Hugues |
| README + diagram | `README.md`, `plots/dqn_architecture.png` | Jacques | Jacques |

## Individual work (required)

| Member | 10 experiments | Model folder | Results markdown | Plots |
|---|---|---|---|---|
| Berissa Muyizere | yes | `experiments/berissa/` | `Berissa_Hyperparameter_experiment_results.md` | `plots/berissa/` |
| Hugues Munezero | yes | `experiments/hugues/` | `Hugues_Hyperparameter_experiment_results.md` | `plots/hugues/` |
| Jacques Twizeyimana | yes | `experiments/jacques/` | `Jacques_Hyperparameter_experiment_results.md` | `plots/jacques/` |

## Presentation (10 min)

| Section | Time | Speaker |
|---|---|---|
| Intro / why Pong | 1 min | Berissa |
| DQN components | 2 min | Hugues + Jacques |
| Experiments (each member) | 2 min x 3 | Berissa, Hugues, Jacques |
| Live `play.py` demo | 2 min | Hugues |
| Q&A | 10 min | all |

## Still left to finish

1. Add GitHub usernames for Hugues and Jacques
2. Hugues and Jacques: run their 10 experiments, save models/plots, fill their results tables
3. Record demo video of `play.py` with `dqn_model.pth`
4. Book presentation slot and add PDF

## Done (Berissa)

- 10 experiments in `experiments/berissa/exp01..exp10`
- Plots in `plots/berissa/`
- Results table filled in `Berissa_Hyperparameter_experiment_results.md`
- Best model copied to `dqn_model.pth` (from exp01)
