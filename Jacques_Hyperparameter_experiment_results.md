# Hyperparameter Experiment Results

Member Name: Jacques Twizeyimana  
Environment: ALE/Pong-v5

Notes:
- All 10 experiments: 250,000 steps on Mac GPU (`mps`), one seed (42) each
- "First 50 → last 50" is the mean episode reward (and length) over the first and last 50 episodes of the run
- Epsilon decays linearly per step. With decay 3e-6, epsilon only reaches 0.25 by the end of a 250k-step run, so experiments 1–7 still acted randomly about 25% of the time at the end

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 3e-6 | Baseline. 266 episodes. Reward -20.5 → -20.2, length 897 → 1035. Only a faint upward trend. last-10 = -19.6, final eps = 0.25. Plot: `plots/jacques/exp01_training.png` |
| 2 | 1e-3 | 0.99 | 32 | 1.0 | 0.01 | 3e-6 | High LR was unstable and got worse. 286 episodes. Reward -20.4 → -20.6, length 925 → 836. last-10 = -20.8. |
| 3 | 1e-5 | 0.99 | 32 | 1.0 | 0.01 | 3e-6 | LR too small to learn in 250k steps. 280 episodes. Reward -20.3 → -20.4, length 914 → 885. last-10 = -20.5. |
| 4 | 1e-4 | 0.999 | 32 | 1.0 | 0.01 | 3e-6 | Longer horizon helped slightly. 271 episodes. Reward -20.5 → -20.0, length 885 → 976. last-10 = -18.9, peak = -17. |
| 5 | 1e-4 | 0.85 | 32 | 1.0 | 0.01 | 3e-6 | Short horizon (about 7 steps) cannot link returning the ball to scoring. 272 episodes. Reward -20.2 → -20.4, length 957 → 912. last-10 = -20.4. |
| 6 | 1e-4 | 0.99 | 8 | 1.0 | 0.01 | 3e-6 | Tiny batch gave noisy updates and no learning. 288 episodes. Reward -20.3 → -20.7, length 920 → 856. last-10 = -20.7. |
| 7 | 1e-4 | 0.99 | 96 | 1.0 | 0.01 | 3e-6 | **Best run, the only clear learning curve.** 211 episodes. Reward -20.2 → -18.1, length 937 → 1660 (longer rallies). last-10 = -17.1, peak = -13. Plot: `plots/jacques/exp07_training.png` |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.001 | 8e-6 | Decay raised to 8e-6 so epsilon actually reaches 0.001 (around step 125k); at 3e-6 this run would repeat exp 1. Going greedy early did not help. 281 episodes. Reward -20.4 → -20.9, length 904 → 909. last-10 = -20.7. |
| 9 | 5e-5 | 0.95 | 64 | 1.0 | 0.01 | 4e-6 | Second best. Larger batch and full epsilon decay (0.01 by the end) gave steady improvement despite the lower LR and gamma. 235 episodes. Reward -20.5 → -18.7, length 902 → 1377. last-10 = -19.2, peak = -15. |
| 10 | 2e-4 | 0.99 | 32 | 0.5 | 0.01 | 4e-6 | **Worst run.** Starting at half exploration plus a higher LR got worse over time. 306 episodes. Reward -20.7 → -21.0, length 885 → 777 (shorter rallies). last-10 = -21.0. |

## Best config
- Experiment: **7**
- Settings: lr=1e-4, gamma=0.99, batch=96, eps 1.0→0.01, decay=3e-6, 250k steps
- Why: the only run whose reward and episode length rose clearly. The larger batch gives a smoother gradient estimate on every update. Experiment 9 (batch 64) was second, which points to batch size as the most useful setting in short runs. Berissa found the same (batch 64 was her best 250k run, batch 16 her worst).
- Model: `experiments/jacques/exp07/dqn_model.pth` (final weights) and `dqn_model_best.pth`
- Plot: `plots/jacques/exp07_training.png`

### Best-run training stats
| Metric | Value |
|---|---|
| Episodes | 211 |
| Steps | 250,000 |
| Best mean100 reward | -18.67 |
| Last-10 mean reward | -17.1 |
| Peak episode reward | -13 |
| Mean episode length | ~1183 |
| Last-10 mean episode length | ~1882 |

## Worst config
- Experiment: **10**
- Settings: lr=2e-4, gamma=0.99, batch=32, eps **0.5**→0.01, decay=4e-6, 250k steps
- Why: reward and episode length both went down during training. Last-10 reward was -21.0, so it lost every point.

## What helped / hurt
- Helped: larger batch size (64–96), gamma 0.99–0.999, learning rate around 1e-4
- Hurt: high (1e-3) or very low (1e-5) learning rate, gamma 0.85, batch 8, starting epsilon at 0.5
- Epsilon: in experiments 1–7 epsilon ended at 0.25, which held back every run including the best one. Experiment 8 shows that decaying faster alone does not fix a short run.

## Final configuration and limits
- Recommended: lr=1e-4, gamma=0.99, batch 64–96, eps 1.0→0.01 with a decay that reaches the floor before the end (4e-6 for 250k steps), and 1M steps if time allows. Hugues's 1M-step experiment 1 reached an average near 0, so training length mattered more than any single hyperparameter.
- Each setting was run once with one seed, so differences of about 1 point are within noise.
- Best mean100 is not a reliable comparison here: in experiments 1, 3, 5, 6 and 10 the best checkpoint was saved before learning started (step 10,000), from early random episodes. The first 50 → last 50 trend and last-10 reward are better measures.

## Commands used

```bash
uv run python train.py --member jacques --experiment 1 --total-steps 250000 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 3e-6
uv run python train.py --member jacques --experiment 2 --total-steps 250000 --learning-rate 1e-3 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 3e-6
uv run python train.py --member jacques --experiment 3 --total-steps 250000 --learning-rate 1e-5 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 3e-6
uv run python train.py --member jacques --experiment 4 --total-steps 250000 --learning-rate 1e-4 --gamma 0.999 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 3e-6
uv run python train.py --member jacques --experiment 5 --total-steps 250000 --learning-rate 1e-4 --gamma 0.85 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 3e-6
uv run python train.py --member jacques --experiment 6 --total-steps 250000 --learning-rate 1e-4 --gamma 0.99 --batch-size 8 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 3e-6
uv run python train.py --member jacques --experiment 7 --total-steps 250000 --learning-rate 1e-4 --gamma 0.99 --batch-size 96 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 3e-6
uv run python train.py --member jacques --experiment 8 --total-steps 250000 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.001 --epsilon-decay 8e-6
uv run python train.py --member jacques --experiment 9 --total-steps 250000 --learning-rate 5e-5 --gamma 0.95 --batch-size 64 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 4e-6
uv run python train.py --member jacques --experiment 10 --total-steps 250000 --learning-rate 2e-4 --gamma 0.99 --batch-size 32 --epsilon-start 0.5 --epsilon-end 0.01 --epsilon-decay 4e-6
```
