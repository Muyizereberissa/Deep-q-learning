# Hyperparameter Experiment Results

Member Name: Berissa Muyizere  
Environment: ALE/Pong-v5

Notes:
- Experiment 1: 1,000,000 steps on CPU
- Experiments 2–10: 250,000 steps on Mac GPU (`mps`), with faster epsilon decay so exploration still finishes

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 1e-6 | Best overall. 731 episodes / 1M steps. Clear learning after ~500k steps. best mean100 = -16.83, last-10 reward = -15.1, peak episode reward = -11. Episode length rose from ~1000 to ~2400 steps. Plot: `plots/berissa/exp01_training.png` |
| 2 | 5e-5 | 0.99 | 32 | 1.0 | 0.01 | 4e-6 | Lower LR learned slowly in 250k steps. best mean100 = -19.67, last-10 = -19.4. Little improvement vs random losing play. |
| 3 | 2.5e-4 | 0.99 | 32 | 1.0 | 0.01 | 4e-6 | Higher LR looked unstable / no real progress. best mean100 = -19.75, last-10 = -20.9. Short episodes (~777 last-10). |
| 4 | 1e-4 | 0.95 | 32 | 1.0 | 0.01 | 4e-6 | Lower gamma hurt long-term scoring. best mean100 = -20.0, last-10 = -20.1. Episode length rose late, but rewards stayed weak. |
| 5 | 1e-4 | 0.90 | 32 | 1.0 | 0.01 | 4e-6 | Even lower gamma. best mean100 = -20.11, last-10 = -19.0. No clear upward reward trend. |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 4e-6 | Best of the short runs. Batch 64 helped. best mean100 = -17.88, last-10 = -12.0, peak reward = -7. Longer rallies near the end. |
| 7 | 1e-4 | 0.99 | 16 | 1.0 | 0.01 | 4e-6 | Worst run. Small batch was noisy. best mean100 = -20.12, last-10 = -21.0 (still losing almost every point). |
| 8 | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 4e-6 | Higher epsilon_end kept too much exploration. best mean100 = -19.6, last-10 = -18.7. Mild late length increase only. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 2e-6 | Decay too slow for 250k steps (final eps ≈ 0.50). best mean100 = -19.85, last-10 = -20.2. Still exploring too much. |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 2e-5 | Very fast decay. best mean100 = -20.0, last-10 = -19.9. Dropped exploration early and stayed weak. |

## Best config
- Experiment: **1**
- Settings: lr=1e-4, gamma=0.99, batch=32, eps 1.0→0.01, decay=1e-6, **1M steps**
- Why: only run with a clear learning curve; best mean100 reward (-16.83) and improving episode length
- Model: `experiments/berissa/exp01/dqn_model_best.pth` (also copied to `dqn_model.pth`)
- Plot: `plots/berissa/exp01_training.png`

### Best-run training stats (for docs)
| Metric | Value |
|---|---|
| Episodes | 731 |
| Steps | 1,000,000 |
| Best mean100 reward | -16.83 |
| Last-10 mean reward | -15.1 |
| Mean episode length | ~1368 |
| Last-10 mean episode length | ~2429 |

## Worst config
- Experiment: **7**
- Settings: same as baseline but batch size **16**, 250k steps
- Why: noisiest updates; finished at last-10 reward -21.0 with no learning signal

## What helped / hurt
- Helped: longer training (exp1), gamma=0.99, batch 32–64, enough epsilon decay
- Hurt: very low gamma (0.90/0.95), batch 16, too-slow or too-fast epsilon decay on short runs, high learning rate on short runs

## Commands used

```bash
# exp1 (long CPU run)
uv run python train.py --member berissa --experiment 1 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 1e-6

# exp2-10 (faster: 250k steps + mps)
uv run python train.py --member berissa --experiment 2 --total-steps 250000 --learning-rate 5e-5 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 4e-6
uv run python train.py --member berissa --experiment 3 --total-steps 250000 --learning-rate 2.5e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 4e-6
uv run python train.py --member berissa --experiment 4 --total-steps 250000 --learning-rate 1e-4 --gamma 0.95 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 4e-6
uv run python train.py --member berissa --experiment 5 --total-steps 250000 --learning-rate 1e-4 --gamma 0.90 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 4e-6
uv run python train.py --member berissa --experiment 6 --total-steps 250000 --learning-rate 1e-4 --gamma 0.99 --batch-size 64 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 4e-6
uv run python train.py --member berissa --experiment 7 --total-steps 250000 --learning-rate 1e-4 --gamma 0.99 --batch-size 16 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 4e-6
uv run python train.py --member berissa --experiment 8 --total-steps 250000 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 4e-6
uv run python train.py --member berissa --experiment 9 --total-steps 250000 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 2e-6
uv run python train.py --member berissa --experiment 10 --total-steps 250000 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 2e-5
```
