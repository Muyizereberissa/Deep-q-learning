# Hyperparameter Experiment Results

Member Name: Hugues Munezero  
Environment: ALE/Pong-v5

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.02 | 1e-6 | |
| 2 | 3e-4 | 0.99 | 32 | 1.0 | 0.02 | 1e-6 | |
| 3 | 5e-5 | 0.99 | 32 | 1.0 | 0.02 | 1e-6 | |
| 4 | 1e-4 | 0.98 | 32 | 1.0 | 0.02 | 1e-6 | |
| 5 | 1e-4 | 0.92 | 32 | 1.0 | 0.02 | 1e-6 | |
| 6 | 1e-4 | 0.99 | 48 | 1.0 | 0.02 | 1e-6 | |
| 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.02 | 1e-6 | |
| 8 | 1e-4 | 0.99 | 32 | 0.8 | 0.01 | 1e-6 | |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 2e-6 | |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.1 | 1e-6 | |

## Best config
- Experiment:
- Why:

## Worst config
- Experiment:
- Why:

## Commands

```bash
uv run python train.py --member hugues --experiment 1 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.02 --epsilon-decay 1e-6
uv run python train.py --member hugues --experiment 2 --learning-rate 3e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.02 --epsilon-decay 1e-6
uv run python train.py --member hugues --experiment 3 --learning-rate 5e-5 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.02 --epsilon-decay 1e-6
uv run python train.py --member hugues --experiment 4 --learning-rate 1e-4 --gamma 0.98 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.02 --epsilon-decay 1e-6
uv run python train.py --member hugues --experiment 5 --learning-rate 1e-4 --gamma 0.92 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.02 --epsilon-decay 1e-6
uv run python train.py --member hugues --experiment 6 --learning-rate 1e-4 --gamma 0.99 --batch-size 48 --epsilon-start 1.0 --epsilon-end 0.02 --epsilon-decay 1e-6
uv run python train.py --member hugues --experiment 7 --learning-rate 1e-4 --gamma 0.99 --batch-size 128 --epsilon-start 1.0 --epsilon-end 0.02 --epsilon-decay 1e-6
uv run python train.py --member hugues --experiment 8 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 0.8 --epsilon-end 0.01 --epsilon-decay 1e-6
uv run python train.py --member hugues --experiment 9 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 2e-6
uv run python train.py --member hugues --experiment 10 --learning-rate 1e-4 --gamma 0.99 --batch-size 32 --epsilon-start 1.0 --epsilon-end 0.1 --epsilon-decay 1e-6
```
