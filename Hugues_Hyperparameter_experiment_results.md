# Hyperparameter Experiment Results

Member Name: Hugues Munezero  
Environment: ALE/Pong-v5

Pong scores run from -21 (agent loses every point) to +21 (agent wins every point). A score around -20 means the agent has barely learned anything. All runs use seed 42, buffer 100,000, train every 4 steps, target update every 1,000 steps.

**Important:** Experiment 1 was trained for **1,000,000 steps**. Experiments 2 to 10 were trained for **250,000 steps**. The epsilon decay values for experiments 1 and 3 below are taken from each run's `config.json` (4e-6), not from the first draft of this table (1e-6).

Epsilon is decayed linearly per step: `epsilon = start - decay * step`, never lower than epsilon end. With decay 1e-6, epsilon after 250k steps is still 0.75, so the agent is acting randomly about 75% of the time.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.02 | 4e-6 | **Only run that learned.** Ran 1M steps. Stayed near -20 for the first ~200k steps, then rose steadily: 20-episode average passed -15 at ~390k steps, -10 at ~640k, and 0 at ~870k. Final 10-episode average +0.1 (about even with the opponent), best single episode +16, episode length grew from ~900 to ~3,200 steps as rallies got longer. Epsilon reached 0.02 at step 245k, so the agent was almost fully greedy for the remaining 750k steps. |
| 2 | 3e-4 | 0.99 | 32 | 1.0 | 0.02 | 1e-6 | No learning. Last 10 episodes averaged -20.4 and the late-run average stayed at about -20.3. Epsilon was still 0.75 at the end. Higher learning rate gave no visible gain in 250k steps. |
| 3 | 5e-5 | 0.99 | 32 | 1.0 | 0.02 | 4e-6 | No learning in 250k steps (last 10 average -19.9). Epsilon reached 0.02, so exploration was not the problem. The low learning rate was too slow to make progress in this budget. Compare with experiment 1, which used the same decay but took 190k+ steps to start improving at lr 1e-4. |
| 4 | 1e-4 | 0.98 | 32 | 1.0 | 0.02 | 1e-6 | No learning. Last 10 average -20.3, last 50 episodes about -20.2. Gamma 0.98 vs 0.99 made no measurable difference. |
| 5 | 1e-4 | 0.92 | 32 | 1.0 | 0.02 | 1e-6 | No learning. Last 10 average -20.3, last 50 episodes about -20.3 (among the lowest). A low gamma discounts the point-scoring reward heavily, which is a plausible reason it did not improve, but the gap to the other runs is too small to be sure. |
| 6 | 1e-4 | 0.99 | 48 | 1.0 | 0.02 | 1e-6 | Slight improvement late in training (last 10 average -19.5, last 50 about -19.6). Still essentially random play. |
| 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.02 | 1e-6 | Slight improvement (last 10 average -19.1, last 50 about -19.4). Larger batches gave smoother updates but also fewer updates per second of wall time. Still near random. |
| 8 | 1e-4 | 0.99 | 32 | 0.8 | 0.01 | 1e-6 | **Worst result.** Last 10 average -20.6 and last 50 about -20.4, the lowest of all runs. Epsilon was lower than in the other 1e-6 runs (0.55 at the end), so more of its actions came from a barely-trained network, which likely performed no better than random. |
| 9 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 2e-6 | **Best of the 250k-step runs.** Last 10 average -18.4, last 50 about -19.2, and episode length began to grow (about 1,400 steps at the end), a sign the agent started surviving longer. Epsilon fell to 0.5, so it exploited its learned values more than the 1e-6 runs. |
| 10 | 1e-4 | 0.99 | 32 | 1.0 | 0.1 | 1e-6 | No clear learning. Last 10 average -19.2, last 50 about -19.9. Keeping a high epsilon floor of 0.1 mattered little because epsilon never got down to 0.1 in 250k steps at this decay rate. |

## Comparison on equal terms (first 250,000 steps, average of the last 50 episodes)

| Exp | 1 | 9 | 7 | 6 | 10 | 3 | 4 | 5 | 2 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| Epsilon decay | 4e-6 | 2e-6 | 1e-6 | 1e-6 | 1e-6 | 4e-6 | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| Avg reward | -18.8 | -19.2 | -19.4 | -19.6 | -19.9 | -20.2 | -20.2 | -20.3 | -20.3 | -20.4 |

## Best config
- **Experiment: 1** (lr 1e-4, gamma 0.99, batch 32, epsilon 1.0 to 0.02, decay 4e-6, 1M steps).
- **Why:** it is the only experiment that clearly learned to play Pong. Its average reward climbed from about -20 to about 0 (last quarter of training averaged -4.4, final 10 episodes +0.1) and episodes grew from ~900 to ~3,200 steps, meaning the agent returned the ball much more often. Moderate learning rate (1e-4) and gamma 0.99 gave stable improvement, and a fast epsilon decay (greedy by 245k steps) let the agent spend most of the run exploiting what it had learned.
- **Caveat:** experiment 1 also had 4 times more training steps than the others, so part of its advantage is simply training time. Even at 250k steps it was already the best (-18.8), but we cannot separate the effect of the longer run from the effect of the hyperparameters. Among the 250k-step runs, **experiment 9** was best.

## Worst config
- **Experiment: 8** (epsilon start 0.8, epsilon end 0.01, decay 1e-6).
- **Why:** it had the lowest reward (-20.6 over the last 10 episodes), essentially the same as random play. Starting epsilon at 0.8 meant less early exploration, so the replay buffer held less varied experience, and the network was still too untrained for greedy actions to help. This explanation is a hypothesis. Experiments 2, 4 and 5 were also within about 0.2 of it, so the difference may be noise from a single seed.

## What helped and what hurt
- **Epsilon decay speed** gave the clearest pattern: 4e-6 (exp 1) > 2e-6 (exp 9) > 1e-6 (most others). With 1e-6, epsilon was still 0.5 to 0.75 at 250k steps, so most runs were mostly random for their whole length.
- **Learning rate:** 3e-4 (exp 2) and 5e-5 (exp 3) both did no better than 1e-4 within 250k steps. 1e-4 is the only value shown to work over a long run.
- **Gamma:** 0.92, 0.98 and 0.99 gave results within noise of each other in 250k steps. No conclusion.
- **Batch size:** 48 and 128 were slightly better than 32 (last 50 about -19.6 and -19.4 versus -19.9 for the nearest 32 run, experiment 10), but these are small gaps from one seed.
- **Not enough training:** 250,000 steps is too few for Pong. Experiment 1 only started to improve after about 200k steps and needed ~870k to reach 0. Most of experiments 2 to 10 did not have a chance to show differences.

## Final configuration and justification
Use experiment 1: lr 1e-4, gamma 0.99, batch 32, epsilon 1.0 to 0.02 with decay 4e-6, trained for 1,000,000 steps. It is the only run that reached positive-level play, and its saved model (`exp01/dqn_model_best.pth`) is the one to use for `dqn_model.pth` and the live demo.

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

> Note: the saved `config.json` files show experiments 1 and 3 actually ran with `--epsilon-decay 4e-6`, and experiment 1 with 1,000,000 total steps (the others used 250,000). Use those values if you re-run them.
