# ami-trim-5

Lab 1: Implementation and Performance Analysis of Q-Learning Using an Epsilon-Greedy Policy.

## Contents

- `Q_Learning_Epsilon_Greedy.ipynb` - the lab notebook. Trains a tabular Q-learning agent on Gymnasium's `FrozenLake-v1` (slippery, discrete, non-continuous) across five training budgets (500, 1000, 2000, 5000, 10000 episodes), with an epsilon-greedy policy that starts fully exploratory and decays toward a small exploration floor. Records per-episode reward, plots cumulative reward and a 100-episode moving average, tests each learned policy for 100 greedy episodes, and visualises the environment and the extracted policy.

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Q_Learning_Epsilon_Greedy.ipynb
```
