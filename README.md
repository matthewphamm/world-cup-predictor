# National Football Match Predictor

A football match outcome predictor built on Elo ratings, Poisson regression, and Monte Carlo simulation - trained and validated on 49,000+ international matches (1872-2026).

## Overview

This project predicts win/draw/loss probabilities for international football matches by:
1. Building an Elo-style strength rating for every national team from previous match results
2. Using those ratings to predict expected goals via a Poisson regression model
3. Running Monte Carlo simulation (10,000 trials per match) to convert expected goals into win/draw/loss probabilities

Two prediction methods were built and compared: a lightweight Elo-based heuristic, and a more complex Elo → Poisson regression → Monte Carlo pipeline. Both were validated using a walk-forward backtest.

## Results

| Method | Test period | Matches | Top-pick accuracy |
|---|---|---|---|
| Elo heuristic | 2010–2026 | 15,929 | **58.6%** |
| Elo → Poisson regression → Monte Carlo | 2010–2026 | 15,929 | 57.5% |

**Key finding:** the simpler Elo-based heuristic outperformed the more complex Poisson + Monte Carlo pipeline on a fair, identical test set. This is attributed to two design trade-offs made for computational practicality: the Poisson model's coefficients were trained on a single fixed split (after 2010) rather than continuously updated, while Elo ratings updated after every match throughout the entire test and the Monte Carlo simulation treats home/away goals as statistically independent, which is known to under-predict low-scoring draws.

## How it works
 
**Elo ratings** — every team starts at 1500. After each historical match, both teams' ratings update based on the gap between the expected result (from the Elo logistic formula) and the actual result, scaled by a K-factor. A home-advantage adjustment is applied at prediction time for non-neutral matches.
 
**Poisson regression** — a separate scikit-learn `PoissonRegressor` is trained for home and away expected goals, using standardized Elo rating difference and a home/away flag as features.
 
**Monte Carlo simulation** — for a given matchup, 10,000 independent scorelines are sampled from `Poisson(λ_home)` and `Poisson(λ_away)`, and outcome probabilities are computed from the resulting win/draw/loss tally.
 
**Validation** — both methods were tested with a walk-forward backtest: for every match in the test, predictions were made using only data available before that match's date, then updated with the real result — avoiding lookahead bias.

## Project Structure
```
national-mens-team-predictor/
├── data/
│   ├── data_prep.py
│   └── results.csv
├── elo_src/          
│   ├── elo.py
│   ├── predict.py
│   └── validate.py
├── poisson_src/       
│   ├── simulate.py
│   ├── train_data.py
│   └── train_model.py
├── tests/
│   ├── backtest_poisson.py
│   └── backtest.py
├── README.md
└── requirements.txt
```
## Setup
 
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
## Usage
 
```bash
python3 -m elo_src.predict             # single-match prediction, Elo heuristic
python3 -m poisson_src.simulate        # single-match prediction, Poisson + Monte Carlo
python3 -m tests.backtest              # run the Elo walk-forward backtest
python3 -m tests.backtest_poisson      # run the Poisson train/test backtest
```
