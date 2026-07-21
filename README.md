# National Football Match Predictor

A football match outcome predictor built on Elo ratings, Poisson regression, and Monte Carlo simulation - trained and validated on 49,000+ international matches (1872-2026).

## Overview

This project predicts win/draw/loss probabilities for international football matches by:
1. Building an Elo-style strength rating for every national team from previous match results
2. Using those ratings to predict expected goals via a Poisson regression model
3. Running Monte Carlo simulation (10,000 trials per match) to convert expected goals into win/draw/loss probabilities
Two prediction methods were built and compared: a lightweight Elo-based heuristic, and a more complex Elo → Poisson regression → Monte Carlo pipeline. Both were validated using a walk-forward backtest.
