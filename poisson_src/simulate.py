import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from data.data_prep import load_data
from elo_src.elo import build_ratings, STARTING_RATING
from poisson_src.train_data import build_training_df
from poisson_src.train_model import train_goal_models

N_SIMULATIONS = 10_000

def get_expected_goals(rating_home: float, rating_away: float, is_home: int,
                   home_model: PoissonRegressor, away_model: PoissonRegressor,
                   scaler: StandardScaler) -> tuple[float, float]:
    rating_diff = rating_home - rating_away

    X_df = pd.DataFrame([{"rating_diff": rating_diff, "is_home": is_home}])

    X_scaled_df = scaler.transform(X_df)

    lambda_home = home_model.predict(X_scaled_df)[0]
    lambda_away = away_model.predict(X_scaled_df)[0]

    return lambda_home, lambda_away

def simulate_match(lambda_home: float, lambda_away: float,
                   n_simulations: int = N_SIMULATIONS) -> dict[str, float]:
    simulated_home = np.random.poisson(lambda_home, size=n_simulations)
    simulated_away = np.random.poisson(lambda_away, size=n_simulations)

    home_wins = (simulated_home > simulated_away).sum()
    draws     = (simulated_home == simulated_away).sum()
    away_wins = (simulated_home < simulated_away).sum()

    home_wins_prob = home_wins / n_simulations
    draws_prob     = draws / n_simulations
    away_wins_prob = away_wins / n_simulations

    results = {
        "home_win": home_wins_prob,
        "draw": draws_prob,
        "away_win": away_wins_prob
    }

    return results

if __name__ == "__main__":

    df = load_data("data/results.csv")
    ratings = build_ratings(df)

    training_df = build_training_df(df)
    home_model, away_model, scaler = train_goal_models(training_df)

    team_a = "Argentina"
    team_b = "France"
    rating_a = ratings.get(team_a, STARTING_RATING)
    rating_b = ratings.get(team_b, STARTING_RATING)

    lambda_home, lambda_away = get_expected_goals(
        rating_a, rating_b, is_home=0,  # neutral venue, like a WC match
        home_model=home_model, away_model=away_model, scaler=scaler
    )

    print(f"Expected goals: {team_a}: {lambda_home:.2f}, {team_b}: {lambda_away:.2f}")

    result = simulate_match(lambda_home, lambda_away)

    print(f"\n{team_a} vs {team_b} (Monte Carlo, {N_SIMULATIONS:,} simulations):")
    print(f"  {team_a} win: {result['home_win']:.1%}")
    print(f"  Draw:          {result['draw']:.1%}")
    print(f"  {team_b} win: {result['away_win']:.1%}")
