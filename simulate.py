import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from data_prep import load_data
from elo import build_ratings, STARTING_RATING
from train_data import build_training_df
from train_model import train_goal_models

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
    pass

if __name__ == "__main__":

    df = load_data("data/results.csv")
    ratings = build_ratings(df)

    training_df = build_training_df(df)
    home_model, away_model, scaler = train_goal_models(training_df)

    team_a = "Argentina"
    team_b = "Canada"
    rating_a = ratings.get(team_a, STARTING_RATING)
    rating_b = ratings.get(team_b, STARTING_RATING)

    lambda_home, lambda_away = get_expected_goals(
        rating_a, rating_b, is_home=0,  # neutral venue, like a WC match
        home_model=home_model, away_model=away_model, scaler=scaler
    )

    print(f"Expected goals: {team_a}: {lambda_home:.2f}, {team_b}: {lambda_away:.2f}")
