import numpy as np
import pandas as pd
from data.data_prep import load_data,DATA_PATH
from poisson_src.train_data import build_training_df
from poisson_src.train_model import train_goal_models
from poisson_src.simulate import simulate_match

CUTOFF_DATE = "2010-01-01"
N_SIMULATIONS = 1000 


def get_actual_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    elif home_goals < away_goals:
        return "away_win"
    else:
        return "draw"


def run_poisson_backtest(training_df: pd.DataFrame) -> None:
    train_subset = training_df[training_df["date"] < CUTOFF_DATE]
    test_subset = training_df[training_df["date"] >= CUTOFF_DATE]

    home_model, away_model, scaler = train_goal_models(train_subset)

    X_test = test_subset[["rating_diff", "is_home"]]
    X_test_scaled = scaler.transform(X_test)

    lambda_home_array = home_model.predict(X_test_scaled)
    lambda_away_array = away_model.predict(X_test_scaled)

    correct = 0
    scored = 0

    for lambda_home, lambda_away, home_goals, away_goals in zip(
        lambda_home_array,
        lambda_away_array,
        test_subset["home_goals"],
        test_subset["away_goals"],
    ):
        result = simulate_match(lambda_home, lambda_away, n_simulations=N_SIMULATIONS)
        predicted_result = max(result, key=result.get)
        actual_result = get_actual_result(home_goals, away_goals)

        scored += 1
        if predicted_result == actual_result:
            correct += 1

    print(f"Train set size: {len(train_subset)}, Test set size: {len(test_subset)}")
    print(f"Matches scored: {scored}")
    print(f"Correct top-pick predictions: {correct}")
    print(f"Accuracy: {correct/scored:.1%}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    training_df = build_training_df(df)
    run_poisson_backtest(training_df)