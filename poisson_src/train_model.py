import pandas as pd
from sklearn.linear_model import PoissonRegressor #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from data.data_prep import load_data, DATA_PATH
from poisson_src.train_data import build_training_df

def train_goal_models(training_df: pd.DataFrame):
    X = training_df[["rating_diff", "is_home"]]
    y_home = training_df["home_goals"]
    y_away = training_df["away_goals"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    home_model = PoissonRegressor()
    home_model.fit(X_scaled, y_home)

    away_model = PoissonRegressor()
    away_model.fit(X_scaled, y_away)

    return home_model, away_model, scaler

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    training_df = build_training_df(df)

    home_model, away_model, scaler = train_goal_models(training_df)

    print("Home goals model coefficients:")
    print("  rating_diff:", home_model.coef_[0])
    print("  is_home:", home_model.coef_[1])
    print("  intercept:", home_model.intercept_)

    print("\nAway goals model coefficients:")
    print("  rating_diff:", away_model.coef_[0])
    print("  is_home:", away_model.coef_[1])
    print("  intercept:", away_model.intercept_)
