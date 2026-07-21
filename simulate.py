import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore

N_SIMULATIONS = 10_000

def expected_goals(rating_home: float, rating_away: float, is_home: int,
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

