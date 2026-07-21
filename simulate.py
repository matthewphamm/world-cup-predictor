import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore

N_SIMULATIONS = 10_000

def expected_goals(rating_home: float, rating_away: float, is_home: int,
                   home_model: PoissonRegressor, away_model: PoissonRegressor,
                   scaler: StandardScaler) -> tuple[float, float]:
    pass