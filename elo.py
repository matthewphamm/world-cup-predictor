import pandas as pd
from data_prep import load_data

DATA_PATH = 'data/results.csv'
STARTING_RATE = 1500 # can be changed
K_FACTOR = 20 # can be changed

def expected_score(rating_a: float, rating_b: float) -> float: # type: ignore
    pass

def actual_score(home_score: int, away_score: int) -> float: # type: ignore
    pass

def update_ratings(rating_home: float, rating_away: float,
                   home_score: int, away_score: int, 
                    k: float = K_FACTOR) -> tuple[float, float]: # type: ignore
    pass

def build_ratings(df: pd.DataFrame) -> dict[str, float]: # type: ignore
    pass