import pandas as pd
from data_prep import load_data

DATA_PATH = 'data/results.csv'
STARTING_RATE = 1500 # can be changed
K_FACTOR = 20 # can be changed

def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Return team A's probability of winning, given both Elo ratings.
    """
    results = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    return results

def actual_score(home_score: int, away_score: int) -> float: 
    """
    Convert a real match result into an Elo 'actual score' for the home team.
    """
    win = 1.0
    draw = 0.5
    loss = 0.0

    if home_score > away_score:
        return win
    elif home_score == away_score:
        return draw
    else:
        return loss

def update_ratings(rating_home: float, rating_away: float,
                   home_score: int, away_score: int, 
                    k: float = K_FACTOR) -> tuple[float, float]: # type: ignore
    pass

def build_ratings(df: pd.DataFrame) -> dict[str, float]: # type: ignore
    pass