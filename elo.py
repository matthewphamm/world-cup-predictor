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
                    k: float = K_FACTOR) -> tuple[float, float]:
    """
    Return the new (rating_home, rating_away) after applying the Elo update.
    """
    expected_home = expected_score(rating_home, rating_away)
    expected_away = 1.0 - expected_home
    
    actual_home = actual_score(home_score, away_score)
    actual_away = 1.0 - actual_home
    
    new_rating_home = rating_home + k * (actual_home - expected_home)
    new_rating_away = rating_away + k * (actual_away - expected_away)

    new_ratings = (new_rating_home, new_rating_away)
    return new_ratings

def build_ratings(df: pd.DataFrame) -> dict[str, float]: 
    ratings: dict[str, float] = {}

    for row in df.itertuples():
        home_team = row.home_team
        away_team = row.away_team
        home_score = row.home_score
        away_score = row.away_score

        if home_team not in ratings:
            rating_home = ratings.get(home_team, STARTING_RATE)
        else:
            rating_home = ratings.get(home_team) 
        
        if away_team not in ratings:
            rating_away = ratings.get(away_team, STARTING_RATE) 
        else:
            rating_away = ratings.get(away_team) 
        
        new_rating_home, new_rating_away = update_ratings(rating_home, rating_away, home_score, away_score) 

        ratings[home_team] = new_rating_home
        ratings[away_team] = new_rating_away

    return ratings