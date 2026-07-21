import pandas as pd
from data.data_prep import load_data, DATA_PATH
from elo_src.elo import update_ratings, STARTING_RATING

def build_training_df(df: pd.DataFrame) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    rows = []

    for row in df.itertuples():
        rating_home = ratings.get(row.home_team, STARTING_RATING)
        rating_away = ratings.get(row.away_team, STARTING_RATING)

        rating_diff = rating_home - rating_away

        if row.neutral:
            is_home = 0
        else:
            is_home = 1

        rows.append({
            "date": row.date,
            "rating_diff": rating_diff,
            "is_home": is_home,
            "home_goals": row.home_score,
            "away_goals": row.away_score,
        })

        new_rating_home, new_rating_away = update_ratings(rating_home, rating_away, 
                                                          row.home_score, row.away_score,
                                                          row.neutral)
        ratings[row.home_team] = new_rating_home
        ratings[row.away_team] = new_rating_away
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    training_df = build_training_df(df)
    print(training_df.head())
    print(training_df.describe())