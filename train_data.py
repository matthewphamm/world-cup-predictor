import pandas as pd
from data_prep import load_data, DATA_PATH
from elo import build_ratings, STARTING_RATING

def build_training_df(df: pd.DataFrame) -> pd.DataFrame:
    pass

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    training_df = build_training_df(df)
    print(training_df.head())
    print(training_df.describe())