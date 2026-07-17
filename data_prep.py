import pandas as pd

DATA_PATH = 'data/results.csv'

def load_data(path: str) -> pd.DataFrame:
    pass

def clean_data(path: pd.DataFrame) -> pd.DataFrame:
    pass

def summarize(df: pd.DataFrame) -> None:
    pass

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df = clean_data(df)
    summarize(df)