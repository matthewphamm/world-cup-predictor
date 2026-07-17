import pandas as pd
import numpy as np

DATA_PATH = 'data/results.csv'

def load_data(path: str) -> pd.DataFrame:
    results = pd.read_csv(DATA_PATH)
    results['date'] = pd.to_datetime(results['date'])
    return results

def summarize(df: pd.DataFrame) -> None:
    total_rows = len(df)
    min_val = df['date'].min()
    max_val = df['date'].max()
    unique_teams = len(np.unique(df[['home_team', 'away_team']]))

    print('Total Rows:', total_rows)
    print('Date Range:', min_val, 'to', max_val)
    print('Number of Unique Teams:', unique_teams)
    
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    summarize(df)