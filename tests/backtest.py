from data.data_prep import load_data, DATA_PATH
from elo_src.elo import update_ratings, STARTING_RATING
from elo_src.predict import predict_match

SCORING_START_YEAR = 2010

def get_actual_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    elif home_score < away_score:
        return "away_win"
    else:
        return "draw"


def run_backtest(df) -> None:
    ratings: dict[str, float] = {}
    correct = 0
    scored = 0

    for row in df.itertuples():
        rating_home = ratings.get(row.home_team, STARTING_RATING)
        rating_away = ratings.get(row.away_team, STARTING_RATING)

        if (row.date.year >= SCORING_START_YEAR):
            result = predict_match(rating_home, rating_away, row.neutral)
            predicted_result = max(result, key=result.get)
            actual_result = get_actual_result(row.home_score, row.away_score)

            scored += 1
            if predicted_result == actual_result:
                correct += 1

        new_rating_home, new_rating_away = update_ratings(
            rating_home, rating_away,
            row.home_score, row.away_score,
            row.neutral
        )
        ratings[row.home_team] = new_rating_home
        ratings[row.away_team] = new_rating_away

    print(f"Matches scored (from {SCORING_START_YEAR} onward): {scored}")
    print(f"Correct top-pick predictions: {correct}")
    print(f"Accuracy: {correct/scored:.1%}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    run_backtest(df)