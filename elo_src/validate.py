from data.data_prep import load_data, DATA_PATH
from elo_src.elo import build_ratings
from elo_src.predict import predict_match

TEST_MATCHES = [
    ("Argentina", "France", "draw", True),             # 2022 WC Final (3-3, Argentina won on penalties)
    ("Argentina", "Croatia", "home_win", True),        # 2022 WC Semifinal (3-0)
    ("France", "Morocco", "home_win", True),           # 2022 WC Semifinal (2-0)
    ("Morocco", "Spain", "draw", True),                # 2022 WC Round of 16 (0-0, Morocco won on penalties)
    ("Germany", "Japan", "away_win", True),            # 2022 WC Group Stage (1-2, major upset)
    ("Saudi Arabia", "Argentina", "home_win", True),   # 2022 WC Group Stage (2-1, huge upset)
    ("Brazil", "Croatia", "draw", True),               # 2022 WC Quarterfinal (1-1, Croatia won on penalties)
    ("Spain", "England", "away_win", True),            # Euro 2024 Final (1-2)
    ("England", "Netherlands", "home_win", True),      # Euro 2024 Semifinal (2-1)
    ("Argentina", "Colombia", "home_win", True),       # Copa America 2024 Final (1-0)
]

def run_validation(ratings: dict[str, float], matches: list[tuple]) -> None:
    correct = 0
    total = len(matches)

    for home, away, actual_result, is_neutral in matches:
        rating_home = ratings.get(home)
        rating_away = ratings.get(away)

        result = predict_match(rating_home, rating_away, is_neutral)

        predicted_result = max(result, key=result.get)
        is_correct = predicted_result == actual_result
        if is_correct:
            correct += 1

        print(f"{home} vs {away}")
        print(f"  Home win: {result['home_win']:.1%}  "
              f"Draw: {result['draw']:.1%}  "
              f"Away win: {result['away_win']:.1%}")
        print(f"  Actual: {actual_result}  |  "
              f"Predicted top pick: {predicted_result}  |  "    
              f"{'✓' if is_correct else '✗'}")
        print()

    print(f"Top-pick accuracy: {correct}/{total} ({correct/total:.1%})")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    ratings = build_ratings(df)
    run_validation(ratings, TEST_MATCHES)