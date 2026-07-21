from elo_src.elo import expected_score, build_ratings, HOME_ADVANTAGE
from data.data_prep import load_data, DATA_PATH

# Tunable constants for draw probability 
DRAW_MAX = 0.34             # highest possible draw probability, for two evenly matched teams
DRAW_MIN = 0.05             # lowest possible draw probability, for very mismatched teams
DRAW_SENSITIVITY = 0.0015   # how fast draw probability shrinks as rating gap grows

def estimate_draw_probability(rating_diff: float) -> float:
    rating_diff = abs(rating_diff)

    draw_prob = DRAW_MAX - DRAW_SENSITIVITY * rating_diff

    results = max(DRAW_MIN, min(DRAW_MAX, draw_prob))
    return results

def predict_match(rating_home: float, rating_away: float,
                  is_neutral: bool) -> dict[str, float]:
    if is_neutral is True:
        effective_rating_home = rating_home
    else:
        effective_rating_home = rating_home + HOME_ADVANTAGE
    
    expected_home = expected_score(effective_rating_home, rating_away)

    rating_diff = effective_rating_home - rating_away

    draw_prob = estimate_draw_probability(rating_diff)

    prob_home_win = expected_home - 0.5 * draw_prob
    prob_away_win = 1 - prob_home_win - draw_prob
    
    results = {
        "home_win": prob_home_win,
        "draw": draw_prob,
        "away_win": prob_away_win
    }

    return results

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    ratings = build_ratings(df)

    while True:
        team_a = input("Enter a Home National Team: ").strip().title()
        team_b = input("Enter an Away National Team: ").strip().title()
        is_neutral = (input("Is the Match a Neutral Venue? (yes/no): ").strip().lower()) in ("yes", "y", "true", "1")

        if team_a in ratings and team_b in ratings and team_a != team_b:
            break
        print("Please enter valid (and different) teams.\n")

    rating_a = ratings.get(team_a)
    rating_b = ratings.get(team_b)

    print("\n")
    print(f"{team_a} rating: {rating_a:.1f}")
    print(f"{team_b} rating: {rating_b:.1f}")

    result = predict_match(rating_a, rating_b, is_neutral)

    print(f"\n{team_a} vs {team_b} (Neutral Venue: {is_neutral}):")
    print(f"  {team_a} Win: {result['home_win']:.1%}")
    print(f"  Draw:          {result['draw']:.1%}")
    print(f"  {team_b} Win: {result['away_win']:.1%}")