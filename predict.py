from elo import expected_score, HOME_ADVANTAGE

# Tunable constants for draw probability 
DRAW_MAX = 0.34             # highest possible draw probability, for two evenly matched teams
DRAW_MIN = 0.05             # lowest possible draw probability, for very mismatched teams
DRAW_SENSITIVITY = 0.0015   # how fast draw probability shrinks as rating gap grows

def estimate_draw_probability(rating_diff: float) -> float:
    pass

def predict_match(rating_home: float, rating_away: float,
                  is_neutral: bool) -> dict[str, float]:
    pass