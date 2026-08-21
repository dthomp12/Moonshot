# To update, run the scraper script in scraper_helpers/pull_season_base_pitches.py and copy the output into this file.
BASE_PITCHES = {
    "K":   4.8527,
    "BB":  5.75188,
    "HBP": 3.09,
    "1B":  3.3437,
    "XBH":  3.3437,
    "HR":  3.2703,
    "OUT": 3.3934,
}

MIN_PITCHES = {
    "K":   3,
    "BB":  4,
    "HBP": 1,
    "1B":  1,
    "XBH":  1,
    "HR":  1,
    "OUT": 1,
}

# Default neutral prior (used for shrinking small samples)
LEAGUE_RATES = {
    "K": 0.2211, "BB": 0.0892, "HBP": 0.0114,
    "1B": 0.1414, "XBH": 0.0446, "HR": 0.0304, "OUT": 0.462
}

# Optional league-level expected profile stats used to derive LEAGUE_X_RATES.
# Fill these when available. Leave as None to fall back to observed LEAGUE_RATES.
LEAGUE_X_PROFILE = {
    "ba": 0.2438,
    "xba": 0.2434,
    "bacon": 0.3222,
    "slg": 0.4013,
    "xslg": 0.396,
    "xbacon": 0.3242,
}

num_dubs = 5932
num_trips = 513
DOUBLE_TRIPLE_RATIO = num_dubs / (num_dubs + num_trips) # % of XBH that are doubles

###############################
# Moonshot parameters
PITCH_VALUE = 0.25

OUTCOME_MULTIPLIERS = {
    "K": 0.0,
    "BB": 1.0,
    "HBP": 1.0,
    "1B": 1.0,
    "2B": 2.0,
    "3B": 3.0,
    "HR": 5.0,
    "OUT": 0.0,
}