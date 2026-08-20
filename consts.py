BASE_PITCHES = {
    "K":   4.80,
    "BB":  5.50,
    "HBP": 4.10,
    "1B":  3.3,
    "XBH":  3.3,
    "HR":  3.3,
    "OUT": 3.3,
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
    "K": 0.22116, "BB": 0.089, "HBP": 0.0114,
    "1B": 0.141, "XBH": 0.0446, "HR": 0.0305, "OUT": 0.4639
}

num_dubs = 5890
num_trips = 511
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