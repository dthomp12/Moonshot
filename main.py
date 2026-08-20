from moonshot import simulate_moonshot_round, run_simulations, find_optimal_starting_batter
from vis import print_round, print_ev_analysis, print_optimal_start_results

import numpy as np

from scraper import make_batter_from_savant, make_pitcher_from_savant

if __name__ == "__main__":

    pitcher_name = "Williamson, Brandon"
    lineup_names = [
        "Schwarber, Kyle",
        "Turner, Trea",
        "Harper, Bryce",
        "Arraez, Luis",
        "Bohm, Alec",
        "Stott, Bryson",
        "Marsh, Brandon",
        "Crawford, Justin",
        "Stubbs, Garrett"
    ]

    pitcher = make_pitcher_from_savant(name=pitcher_name)

    lineup = []
    for name in lineup_names:
        batter = make_batter_from_savant(name=name)
        lineup.append(batter)

    rand_seed = np.random.randint(0, 1_000_000)
    opt = find_optimal_starting_batter(
        pitcher=pitcher,
        batters=lineup,
        n_sims=5_000,
        min_mult=1.25,
        max_mult=15.0,
        seed=rand_seed,
        use_expected=True,
    )
    print_optimal_start_results(opt)