from moonshot import simulate_moonshot_round, run_simulations, find_optimal_starting_batter
from vis import (
    print_round,
    print_ev_analysis,
    print_optimal_start_results,
    plot_starting_batter_distributions,
)

import numpy as np
import argparse

from scraper import make_batter_from_savant, make_pitcher_from_savant
from live_game_scraper import get_current_lineups

if __name__ == "__main__":

    pitcher_name = "Andrew Painter"
    lineup_names = [
        "Ronald Acuna Jr.",
        "Bryce Harper",
        "Matt Olson",
        'Shohei Ohtani',
        'Aaron Judge',
        'Kyle Schwarber',
        'Freddie Freeman',
    ]
 
    pitcher = make_pitcher_from_savant(name=pitcher_name)
    pitcher.shrink(None, final_pas=100, min_prior_pas=40)

    lineup = [
        make_batter_from_savant(name=batter_name)
        for batter_name in lineup_names
    ]

    for batter in lineup:
        batter.shrink(None, final_pas=100, min_prior_pas=40)

    rand_seed = np.random.randint(0, 1_000_000)

    opt = find_optimal_starting_batter(
        pitcher=pitcher,
        batters=lineup,
        n_sims=20_000,
        min_mult=1.00,
        max_mult=20.0,
        seed=rand_seed,
        use_expected=True,
    )

    print_optimal_start_results(opt)
    plot_starting_batter_distributions(opt)