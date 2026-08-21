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

def get_matchup(lineups, batting_side):
    pitching_side = "away" if batting_side == "home" else "home"

    batter_names = [b[1] for b in lineups[batting_side]["lineup"]]
    batter_ids = [b[0] for b in lineups[batting_side]["lineup"]]

    return (
        lineups[batting_side]["team"],
        batter_ids,
        batter_names,
        lineups[pitching_side]["team"],
        lineups[pitching_side]["pitcher"][1],  # Full name
        lineups[pitching_side]["pitcher"][0],  # ID
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run lineup optimization for an MLB game."
    )

    parser.add_argument(
        "side",
        choices=["home", "away"],
        help="Which team's lineup to analyze."
    )

    parser.add_argument(
        "--team",
        default="Washington",
        help="Team to find today's game for."
    )

    args = parser.parse_args()

    team = args.team
    batting_side = args.side

    lineups = get_current_lineups(team=team)

    # Print both lineups
    for side in ["away", "home"]:
        team_name = lineups[side]["team"]
        pitcher = lineups[side]["pitcher"][1]
        lineup = lineups[side]["lineup"]

        print(f"\n{'=' * 50}")
        print(team_name)
        print(f"Pitcher: {pitcher}")
        print("-" * 50)

        for i, player in enumerate(lineup, 1):
            print(f"{i}. {player[1]}")

    print(f"{'=' * 50}\n")

    # Only run the requested side
    (
        batting_team,
        lineup_ids,
        lineup_names,
        pitching_team,
        pitcher_name,
        pitcher_id,
    ) = get_matchup(lineups, batting_side)

    print(f"\n{batting_team} vs {pitcher_name} ({pitching_team})")

    pitcher = make_pitcher_from_savant(player_id=pitcher_id)

    lineup = [
        make_batter_from_savant(player_id=batter_id)
        for batter_id in lineup_ids
    ]

    # Set this when expected league-rate priors are available.
    expected_league_rates = None
    shrink_expected = expected_league_rates is not None

    pitcher.shrink(
        prior=None,
        final_pas=100,
        min_prior_pas=40,
        x_prior=expected_league_rates,
        shrink_expected=shrink_expected,
    )

    for batter in lineup:
        batter.shrink(
            prior=None,
            final_pas=100,
            min_prior_pas=40,
            x_prior=expected_league_rates,
            shrink_expected=shrink_expected,
        )

    rand_seed = np.random.randint(0, 1_000_000)

    opt = find_optimal_starting_batter(
        pitcher=pitcher,
        batters=lineup,
        n_sims=100_000,
        min_mult=1.00,
        max_mult=20.0,
        seed=rand_seed,
        use_expected=True,
    )

    print_optimal_start_results(opt)
    plot_starting_batter_distributions(opt)