import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from helpers import get_scaled_pitches
from player import Pitcher, Batter, matchup_rates, matchup_ppa

from consts import MIN_PITCHES, DOUBLE_TRIPLE_RATIO

def sample_outcome(rates: Dict[str, float], rng: np.random.Generator) -> str:
    outcomes = list(rates.keys())
    probs = list(rates.values())
    outcome_choice = rng.choice(outcomes, p=probs)
    if outcome_choice == "XBH":
        # Split XBH into doubles vs triples
        if rng.random() < DOUBLE_TRIPLE_RATIO:
            return "2B"
        else:
            return "3B"
    return outcome_choice


def sample_pitches_poisson(
    outcome: str,
    scaled_pitches: Dict[str, float],
    rng: np.random.Generator,
    extra_variance: float = 0.2,
) -> int:
    """Shifted Poisson (or Negative Binomial) that respects minimums."""
    if outcome == "2B" or outcome == "3B":
        outcome = "XBH"  # Use the same pitch distribution for doubles/triples
    mu = scaled_pitches[outcome]
    minimum = MIN_PITCHES.get(outcome, 1)
    lam = max(0.3, mu - minimum)

    if extra_variance <= 0:
        extra = rng.poisson(lam)
    else:
        # Negative Binomial for overdispersion
        r = 1.0 / (extra_variance + 1e-8)
        p = r / (r + lam)
        extra = rng.negative_binomial(r, p)

    pitches = minimum + extra
    return min(pitches, 16)

def sample_pa(
    pitcher: Pitcher,
    batter: Batter,
    rng: np.random.Generator,
    league: Dict[str, float] = None,
    use_expected: bool = True,
    extra_pitch_variance: float = 0.2,
) -> Tuple[str, int]:
    """
    Full plate-appearance sample: outcome + number of pitches.
    Returns (outcome, pitches) where outcome is one of
    "K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT"
    """
    rates = matchup_rates(pitcher, batter, league, use_expected)
    outcome = sample_outcome(rates, rng)

    target_ppa = matchup_ppa(pitcher, batter)
    scaled = get_scaled_pitches(rates, target_ppa)  # your existing scaler
    pitches = sample_pitches_poisson(outcome, scaled, rng, extra_pitch_variance)

    return outcome, pitches