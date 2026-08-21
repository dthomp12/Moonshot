from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Optional

from helpers import get_scaled_pitches
from player import Pitcher, Batter, matchup_rates, matchup_ppa

from consts import MIN_PITCHES, DOUBLE_TRIPLE_RATIO

@dataclass
class MatchupParams:
    rates: Dict[str, float]
    scaled_pitches: Dict[str, float]
    pitch_params: Dict[str, Tuple[int, float]]


def _normalize_pitch_outcome(outcome: str) -> str:
    if outcome == "2B" or outcome == "3B":
        return "XBH"
    return outcome


def _build_pitch_params(scaled_pitches: Dict[str, float]) -> Dict[str, Tuple[int, float]]:
    pitch_params: Dict[str, Tuple[int, float]] = {}
    for outcome, mu in scaled_pitches.items():
        minimum = MIN_PITCHES.get(outcome, 1)
        lam = max(0.3, mu - minimum)
        pitch_params[outcome] = (minimum, lam)
    return pitch_params

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
    outcome = _normalize_pitch_outcome(outcome)
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


def sample_pitches_poisson_precomputed(
    outcome: str,
    pitch_params: Dict[str, Tuple[int, float]],
    rng: np.random.Generator,
    extra_variance: float = 0.2,
) -> int:
    """Pitch sampler using precomputed (minimum, lambda) per outcome."""
    outcome = _normalize_pitch_outcome(outcome)
    minimum, lam = pitch_params[outcome]

    if extra_variance <= 0:
        extra = rng.poisson(lam)
    else:
        # Negative Binomial for overdispersion
        r = 1.0 / (extra_variance + 1e-8)
        p = r / (r + lam)
        extra = rng.negative_binomial(r, p)

    pitches = minimum + extra
    return min(pitches, 16)

def precompute_matchups(
    pitcher: Pitcher,
    batters: List[Batter],
    league: Optional[Dict[str, float]] = None,
    use_expected: bool = True,
) -> List[MatchupParams]:

    matchups = []

    for batter in batters:
        if league is None:
            rates = matchup_rates(
                pitcher,
                batter,
                use_expected=use_expected,
            )
        else:
            rates = matchup_rates(
                pitcher,
                batter,
                league,
                use_expected,
            )

        target_ppa = matchup_ppa(pitcher, batter)
        scaled = get_scaled_pitches(rates, target_ppa)
        pitch_params = _build_pitch_params(scaled)

        matchups.append(
            MatchupParams(
                rates=rates,
                scaled_pitches=scaled,
                pitch_params=pitch_params,
            )
        )

    return matchups

def sample_pa(
    pitcher: Pitcher,
    batter: Batter,
    rng: np.random.Generator,
    league: Optional[Dict[str, float]] = None,
    use_expected: bool = True,
    extra_pitch_variance: float = 0.2,
) -> Tuple[str, int]:
    """
    Full plate-appearance sample: outcome + number of pitches.
    Returns (outcome, pitches) where outcome is one of
    "K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT"
    """
    if league is None:
        rates = matchup_rates(
            pitcher,
            batter,
            use_expected=use_expected,
        )
    else:
        rates = matchup_rates(
            pitcher,
            batter,
            league,
            use_expected,
        )
    outcome = sample_outcome(rates, rng)

    target_ppa = matchup_ppa(pitcher, batter)
    scaled = get_scaled_pitches(rates, target_ppa)  # your existing scaler
    pitches = sample_pitches_poisson(outcome, scaled, rng, extra_pitch_variance)

    return outcome, pitches

def sample_pa_precomputed(
    matchup: MatchupParams,
    rng: np.random.Generator,
    extra_pitch_variance: float = 0.2,
) -> Tuple[str, int]:

    outcome = sample_outcome(matchup.rates, rng)

    pitches = sample_pitches_poisson_precomputed(
        outcome,
        matchup.pitch_params,
        rng,
        extra_pitch_variance,
    )

    return outcome, pitches