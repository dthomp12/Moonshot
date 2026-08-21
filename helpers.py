import numpy as np
from typing import Dict, Tuple, Optional

from consts import BASE_PITCHES, LEAGUE_RATES

# ============================================================
# 2. PITCHER-SPECIFIC INPUT HELPERS
# ============================================================

def rates_from_counts(
    BF: float,
    singles: float,
    XBH: float,
    HR: float,
    SO: float,
    BB: float,
    HBP: float = 0.0,
    TP: float = None,
) -> Tuple[Dict[str, float], Optional[float], float]:
    """
    Build outcome rates + P/PA from counting stats.
    BF is used as PA.
    """
    PA = BF
    if PA <= 0:
        raise ValueError("BF/PA must be > 0")

    rates = {
        "K":   SO / PA,
        "BB":  BB / PA,
        "HBP": HBP / PA,
        "1B":  singles / PA,
        "XBH":  XBH  / PA,
        "HR":  HR / PA,
    }

    contact_outs = PA - (SO + BB + HBP + singles + XBH + HR)
    rates["OUT"] = max(0.0, contact_outs / PA)

    # Normalize
    total = sum(rates.values())
    rates = {k: v / total for k, v in rates.items()}

    ppa = (TP / PA) if (TP is not None and TP > 0) else None
    return rates, ppa, PA


def shrink_rates(
    observed: Dict[str, float],
    PA: float,
    final_pas: float = 100.0,
    prior: Dict[str, float] = LEAGUE_RATES,
) -> Dict[str, float]:
    """Simple empirical-Bayes shrink toward a prior when sample is small."""
    if PA >= final_pas:
        return observed

    if prior is None:
        prior = LEAGUE_RATES

    num_prior_pas = max(0.0, final_pas - PA)

    w_obs = PA / (PA + num_prior_pas)
    w_prior = 1.0 - w_obs

    shrunk = {
        k: w_obs * observed.get(k, 0.0) + w_prior * prior.get(k, 0.0)
        for k in observed
    }
    total = sum(shrunk.values())
    return {k: v / total for k, v in shrunk.items()}


def get_scaled_pitches(rates: Dict[str, float], target_ppa: float) -> Dict[str, float]:
    """Scale base averages so expected P/PA matches the pitcher's real value."""
    expected = sum(rates[o] * BASE_PITCHES[o] for o in rates)
    if expected <= 0:
        return BASE_PITCHES.copy()
    scale = target_ppa / expected
    return {o: max(1.5, BASE_PITCHES[o] * scale) for o in BASE_PITCHES}

def log5(p_batter: float, p_pitcher: float, p_league: float) -> float:
    """
    Classic log5 / odds-ratio form.
    Returns the matchup probability for a binary event.
    Handles edge cases near 0/1.
    """
    # Clamp to avoid division by zero
    eps = 1e-6
    p_b = np.clip(p_batter, eps, 1 - eps)
    p_p = np.clip(p_pitcher, eps, 1 - eps)
    p_l = np.clip(p_league, eps, 1 - eps)

    # Odds form (equivalent to Bill James log5)
    odds_b = p_b / (1 - p_b)
    odds_p = p_p / (1 - p_p)
    odds_l = p_l / (1 - p_l)

    matchup_odds = (odds_b * odds_p) / odds_l
    return matchup_odds / (1 + matchup_odds)