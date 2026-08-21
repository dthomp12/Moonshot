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
    TP: Optional[float] = None,
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
    min_added_pas: float = 40.0,
    prior: Dict[str, float] = LEAGUE_RATES,
) -> Dict[str, float]:
    """Simple empirical-Bayes shrink toward a prior when sample is small."""
    if prior is None:
        prior = LEAGUE_RATES

    if PA >= final_pas:
        num_prior_pas = min_added_pas
    else:
        num_prior_pas = max(0.0, final_pas - PA)

    if num_prior_pas < min_added_pas:
        num_prior_pas = min_added_pas

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


def build_expected_rates_from_profile(
    observed_rates: Dict[str, float],
    ba: Optional[float] = None,
    xba: Optional[float] = None,
    bacon: Optional[float] = None,
    slg: Optional[float] = None,
    xslg: Optional[float] = None,
    xbacon: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """
    Build expected outcome rates from observed rates + expected contact/power stats.
    Returns None when there is no expected signal.
    """
    if xbacon is None and xba is None and xslg is None:
        return None

    r = observed_rates.copy()
    k = r["K"]
    bb = r["BB"]
    hbp = r.get("HBP", 0.0)
    contact = max(1e-9, 1.0 - k - bb - hbp)

    obs_hits = r["1B"] + r["XBH"] + r["HR"]
    obs_hit_frac = obs_hits / contact

    if xbacon is not None and bacon is not None and bacon > 0:
        # Prefer relative adjustment when both expected and observed BACON exist.
        target_hit_frac = obs_hit_frac * (xbacon / bacon)
    elif xbacon is not None:
        target_hit_frac = xbacon
    elif xba is not None and ba is not None and ba > 0:
        # Use BA->xBA ratio to move observed contact-hit rate.
        target_hit_frac = obs_hit_frac * (xba / ba)
    elif xba is not None and (1.0 - k) > 0:
        # Fallback when BA is unavailable.
        target_hit_frac = xba / (1.0 - k)
    else:
        target_hit_frac = (r["1B"] + r["XBH"] + r["HR"]) / contact

    target_hit_frac = float(np.clip(target_hit_frac, 0.05, 0.55))

    scale = target_hit_frac / obs_hit_frac if obs_hit_frac > 1e-9 else 1.0

    new_1b = r["1B"] * scale
    new_xbh = r["XBH"] * scale
    new_hr = r["HR"] * scale

    power_scale = 1.0
    if xslg is not None and slg is not None and slg > 0:
        power_scale = xslg / slg
    power_scale = float(np.clip(power_scale, 0.6, 1.6))

    new_xbh *= power_scale
    new_hr *= power_scale

    hit_total = new_1b + new_xbh + new_hr
    target_hits = target_hit_frac * contact
    if hit_total > 1e-9:
        factor = target_hits / hit_total
        new_1b *= factor
        new_xbh *= factor
        new_hr *= factor

    new_rates = {
        "K": k,
        "BB": bb,
        "HBP": hbp,
        "1B": max(0.0, new_1b),
        "XBH": max(0.0, new_xbh),
        "HR": max(0.0, new_hr),
        "OUT": max(0.0, contact - (new_1b + new_xbh + new_hr)),
    }

    total = sum(new_rates.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in new_rates.items()}