from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from helpers import shrink_rates, rates_from_counts, log5
from consts import LEAGUE_RATES

@dataclass
class Player:
    name: Optional[str] = None
    rates: Dict[str, float] = field(default_factory=dict)          # observed
    x_rates: Optional[Dict[str, float]] = None                     # expected
    ppa: Optional[float] = None
    pa: float = 0.0
    handedness: Optional[str] = None

    # ---- raw counting stats (optional, useful for debugging) ----
    raw_counts: Optional[Dict[str, float]] = None

    # ---- traditional + expected slash / value stats ----
    ba: Optional[float] = None
    xba: Optional[float] = None
    slg: Optional[float] = None
    xslg: Optional[float] = None
    bacon: Optional[float] = None
    xbacon: Optional[float] = None

    # ---------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------
    @classmethod
    def from_counts(
        cls,
        BF_or_PA: float,
        singles: float,
        XBH: float,
        HR: float,
        SO: float,
        BB: float,
        HBP: float = 0.0,
        TP: Optional[float] = None,
        name: Optional[str] = None,
        # optional expected / traditional stats
        ba: Optional[float] = None,
        xba: Optional[float] = None,
        slg: Optional[float] = None,
        xslg: Optional[float] = None,
        bacon: Optional[float] = None,
        xbacon: Optional[float] = None,
        handedness: Optional[str] = None,
        **kwargs,
    ) -> "Player":
        rates, ppa, pa = rates_from_counts(
            BF_or_PA, singles, XBH, HR, SO, BB, HBP, TP
        )

        player = cls(
            name=name,
            rates=rates,
            ppa=ppa,
            pa=pa,
            handedness=handedness,
            ba=ba, xba=xba,
            slg=slg, xslg=xslg,
            bacon=bacon, xbacon=xbacon,
            raw_counts={
                "PA": BF_or_PA, "1B": singles, "XBH": XBH, "HR": HR,
                "SO": SO, "BB": BB, "HBP": HBP, "TP": TP,
            },
            **kwargs,
        )

        # Automatically build expected rates if we have enough signal
        player.x_rates = player._build_expected_rates()
        return player

    # ---------------------------------------------------------------
    # Core expected-rate construction
    # ---------------------------------------------------------------
    def _build_expected_rates(self) -> Optional[Dict[str, float]]:
        """
        Build an expected rate vector.

        Returns None if there is not enough expected information.
        """
        has_contact_signal = any([
            self.xbacon is not None,
            self.xba is not None,
            self.xslg is not None,
        ])
        if not has_contact_signal:
            return None

        # Start from observed rates
        r = self.rates.copy()
        k   = r["K"]
        bb  = r["BB"]
        hbp = r.get("HBP", 0.0)
        contact = max(1e-9, 1.0 - k - bb - hbp)

        # ----- 1. Decide target hit-rate on contact -----
        if self.xbacon is not None:
            target_hit_frac = self.xbacon
        elif self.xba is not None and (1.0 - k) > 0:
            # crude but usable conversion
            target_hit_frac = self.xba / (1.0 - k)
        else:
            # fall back to observed
            target_hit_frac = (r["1B"] + r["XBH"] + r["HR"]) / contact

        target_hit_frac = float(np.clip(target_hit_frac, 0.05, 0.55))

        # ----- 2. Scale the three hit types -----
        obs_hits = r["1B"] + r["XBH"] + r["HR"]
        obs_hit_frac = obs_hits / contact
        scale = target_hit_frac / obs_hit_frac if obs_hit_frac > 1e-9 else 1.0

        new_1b  = r["1B"]  * scale
        new_xbh = r["XBH"] * scale
        new_hr  = r["HR"]  * scale

        # ----- 3. Extra power adjustment (SLG) -----
        power_scale = 1.0

        if self.xslg is not None and self.slg is not None and self.slg > 0:
            power_scale = self.xslg / self.slg

        power_scale = float(np.clip(power_scale, 0.6, 1.6))

        # Apply extra scale only to extra-base hits
        new_xbh *= power_scale
        new_hr  *= power_scale

        # Re-normalize the three hit types so total hit rate stays correct
        hit_total = new_1b + new_xbh + new_hr
        target_hits = target_hit_frac * contact
        if hit_total > 1e-9:
            factor = target_hits / hit_total
            new_1b  *= factor
            new_xbh *= factor
            new_hr  *= factor

        # ----- 4. Assemble final vector -----
        new_rates = {
            "K":   k,
            "BB":  bb,
            "HBP": hbp,
            "1B":  max(0.0, new_1b),
            "XBH": max(0.0, new_xbh),
            "HR":  max(0.0, new_hr),
            "OUT": max(0.0, contact - (new_1b + new_xbh + new_hr)),
        }

        # Final safety renormalization
        total = sum(new_rates.values())
        if total <= 0:
            return None
        return {k: v / total for k, v in new_rates.items()}

    def get_rates(self, use_expected: bool = True) -> Dict[str, float]:
        if use_expected and self.x_rates is not None:
            return self.x_rates
        return self.rates

    def shrink(
        self,
        prior: Optional[Dict[str, float]] = None,
        prior_strength: float = 40.0,
    ) -> None:
        """In-place empirical-Bayes shrink on both observed and expected rates."""
        if prior is None:
            prior = LEAGUE_RATES
        self.rates = shrink_rates(self.rates, self.pa, prior_strength, prior)
        if self.x_rates is not None:
            self.x_rates = shrink_rates(self.x_rates, self.pa, prior_strength, prior)

    def refresh_expected_rates(self) -> None:
        """Call this if you mutate any of the x* attributes after construction."""
        self.x_rates = self._build_expected_rates()

class Pitcher(Player):
    pass

class Batter(Player):
    pass

def matchup_rates(
    pitcher: Player,
    batter: Player,
    league: Dict[str, float] = None,
    use_expected: bool = True,
) -> Dict[str, float]:
    """
    Produce a full outcome probability vector for this pitcher vs batter.
    Applies log5 independently to each rate component, then renormalizes.
    """
    if league is None:
        league = LEAGUE_RATES

    p_rates = pitcher.get_rates(use_expected)
    b_rates = batter.get_rates(use_expected)

    # Apply log5 to each mutually exclusive outcome
    match = {}
    for outcome in p_rates:
        match[outcome] = log5(
            b_rates.get(outcome, league[outcome]),
            p_rates.get(outcome, league[outcome]),
            league[outcome],
        )

    # Renormalize so they sum to 1 (important because log5 is applied component-wise)
    total = sum(match.values())
    return {k: v / total for k, v in match.items()}

def matchup_ppa(
    pitcher: Player,
    batter: Player,
    method: str = "harmonic",  # or "average", "pitcher", "batter"
) -> float:
    """Blend pitcher and batter observed P/PA."""
    pp = pitcher.ppa
    bp = batter.ppa
    if pp is None and bp is None:
        return 3.85  # rough MLB average
    elif pp is None:
        return bp
    elif bp is None:
        return pp

    if method == "average":
        return (pp + bp) / 2
    if method == "harmonic":
        return 2 / (1/pp + 1/bp)
    if method == "pitcher":
        return pp
    return bp
