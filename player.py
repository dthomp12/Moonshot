from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

from helpers import shrink_rates, rates_from_counts, log5, build_expected_rates_from_profile
from consts import LEAGUE_RATES, LEAGUE_X_PROFILE


LEAGUE_X_RATES = build_expected_rates_from_profile(
    observed_rates=LEAGUE_RATES,
    ba=LEAGUE_X_PROFILE.get("ba"),
    xba=LEAGUE_X_PROFILE.get("xba"),
    bacon=LEAGUE_X_PROFILE.get("bacon"),
    slg=LEAGUE_X_PROFILE.get("slg"),
    xslg=LEAGUE_X_PROFILE.get("xslg"),
    xbacon=LEAGUE_X_PROFILE.get("xbacon"),
)

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
        return build_expected_rates_from_profile(
            observed_rates=self.rates,
            ba=self.ba,
            xba=self.xba,
            bacon=self.bacon,
            slg=self.slg,
            xslg=self.xslg,
            xbacon=self.xbacon,
        )

    def get_rates(self, use_expected: bool = True) -> Dict[str, float]:
        if use_expected and self.x_rates is not None:
            return self.x_rates
        return self.rates

    def shrink(
        self,
        prior: Optional[Dict[str, float]] = None,
        final_pas: float = 40.0,
        min_prior_pas: float = 0.0,
        x_prior: Optional[Dict[str, float]] = None,
        shrink_expected: bool = False,
    ) -> None:
        """
        In-place empirical-Bayes shrink.

        Default behavior shrinks observed rates only, then rebuilds expected rates from
        expected metrics. If expected league priors are available, set
        shrink_expected=True and pass x_prior.
        """
        if prior is None:
            prior = LEAGUE_RATES
        self.rates = shrink_rates(self.rates, self.pa, final_pas=final_pas, min_added_pas=min_prior_pas, prior=prior)
        if self.x_rates is not None:
            if shrink_expected:
                prior_for_expected = x_prior if x_prior is not None else prior
                self.x_rates = shrink_rates(
                    self.x_rates,
                    self.pa,
                    final_pas=final_pas,
                    min_added_pas=min_prior_pas,
                    prior=prior_for_expected,
                )
            else:
                self.refresh_expected_rates()

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
    league: Optional[Dict[str, float]] = None,
    use_expected: bool = True,
) -> Dict[str, float]:
    """
    Produce a full outcome probability vector for this pitcher vs batter.
    Applies log5 independently to each rate component, then renormalizes.
    """
    if league is None:
        if use_expected and LEAGUE_X_RATES is not None:
            league = LEAGUE_X_RATES
        else:
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
