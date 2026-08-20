import numpy as np
from typing import Dict, List, Optional, Any

from consts import PITCH_VALUE, OUTCOME_MULTIPLIERS
from player import Pitcher, Batter
from sampling import sample_pa


def simulate_moonshot_round(
    pitcher: Pitcher,
    batters: List[Batter],
    starting_batter_idx: int = 0,
    rng: Optional[np.random.Generator] = None,
    max_batters: int = 25,
    use_expected: bool = True,
) -> Dict[str, Any]:
    """
    Simulate one moonshot round.

    Precise bonus handling:
    - When a 10-pitch threshold is crossed mid-PA, only the pitches
      *after* the threshold (plus the outcome) receive the new bonus.
    - A HR doubles the multiplier for everything that comes after it.
    """
    if rng is None:
        rng = np.random.default_rng()

    multiplier = 0.0
    total_pitches = 0
    path = []
    bonus_mult = 1.0
    pitch_count = 0                 # cumulative pitches in the round
    next_bonus_at = 10

    n_batters = len(batters)
    batter_idx = starting_batter_idx
    batters_seen = 0

    while batters_seen < max_batters:
        batter = batters[batter_idx % n_batters]

        # ----- Sample the full PA using the new matchup logic -----
        outcome, pitches = sample_pa(
            pitcher,
            batter,
            rng,
            use_expected=use_expected,
        )

        # ----- Split this PA around any bonus thresholds -----
        remaining_pitches = pitches
        mult_added = 0.0

        while remaining_pitches > 0:
            pitches_until_bonus = next_bonus_at - pitch_count

            if pitches_until_bonus <= 0:
                # Already at or past threshold (shouldn't normally happen)
                take = remaining_pitches
                current_bonus = bonus_mult
            elif remaining_pitches <= pitches_until_bonus:
                # Entire remaining portion is before the next threshold
                take = remaining_pitches
                current_bonus = bonus_mult
            else:
                # We will cross the threshold during this PA
                take = pitches_until_bonus
                current_bonus = bonus_mult

            # Add the pitch contribution at the *current* bonus
            mult_added += take * PITCH_VALUE * current_bonus
            pitch_count += take
            remaining_pitches -= take

            if take == pitches_until_bonus and pitches_until_bonus > 0:
                # We just hit the threshold → double for everything after
                bonus_mult *= 2
                next_bonus_at += 10

        # Outcome is valued with the *final* bonus_mult that applies after all pitches
        outcome_val = OUTCOME_MULTIPLIERS.get(outcome, 0.0) * bonus_mult
        mult_added += outcome_val

        # Record
        multiplier += mult_added
        total_pitches += pitches
        path.append({
            "batter": batter.name,
            "outcome": outcome,
            "pitches": pitches,
            "mult_added": round(mult_added, 4),
            "running_mult": round(multiplier, 4),
        })

        # HR triggers an additional double for future actions
        if outcome == "HR":
            bonus_mult *= 2

        # End the round on an out
        if outcome in ("K", "OUT"):
            break

        batters_seen += 1
        batter_idx += 1

    return {
        "final_multiplier": round(multiplier, 4),
        "total_pitches": total_pitches,
        "batters_faced": batters_seen + 1,
        "path": path,
    }


def run_simulations(
    pitcher: Pitcher,
    batters: List[Batter],
    n_sims: int = 8000,
    targets: Optional[List[float]] = None,
    seed: int = 42,
    use_expected: bool = True,
    starting_batter_idx: int = 0,
) -> Dict[str, Any]:
    """
    Run many moonshot rounds and return summary statistics + reach probabilities.
    """
    if targets is None:
        targets = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]

    rng = np.random.default_rng(seed)

    results = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        # Use a child RNG so each sim is independent but the whole run is reproducible
        child_rng = np.random.default_rng(rng.integers(0, 2**63))
        sim = simulate_moonshot_round(
            pitcher=pitcher,
            batters=batters,
            starting_batter_idx=starting_batter_idx,
            rng=child_rng,
            use_expected=use_expected,
        )
        results[i] = sim["final_multiplier"]

    summary = {
        "mean": float(np.mean(results)),
        "median": float(np.median(results)),
        "std": float(np.std(results)),
        "p75": float(np.percentile(results, 75)),
        "p90": float(np.percentile(results, 90)),
        "p95": float(np.percentile(results, 95)),
        "p99": float(np.percentile(results, 99)),
        "max": float(np.max(results)),
    }

    reach = {}
    for t in targets:
        p = float(np.mean(results >= t))
        reach[t] = p
        reach[f"EV_{t}x"] = p * t - 1.0   # simple EV assuming even-money at that multiplier

    summary["best_ev"] = max(reach.get(f"EV_{t}x", -np.inf) for t in targets)
    summary["best_mult"] = max(targets, key=lambda t: reach.get(f"EV_{t}x", -np.inf))

    summary["reach"] = reach
    summary["raw"] = results          # keep the full array if you want histograms later
    return summary


def find_optimal_starting_batter(
    pitcher: Pitcher,
    batters: List[Batter],
    n_sims: int = 5000,
    min_mult: float = 1.0,
    max_mult: float = 30.0,
    step: float = 0.25,
    seed: int = 42,
    use_expected: bool = True,
) -> Dict[str, Any]:
    """
    For every possible starting batter, find the multiplier that gives the highest EV.
    Returns ranking sorted by that best EV.
    """
    targets = np.arange(min_mult, max_mult + step, step).tolist()

    results_by_start = []
    best_score = -np.inf
    best_idx = 0
    best_summary = None

    for start_idx in range(len(batters)):
        print(f"Testing starting batter {start_idx}: {batters[start_idx].name}...")

        summary = run_simulations(
            pitcher=pitcher,
            batters=batters,
            n_sims=n_sims,
            targets=targets,
            seed=seed + start_idx,
            use_expected=use_expected,
            starting_batter_idx=start_idx,
        )

        # best_ev and best_mult are already calculated inside run_simulations
        best_ev = summary["best_ev"]
        best_mult = summary["best_mult"]

        results_by_start.append({
            "starting_idx": start_idx,
            "batter_name": batters[start_idx].name,
            "best_ev": best_ev,
            "best_mult": best_mult,
            "mean": summary["mean"],
            "median": summary["median"],
            "p90": summary["p90"],
            "p95": summary["p95"],
        })

        if best_ev > best_score:
            best_score = best_ev
            best_idx = start_idx
            best_summary = summary

    # Sort by best EV descending
    results_by_start.sort(key=lambda x: x["best_ev"], reverse=True)

    return {
        "best_starting_idx": best_idx,
        "best_batter_name": batters[best_idx].name,
        "best_ev": best_score,
        "best_mult": best_summary["best_mult"] if best_summary else None,
        "best_summary": best_summary,
        "all_starts": results_by_start,
        "pitcher_name": pitcher.name,
    }