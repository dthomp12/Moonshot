from pathlib import Path
import pandas as pd

from player import Batter, Pitcher

DATA_DIR = Path("data")  # adjust if needed

# Load once
batters_df = pd.read_csv(DATA_DIR / "batter_stats.csv")   # or whatever you named it
pitchers_df = pd.read_csv(DATA_DIR / "pitcher_stats.csv")

# Optional: make player_id the index for faster lookups
batters_df = batters_df.set_index("player_id")
pitchers_df = pitchers_df.set_index("player_id")

# Create new column for full name in "First Last" format for easier searching
import unidecode
batters_df["full_name"] = batters_df["last_name, first_name"].apply(lambda x: " ".join(part.strip() for part in x.split(",")[::-1]))
pitchers_df["full_name"] = pitchers_df["last_name, first_name"].apply(lambda x: " ".join(part.strip() for part in x.split(",")[::-1]))

# Convert accented characters to ASCII for easier searching

batters_df["full_name"] = batters_df["full_name"].apply(unidecode.unidecode)
pitchers_df["full_name"] = pitchers_df["full_name"].apply(unidecode.unidecode)

def make_batter_from_savant(player_id: int = None, name: str = None):
    if player_id is not None:
        row = batters_df.loc[player_id]
    elif name is not None:
        if "," in name:
            name = " ".join(part.strip() for part in name.split(",")[::-1])  # convert "Last, First" to "First Last"
        matches = batters_df[batters_df["full_name"].str.contains(name, case=False)]
        if len(matches) == 0:
            raise ValueError(f"No batter found for '{name}'")
        if len(matches) > 1:
            print("Multiple matches, taking first:")
            print(matches[["full_name"]])
        row = matches.iloc[0]
    else:
        raise ValueError("Provide player_id or name")

    singles = row["single"]
    doubles = row["double"]
    triples = row["triple"]
    xbh = doubles + triples

    return Batter.from_counts(
        BF_or_PA=row["pa"],
        singles=singles,
        XBH=xbh,
        HR=row["home_run"],
        SO=row["strikeout"],
        BB=row["walk"],
        HBP=row["b_hit_by_pitch"],
        TP=row["b_total_pitches"],
        name=row["full_name"],
        ba=row["batting_avg"],
        xba=row["xba"],
        slg=row["slg_percent"],
        xslg=row["xslg"],
        bacon=row["bacon"],
        xbacon=row["xbacon"],
    )


def make_pitcher_from_savant(player_id: int = None, name: str = None):
    if player_id is not None:
        row = pitchers_df.loc[player_id]
    elif name is not None:
        if "," in name:
            name = " ".join(part.strip() for part in name.split(",")[::-1])  # convert "Last, First" to "First Last"
        matches = pitchers_df[pitchers_df["full_name"].str.contains(name, case=False)]
        if len(matches) == 0:
            raise ValueError(f"No pitcher found for '{name}'")
        if len(matches) > 1:
            print("Multiple matches, taking first:")
            print(matches[["full_name"]])
        row = matches.iloc[0]
    else:
        raise ValueError("Provide player_id or name")

    singles = row["single"]
    doubles = row["double"]
    triples = row["triple"]
    xbh = doubles + triples

    return Pitcher.from_counts(
        BF_or_PA=row["pa"],
        singles=singles,
        XBH=xbh,
        HR=row["home_run"],
        SO=row["strikeout"],
        BB=row["walk"],
        HBP=row["p_hit_by_pitch"],
        TP=row["p_total_pitches"],
        name=row["full_name"],
        ba=row["batting_avg"],
        xba=row["xba"],
        slg=row["slg_percent"],
        xslg=row["xslg"],
        bacon=row["bacon"],
        xbacon=row["xbacon"],
    )

if __name__ == "__main__":
    # Get the league-average expected rates for reference

    # Count-based rates are computed from league totals.
    total_pa = float(batters_df["pa"].sum())

    k_rate = float(batters_df["strikeout"].sum() / total_pa)
    bb_rate = float(batters_df["walk"].sum() / total_pa)
    hbp_rate = float(batters_df["b_hit_by_pitch"].sum() / total_pa)
    one_b_rate = float(batters_df["single"].sum() / total_pa)
    xbh_rate = float((batters_df["double"] + batters_df["triple"]).sum() / total_pa)
    hr_rate = float(batters_df["home_run"].sum() / total_pa)
    out_rate = max(0.0, 1.0 - (k_rate + bb_rate + hbp_rate + one_b_rate + xbh_rate + hr_rate))

    def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
        valid = series.notna() & weights.notna() & (weights > 0)
        if not valid.any():
            return float("nan")
        s = series[valid].astype(float)
        w = weights[valid].astype(float)
        return float((s * w).sum() / w.sum())

    pa_weights = batters_df["pa"]

    league_rates = {
        "K": k_rate,
        "BB": bb_rate,
        "HBP": hbp_rate,
        "1B": one_b_rate,
        "XBH": xbh_rate,
        "HR": hr_rate,
        "OUT": out_rate,
        # For profile stats, PA-weighted means are a practical default.
        # If you later add exact denominators (AB, BIP), use those instead.
        "BA": weighted_mean(batters_df["batting_avg"], pa_weights),
        "xBA": weighted_mean(batters_df["xba"], pa_weights),
        "SLG": weighted_mean(batters_df["slg_percent"], pa_weights),
        "xSLG": weighted_mean(batters_df["xslg"], pa_weights),
        "BACON": weighted_mean(batters_df["bacon"], pa_weights),
        "xBACON": weighted_mean(batters_df["xbacon"], pa_weights),
    }

    for key, value in league_rates.items():
        print(f"{key}: {value:.4f}")