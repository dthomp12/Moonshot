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

def make_batter_from_savant(player_id: int = None, name: str = None):
    if player_id is not None:
        row = batters_df.loc[player_id]
    elif name is not None:
        matches = batters_df[batters_df["last_name, first_name"].str.contains(name, case=False)]
        if len(matches) == 0:
            raise ValueError(f"No batter found for '{name}'")
        if len(matches) > 1:
            print("Multiple matches, taking first:")
            print(matches[["last_name, first_name"]])
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
        name=row["last_name, first_name"],
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
        matches = pitchers_df[pitchers_df["last_name, first_name"].str.contains(name, case=False)]
        if len(matches) == 0:
            raise ValueError(f"No pitcher found for '{name}'")
        if len(matches) > 1:
            print("Multiple matches, taking first:")
            print(matches[["last_name, first_name"]])
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
        name=row["last_name, first_name"],
        ba=row["batting_avg"],
        xba=row["xba"],
        slg=row["slg_percent"],
        xslg=row["xslg"],
        bacon=row["bacon"],
        xbacon=row["xbacon"],
    )