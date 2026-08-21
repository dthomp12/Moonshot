from pybaseball import statcast
import pandas as pd

# Pull whatever date range you want
df = statcast("2026-03-01", "2026-08-20")

# Keep only rows that belong to completed PAs
pa_data = df[df["events"].notna()].copy()

# One row per PA
pa_data = (
    pa_data
    .groupby(["game_pk", "at_bat_number"], as_index=False)
    .agg(
        outcome=("events", "last"),
        pitches=("pitch_number", "max")
    )
)

# Mean pitches by outcome
result = (
    pa_data
    .groupby("outcome")["pitches"]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
)

print(result)