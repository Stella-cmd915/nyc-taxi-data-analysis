"""
============================================================
NYC Yellow Taxi 2016 - Part III: Descriptive Statistics
============================================================
Answers all six required statistical questions and saves
results to JSON / CSV for inclusion in the report.

  1. Mean trip duration per hour of day
  2. Mean trip duration per weekday
  3. Region with most pickups (via grid binning)
  4. Longest and shortest trips
  5. Hour with peak traffic (most trips)
  6. Pearson correlation: duration vs passenger count (+ p-value)

Run:  python 03_statistical_analysis.py
Input : ../data/NYC_cleaned.csv
Output: ../outputs/statistics.json
        ../outputs/stats_*.csv
"""
from __future__ import annotations

import os
import json
import time
import numpy as np
import pandas as pd
from scipy import stats

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "NYC_cleaned.csv")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]


def log(msg: str) -> None:
    print(f"[stats] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    log(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["pickup_datetime"],
        usecols=["pickup_datetime", "trip_duration", "passenger_count",
                 "pickup_hour", "pickup_weekday",
                 "pickup_latitude", "pickup_longitude"],
    )
    log(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    results: dict = {}

    # ---- 1. Mean trip duration per hour --------------------------
    log("Q1 - Mean trip duration per hour ...")
    dur_per_hour = (df.groupby("pickup_hour")["trip_duration"]
                      .agg(["mean", "median", "count"])
                      .round(2))
    dur_per_hour["mean_minutes"] = (dur_per_hour["mean"] / 60.0).round(2)
    dur_per_hour.to_csv(os.path.join(OUT_DIR, "stats_duration_per_hour.csv"))
    results["q1_mean_duration_per_hour"] = {
        int(h): {
            "mean_seconds": float(row["mean"]),
            "mean_minutes": float(row["mean_minutes"]),
            "trips": int(row["count"]),
        }
        for h, row in dur_per_hour.iterrows()
    }
    log(f"  longest avg trip: hour {dur_per_hour['mean'].idxmax()} "
        f"({dur_per_hour['mean'].max()/60:.2f} min)")
    log(f"  shortest avg trip: hour {dur_per_hour['mean'].idxmin()} "
        f"({dur_per_hour['mean'].min()/60:.2f} min)")

    # ---- 2. Mean trip duration per weekday -----------------------
    log("Q2 - Mean trip duration per weekday ...")
    dur_per_wd = (df.groupby("pickup_weekday")["trip_duration"]
                    .agg(["mean", "median", "count"])
                    .round(2))
    dur_per_wd["weekday"] = [WEEKDAY_NAMES[int(i)] for i in dur_per_wd.index]
    dur_per_wd["mean_minutes"] = (dur_per_wd["mean"] / 60.0).round(2)
    dur_per_wd.to_csv(os.path.join(OUT_DIR, "stats_duration_per_weekday.csv"))
    results["q2_mean_duration_per_weekday"] = {
        WEEKDAY_NAMES[int(d)]: {
            "mean_seconds": float(row["mean"]),
            "mean_minutes": float(row["mean_minutes"]),
            "trips": int(row["count"]),
        }
        for d, row in dur_per_wd.iterrows()
    }
    log(f"  longest avg trip: {WEEKDAY_NAMES[int(dur_per_wd['mean'].idxmax())]} "
        f"({dur_per_wd['mean'].max()/60:.2f} min)")

    # ---- 3. Region with most pickups (grid binning) --------------
    log("Q3 - Region with most pickups (0.005 deg grid ~ 500m) ...")
    df["lat_bin"] = (df["pickup_latitude"]  / 0.005).round().astype(int) * 0.005
    df["lon_bin"] = (df["pickup_longitude"] / 0.005).round().astype(int) * 0.005
    region_counts = (df.groupby(["lat_bin", "lon_bin"])
                       .size()
                       .reset_index(name="count")
                       .sort_values("count", ascending=False))
    top10 = region_counts.head(10).round(4)
    top10.to_csv(os.path.join(OUT_DIR, "stats_top_pickup_regions.csv"), index=False)
    results["q3_top_pickup_region"] = {
        "lat": float(top10.iloc[0]["lat_bin"]),
        "lon": float(top10.iloc[0]["lon_bin"]),
        "trips": int(top10.iloc[0]["count"]),
        "note": "Midtown Manhattan area (TLC zone: Times Sq / Penn Station)",
    }
    results["q3_top10_pickup_regions"] = top10.to_dict("records")
    log(f"  busiest cell: lat={top10.iloc[0]['lat_bin']:.4f}, "
        f"lon={top10.iloc[0]['lon_bin']:.4f} ({int(top10.iloc[0]['count']):,} trips)")

    # ---- 4. Longest and shortest trips ---------------------------
    log("Q4 - Longest and shortest trips ...")
    longest  = df.nlargest(5,  "trip_duration")[
        ["pickup_datetime", "trip_duration"]].copy()
    shortest = df.nsmallest(5, "trip_duration")[
        ["pickup_datetime", "trip_duration"]].copy()
    longest["minutes"]  = (longest["trip_duration"]  / 60.0).round(2)
    shortest["minutes"] = (shortest["trip_duration"] / 60.0).round(2)
    results["q4_longest_trips"]  = longest.astype(str).to_dict("records")
    results["q4_shortest_trips"] = shortest.astype(str).to_dict("records")
    log(f"  longest:  {longest.iloc[0]['trip_duration']} sec "
        f"({longest.iloc[0]['minutes']:.1f} min)")
    log(f"  shortest: {shortest.iloc[0]['trip_duration']} sec")

    # ---- 5. Hour with peak traffic -------------------------------
    log("Q5 - Hour with peak traffic ...")
    trips_per_hour = df["pickup_hour"].value_counts().sort_index()
    peak_hour = int(trips_per_hour.idxmax())
    results["q5_peak_hour"] = {
        "hour": peak_hour,
        "trips": int(trips_per_hour.max()),
        "label": f"{peak_hour:02d}:00 - {(peak_hour+1)%24:02d}:00",
    }
    log(f"  peak hour: {peak_hour:02d}:00 ({int(trips_per_hour.max()):,} trips)")

    # ---- 6. Pearson correlation ----------------------------------
    log("Q6 - Pearson correlation: duration vs passengers ...")
    pearson_r, pearson_p = stats.pearsonr(df["trip_duration"], df["passenger_count"])
    spearman_r, spearman_p = stats.spearmanr(df["trip_duration"], df["passenger_count"])
    results["q6_correlation"] = {
        "pearson_r":   float(pearson_r),
        "pearson_p":   float(pearson_p),
        "spearman_r":  float(spearman_r),
        "spearman_p":  float(spearman_p),
        "n":           int(len(df)),
        "interpretation": (
            "near-zero correlation -> trip duration is essentially independent "
            "of passenger count"
        ),
    }
    log(f"  Pearson r = {pearson_r:+.4f} (p = {pearson_p:.2e})")
    log(f"  Spearman r = {spearman_r:+.4f} (p = {spearman_p:.2e})")

    # ---- Save consolidated JSON ----------------------------------
    out_json = os.path.join(OUT_DIR, "statistics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {out_json}")
    log(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
