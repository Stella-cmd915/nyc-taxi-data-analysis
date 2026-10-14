"""
============================================================
NYC Yellow Taxi 2016 - Part I: Data Ingestion & Cleaning
============================================================
- Loads the raw NYC.csv with Pandas
- Handles missing values and invalid coordinates
- Removes outliers (trip_duration extremes, geographic outliers)
- Feature engineering: pickup_hour, pickup_day, pickup_month,
  pickup_weekday, trip_distance_km, avg_speed_kmh
- Saves a cleaned CSV ready for the rest of the pipeline.

Run:  python 01_data_cleaning.py
Input : ../data/NYC.csv
Output: ../data/NYC_cleaned.csv
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd

# ----------------------------- Config -----------------------------
INPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "NYC.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "NYC_cleaned.csv")

# NYC bounding box (rough): covers all 5 boroughs + JFK/LGA/EWR
LAT_MIN, LAT_MAX = 40.50, 40.95
LON_MIN, LON_MAX = -74.30, -73.65

# Trip duration bounds: realistic taxi trip is 30 sec to 4 hours
MIN_DURATION_SEC = 30
MAX_DURATION_SEC = 4 * 3600

# Passenger count bounds
MIN_PASSENGERS = 1
MAX_PASSENGERS = 6  # NYC TLC max for medallion cabs

# Speed sanity check (km/h)
MAX_SPEED_KMH = 120


# --------------------------- Helpers -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in kilometres."""
    R = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def log(msg: str) -> None:
    print(f"[clean] {msg}", flush=True)


# --------------------------- Pipeline ----------------------------
def main() -> None:
    t0 = time.time()
    log(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["pickup_datetime", "dropoff_datetime"],
        dtype={"vendor_id": "int8", "passenger_count": "Int8",
               "store_and_fwd_flag": "category"},
    )
    n0 = len(df)
    log(f"Loaded {n0:,} rows, {df.shape[1]} columns in {time.time()-t0:.1f}s")

    # ---- 1. Missing-value handling --------------------------------
    na_per_col = df.isna().sum()
    log(f"Missing values per column:\n{na_per_col[na_per_col > 0].to_string() or '  (none)'}")
    df = df.dropna(subset=[
        "pickup_datetime", "dropoff_datetime",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "trip_duration",
    ])
    log(f"After dropna of essential columns: {len(df):,} rows")

    # ---- 2. Invalid coordinates -----------------------------------
    geo_mask = (
        df["pickup_latitude"].between(LAT_MIN, LAT_MAX)
        & df["pickup_longitude"].between(LON_MIN, LON_MAX)
        & df["dropoff_latitude"].between(LAT_MIN, LAT_MAX)
        & df["dropoff_longitude"].between(LON_MIN, LON_MAX)
    )
    n_geo_removed = (~geo_mask).sum()
    df = df.loc[geo_mask].copy()
    log(f"Removed {n_geo_removed:,} rows with out-of-bounds coordinates")

    # ---- 3. Outlier removal on trip_duration ----------------------
    dur_mask = df["trip_duration"].between(MIN_DURATION_SEC, MAX_DURATION_SEC)
    n_dur_removed = (~dur_mask).sum()
    df = df.loc[dur_mask].copy()
    log(f"Removed {n_dur_removed:,} rows with trip_duration outside "
        f"[{MIN_DURATION_SEC}s, {MAX_DURATION_SEC}s]")

    # ---- 4. Passenger count sanity --------------------------------
    pax_mask = df["passenger_count"].between(MIN_PASSENGERS, MAX_PASSENGERS)
    n_pax_removed = (~pax_mask).sum()
    df = df.loc[pax_mask].copy()
    log(f"Removed {n_pax_removed:,} rows with passenger_count outside "
        f"[{MIN_PASSENGERS}, {MAX_PASSENGERS}]")

    # ---- 5. Feature engineering -----------------------------------
    log("Feature engineering ...")
    df["pickup_hour"]    = df["pickup_datetime"].dt.hour.astype("int8")
    df["pickup_day"]     = df["pickup_datetime"].dt.day.astype("int8")
    df["pickup_month"]   = df["pickup_datetime"].dt.month.astype("int8")
    df["pickup_weekday"] = df["pickup_datetime"].dt.dayofweek.astype("int8")  # 0=Mon
    df["pickup_date"]    = df["pickup_datetime"].dt.date

    df["trip_distance_km"] = haversine_km(
        df["pickup_latitude"].values, df["pickup_longitude"].values,
        df["dropoff_latitude"].values, df["dropoff_longitude"].values,
    ).round(4)

    df["avg_speed_kmh"] = (df["trip_distance_km"] / (df["trip_duration"] / 3600.0)).round(2)

    # Remove physically impossible speeds (GPS noise / data error)
    speed_mask = df["avg_speed_kmh"].between(0, MAX_SPEED_KMH)
    n_speed_removed = (~speed_mask).sum()
    df = df.loc[speed_mask].copy()
    log(f"Removed {n_speed_removed:,} rows with avg_speed > {MAX_SPEED_KMH} km/h "
        "(GPS noise / impossible)")

    # ---- 6. Save cleaned dataset ----------------------------------
    log(f"Final cleaned rows: {len(df):,} (kept {len(df)/n0*100:.2f}% of original)")
    # Use pyarrow for ~3x faster CSV writing on large frames; fall back to pandas
    try:
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        # Convert datetime columns to ISO strings for portable CSV
        df_out = df.copy()
        df_out["pickup_datetime"]  = df_out["pickup_datetime"].astype(str)
        df_out["dropoff_datetime"] = df_out["dropoff_datetime"].astype(str)
        df_out["pickup_date"]      = df_out["pickup_date"].astype(str)
        table = pa.Table.from_pandas(df_out, preserve_index=False)
        pa_csv.write_csv(table, OUTPUT_PATH)
    except Exception as exc:
        log(f"pyarrow write failed ({exc}); falling back to pandas")
        df.to_csv(OUTPUT_PATH, index=False)
    log(f"Wrote {OUTPUT_PATH}  ({os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")

    # ---- 7. Summary -----------------------------------------------
    log("=" * 60)
    log("CLEANING SUMMARY")
    log("=" * 60)
    log(f"Original rows                : {n0:>10,}")
    log(f"Dropped - missing values      : {n0 - len(df) - n_geo_removed - n_dur_removed - n_pax_removed - n_speed_removed:>10,}")
    log(f"Dropped - invalid coordinates : {n_geo_removed:>10,}")
    log(f"Dropped - bad trip_duration   : {n_dur_removed:>10,}")
    log(f"Dropped - bad passenger_count : {n_pax_removed:>10,}")
    log(f"Dropped - impossible speed    : {n_speed_removed:>10,}")
    log(f"Final cleaned rows            : {len(df):>10,}")
    log(f"Total time                    : {time.time()-t0:>10.1f}s")


if __name__ == "__main__":
    main()
