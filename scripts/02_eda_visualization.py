"""
============================================================
NYC Yellow Taxi 2016 - Part II: EDA & Visualisation
============================================================
Generates all figures for the report:

  Distribution analysis (8 pt)
    - trips_per_hour.png
    - trips_per_weekday.png
    - passenger_count_distribution.png

  Trip duration vs passengers (8 pt)
    - duration_vs_passengers_scatter.png  (+ regression line)
    - duration_vs_passengers_box.png

  Geospatial analysis (6 pt)
    - pickup_heatmap.html   (interactive Folium)
    - dropoff_heatmap.html
    - pickup_density_static.png
    - dropoff_density_static.png

  Traffic analysis (6 pt)
    - peak_vs_offpeak.png
    - weekday_vs_weekend.png

Run:  python 02_eda_visualization.py
Input : ../data/NYC_cleaned.csv
Output: ../figures/*.png and *.html
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "NYC_cleaned.csv")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def log(msg: str) -> None:
    print(f"[eda] {msg}", flush=True)


def save(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    log(f"  saved {name}")
    return path


# ---------------- 1. Load cleaned dataset --------------------------
def load_data() -> pd.DataFrame:
    log(f"Loading {INPUT_PATH} ...")
    t0 = time.time()
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["pickup_datetime", "dropoff_datetime"],
    )
    log(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


# ---------------- 2. Distribution plots ----------------------------
def plot_distributions(df: pd.DataFrame) -> None:
    log("Distribution analysis ...")

    # Trips per hour
    hourly = df["pickup_hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(hourly.index, hourly.values, color="#1f77b4", edgecolor="white")
    ax.set_xticks(range(24))
    ax.set_xlabel("Pickup hour (0-23)")
    ax.set_ylabel("Number of trips")
    ax.set_title("Trips per hour of day - NYC Yellow Taxi 2016")
    save(fig, "trips_per_hour.png")

    # Trips per weekday
    weekday = df["pickup_weekday"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([WEEKDAY_NAMES[int(i)] for i in weekday.index], weekday.values,
           color="#2ca02c", edgecolor="white")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Number of trips")
    ax.set_title("Trips per weekday - NYC Yellow Taxi 2016")
    save(fig, "trips_per_weekday.png")

    # Passenger count distribution
    pax = df["passenger_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(pax.index.astype(int), pax.values, color="#ff7f0e", edgecolor="white")
    ax.set_xlabel("Passenger count")
    ax.set_ylabel("Number of trips")
    ax.set_title("Distribution of passenger count per trip")
    for x, y in zip(pax.index, pax.values):
        ax.text(x, y, f"{y/pax.sum()*100:.1f}%", ha="center", va="bottom", fontsize=9)
    save(fig, "passenger_count_distribution.png")


# ---------------- 3. Duration vs passengers ------------------------
def plot_duration_vs_passengers(df: pd.DataFrame) -> None:
    log("Duration vs passengers ...")

    # Sample for scatter (1M+ points is too dense)
    sample = df.sample(n=min(50_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample["passenger_count"] + np.random.uniform(-0.2, 0.2, len(sample)),
               sample["trip_duration"] / 60.0,
               alpha=0.10, s=8, color="#1f77b4")

    # Linear regression line
    x = sample["passenger_count"].values.astype(float)
    y = (sample["trip_duration"] / 60.0).values
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.array([0.5, 6.5])
    ax.plot(xs, slope * xs + intercept, color="red", lw=2,
            label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")

    ax.set_xlabel("Passenger count")
    ax.set_ylabel("Trip duration (minutes)")
    ax.set_title("Trip duration vs passenger count (50K sample)")
    ax.set_ylim(0, 60)  # focus on bulk of distribution
    ax.legend()
    save(fig, "duration_vs_passengers_scatter.png")

    # Boxplot of duration per passenger count
    fig, ax = plt.subplots(figsize=(8, 5))
    df_box = df[df["trip_duration"] < 3600].copy()  # focus on <1h trips
    df_box["duration_min"] = df_box["trip_duration"] / 60.0
    sns.boxplot(data=df_box, x="passenger_count", y="duration_min",
                ax=ax, color="#1f77b4", showfliers=False)
    ax.set_xlabel("Passenger count")
    ax.set_ylabel("Trip duration (minutes)")
    ax.set_title("Trip duration distribution per passenger count")
    save(fig, "duration_vs_passengers_box.png")


# ---------------- 4. Geospatial heatmaps ---------------------------
def plot_geospatial(df: pd.DataFrame) -> None:
    log("Geospatial analysis ...")

    # Static density (hexbin) maps
    sample = df.sample(n=min(200_000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(sample["pickup_longitude"], sample["pickup_latitude"],
                   gridsize=120, bins="log", cmap="hot",
                   extent=[-74.05, -73.75, 40.63, 40.88])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Pickup density (log scale)")
    plt.colorbar(hb, ax=ax, label="log10(trip count)")
    save(fig, "pickup_density_static.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(sample["dropoff_longitude"], sample["dropoff_latitude"],
                   gridsize=120, bins="log", cmap="hot",
                   extent=[-74.05, -73.75, 40.63, 40.88])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Dropoff density (log scale)")
    plt.colorbar(hb, ax=ax, label="log10(trip count)")
    save(fig, "dropoff_density_static.png")

    # Interactive Folium heatmaps (optional but standard)
    try:
        import folium
        from folium.plugins import HeatMap

        sample_small = df.sample(n=min(40_000, len(df)), random_state=42)

        m_pick = folium.Map(location=[40.75, -73.97], zoom_start=11, tiles="cartodbpositron")
        HeatMap(sample_small[["pickup_latitude", "pickup_longitude"]].values.tolist(),
                radius=7, blur=10, max_zoom=13).add_to(m_pick)
        path = os.path.join(FIG_DIR, "pickup_heatmap.html")
        m_pick.save(path)
        log(f"  saved pickup_heatmap.html")

        m_drop = folium.Map(location=[40.75, -73.97], zoom_start=11, tiles="cartodbpositron")
        HeatMap(sample_small[["dropoff_latitude", "dropoff_longitude"]].values.tolist(),
                radius=7, blur=10, max_zoom=13).add_to(m_drop)
        path = os.path.join(FIG_DIR, "dropoff_heatmap.html")
        m_drop.save(path)
        log(f"  saved dropoff_heatmap.html")
    except ImportError:
        log("  folium not installed - skipping interactive heatmaps")


# ---------------- 5. Traffic analysis ------------------------------
def plot_traffic(df: pd.DataFrame) -> None:
    log("Traffic analysis ...")

    # Peak hours: 7-9 & 17-19; off-peak: 0-5
    peak = df[df["pickup_hour"].isin([7, 8, 9, 17, 18, 19])]
    off  = df[df["pickup_hour"].isin([0, 1, 2, 3, 4, 5])]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cats = ["Peak (7-9, 17-19)", "Off-peak (0-5)"]
    vals = [peak["trip_duration"].mean() / 60.0, off["trip_duration"].mean() / 60.0]
    bars = ax.bar(cats, vals, color=["#d62728", "#1f77b4"], edgecolor="white")
    ax.set_ylabel("Average trip duration (minutes)")
    ax.set_title("Average trip duration: peak vs off-peak hours")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f} min",
                ha="center", va="bottom", fontsize=11)
    save(fig, "peak_vs_offpeak.png")

    # Weekday vs weekend - trips per hour curves
    weekday = df[df["pickup_weekday"] < 5].groupby("pickup_hour").size()
    weekend = df[df["pickup_weekday"] >= 5].groupby("pickup_hour").size()
    weekday /= 5.0  # normalise per day
    weekend /= 2.0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(weekday.index, weekday.values, marker="o", label="Weekday (avg/day)", color="#1f77b4")
    ax.plot(weekend.index, weekend.values, marker="s", label="Weekend (avg/day)", color="#ff7f0e")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average trips per day")
    ax.set_title("Trip volume by hour: weekday vs weekend")
    ax.legend()
    save(fig, "weekday_vs_weekend.png")


# ---------------------- Main --------------------------------------
def main() -> None:
    df = load_data()
    plot_distributions(df)
    plot_duration_vs_passengers(df)
    plot_geospatial(df)
    plot_traffic(df)
    log(f"All figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
