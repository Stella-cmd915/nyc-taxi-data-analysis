"""
============================================================
NYC Yellow Taxi 2016 - Part IV: Interactive Streamlit Dashboard
============================================================
Run:
    streamlit run 04_dashboard.py

Provides:
  * Interactive filters (date range, hour, passengers, duration)
  * KPI cards
  * Plotly charts (hourly, weekday, duration distribution)
  * Statistics table
  * Filtered raw-data table
  * Folium pickup density map of the filtered subset
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "NYC_cleaned.csv")

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ----------------------------------------------------------------------
st.set_page_config(
    page_title="NYC Yellow Taxi 2016 - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading cleaned dataset ...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["pickup_datetime"],
        usecols=[
            "pickup_datetime", "pickup_hour", "pickup_weekday",
            "passenger_count", "trip_duration",
            "pickup_latitude", "pickup_longitude",
            "dropoff_latitude", "dropoff_longitude",
            "trip_distance_km", "avg_speed_kmh",
        ],
    )
    df["pickup_date"] = df["pickup_datetime"].dt.date
    df["duration_min"] = (df["trip_duration"] / 60.0).round(2)
    return df


# ----------------------------------------------------------------------
def main() -> None:
    st.title("NYC Yellow Taxi 2016 - Big Data Dashboard")
    st.caption("Interactive exploration of 1.45M cleaned NYC Yellow Cab trips.")

    df = load_data()

    # ------------------------- Sidebar filters ------------------------
    st.sidebar.header("Filters")

    min_date, max_date = df["pickup_date"].min(), df["pickup_date"].max()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d_from, d_to = date_range
    else:
        d_from, d_to = min_date, max_date

    hour_range = st.sidebar.slider("Pickup hour", 0, 23, (0, 23))
    pax_range  = st.sidebar.slider("Passenger count", 1, 6, (1, 6))
    dur_range  = st.sidebar.slider("Trip duration (minutes)", 0, 240, (0, 60))

    # ------------------------- Apply filters --------------------------
    mask = (
        (df["pickup_date"] >= d_from) & (df["pickup_date"] <= d_to)
        & df["pickup_hour"].between(hour_range[0], hour_range[1])
        & df["passenger_count"].between(pax_range[0], pax_range[1])
        & df["duration_min"].between(dur_range[0], dur_range[1])
    )
    f = df.loc[mask]

    if len(f) == 0:
        st.warning("No trips match the current filter. Loosen the criteria.")
        return

    # ------------------------- KPI cards ------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trips",                f"{len(f):,}")
    c2.metric("Avg duration (min)",   f"{f['duration_min'].mean():.2f}")
    c3.metric("Avg distance (km)",    f"{f['trip_distance_km'].mean():.2f}")
    c4.metric("Avg passengers",       f"{f['passenger_count'].mean():.2f}")

    st.markdown("---")

    # ------------------------- Charts ---------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Trips per hour")
        hourly = f.groupby("pickup_hour").size().reset_index(name="trips")
        fig = px.bar(hourly, x="pickup_hour", y="trips",
                     labels={"pickup_hour": "Hour", "trips": "Trips"})
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Trips per weekday")
        weekly = (f.groupby("pickup_weekday").size()
                   .reset_index(name="trips"))
        weekly["weekday"] = [WEEKDAY_NAMES[int(i)] for i in weekly["pickup_weekday"]]
        fig = px.bar(weekly, x="weekday", y="trips",
                     category_orders={"weekday": WEEKDAY_NAMES})
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Trip duration distribution")
        fig = px.histogram(f, x="duration_min", nbins=60,
                           labels={"duration_min": "Duration (min)"})
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.subheader("Duration vs passengers")
        sample = f.sample(min(5000, len(f)), random_state=42)
        fig = px.box(sample, x="passenger_count", y="duration_min",
                     labels={"passenger_count": "Passengers",
                             "duration_min": "Duration (min)"})
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------- Statistics table -----------------------
    st.subheader("Descriptive statistics")
    desc = (f[["duration_min", "trip_distance_km", "avg_speed_kmh",
               "passenger_count"]]
            .describe()
            .round(2))
    st.dataframe(desc, use_container_width=True)

    # ------------------------- Raw data table -------------------------
    with st.expander("Raw filtered data (first 1000 rows)"):
        st.dataframe(f.head(1000), use_container_width=True)

    # ------------------------- Map -----------------------------------
    st.subheader("Pickup density map (sample of 10K filtered trips)")
    map_sample = f.sample(min(10_000, len(f)), random_state=42).rename(
        columns={"pickup_latitude": "lat", "pickup_longitude": "lon"}
    )
    st.map(map_sample[["lat", "lon"]], zoom=10)


if __name__ == "__main__":
    main()
