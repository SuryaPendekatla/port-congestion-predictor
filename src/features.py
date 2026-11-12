"""
features.py - Turns raw port-call records into model inputs.

Rule we follow everywhere: a feature may only use information that is
available 48 hours before the ship arrives. Outcome columns (wait_hours,
turnaround_hours, ...) are never used as inputs - that would be "leakage"
(the model would be peeking at the answer).
"""
import numpy as np
import pandas as pd

from src import config as C

PORT_NAMES = list(C.PORTS.keys())

# Raw inputs a planner fills in (or that come from data feeds) at prediction time.
RAW_INPUTS = [
    "port", "ships_waiting_now", "berths_busy_now", "expected_arrivals_48h",
    "berths_total", "berths_available", "forecast_max_wind_kmh", "forecast_storm_hours",
    "strike_notice",
    "vessel_teu", "needs_deep_berth", "recent_avg_wait_7d_h",
    "arrival_hour", "day_of_week", "month",
]

# Columns the model actually sees (after engineering).
MODEL_FEATURES = [
    # traffic & queue
    "ships_waiting_now", "expected_arrivals_48h", "load_per_berth", "occupancy_now",
    # capacity
    "berths_available", "berth_out_for_maintenance",
    # weather
    "forecast_max_wind_kmh", "forecast_storm_hours", "storm_warning", "storm_x_load",
    # labour
    "strike_notice",
    # vessel
    "vessel_teu", "needs_deep_berth",
    # timing / seasonality
    "is_peak_season", "is_night_arrival", "night_at_daylight_only_port", "is_weekend",
    # recent history of this port
    "recent_avg_wait_7d_h",
] + [f"port_{p}" for p in PORT_NAMES]

# Groups used for explanations ("which kind of factor drove this prediction?").
# Each group lists the RAW inputs that get reset to "a normal day" in the what-if test.
FACTOR_GROUPS = {
    "Traffic & queue": ["ships_waiting_now", "berths_busy_now", "expected_arrivals_48h"],
    "Berth capacity": ["berths_available"],
    "Weather": ["forecast_max_wind_kmh", "forecast_storm_hours"],
    "Labour": ["strike_notice"],
    "Vessel size": ["vessel_teu", "needs_deep_berth"],
    "Timing & season": ["arrival_hour", "day_of_week", "month"],
    "Recent port backlog": ["recent_avg_wait_7d_h"],
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw records -> table of MODEL_FEATURES (missing values stay as NaN)."""
    out = pd.DataFrame(index=df.index)
    berths_avail = df["berths_available"].clip(lower=1)

    # Traffic & queue: the core congestion signal is "how many ships are
    # competing for each available berth" (traffic x capacity interaction).
    out["ships_waiting_now"] = df["ships_waiting_now"]
    out["expected_arrivals_48h"] = df["expected_arrivals_48h"]
    out["load_per_berth"] = (df["ships_waiting_now"] + df["expected_arrivals_48h"]) / berths_avail
    out["occupancy_now"] = df["berths_busy_now"] / df["berths_total"]

    # Capacity
    out["berths_available"] = df["berths_available"]
    out["berth_out_for_maintenance"] = (df["berths_available"] < df["berths_total"]).astype(int)

    # Weather: how strong and how LONG the forecast storm is. A storm hurts much
    # more when the port is already busy (days of closure x ships per berth).
    wind = df["forecast_max_wind_kmh"]
    out["forecast_max_wind_kmh"] = wind
    out["forecast_storm_hours"] = df["forecast_storm_hours"]
    out["storm_warning"] = np.where(wind.isna(), np.nan,
                                    (wind >= C.STORM_CLOSURE_WIND_KMH).astype(float))
    out["storm_x_load"] = df["forecast_storm_hours"] / 24 * out["load_per_berth"]

    out["strike_notice"] = df["strike_notice"]
    out["vessel_teu"] = df["vessel_teu"]
    out["needs_deep_berth"] = df["needs_deep_berth"]

    # Timing & seasonality
    out["is_peak_season"] = df["month"].isin(C.PEAK_SEASON_MONTHS).astype(int)
    night = ((df["arrival_hour"] < 6) | (df["arrival_hour"] >= 18)).astype(int)
    out["is_night_arrival"] = night
    daylight_port = df["port"].map(lambda p: C.PORTS[p]["daylight_only"]).astype(int)
    out["night_at_daylight_only_port"] = night * daylight_port
    out["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    out["recent_avg_wait_7d_h"] = df["recent_avg_wait_7d_h"]
    for p in PORT_NAMES:
        out[f"port_{p}"] = (df["port"] == p).astype(int)
    return out[MODEL_FEATURES].astype(float)


def time_split(df: pd.DataFrame):
    """Development period (past) vs test period (future). Never a random split:
    a random split would let the model learn from the future."""
    t = pd.to_datetime(df["arrival_time"])
    # a ship can only be used for training once its outcome is known (it got a berth);
    # the few ships that arrived before the test start but were still waiting are dropped
    known = t + pd.to_timedelta(df["wait_hours"], unit="h")
    start = pd.Timestamp(C.TEST_START)
    return df[known < start], df[t >= start]


def walk_forward_folds(dev: pd.DataFrame):
    """Yield (name, train_part, validation_part) for each walk-forward fold:
    train on everything before the fold date, validate on the next few months."""
    t = pd.to_datetime(dev["arrival_time"])
    for start in C.WALK_FORWARD_FOLDS:
        s = pd.Timestamp(start)
        e = s + pd.DateOffset(months=C.FOLD_MONTHS)
        # a ship only counts as "past" once its outcome is known (it got a berth)
        known = pd.to_datetime(dev["arrival_time"]) + pd.to_timedelta(dev["wait_hours"], unit="h")
        name = f"{s:%b %Y}-{(e - pd.Timedelta(days=1)):%b %Y}"
        yield name, dev[known < s], dev[(t >= s) & (t < e)]


def normal_day_reference(train: pd.DataFrame) -> dict:
    """What 'a normal day' looks like at each port (from training data only).
    Used by the what-if explanations: 'if traffic were normal, risk would be X'."""
    ref = {}
    for p, g in train.groupby("port"):
        ref[p] = dict(
            ships_waiting_now=float(g["ships_waiting_now"].median()),
            berths_busy_now=float(g["berths_busy_now"].median()),
            expected_arrivals_48h=float(g["expected_arrivals_48h"].median()),
            berths_available=float(C.PORTS[p]["berths"]),
            forecast_max_wind_kmh=float(g["forecast_max_wind_kmh"].median()),
            forecast_storm_hours=0.0,
            strike_notice=0.0,
            vessel_teu=float(g["vessel_teu"].median()),
            needs_deep_berth=0.0,
            arrival_hour=12.0, day_of_week=2.0, month=4.0,   # a weekday, midday, off-peak month
            recent_avg_wait_7d_h=float(g["recent_avg_wait_7d_h"].median()),
        )
    return ref
