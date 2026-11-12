"""
config.py - All the "knobs" of the project in one place.

If an interviewer asks "what if the delay threshold were 12 hours instead of 24?"
you change DELAY_THRESHOLD_H here and re-run. Nothing else needs to change.
"""

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. Definition of congestion (the "ground truth")
# ---------------------------------------------------------------------------
# A port call is DELAYED (= affected by congestion) if the ship waits at
# anchorage for MORE than 24 hours before it gets a berth.
# Why waiting time and not total turnaround? Turnaround also includes the time
# spent unloading, which depends on how much cargo the ship carries, not on
# congestion. Waiting time is the part caused purely by the port being busy.
DELAY_THRESHOLD_H = 24

# We predict 48 hours before the ship arrives. That is early enough to act
# (slow down, warn the customer, re-plan trucks) but late enough that the
# port situation is reasonably known.
PREDICTION_HORIZON_H = 48

# ---------------------------------------------------------------------------
# 2. Simulation settings
# ---------------------------------------------------------------------------
SIM_START = "2024-11-01"
SIM_DAYS = 730                 # Nov 2024 - Oct 2026 (24 months)
WARMUP_DAYS = 10               # the simulated port starts empty; skip the first days
STORM_CLOSURE_WIND_KMH = 45    # above this wind speed, pilots and cranes stop
ROUGH_WEATHER_WIND_KMH = 35    # between 35 and 45 km/h cranes work at half speed (swell, gusts)
DEEP_BERTH_TEU = 8000          # ships bigger than this need a deep-water berth

# Each port behaves differently - this is how we model "variability across ports".
PORTS = {
    # Big, efficient hub. Lots of traffic, few weather problems.
    "Northbay": dict(
        berths=4, deep_berths=2, arrivals_per_day=1.00, productivity=1.15,
        teu_median=5200, teu_max=15000, storm_exposure=0.6,
        strikes_per_year=0.7, maintenance_per_year=2, daylight_only=False,
    ),
    # Medium port on an exposed coast - monsoon storms close it often.
    "Eastgate": dict(
        berths=3, deep_berths=1, arrivals_per_day=0.75, productivity=1.0,
        teu_median=4200, teu_max=12000, storm_exposure=1.7,
        strikes_per_year=0.7, maintenance_per_year=2, daylight_only=False,
    ),
    # Small, slower port. Labour disputes are common and ships can only
    # berth in daylight (06:00-18:00) because pilots don't work at night.
    "Southreach": dict(
        berths=2, deep_berths=0, arrivals_per_day=0.48, productivity=0.9,
        teu_median=2400, teu_max=7500, storm_exposure=1.0,
        strikes_per_year=3.0, maintenance_per_year=1, daylight_only=True,
    ),
    # Quiet regional port - rarely congested. Useful contrast for the model.
    "Westhaven": dict(
        berths=2, deep_berths=1, arrivals_per_day=0.38, productivity=1.0,
        teu_median=3400, teu_max=10000, storm_exposure=0.8,
        strikes_per_year=0.5, maintenance_per_year=1, daylight_only=False,
    ),
}

# Seasonality of ship arrivals (multiplier per month).
# Aug-Oct = pre-holiday peak shipping season; Feb = post Lunar-New-Year lull.
MONTH_TRAFFIC = {1: 1.0, 2: 0.8, 3: 0.95, 4: 1.0, 5: 1.0, 6: 1.0,
                 7: 1.05, 8: 1.2, 9: 1.25, 10: 1.2, 11: 1.05, 12: 1.0}
# Storm frequency per month (monsoon Jun-Sep, cyclone season Oct-Nov).
MONTH_STORMS = {1: 0.4, 2: 0.3, 3: 0.3, 4: 0.4, 5: 0.7, 6: 1.6,
                7: 2.0, 8: 2.0, 9: 1.6, 10: 1.3, 11: 1.2, 12: 0.5}
PEAK_SEASON_MONTHS = [8, 9, 10]

# ---------------------------------------------------------------------------
# 3. Real-world data problems we inject on purpose ("incomplete or noisy data")
# ---------------------------------------------------------------------------
MISSING_RATE = {
    "forecast_max_wind_kmh": 0.08,   # weather feed outage
    "expected_arrivals_48h": 0.05,   # AIS coverage gap
    "vessel_teu": 0.03,              # incomplete vessel registry
}
ETA_ERROR_SD_H = 8          # ships' announced arrival times are off by ~8 h
WIND_FORECAST_ERROR_SD = 7  # a 2-day-ahead wind forecast is off by ~7 km/h
UNANNOUNCED_STRIKE_SHARE = 0.3  # 30% of strikes are sudden (no advance notice)

# ---------------------------------------------------------------------------
# 4. Modelling settings
# ---------------------------------------------------------------------------
# Time-based split: learn from the past, test on the future. Never a random split.
# Development period (Nov 2024 - Apr 2026): used to build and choose the model.
# Test period (May - Oct 2026): touched ONCE at the very end. It deliberately covers
# the hardest months - monsoon storms AND peak shipping season.
TEST_START = "2026-05-01"
# Walk-forward validation inside the development period: train on everything
# before each date, validate on the next 3 months. 3 folds = 3 different seasons
# (peak/monsoon, winter, spring), so the model choice doesn't hinge on one season.
WALK_FORWARD_FOLDS = ["2025-08-01", "2025-11-01", "2026-02-01"]
FOLD_MONTHS = 3

# Missing a real delay is worse than a false alarm: a surprise delay means
# idle trucks, missed connections and an unhappy customer; a false alarm
# mostly costs a phone call. We assume a miss costs 3x a false alarm.
COST_FALSE_NEGATIVE = 3.0
COST_FALSE_POSITIVE = 1.0

N_BOOTSTRAP_MODELS = 25   # used for the confidence score

DATA_PATH = "data/port_calls.csv"
MODEL_PATH = "models/model_bundle.joblib"
OUTPUT_DIR = "outputs"
