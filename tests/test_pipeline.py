"""
Automated checks for the rules that make the results trustworthy.

Run from the project folder:   python -m pytest -q

  - the label matches the definition of congestion (wait > 24 h)
  - no outcome column can leak into the model's inputs
  - the time split never lets the model learn from the future
  - features, threshold and predictions behave as designed
  - the model reacts sensibly (more ships waiting -> higher risk)
"""
import os

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import MODEL_FEATURES, add_features, time_split, walk_forward_folds
from src.predictor import ALERT_THRESHOLD, PortDelayPredictor, risk_level

OUTCOME_COLUMNS = ["wait_hours", "turnaround_hours", "storm_hours_while_waiting",
                   "strike_hours_while_waiting", "delayed"]


def scenario(**changes):
    """One port call with ordinary values; override any field with keyword arguments."""
    row = dict(port="Northbay", ships_waiting_now=1, berths_busy_now=4, expected_arrivals_48h=2,
               berths_total=4, berths_available=4, forecast_max_wind_kmh=25.0, forecast_storm_hours=0.0,
               strike_notice=0, vessel_teu=4000.0, needs_deep_berth=0, recent_avg_wait_7d_h=5.0,
               arrival_hour=10, day_of_week=2, month=4)
    row.update(changes)
    return pd.Series(row)


@pytest.fixture(scope="session")
def data():
    if not os.path.exists(C.DATA_PATH):
        from src import simulate
        simulate.main()
    return pd.read_csv(C.DATA_PATH, parse_dates=["arrival_time", "prediction_time"])


@pytest.fixture(scope="session")
def predictor(data):
    if not os.path.exists(C.MODEL_PATH):
        from src import train
        train.main()
    return PortDelayPredictor()


# --- the data -------------------------------------------------------------
def test_label_matches_definition(data):
    assert (data["wait_hours"] >= 0).all()
    assert (data["delayed"] == (data["wait_hours"] > C.DELAY_THRESHOLD_H).astype(int)).all()


def test_prediction_is_made_48h_before_arrival(data):
    gap_h = (data["arrival_time"] - data["prediction_time"]).dt.total_seconds() / 3600
    assert (gap_h == C.PREDICTION_HORIZON_H).all()


# --- no leakage -----------------------------------------------------------
def test_no_outcome_column_is_a_model_input(data):
    assert not set(OUTCOME_COLUMNS) & set(MODEL_FEATURES)
    # stronger check: scrambling every outcome column must not change a single feature value
    scrambled = data.copy()
    rng = np.random.default_rng(0)
    for col in OUTCOME_COLUMNS:
        scrambled[col] = rng.permutation(scrambled[col].values)
    pd.testing.assert_frame_equal(add_features(data), add_features(scrambled))


def test_time_split_never_trains_on_the_future(data):
    dev, test = time_split(data)
    start = pd.Timestamp(C.TEST_START)
    outcome_known = dev["arrival_time"] + pd.to_timedelta(dev["wait_hours"], unit="h")
    assert (outcome_known < start).all()          # every training answer was known before the test period
    assert (test["arrival_time"] >= start).all()
    assert not set(dev["call_id"]) & set(test["call_id"])


def test_walk_forward_folds_only_use_the_past(data):
    dev, _ = time_split(data)
    for name, train_part, valid_part in walk_forward_folds(dev):
        outcome_known = train_part["arrival_time"] + pd.to_timedelta(train_part["wait_hours"], unit="h")
        assert outcome_known.max() < valid_part["arrival_time"].min(), name


# --- features and decision rule -----------------------------------------------
def test_ships_per_berth_feature():
    X = add_features(pd.DataFrame([scenario(port="Eastgate", ships_waiting_now=3, expected_arrivals_48h=1,
                                            berths_total=3, berths_available=3,
                                            forecast_max_wind_kmh=78.0, forecast_storm_hours=23.0)]))
    assert X.loc[0, "load_per_berth"] == pytest.approx(4 / 3)
    assert X.loc[0, "storm_warning"] == 1
    assert list(X.columns) == MODEL_FEATURES


def test_alert_threshold_follows_cost_ratio():
    ratio = C.COST_FALSE_NEGATIVE / C.COST_FALSE_POSITIVE
    assert ALERT_THRESHOLD == pytest.approx(1 / (1 + ratio))
    assert [risk_level(p) for p in (0.10, 0.30, 0.80)] == ["Low", "Elevated", "High"]


# --- the trained system ---------------------------------------------------
def test_predictions_are_valid(data, predictor):
    _, test = time_split(data)
    out = predictor.predict(test.head(50))
    assert out["probability"].between(0, 1).all()
    assert set(out["confidence_label"]) <= {"High", "Medium", "Low"}


def test_more_ships_waiting_means_more_risk(predictor):
    risks = [predictor.explain(scenario(ships_waiting_now=n))["probability"] for n in range(0, 9, 2)]
    assert all(a < b for a, b in zip(risks, risks[1:])), risks


def test_explanation_is_complete(predictor):
    r = predictor.explain(scenario(ships_waiting_now=6, strike_notice=1))
    assert len(r["factors"]) == 7
    assert r["action"]
    assert r["risk"] in {"Low", "Elevated", "High"}


def test_missing_data_is_reported(predictor):
    r = predictor.explain(scenario(expected_arrivals_48h=np.nan, forecast_max_wind_kmh=np.nan,
                                   forecast_storm_hours=np.nan))
    assert "incoming-traffic data (AIS) missing" in r["confidence_notes"]
    assert "weather forecast missing" in r["confidence_notes"]
