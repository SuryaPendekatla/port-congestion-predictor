"""
predict.py - Ask the system about one ship from the command line.

Examples
  # explain some real cases from the test period
  python -m src.predict --examples

  # describe your own situation
  python -m src.predict --port Eastgate --waiting 4 --arriving 3 --wind 70 --storm-hours 20

  # look up a specific port call from the dataset
  python -m src.predict --call-id PC01616
"""
import argparse

import pandas as pd

from src import config as C
from src.features import time_split
from src.predictor import PortDelayPredictor


def scenario(a) -> pd.Series:
    ts = pd.Timestamp(a.arrival)
    p = C.PORTS[a.port]
    return pd.Series(dict(
        port=a.port, ships_waiting_now=a.waiting, berths_busy_now=a.busy if a.busy is not None else p["berths"],
        expected_arrivals_48h=a.arriving, berths_total=p["berths"],
        berths_available=a.berths if a.berths is not None else p["berths"],
        forecast_max_wind_kmh=a.wind, forecast_storm_hours=a.storm_hours,
        strike_notice=int(a.strike), vessel_teu=a.teu, needs_deep_berth=int(a.teu > C.DEEP_BERTH_TEU),
        recent_avg_wait_7d_h=a.recent_wait, arrival_hour=ts.hour, day_of_week=ts.dayofweek,
        month=ts.month))


def main():
    ap = argparse.ArgumentParser(description="Predict the delay risk for one port call.")
    ap.add_argument("--examples", action="store_true", help="explain 4 real cases from the test period")
    ap.add_argument("--call-id", help="explain one port call from data/port_calls.csv")
    ap.add_argument("--port", choices=list(C.PORTS), default="Northbay")
    ap.add_argument("--arrival", default="2026-09-15 10:00", help="planned arrival time")
    ap.add_argument("--waiting", type=float, default=1, help="ships waiting at anchorage now")
    ap.add_argument("--busy", type=float, default=None, help="berths occupied now (default: all)")
    ap.add_argument("--arriving", type=float, default=2, help="other ships due in the next 48 h")
    ap.add_argument("--berths", type=float, default=None, help="berths available (default: all)")
    ap.add_argument("--wind", type=float, default=25, help="forecast max wind, km/h")
    ap.add_argument("--storm-hours", type=float, default=0, help="forecast hours above 45 km/h")
    ap.add_argument("--strike", action="store_true", help="a strike has been announced")
    ap.add_argument("--teu", type=float, default=4000, help="vessel size in TEU")
    ap.add_argument("--recent-wait", type=float, default=5, help="avg wait at this port last 7 days, h")
    a = ap.parse_args()

    P = PortDelayPredictor()
    if a.examples or a.call_id:
        df = pd.read_csv(C.DATA_PATH, parse_dates=["arrival_time", "prediction_time"])
        if a.call_id:
            rows = [df[df.call_id == a.call_id].iloc[0]]
        else:
            _, test = time_split(df)
            rows = [test[(test.port == "Eastgate") & (test.forecast_storm_hours > 12)].iloc[0],
                    test[test.port == "Northbay"].sort_values("ships_waiting_now").iloc[-3],
                    test[(test.port == "Southreach") & (test.strike_notice == 1)].iloc[0]
                    if ((test.port == "Southreach") & (test.strike_notice == 1)).any()
                    else test[test.strike_notice == 1].iloc[0],
                    test[(test.port == "Westhaven") & (test.delayed == 0)].iloc[5]]
        for r in rows:
            print("=" * 90)
            print(f"{r.call_id} | {r.port} | arrives {r.arrival_time:%d %b %Y %H:%M} | "
                  f"predicted at {r.prediction_time:%d %b %H:%M}")
            print(P.to_text(P.explain(r)))
            print(f">> What actually happened: waited {r.wait_hours:.0f} h "
                  f"({'DELAYED' if r.delayed else 'on time'})")
    else:
        print(P.to_text(P.explain(scenario(a))))


if __name__ == "__main__":
    main()
