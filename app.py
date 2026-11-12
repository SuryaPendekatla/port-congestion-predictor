"""
app.py - Interactive demo (Streamlit).

    pip install -r requirements.txt
    streamlit run app.py

Two modes:
  * "What-if planner": set the situation with sliders, see risk, confidence,
    reasons and the recommended action update live.
  * "Replay a real test case": pick a port call from the test period
    (May-Oct 2026), see what the system predicted 48 h ahead, then what happened.
"""
import os

import pandas as pd
import streamlit as st

from src import config as C
from src.features import time_split
from src.predictor import ALERT_THRESHOLD, HIGH_RISK, PortDelayPredictor

st.set_page_config(page_title="Port congestion & delay predictor", layout="wide")

if not os.path.exists(C.MODEL_PATH):
    # First run (e.g. on Streamlit Cloud or a fresh clone): build the data and the model
    # automatically so the demo works with one command. Takes about 15 seconds.
    with st.spinner("First run: generating data and training the model (about 15 s)..."):
        from src import simulate, train
        if not os.path.exists(C.DATA_PATH):
            simulate.main()
        train.main()


@st.cache_resource
def load():
    df = pd.read_csv(C.DATA_PATH, parse_dates=["arrival_time", "prediction_time"])
    return PortDelayPredictor(), df


P, df = load()

st.title("Port congestion & shipment delay predictor")
st.caption(f"Predicts, 48 h before arrival, whether a ship will wait more than "
           f"{C.DELAY_THRESHOLD_H} h at anchorage for a berth. Synthetic data - 4 fictional ports.")

mode = st.sidebar.radio("Mode", ["What-if planner", "Replay a real test case"])

if mode == "What-if planner":
    sb = st.sidebar
    port = sb.selectbox("Port", list(C.PORTS))
    pc = C.PORTS[port]
    sb.caption(f"{pc['berths']} berths ({pc['deep_berths']} deep-water)"
               + (" - daylight-only berthing" if pc["daylight_only"] else ""))
    date = sb.date_input("Planned arrival date", pd.Timestamp("2026-09-15"))
    hour = sb.slider("Arrival hour", 0, 23, 10)
    sb.markdown("**Port right now (48 h before arrival)**")
    waiting = sb.slider("Ships waiting at anchorage", 0, 12, 1)
    busy = sb.slider("Berths busy", 0, pc["berths"], pc["berths"])
    arriving = sb.slider("Other ships due in next 48 h", 0, 12, 2)
    ais_missing = sb.checkbox("AIS feed down (incoming traffic unknown)")
    berths = sb.slider("Berths available at arrival", 1, pc["berths"], pc["berths"])
    sb.markdown("**Conditions**")
    wind = sb.slider("Forecast max wind (km/h)", 0, 100, 25)
    storm_h = sb.slider("Forecast hours above 45 km/h", 0, 72, 0) if wind >= 45 else 0
    weather_missing = sb.checkbox("Weather forecast unavailable")
    strike = sb.checkbox("Strike announced")
    teu = sb.slider("Vessel size (TEU)", 600, int(pc["teu_max"]), min(4000, int(pc["teu_max"])), step=100)
    recent = sb.slider("Avg wait at this port last 7 days (h)", 0, 120, 5)
    ts = pd.Timestamp(date) + pd.Timedelta(hours=hour)
    row = pd.Series(dict(
        port=port, ships_waiting_now=waiting, berths_busy_now=busy,
        expected_arrivals_48h=float("nan") if ais_missing else arriving,
        berths_total=pc["berths"], berths_available=berths,
        forecast_max_wind_kmh=float("nan") if weather_missing else wind,
        forecast_storm_hours=float("nan") if weather_missing else storm_h,
        strike_notice=int(strike), vessel_teu=teu, needs_deep_berth=int(teu > C.DEEP_BERTH_TEU),
        recent_avg_wait_7d_h=recent, arrival_hour=ts.hour, day_of_week=ts.dayofweek, month=ts.month))
    actual = None
else:
    _, test = time_split(df)
    test = test.sort_values("arrival_time")
    port = st.sidebar.selectbox("Port", ["All"] + list(C.PORTS))
    only = st.sidebar.selectbox("Show", ["All cases", "Delayed only", "On time only"])
    t = test if port == "All" else test[test.port == port]
    t = t if only == "All cases" else t[t.delayed == (1 if only == "Delayed only" else 0)]
    labels = [f"{r.call_id} - {r.port} - {r.arrival_time:%d %b %Y %H:%M}" for r in t.itertuples()]
    pick = st.sidebar.selectbox("Port call", labels)
    row = t.iloc[labels.index(pick)]
    actual = row

r = P.explain(row)

c1, c2, c3 = st.columns(3)
c1.metric("Chance of >24 h wait", f"{r['probability']:.0%}")
c2.metric("Risk level", r["risk"],
          help=f"Low < {ALERT_THRESHOLD:.0%} <= Elevated < {HIGH_RISK:.0%} <= High. The 25% alert line "
               "comes from the cost ratio (a missed delay costs 3x a false alarm).")
c3.metric("Confidence", r["confidence_label"],
          help="Share of 25 model versions (different training resamples, different fills for "
               "missing data) that reach the same alert decision.")
st.caption(f"{r['confidence'] * 25:.0f} of 25 model versions agree with this decision."
           + (" Note: " + "; ".join(r["confidence_notes"]) if r["confidence_notes"] else ""))

st.subheader("Recommended action")
box = st.error if r["risk"] == "High" else st.warning if r["risk"] == "Elevated" else st.success
box(r["action"])

st.subheader("Why - what if each factor were normal for this port?")
rows = []
for e in r["factors"]:
    rows.append({"Factor": e["factor"], "Right now": e["description"],
                 "Effect on risk": f"{e['effect'] * 100:+.0f} pts",
                 "Risk if this were normal": f"{e['prob_if_normal']:.0%}"})
st.table(pd.DataFrame(rows).set_index("Factor"))
st.bar_chart(pd.DataFrame({"Effect on risk (points)": [e["effect"] * 100 for e in r["factors"]]},
                          index=[e["factor"] for e in r["factors"]]), horizontal=True)

s = r["similar"]
st.caption(f"Past ships at this port with similar predicted risk ({s['k']} cases): median wait "
           f"{s['median_wait_h']:.0f} h, middle half {s['p25']:.0f}-{s['p75']:.0f} h, "
           f"{s['share_delayed']:.0%} waited more than {C.DELAY_THRESHOLD_H} h.")

if actual is not None:
    st.subheader("What actually happened")
    verdict = "DELAYED" if actual.delayed else "on time"
    alert = r["probability"] >= ALERT_THRESHOLD
    correct = alert == bool(actual.delayed)
    st.info(f"The ship waited **{actual.wait_hours:.0f} h** for a berth ({verdict}). "
            f"The system {'raised' if alert else 'did not raise'} an alert -> "
            f"**{'correct' if correct else 'wrong'}**."
            + (f" Storm hit while waiting: {actual.storm_hours_while_waiting:.0f} h."
               if actual.storm_hours_while_waiting else "")
            + (f" Strike hours while waiting: {actual.strike_hours_while_waiting:.0f} h."
               if actual.strike_hours_while_waiting else ""))
