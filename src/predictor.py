"""
predictor.py - Everything needed to answer one question for a planner:

    "This ship arrives at port X in 48 hours. Will it be stuck waiting,
     how sure are we, why, and what should we do?"

It returns four things for every port call:
  1. probability  - chance that the ship waits more than 24 h for a berth
  2. confidence   - how much to trust that probability (High / Medium / Low)
  3. reasons      - which factors pushed the risk up or down, in plain English
  4. action       - what a planner should do about it

The trained model and helpers are loaded from models/model_bundle.joblib
(created by `python -m src.train`).
"""
import numpy as np
import pandas as pd
import joblib

from src import config as C
from src.features import add_features, FACTOR_GROUPS

# Risk bands shown to users. The 25% alert line is not arbitrary: if missing a
# delay costs 3x a false alarm, it is worth acting whenever
#   P(delay) x 3  >  (1 - P(delay)) x 1   <=>   P(delay) > 1 / (1 + 3) = 25%
ALERT_THRESHOLD = C.COST_FALSE_POSITIVE / (C.COST_FALSE_POSITIVE + C.COST_FALSE_NEGATIVE)
HIGH_RISK = 0.60

# Inputs that are sometimes missing in real feeds. When one is missing, each of
# the 25 model versions fills it with a different plausible value from history.
FILLABLE = {
    "expected_arrivals_48h": "incoming-traffic data (AIS) missing",
    "forecast_max_wind_kmh": "weather forecast missing",
    "recent_avg_wait_7d_h": "no recent waiting-time history",
    "vessel_teu": "vessel size unknown",
}
UNUSUAL_PENALTY = 0.20
CONF_HIGH, CONF_MEDIUM = 0.90, 0.70     # >= 23 of 25 votes agree = High; >= 18 = Medium



def risk_level(p):
    if p >= HIGH_RISK:
        return "High"
    if p >= ALERT_THRESHOLD:
        return "Elevated"
    return "Low"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def confidence_scores(bundle, raw: pd.DataFrame):
    """Confidence = share of 25 slightly different models that reach the SAME decision.

    Each of the 25 models was trained on a different random resample of the
    history (bootstrap), and each fills any missing input with a different
    plausible value from that port's history. If (almost) all 25 agree with the
    final alert / no-alert decision, the prediction is robust -> High confidence.
    If they split, small changes in data would flip the answer -> Low confidence.

    This captures three sources of doubt with one simple, explainable number:
      - the model itself is uncertain (little similar history),
      - the probability sits close to the 25% alert line,
      - important data is missing (filling it differently changes the answer).

    Note: probability and confidence are different things.
      probability = "how likely is the delay?"
      confidence  = "how stable is our alert decision?"
    """
    raw = raw.reset_index(drop=True)
    p_final = bundle["model"].predict_proba(add_features(raw))[:, 1]
    decision = p_final >= ALERT_THRESHOLD
    boot_p = []
    for i, m in enumerate(bundle["bootstrap_models"]):
        rng = np.random.default_rng(1000 + i)
        r = raw.copy()
        for col in FILLABLE:
            miss = r[col].isna().to_numpy()
            for j in np.where(miss)[0]:
                pool = bundle["fill_values"][r.at[j, "port"]][col]
                r.at[j, col] = rng.choice(pool)
                if col == "forecast_max_wind_kmh":   # keep the weather pair consistent
                    r.at[j, "forecast_storm_hours"] = bundle["fill_storm_hours"][r.at[j, "port"]].get(
                        round(float(r.at[j, col])), 0.0)
        boot_p.append(m.predict_proba(add_features(r))[:, 1])
    boot_p = np.column_stack(boot_p)
    agreement = ((boot_p >= ALERT_THRESHOLD) == decision[:, None]).mean(axis=1)
    spread = boot_p.std(axis=1)

    notes = [[] for _ in range(len(raw))]
    for col, msg in FILLABLE.items():
        for j in np.where(raw[col].isna().to_numpy())[0]:
            notes[j].append(msg)
    # situations far outside anything seen in training -> the model is extrapolating
    X = add_features(raw)
    unusual = (X["load_per_berth"] > bundle["load_p99"]).to_numpy() | \
              (X["ships_waiting_now"] > bundle["waiting_max"]).to_numpy()
    for j in np.where(unusual)[0]:
        notes[j].append("traffic level beyond almost everything in the training data")

    conf = np.clip(agreement - unusual * UNUSUAL_PENALTY, 0, 1)
    labels = np.where(conf >= CONF_HIGH, "High", np.where(conf >= CONF_MEDIUM, "Medium", "Low"))
    return conf, labels, spread, notes


# ---------------------------------------------------------------------------
# Explanations: "what if this factor were normal?"
# ---------------------------------------------------------------------------
def what_if_effects(bundle, row: pd.Series):
    """For each factor group, reset it to a normal day at this port and see how
    much the probability changes. Big change = that factor matters for THIS ship.
    Works for any model, and every number is something you can say out loud:
    'if traffic were normal, risk would drop from 78% to 21%'."""
    model = bundle["model"]
    base = pd.DataFrame([row])
    p = model.predict_proba(add_features(base))[0, 1]
    ref = bundle["normal_day"][row["port"]]
    effects = []
    for group, cols in FACTOR_GROUPS.items():
        alt = base.copy()
        for c in cols:
            alt[c] = ref[c]
        if group == "Vessel size":
            alt["needs_deep_berth"] = float(ref["vessel_teu"] > C.DEEP_BERTH_TEU)
        p_alt = model.predict_proba(add_features(alt))[0, 1]
        effects.append(dict(factor=group, effect=p - p_alt, prob_if_normal=p_alt))
    effects.sort(key=lambda e: -abs(e["effect"]))
    return p, effects


def describe_factor(group, row, bundle):
    """Plain-English description of the current value of a factor."""
    port = row["port"]
    ref = bundle["normal_day"][port]
    pc = C.PORTS[port]
    if group == "Traffic & queue":
        w, e, b = row["ships_waiting_now"], row["expected_arrivals_48h"], max(row["berths_available"], 1)
        busy = f"{row['berths_busy_now']:.0f} of {pc['berths']} berths busy now"
        normal = (ref["ships_waiting_now"] + ref["expected_arrivals_48h"]) / pc["berths"]
        if pd.isna(e):
            return f"{busy}, {w:.0f} ship{'s' if w != 1 else ''} waiting; incoming-traffic data missing"
        return (f"{busy}; {w:.0f} waiting + {e:.0f} due in 48 h for {b:.0f} "
                f"berth{'s' if b != 1 else ''} = {(w + e) / b:.1f} ships per berth "
                f"(normal here: {normal:.1f})")
    if group == "Berth capacity":
        if row["berths_available"] < pc["berths"]:
            return f"only {row['berths_available']:.0f} of {pc['berths']} berths open (maintenance)"
        return f"all {pc['berths']} berths open"
    if group == "Weather":
        w = row["forecast_max_wind_kmh"]
        if pd.isna(w):
            return "weather forecast unavailable"
        sh = row.get("forecast_storm_hours", 0)
        if w >= C.STORM_CLOSURE_WIND_KMH:
            return (f"storm forecast - about {sh:.0f} h of port closure expected "
                    f"(winds up to {w:.0f} km/h; port stops above {C.STORM_CLOSURE_WIND_KMH})")
        return f"no storm expected (max wind {w:.0f} km/h)"
    if group == "Labour":
        return "strike announced at the port in this window" if row["strike_notice"] else "no strike announced"
    if group == "Vessel size":
        t = row["vessel_teu"]
        if pd.isna(t):
            return "vessel size unknown"
        if row["needs_deep_berth"]:
            return f"large ship ({t:,.0f} TEU) can only use the {pc['deep_berths']} deep-water berth(s)"
        return f"{t:,.0f} TEU ship - fits any berth"
    if group == "Timing & season":
        bits = []
        if row["month"] in C.PEAK_SEASON_MONTHS:
            bits.append("peak shipping season")
        night = row["arrival_hour"] < 6 or row["arrival_hour"] >= 18
        if night and pc["daylight_only"]:
            bits.append("night arrival at a daylight-only port (must wait for morning)")
        elif night:
            bits.append("night arrival")
        if row["day_of_week"] >= 5:
            bits.append("weekend arrival")
        return ", ".join(bits) if bits else "ordinary weekday, off-peak"
    if group == "Recent port backlog":
        r = row["recent_avg_wait_7d_h"]
        if pd.isna(r):
            return "no recent waiting-time history"
        return f"ships waited {r:.0f} h on average last week (normal here: {ref['recent_avg_wait_7d_h']:.0f} h)"
    return ""


# ---------------------------------------------------------------------------
# Similar past cases (gives a realistic "how long" estimate)
# ---------------------------------------------------------------------------
def similar_cases(bundle, row: pd.Series, p: float, window=0.10, k_min=15):
    """Look up past port calls at the same port that the model rated at a similar
    risk (within +/-10 points), and report how long those ships ACTUALLY waited.
    Turns a probability into an hours estimate a customer understands."""
    s = bundle["similar"][row["port"]]
    dist = np.abs(s["probs"] - p)
    idx = np.where(dist <= window)[0]
    if len(idx) < k_min:
        idx = np.argsort(dist)[:k_min]
    w = s["waits"][idx]
    return dict(k=int(len(idx)), median_wait_h=float(np.median(w)), p25=float(np.percentile(w, 25)),
                p75=float(np.percentile(w, 75)), p90=float(np.percentile(w, 90)),
                share_delayed=float((w > C.DELAY_THRESHOLD_H).mean()))


# ---------------------------------------------------------------------------
# Recommended action
# ---------------------------------------------------------------------------
ACTIONS_BY_DRIVER = {
    "Traffic & queue": "Too many ships for the berths. Slow-steam (reduce speed) to arrive nearer the "
                       "likely berthing time - same berth time, less fuel and anchorage cost. "
                       "If the cargo allows, compare with a nearby alternative port.",
    "Weather": "Delay is weather-driven and usually clears 1-2 days after the storm. Plan for a short "
               "slip rather than re-routing; re-check the forecast in 24 h.",
    "Labour": "Delay is strike-driven. Track union/port notices and line up a backup port or "
              "inland transport plan in case the strike is extended.",
    "Berth capacity": "A berth is out for maintenance. Ask the terminal for a berth window after it reopens.",
    "Vessel size": "Deep-water berths are the bottleneck for this ship. Request a berth window early.",
    "Timing & season": "Timing is the issue (peak season / night arrival). Adjusting arrival time can avoid the wait.",
    "Recent port backlog": "The port has been running behind all week. Expect knock-on delays and pad the ETA.",
}


def recommend(p, effects, similar):
    level = risk_level(p)
    if level == "Low":
        return level, "No action needed - proceed as planned and re-check 24 h before arrival."
    drivers = [e for e in effects if e["effect"] >= 0.10][:2] or \
              [e for e in effects if e["effect"] > 0.03][:1]
    tips = " ".join(ACTIONS_BY_DRIVER[e["factor"]] for e in drivers)
    if level == "Elevated":
        return level, (f"Monitor. Tell the customer a delay is possible - past ships at this port with "
                       f"similar risk waited up to ~{similar['p75']:.0f} h (1 in 4 cases). "
                       f"Re-check 24 h before arrival. {tips}")
    return level, (f"Act now. Move the customer's delivery date back - past ships at this port with "
                   f"similar risk waited ~{similar['median_wait_h']:.0f} h (typical range "
                   f"{similar['p25']:.0f}-{similar['p75']:.0f} h). Re-book trucks/rail for the later "
                   f"slot. {tips}")


# ---------------------------------------------------------------------------
# One call that does everything
# ---------------------------------------------------------------------------
class PortDelayPredictor:
    def __init__(self, bundle_path=C.MODEL_PATH):
        self.bundle = joblib.load(bundle_path)

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        X = add_features(raw)
        p = self.bundle["model"].predict_proba(X)[:, 1]
        conf, labels, spread, notes = confidence_scores(self.bundle, raw)
        return pd.DataFrame(dict(probability=p, risk=[risk_level(x) for x in p],
                                 alert=p >= ALERT_THRESHOLD, confidence=conf,
                                 confidence_label=labels, model_spread=spread,
                                 confidence_notes=notes), index=raw.index)

    def explain(self, row: pd.Series) -> dict:
        """Full answer for one port call."""
        p, effects = what_if_effects(self.bundle, row)
        conf, label, spread, notes = confidence_scores(self.bundle, pd.DataFrame([row]))
        sim = similar_cases(self.bundle, row, p)
        level, action = recommend(p, effects, sim)
        for e in effects:
            e["description"] = describe_factor(e["factor"], row, self.bundle)
        return dict(probability=p, risk=level, confidence=float(conf[0]),
                    confidence_label=str(label[0]), confidence_notes=notes[0],
                    factors=effects, similar=sim, action=action)

    @staticmethod
    def to_text(r: dict) -> str:
        lines = [f"Delay risk: {r['probability']:.0%}  ({r['risk']})   "
                 f"Confidence: {r['confidence_label']} ({r['confidence'] * 25:.0f} of 25 model versions agree)"]
        if r["confidence_notes"]:
            lines.append("  Note: " + "; ".join(r["confidence_notes"]))
        lines.append("Why:")
        shown = [e for e in r["factors"] if abs(e["effect"]) >= 0.03][:4]
        for e in shown:
            arrow = "raises" if e["effect"] > 0 else "lowers"
            lines.append(f"  - {e['factor']}: {e['description']} -> {arrow} risk by "
                         f"{abs(e['effect']) * 100:.0f} pts (if normal: {e['prob_if_normal']:.0%})")
        if not shown:
            lines.append("  - No single factor stands out; conditions are close to normal for this port.")
        s = r["similar"]
        lines.append(f"Past ships at this port with similar risk ({s['k']} cases): median wait "
                     f"{s['median_wait_h']:.0f} h, {s['share_delayed']:.0%} waited >24 h")
        lines.append(f"Recommended action: {r['action']}")
        return "\n".join(lines)
