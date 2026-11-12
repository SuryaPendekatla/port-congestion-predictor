"""
build_report.py - Builds report/Report.pdf from outputs/metrics.json and the figures,
so every number in the report matches what the code actually produced.

Run (after src.train):  python -m report.build_report
"""
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from src import config as C
from src.predictor import PortDelayPredictor

def _register_fonts():
    """Liberation Sans on Linux; otherwise the DejaVu Sans that ships with matplotlib
    (works on Windows/Mac too). Both contain the arrows and maths symbols used below."""
    import os
    import matplotlib
    lib = "/usr/share/fonts/truetype/liberation/LiberationSans-"
    mpl = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans")
    for regular, bold, italic in [(lib + "Regular.ttf", lib + "Bold.ttf", lib + "Italic.ttf"),
                                  (mpl + ".ttf", mpl + "-Bold.ttf", mpl + "-Oblique.ttf")]:
        if all(os.path.exists(f) for f in (regular, bold, italic)):
            pdfmetrics.registerFont(TTFont("Body", regular))
            pdfmetrics.registerFont(TTFont("Body-Bold", bold))
            pdfmetrics.registerFont(TTFont("Body-Italic", italic))
            pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")
            return "Body", "Body-Bold"
    return "Helvetica", "Helvetica-Bold"


BODY, BOLD = _register_fonts()

INK, INK2, BLUE, RULE, TINT = (colors.HexColor(h) for h in
                               ["#0b0b0b", "#52514e", "#1c5cab", "#d9d8d3", "#eef4fc"])
base = ParagraphStyle("b", fontName=BODY, fontSize=8.9, leading=11.4, textColor=INK, alignment=TA_LEFT,
                      spaceAfter=3.5)
S = dict(
    title=ParagraphStyle("t", parent=base, fontName=BOLD, fontSize=16, leading=19, spaceAfter=2),
    sub=ParagraphStyle("s", parent=base, fontSize=8.8, textColor=INK2, spaceAfter=7),
    h=ParagraphStyle("h", parent=base, fontName=BOLD, fontSize=10.6, leading=13, textColor=BLUE,
                     spaceBefore=6, spaceAfter=2.5),
    body=base,
    bullet=ParagraphStyle("bl", parent=base, leftIndent=10, bulletIndent=2, spaceAfter=1.8),
    cap=ParagraphStyle("c", parent=base, fontSize=7.8, leading=9.6, textColor=INK2, spaceAfter=5),
    cell=ParagraphStyle("cl", parent=base, fontSize=7.9, leading=9.6, spaceAfter=0),
    cellb=ParagraphStyle("cb", parent=base, fontName=BOLD, fontSize=7.9, leading=9.6, spaceAfter=0),
    mono=ParagraphStyle("m", parent=base, fontSize=7.9, leading=10, spaceAfter=0),
)


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, bold_row=None, size_style="cell"):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) or i == bold_row else S[size_style])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK2),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if bold_row is not None:
        style.append(("BACKGROUND", (0, bold_row), (-1, bold_row), TINT))
    t.setStyle(TableStyle(style))
    return t


def fig(path, width_cm, caption):
    img = Image(path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    return KeepTogether([img, P(caption, "cap")])


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def main(repo_url, demo_url=None):
    m = json.load(open(f"{C.OUTPUT_DIR}/metrics.json"))
    d, te = m["data"], m["test"]
    lr, gb, rule = te["Logistic regression"], te["Gradient boosting"], te["Rule baseline"]
    wf = {r["model"]: r for r in m["walk_forward"]["average"]}
    conf = {r["confidence"]: r for r in m["confidence_check"]}
    nconf = sum(r["n"] for r in m["confidence_check"])
    th = m["threshold"]
    err = m["error_analysis"]
    lc = {r["n"]: r["roc_auc"] for r in m["learning_curve"]}
    pp = {r["port"]: r for r in m["per_port"]}
    imp = {r["factor"]: r["auc_drop"] for r in m["importance"]}
    cost_cut = 1 - lr["cost_per_100"] / rule["cost_per_100"]

    # live example from the trained system (storm + traffic case in the test period)
    df = pd.read_csv(C.DATA_PATH, parse_dates=["arrival_time", "prediction_time"])
    ex_row = df[df.call_id == "PC01616"].iloc[0]
    storm_hit_rate = df.loc[df.storm_hours_while_waiting > 0, "delayed"].mean()
    borderline = err["missed_by_reason"].get("borderline: waited 24-36 h, just over the line", 0)
    ex = PortDelayPredictor().explain(ex_row)
    shown = [e for e in ex["factors"] if abs(e["effect"]) >= 0.03][:3]

    doc = SimpleDocTemplate("report/Report.pdf", pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.3 * cm,
                            title="Predicting Port Congestion and Shipment Delays",
                            author="Surya Pendekatla")
    W = A4[0] - 3.4 * cm
    st = []
    st += [P("Predicting Port Congestion and Shipment Delays", "title"),
           P("Technical report  ·  Surya Pendekatla  ·  Aiqon Jr AI Engineer take-home  ·  November 2026  ·  "
             f"Code: {repo_url}" + (f"  ·  Live demo: {demo_url}" if demo_url else ""), "sub")]

    summary = (f"<b>Summary.</b> The system predicts, <b>48 hours before a ship arrives</b>, whether it will wait "
               f"<b>more than 24 hours</b> at anchorage for a berth, and returns a probability, a confidence level, "
               f"the factors behind it in plain English, and a recommended action. On a held-out test period "
               f"covering monsoon and peak season (May–Oct 2026, {d['test']} port calls), it catches "
               f"<b>{pct(lr['recall'])} of delays at {pct(lr['precision'])} precision</b> (ROC-AUC "
               f"{lr['roc_auc']:.2f}) and cuts the cost of wrong calls by <b>{pct(cost_cut)}</b> versus a "
               f"planner's rule of thumb. The data is <b>synthetic</b>, produced by an hour-by-hour port "
               f"simulation; Section 7 sets out what that does and does not prove.")
    box = Table([[P(summary)]], colWidths=[W])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 5),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    st += [box, Spacer(1, 3)]

    # 1 ------------------------------------------------------------------
    st.append(P("1. Problem definition: what counts as congestion", "h"))
    st.append(P(
        "The unit of prediction is one <b>port call</b> (one ship visiting one port). A call is <b>delayed</b> "
        "if the ship waits <b>more than 24 h at anchorage</b> before getting a berth. Waiting time is used "
        "instead of total turnaround because turnaround also includes unloading time, which depends on cargo "
        "volume, not congestion. 24 h is the point where delivery dates, truck bookings and onward connections "
        "start to break. The prediction is made <b>48 h before arrival</b>: early enough to act (slow down, "
        "warn the customer, re-plan trucks), late enough that the port situation is reasonably known. "
        "Both numbers are single settings in <i>src/config.py</i>."))

    # 2 ------------------------------------------------------------------
    st.append(P("2. Data: how delays were simulated and how ground truth was defined", "h"))
    st.append(P(
        "If the label were written as a formula of the features (e.g. <i>delay = 3×queue + 2×storm</i>), "
        "any model would just rediscover that formula and prove nothing. Instead, <i>src/simulate.py</i> runs "
        "<b>an hour-by-hour queue simulation</b> of four fictional ports. Ships arrive, wait at anchorage "
        "(first come, first served) until a suitable berth is free and the port is open, are unloaded, and "
        "leave. <b>Waiting time comes out of this process; it is never typed in.</b> The drivers built into "
        "the simulation are:"))
    st += bullets([
        "<b>Arrivals:</b> random (Poisson) with seasonality (Aug–Oct peak, Feb lull), a weekday pattern, and "
        "slowly drifting <b>“waves”</b> that bunch ships together, as happens after upstream disruptions.",
        "<b>Capacity:</b> 2–4 berths per port; ships over 8,000 TEU need one of the few deep-water berths; "
        "planned maintenance removes a berth for 3–10 days; unloading time grows with ship size.",
        "<b>Weather:</b> hourly wind, more storms in monsoon (Jun–Sep) and cyclone season (Oct–Nov). The "
        "port stops above 45 km/h and cranes run at half speed between 35 and 45 km/h.",
        "<b>Labour:</b> strikes of 12–72 h; 70% announced in advance, <b>30% sudden</b> (unpredictable on purpose).",
        "<b>Port differences:</b> a busy hub (Northbay), a storm-exposed port (Eastgate), a small strike-prone "
        "port with <b>daylight-only berthing</b> (Southreach) and a quiet port (Westhaven).",
        f"<b>Messy data, injected deliberately:</b> weather forecast missing {pct(d['missing_pct']['forecast_max_wind_kmh'])}, "
        f"incoming-traffic (AIS) data missing {pct(d['missing_pct']['expected_arrivals_48h'])}, vessel size missing "
        f"{pct(d['missing_pct']['vessel_teu'])}; ship ETAs off by ~8 h; 2-day wind forecast off by ~7 km/h.",
    ])
    st.append(P(
        f"<b>Ground truth:</b> for each simulated call we store (a) only what was knowable 48 h before arrival "
        f"and (b) the simulated waiting time, with <i>delayed = wait &gt; 24 h</i>. Outcome columns are never "
        f"used as model inputs. Result: <b>{d['rows']:,} port calls over 24 months</b> (Nov 2024–Oct 2026), "
        f"{pct(d['delay_rate'])} delayed (by port: Northbay {pct(d['delay_rate_by_port']['Northbay'])}, "
        f"Eastgate {pct(d['delay_rate_by_port']['Eastgate'])}, Southreach "
        f"{pct(d['delay_rate_by_port']['Southreach'])}, Westhaven {pct(d['delay_rate_by_port']['Westhaven'])}). "
        f"The brief suggests 100–500 rows. More were generated because a queue needs realistic traffic "
        f"to produce congestion, and the learning curve shows 500 rows already reaches ROC-AUC "
        f"{lc[500]:.3f} versus {lc[max(lc)]:.3f} with all {max(lc):,}."))
    st.append(fig(f"{C.OUTPUT_DIR}/figures/1_delay_drivers.png", 13.0,
                  "Figure 1. Sanity check of the simulated ground truth: delay risk climbs steeply once more than "
                  "one ship competes for each berth, and is highest in peak season and cyclone season."))

    # 3 ------------------------------------------------------------------
    st.append(P("3. Features (all available 48 h before arrival)", "h"))
    st.append(table([
        ["Group", "Features", "Why it matters"],
        ["Traffic & queue", "ships waiting now, ships due in 48 h (from ETAs), <b>ships per available berth</b>, "
         "berth occupancy", "The core signal is the <i>interaction</i> of traffic and capacity"],
        ["Capacity", "berths available, maintenance flag", "Same traffic, fewer berths → queue"],
        ["Weather", "forecast max wind, <b>forecast hours of closure</b>, closure-days × ships per berth",
         "Duration matters more than peak; storms hurt more at a busy port"],
        ["Labour / vessel", "strike announced; ship size, needs deep-water berth", "Stoppages; competition for few deep berths"],
        ["Timing", "peak season, night arrival, <b>night arrival at a daylight-only port</b>, weekend",
         "Temporal patterns and port rules"],
        ["History / port", "average wait at this port in the last 7 days; port identity", "Momentum; port baseline"],
    ], [2.6 * cm, 8.4 * cm, W - 11.0 * cm]))

    # 4 ------------------------------------------------------------------
    st.append(P("4. Approach", "h"))
    st += bullets([
        "<b>Three approaches compared:</b> (a) a planner's rule: alert if more than 1 ship per berth, a strike "
        "is announced, or a storm is forecast at a busy port; (b) logistic regression; (c) gradient boosting.",
        "<b>Time-based evaluation, never a random split</b> (a random split would let the model learn from the "
        "future). Development: Nov 2024–Apr 2026. <b>Test: May–Oct 2026</b>, deliberately the hardest season "
        "(monsoon + peak), used once at the end. A ship enters training only once its outcome is known.",
        f"<b>Model choice by walk-forward validation:</b> 3 folds (Aug–Oct 2025, Nov–Jan, Feb–Apr), each trained "
        f"only on the past. Logistic regression won on average (ROC-AUC {wf['Logistic regression']['roc_auc']:.3f} "
        f"vs {wf['Gradient boosting']['roc_auc']:.3f} for boosting and {wf['Rule baseline']['roc_auc']:.3f} for "
        f"the rule; Brier {wf['Logistic regression']['brier']:.3f} vs {wf['Gradient boosting']['brier']:.3f}). "
        f"The key interactions are engineered as features, so a linear model can use them, and with ~1,500 rows "
        f"boosting overfits slightly. The simpler model is also easier to explain and maintain.",
        f"<b>Alert threshold = 25%, from decision theory rather than tuning.</b> If a missed delay costs 3× a "
        f"false alarm, alerting pays off when P×3 &gt; (1−P)×1, i.e. P &gt; 1/(1+3). The validation cost curve is "
        f"flat around it (best {th['empirical_best_walk_forward']:.2f}: {th['cost_at_best']:.1f} vs "
        f"{th['cost_at_used']:.1f} per 100 ships), so the principled value was kept to avoid overfitting.",
        "<b>Confidence = share of 25 model versions that reach the same alert decision.</b> Each version is "
        "trained on a different bootstrap resample and fills any missing input with a different plausible value "
        "from that port's history. One number captures three doubts: the model is unsure, the probability sits "
        "near the alert line, or key data is missing. Probability says <i>how likely</i> the delay is; "
        "confidence says <i>how stable</i> that answer is.",
        "<b>Explanations by “what if this factor were normal?”:</b> each factor group is reset to a typical "
        "day at that port and the model re-predicts. This works for any model, and every number can be said "
        "out loud and checked. An <b>hours estimate</b> comes from past ships at the same port with similar "
        "predicted risk and how long they actually waited.",
    ])

    # 5 ------------------------------------------------------------------
    st.append(P("5. Results", "h"))
    st.append(table([
        ["Approach (test: May–Oct 2026)", "ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "Brier",
         "Cost / 100 ships*"],
        ["Never alert", "0.50", "–", "0%", "–", "0.00", "–", f"{te['Never alert']['cost_per_100']:.0f}"],
        ["Always alert", "0.50", "–", "100%", pct(te['Always alert']['precision']),
         f"{te['Always alert']['f1']:.2f}", "–", f"{te['Always alert']['cost_per_100']:.0f}"],
        ["Planner's rule (baseline)", f"{rule['roc_auc']:.3f}", f"{rule['pr_auc']:.3f}", pct(rule['recall']),
         pct(rule['precision']), f"{rule['f1']:.2f}", "–", f"{rule['cost_per_100']:.1f}"],
        ["<b>Logistic regression (chosen)</b>", f"{lr['roc_auc']:.3f}", f"{lr['pr_auc']:.3f}", pct(lr['recall']),
         pct(lr['precision']), f"{lr['f1']:.2f}", f"{lr['brier']:.3f}", f"{lr['cost_per_100']:.1f}"],
        ["Gradient boosting", f"{gb['roc_auc']:.3f}", f"{gb['pr_auc']:.3f}", pct(gb['recall']),
         pct(gb['precision']), f"{gb['f1']:.2f}", f"{gb['brier']:.3f}", f"{gb['cost_per_100']:.1f}"],
    ], [4.6 * cm] + [(W - 4.6 * cm) / 7] * 7, bold_row=4))
    st.append(P(f"*Cost = 3 × missed delays + 1 × false alarms per 100 ships. Test delay rate "
                f"{pct(d['test_delay_rate'])}. Out of {lr['tp'] + lr['fn']} real delays, {lr['tp']} were caught; "
                f"{lr['fp']} false alarms among {lr['fp'] + lr['tn']} on-time ships.", "cap"))
    st.append(fig(f"{C.OUTPUT_DIR}/figures/3_calibration_threshold.png", 13.0,
                  "Figure 2. Left: calibration on the test period. Probabilities are reliable at the extremes, but "
                  "mid-range risk is <b>underestimated</b> in monsoon/peak season (seasonal drift, see Section 7). "
                  "Right: cost per 100 ships vs alert threshold across walk-forward folds; the 25% line sits in the "
                  "flat region."))
    st.append(fig(f"{C.OUTPUT_DIR}/figures/4_importance_confidence.png", 13.0,
                  "Figure 3. Left: global importance (drop in test ROC-AUC when a factor is shuffled). Right: the "
                  "confidence score separates reliable from unreliable predictions."))
    st += bullets([
        f"<b>The confidence score is meaningful:</b> High-confidence calls ({pct(conf['High']['n'] / nconf)} of "
        f"cases) are correct {pct(conf['High']['accuracy'])} of the time, Medium {pct(conf['Medium']['accuracy'])}, "
        f"Low {pct(conf['Low']['accuracy'])}. Low is close to a coin flip, so the system correctly flags it.",
        f"<b>Traffic per berth dominates</b> (AUC drop {imp['Traffic & queue']:.2f}), then berth capacity "
        f"({imp['Berth capacity']:.2f}). Weather and strikes rank low <i>on average</i> because they affect few "
        f"ships, but they are decisive when they occur: when a storm hit during the wait, {pct(storm_hit_rate)} "
        f"of ships were delayed. "
        f"Global importance and per-ship explanations answer different questions.",
        f"<b>Per port</b> ROC-AUC ranges from {min(r['roc_auc'] for r in pp.values()):.2f} (Northbay) to "
        f"{max(r['roc_auc'] for r in pp.values()):.2f} (Southreach). Northbay has the lowest recall "
        f"({pct(pp['Northbay']['recall'])}) because delays there are rare ({pct(pp['Northbay']['delay_rate'])}) "
        f"and come from short traffic waves.",
        f"<b>Error analysis</b> of the {err['missed']} missed delays: "
        + ", ".join(f"{v} {k.split(':')[0].split('(')[0].strip()}" for k, v in err["missed_by_reason"].items())
        + f". {pct(borderline / err['missed'])} were borderline (waited 24–36 h); median wait of missed "
        f"delays {err['missed_median_wait_h']:.0f} h.",
    ])

    # 6 ------------------------------------------------------------------
    st.append(P("6. From prediction to operational decision", "h"))
    st.append(P(
        "Risk is shown in three bands with a matching action: <b>Low</b> (&lt;25%): proceed; <b>Elevated</b> "
        "(25–60%): warn the customer and re-check at 24 h; <b>High</b> (≥60%): move the delivery date and "
        "re-book trucks/rail. The advice depends on the <i>main driver</i>. Traffic → slow-steam to arrive "
        "just in time (same berth time, less fuel and anchorage cost) or compare an alternative port; weather → "
        "expect a 1–2 day slip, don't re-route; strike → line up a backup port; maintenance → request a berth "
        "window after it reopens. A real test case, predicted 48 h ahead:"))
    lines = [f"<b>{ex_row.call_id} · {ex_row.port} · arrives {ex_row.arrival_time:%d %b %Y %H:%M}</b>",
             f"Delay risk <b>{ex['probability']:.0%} ({ex['risk']})</b> · Confidence <b>{ex['confidence_label']}</b> "
             f"({ex['confidence'] * 25:.0f}/25 model versions agree)"]
    for e in shown:
        lines.append(f"• <b>{e['factor']}</b>: {e['description']} → {'+' if e['effect'] > 0 else '−'}"
                     f"{abs(e['effect']) * 100:.0f} pts (risk if normal: {e['prob_if_normal']:.0%})")
    s = ex["similar"]
    lines.append(f"• Similar-risk past ships here waited a median of {s['median_wait_h']:.0f} h "
                 f"(typical {s['p25']:.0f}–{s['p75']:.0f} h)")
    lines.append(f"• <b>Action:</b> move delivery date and re-book transport; slow-steam to arrive near the "
                 f"likely berth time; treat the storm part as a short slip, not a reason to re-route.")
    lines.append(f"<i>What happened: the ship waited {ex_row.wait_hours:.0f} h.</i>")
    exbox = Table([[P("<br/>".join(lines), "mono")]], colWidths=[W])
    exbox.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, RULE), ("BACKGROUND", (0, 0), (-1, -1), TINT),
                               ("LEFTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 4),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    st += [exbox, Spacer(1, 2)]

    # 7 ------------------------------------------------------------------
    st.append(P("7. Assumptions and limitations", "h"))
    st += bullets([
        "<b>Synthetic data is the main limitation.</b> The model can only learn what the simulator contains; "
        "real ports add customs holds, yard congestion, crane breakdowns and priority berthing for liner "
        "contracts. The results show that the <i>method</i> works end to end, not how accurate it would be at a "
        "real port.",
        "<b>Seasonal drift:</b> on the monsoon/peak test period the model underestimates mid-range risk "
        "(Figure 2). In production: recalibrate monthly and track calibration by season and port.",
        "<b>One-factor-at-a-time explanations</b> can understate a factor when another already pushes risk "
        "near 100% (saturation). SHAP values would split credit more fairly; they were left out to keep "
        "dependencies minimal.",
        "<b>Assumed numbers:</b> the 3:1 cost ratio, the 24 h line and the 48 h horizon should come from the "
        "business. All are settings, and the alert threshold follows the cost ratio automatically.",
        "<b>Ports are simulated independently</b>; in reality ships divert between neighbouring ports. Ship "
        "ETAs and forecasts are noisy but unbiased here; real feeds can be systematically late.",
    ])

    # 8 ------------------------------------------------------------------
    st.append(P("8. Next steps with real data", "h"))
    st.append(P(
        "Using AIS data (e.g. MarineCadastre): detect port calls by finding ships that stop inside an anchorage "
        "polygon (speed &lt; 1 knot) and later enter a berth polygon. Waiting time = berth entry − anchorage "
        "entry, and queue length and arrivals-in-48h come from the same feed. Then add weather forecasts and port "
        "notices, and keep the same pipeline: features → walk-forward validation → calibrated probability → "
        "confidence → explanation → action. A per-port model or recalibration and a feedback loop (planners "
        "marking alerts as useful or not) would follow."))

    doc.build(st, onFirstPage=footer, onLaterPages=footer)
    print("Saved report/Report.pdf")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY, 7.5)
    canvas.setFillColor(INK2)
    canvas.drawString(1.7 * cm, 0.75 * cm, "Port congestion & delay prediction · Surya Pendekatla")
    canvas.drawRightString(A4[0] - 1.7 * cm, 0.75 * cm, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="github.com/<your-username>/port-congestion-predictor")
    ap.add_argument("--demo-url", default=None, help="link to the live Streamlit app, if deployed")
    a = ap.parse_args()
    main(a.repo_url, a.demo_url)
