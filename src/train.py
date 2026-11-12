"""
train.py - Trains, compares and evaluates the models, then saves everything.

Steps (each one prints what it found):
  1. Split by time: development (Nov 2024-Apr 2026) / test (May-Oct 2026)
  2. Compare three approaches with WALK-FORWARD validation (3 folds, 3 seasons):
        a) a simple rule a port planner might use (the baseline to beat)
        b) logistic regression (simple, transparent)
        c) gradient boosting (captures interactions like storm x busy port)
  3. Pick the best, retrain it on the whole development period, evaluate ONCE on TEST
  4. Build the confidence score (25 bootstrap models) and check it is meaningful
  5. Which factors matter most overall? (permutation importance)
  6. Per-port results, learning curve, and error analysis (why did we miss?)
  7. Save the model bundle, metrics.json and figures

Run:  python -m src.train
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.features import (FACTOR_GROUPS, MODEL_FEATURES, add_features, normal_day_reference,
                          time_split, walk_forward_folds)
from src.predictor import (ALERT_THRESHOLD, FILLABLE, confidence_scores)

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(C.RANDOM_SEED)
FIG = f"{C.OUTPUT_DIR}/figures"

# chart styling (validated palette; recessive axes and grid)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def candidate_models():
    logreg = make_pipeline(SimpleImputer(strategy="median"),
                           StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
    gboost = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250,
                                            min_samples_leaf=20, l2_regularization=1.0,
                                            random_state=C.RANDOM_SEED)
    return {"Logistic regression": logreg, "Gradient boosting": gboost}


class RuleBaseline:
    """What a port planner might do without ML:
    flag a delay if there is more than 1 ship per berth (waiting + arriving),
    or a strike is announced, or a storm is forecast while the port is busy."""
    def fit(self, X, y):
        self.load_median = X["load_per_berth"].median()
        return self

    def flags(self, X):
        load = X["load_per_berth"].fillna(self.load_median)
        storm = X["storm_warning"].fillna(0)
        return ((load > 1.0) | (X["strike_notice"] == 1) | ((storm == 1) & (load > 0.5))).astype(int)

    def score(self, X):   # a ranking score so we can also compute ROC-AUC for the rule
        load = X["load_per_berth"].fillna(self.load_median)
        return load + 2 * X["strike_notice"] + 0.5 * X["storm_warning"].fillna(0)


def metrics(y, prob=None, flags=None, threshold=ALERT_THRESHOLD, score=None):
    """All evaluation numbers for one model on one dataset."""
    y = np.asarray(y)
    if flags is None:
        flags = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, flags, labels=[0, 1]).ravel()
    rank = prob if prob is not None else score
    out = dict(
        n=int(len(y)), delay_rate=float(y.mean()),
        roc_auc=float(roc_auc_score(y, rank)), pr_auc=float(average_precision_score(y, rank)),
        precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
        f1=float(2 * tp / max(2 * tp + fp + fn, 1)), accuracy=float((tp + tn) / len(y)),
        alerts_pct=float(flags.mean()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        cost_per_100=float((C.COST_FALSE_NEGATIVE * fn + C.COST_FALSE_POSITIVE * fp) / len(y) * 100),
    )
    if prob is not None:
        out["brier"] = float(brier_score_loss(y, prob))
        out["log_loss"] = float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)))
        out["calibration_error"] = expected_calibration_error(y, prob)
    return out


def expected_calibration_error(y, p, bins=10):
    """Average gap between predicted probability and what actually happened."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def best_cost_threshold(y, p):
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    costs = [(C.COST_FALSE_NEGATIVE * ((p < t) & (y == 1)).sum()
              + C.COST_FALSE_POSITIVE * ((p >= t) & (y == 0)).sum()) / len(y) * 100 for t in grid]
    return float(grid[int(np.argmin(costs))]), float(min(costs)), grid, costs


# ---------------------------------------------------------------------------
def main():
    import os
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(os.path.dirname(C.MODEL_PATH), exist_ok=True)
    df = pd.read_csv(C.DATA_PATH, parse_dates=["arrival_time", "prediction_time"])
    dev, test = time_split(df)
    print(f"Development {len(dev)} port calls (Nov 2024-Apr 2026) | "
          f"Test {len(test)} port calls (May-Oct 2026)")
    Xte, yte = add_features(test), test["delayed"].values
    results = {"data": dict(rows=len(df), delay_rate=float(df.delayed.mean()),
                            development=len(dev), test=len(test),
                            test_delay_rate=float(yte.mean()),
                            missing_pct={c: float(df[c].isna().mean()) for c in C.MISSING_RATE},
                            delay_rate_by_port=df.groupby("port").delayed.mean().round(3).to_dict())}

    # ---- 2. walk-forward validation: choose the model ---------------------
    print("\n[Walk-forward validation - used to choose the model]")
    rows, pooled = [], {n: ([], []) for n in candidate_models()}
    for fold, tr, va in walk_forward_folds(dev):
        Xtr, Xva = add_features(tr), add_features(va)
        ytr, yva = tr["delayed"].values, va["delayed"].values
        rule = RuleBaseline().fit(Xtr, ytr)
        rows.append(dict(fold=fold, model="Rule baseline", n_train=len(tr), n_valid=len(va),
                         **metrics(yva, flags=rule.flags(Xva).values, score=rule.score(Xva))))
        for name, m in candidate_models().items():
            p = clone(m).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
            rows.append(dict(fold=fold, model=name, n_train=len(tr), n_valid=len(va),
                             **metrics(yva, prob=p)))
            pooled[name][0].append(yva)
            pooled[name][1].append(p)
    folds = pd.DataFrame(rows)
    for fold, g in folds.groupby("fold", sort=False):
        print(f"  {fold}  (train {g.n_train.iloc[0]}, validate {g.n_valid.iloc[0]}, "
              f"delay rate {g.delay_rate.iloc[0]:.0%})")
        for _, r in g.iterrows():
            print(f"     {r.model:20s} ROC-AUC {r.roc_auc:.3f}  Brier {r.brier:.3f}  "
                  f"recall {r.recall:.2f}  cost/100 {r.cost_per_100:.1f}")
    cols = ["roc_auc", "pr_auc", "brier", "recall", "precision", "cost_per_100"]
    summary = folds.groupby("model")[cols].mean()
    print("  Average over folds:")
    print(summary.round(3).to_string())
    # choose by Brier score (quality of the probability itself);
    # prefer the simpler model unless the complex one is clearly better
    lr_b, gb_b = summary.loc["Logistic regression", "brier"], summary.loc["Gradient boosting", "brier"]
    chosen = "Gradient boosting" if gb_b < lr_b - 0.005 else "Logistic regression"
    print(f"  -> chosen: {chosen}")

    # threshold check: is the cost-optimal threshold close to the 25% from theory?
    y_oof, p_oof = np.concatenate(pooled[chosen][0]), np.concatenate(pooled[chosen][1])
    t_best, c_best, grid, costs = best_cost_threshold(y_oof, p_oof)
    c_theory = costs[int(np.argmin(np.abs(grid - ALERT_THRESHOLD)))]
    print(f"  cost-optimal threshold across folds = {t_best:.2f} (cost {c_best:.1f}/100) | "
          f"theory {ALERT_THRESHOLD:.2f} (cost {c_theory:.1f}/100)")
    results["walk_forward"] = dict(per_fold=folds.drop(columns=["tp", "fp", "fn", "tn"]).to_dict("records"),
                                   average=summary.reset_index().to_dict("records"))
    results["chosen_model"] = chosen
    results["threshold"] = dict(used=ALERT_THRESHOLD, empirical_best_walk_forward=t_best,
                                cost_at_best=c_best, cost_at_used=float(c_theory))

    # ---- 3. final fit on the whole development period, single test --------
    trval = dev
    Xtv, ytv = add_features(trval), trval["delayed"].values
    finals = {n: clone(m).fit(Xtv, ytv) for n, m in candidate_models().items()}
    rule_f = RuleBaseline().fit(Xtv, ytv)
    test_res = {"Rule baseline": metrics(yte, flags=rule_f.flags(Xte).values, score=rule_f.score(Xte))}
    probs_te = {}
    for n, m in finals.items():
        probs_te[n] = m.predict_proba(Xte)[:, 1]
        test_res[n] = metrics(yte, prob=probs_te[n])
    test_res["Always alert"] = metrics(yte, flags=np.ones(len(yte), int), score=np.ones(len(yte)))
    test_res["Never alert"] = metrics(yte, flags=np.zeros(len(yte), int), score=np.zeros(len(yte)))
    print("\n[Test - May-Oct 2026 (monsoon + peak season), never seen before]")
    for name, r in test_res.items():
        print(f"  {name:20s} ROC-AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  "
              f"Brier {r.get('brier', float('nan')):.3f}  recall {r['recall']:.2f}  "
              f"precision {r['precision']:.2f}  F1 {r['f1']:.2f}  cost/100 {r['cost_per_100']:.1f}")
    results["test"] = test_res
    model = finals[chosen]
    p_te = probs_te[chosen]

    # ---- 4. confidence score ----------------------------------------------
    boot = []
    for _ in range(C.N_BOOTSTRAP_MODELS):
        idx = RNG.integers(0, len(Xtv), len(Xtv))
        boot.append(clone(candidate_models()[chosen]).fit(Xtv.iloc[idx], ytv[idx]))
    bundle = dict(model=model, model_name=chosen, bootstrap_models=boot,
                  features=MODEL_FEATURES, normal_day=normal_day_reference(trval),
                  fill_values={p: {c: g[c].dropna().to_numpy() for c in FILLABLE}
                               for p, g in trval.groupby("port")},
                  fill_storm_hours={p: g.dropna(subset=["forecast_max_wind_kmh"])
                                    .assign(w=lambda d: d.forecast_max_wind_kmh.round())
                                    .groupby("w").forecast_storm_hours.median().to_dict()
                                    for p, g in trval.groupby("port")},
                  load_p99=float(Xtv["load_per_berth"].quantile(0.99)),
                  waiting_max=float(Xtv["ships_waiting_now"].max()),
                  threshold=ALERT_THRESHOLD)
    conf, labels, spread, notes = confidence_scores(bundle, test.reset_index(drop=True))
    conf_tab = []
    for lab in ["High", "Medium", "Low"]:
        m = labels == lab
        if m.sum():
            conf_tab.append(dict(confidence=lab, n=int(m.sum()),
                                 accuracy=float(((p_te[m] >= ALERT_THRESHOLD) == yte[m]).mean()),
                                 brier=float(brier_score_loss(yte[m], p_te[m])),
                                 mean_abs_error=float(np.abs(p_te[m] - yte[m]).mean())))
    print("\n[Confidence check - are high-confidence predictions really more reliable?]")
    for r in conf_tab:
        print(f"  {r['confidence']:6s} n={r['n']:3d}  accuracy {r['accuracy']:.2f}  Brier {r['brier']:.3f}")
    results["confidence_check"] = conf_tab

    # ---- similar-risk past cases (per port): "ships we rated like this one,
    #      how long did they actually wait?" -> gives a realistic hours estimate
    p_dev = model.predict_proba(Xtv)[:, 1]
    similar = {port: dict(probs=p_dev[(trval["port"] == port).values],
                          waits=trval.loc[trval["port"] == port, "wait_hours"].values)
               for port in C.PORTS}
    bundle["similar"] = similar
    joblib.dump(bundle, C.MODEL_PATH)
    print(f"\nSaved model bundle -> {C.MODEL_PATH}")

    # ---- 5. global importance: shuffle one factor group, see how much AUC drops
    base_auc = roc_auc_score(yte, p_te)
    imp_rows = []
    test_r = test.reset_index(drop=True)
    for group, cols in FACTOR_GROUPS.items():
        drops = []
        for _ in range(30):
            sh = test_r.copy()
            perm = RNG.permutation(len(sh))
            for c in cols:
                sh[c] = test_r[c].values[perm]
            drops.append(base_auc - roc_auc_score(yte, model.predict_proba(add_features(sh))[:, 1]))
        imp_rows.append(dict(factor=group, auc_drop=float(np.mean(drops)), sd=float(np.std(drops))))
    sh = test_r.copy()   # port identity
    drops = []
    for _ in range(30):
        sh["port"] = test_r["port"].values[RNG.permutation(len(sh))]
        sh["berths_total"] = sh["port"].map(lambda p: C.PORTS[p]["berths"])
        drops.append(base_auc - roc_auc_score(yte, model.predict_proba(add_features(sh))[:, 1]))
    imp_rows.append(dict(factor="Port identity", auc_drop=float(np.mean(drops)), sd=float(np.std(drops))))
    imp = pd.DataFrame(imp_rows).sort_values("auc_drop", ascending=False)
    print("\n[Which factors matter overall? AUC drop when shuffled]")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    results["importance"] = imp.to_dict("records")

    # ---- 6a. per-port results ---------------------------------------------
    per_port = []
    for port in C.PORTS:
        m = (test_r["port"] == port).values
        if m.sum() and len(set(yte[m])) > 1:
            r = metrics(yte[m], prob=p_te[m])
            per_port.append(dict(port=port, n=r["n"], delay_rate=r["delay_rate"], roc_auc=r["roc_auc"],
                                 recall=r["recall"], precision=r["precision"], brier=r["brier"]))
    print("\n[Per-port test results]")
    print(pd.DataFrame(per_port).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    results["per_port"] = per_port

    # ---- 6b. learning curve: how much data do we need? ---------------------
    lc = []
    for n in [150, 300, 500, 800, len(Xtv)]:
        aucs, briers = [], []
        for _ in range(5 if n < len(Xtv) else 1):
            idx = RNG.choice(len(Xtv), n, replace=False) if n < len(Xtv) else np.arange(n)
            if len(set(ytv[idx])) < 2:
                continue
            mm = clone(candidate_models()[chosen]).fit(Xtv.iloc[idx], ytv[idx])
            pp = mm.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, pp))
            briers.append(brier_score_loss(yte, pp))
        lc.append(dict(n=int(n), roc_auc=float(np.mean(aucs)), brier=float(np.mean(briers))))
    print("\n[Learning curve]", [(r["n"], round(r["roc_auc"], 3)) for r in lc])
    results["learning_curve"] = lc

    # ---- 6c. error analysis: why were delays missed? -----------------------
    flags = p_te >= ALERT_THRESHOLD
    missed = test_r[(~flags) & (yte == 1)].copy()
    false_alarm = test_r[flags & (yte == 0)].copy()

    def miss_reason(r):
        if r.strike_hours_while_waiting > 0 and r.strike_notice == 0:
            return "sudden strike (no advance notice)"
        if r.storm_hours_while_waiting > 0 and not (r.forecast_max_wind_kmh >= C.STORM_CLOSURE_WIND_KMH):
            return "storm not in forecast / forecast missing"
        if pd.isna(r.expected_arrivals_48h):
            return "incoming-traffic data missing"
        if r.wait_hours <= 36:
            return "borderline: waited 24-36 h, just over the line"
        return "queue built up faster than expected (ETA noise, bunching)"
    missed["reason"] = missed.apply(miss_reason, axis=1)
    reasons = missed["reason"].value_counts().to_dict()
    print(f"\n[Error analysis] missed delays: {len(missed)}  false alarms: {len(false_alarm)}")
    for k, v in reasons.items():
        print(f"  {v:3d}  {k}")
    results["error_analysis"] = dict(missed=len(missed), false_alarms=len(false_alarm),
                                     missed_by_reason=reasons,
                                     missed_median_wait_h=float(missed.wait_hours.median()) if len(missed) else None)

    # ---- 7. figures ---------------------------------------------------------
    make_figures(df, yte, probs_te, test_res, imp, conf_tab, lc, grid, costs, chosen)
    with open(f"{C.OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    test_out = test_r[["call_id", "port", "arrival_time", "wait_hours", "delayed"]].copy()
    test_out["probability"] = p_te.round(3)
    test_out["confidence"] = labels
    test_out.to_csv(f"{C.OUTPUT_DIR}/test_predictions.csv", index=False)
    print(f"Saved metrics -> {C.OUTPUT_DIR}/metrics.json, figures -> {FIG}/")


def make_figures(df, yte, probs_te, test_res, imp, conf_tab, lc, grid, costs, chosen):
    # (1) what drives delays in the data
    X = add_features(df)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    bins = [0, 0.5, 1.0, 1.5, 2.0, 10]
    lab = ["<0.5", "0.5-1", "1-1.5", "1.5-2", ">2"]
    g = df.assign(b=pd.cut(X["load_per_berth"], bins, labels=lab)).groupby("b", observed=True).delayed.mean()
    ax[0].bar(lab, g.values * 100, color=BLUE, width=0.6)
    ax[0].set_title("Delay rate vs ships per berth")
    ax[0].set_xlabel("(ships waiting + arriving in 48 h) / available berths")
    ax[0].set_ylabel("% of ships delayed >24 h")
    for i, v in enumerate(g.values):
        ax[0].text(i, v * 100 + 2, f"{v:.0%}", ha="center", color=INK2, fontsize=8)
    ax[0].set_ylim(0, 110)
    m = df.groupby(df["arrival_time"].dt.month).delayed.mean()
    ax[1].bar(m.index, m.values * 100, color=[ORANGE if i in C.PEAK_SEASON_MONTHS else BLUE for i in m.index],
              width=0.6)
    ax[1].set_title("Delay rate by month (orange = peak season)")
    ax[1].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax[1].set_ylabel("% delayed")
    fig.tight_layout()
    fig.savefig(f"{FIG}/1_delay_drivers.png", dpi=200)
    plt.close(fig)

    # (2) model comparison on test
    names = ["Rule baseline", "Logistic regression", "Gradient boosting"]
    cols = [GREY, BLUE, ORANGE]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for a, (key, title) in zip(ax, [("roc_auc", "ROC-AUC (higher = better)"),
                                    ("recall", "Delays caught (recall)"),
                                    ("cost_per_100", "Cost per 100 ships (lower = better)".replace(" = ", "="))]):
        vals = [test_res[n][key] for n in names]
        a.bar(range(3), vals, color=cols, width=0.6)
        a.set_xticks(range(3), ["Rule", "LogReg", "GBoost"])
        a.set_title(title, fontsize=8.5)
        for i, v in enumerate(vals):
            a.text(i, v * 1.02, f"{v:.2f}" if key != "cost_per_100" else f"{v:.0f}",
                   ha="center", color=INK2, fontsize=8)
        a.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(f"{FIG}/2_model_comparison.png", dpi=200)
    plt.close(fig)

    # (3) calibration + cost curve
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for n, c in [("Logistic regression", BLUE), ("Gradient boosting", ORANGE)]:
        fr, mp = calibration_curve(yte, probs_te[n], n_bins=8, strategy="quantile")
        ax[0].plot(mp, fr, "-o", color=c, lw=2, ms=4, label=n)
    ax[0].plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="perfect")
    ax[0].set_title("Calibration on test: predicted vs actual")
    ax[0].set_xlabel("predicted probability of delay")
    ax[0].set_ylabel("actual share delayed")
    ax[0].legend(fontsize=7, loc="upper left")
    ax[1].plot(grid, costs, color=BLUE, lw=2)
    ax[1].axvline(ALERT_THRESHOLD, color=ORANGE, lw=1.5, ls="--")
    ax[1].text(ALERT_THRESHOLD + 0.02, max(costs) * 0.92, "25% alert line\n(3:1 cost ratio)", color=INK2, fontsize=8)
    ax[1].set_title("Cost vs alert threshold (walk-forward folds)", fontsize=9)
    ax[1].set_xlabel("alert if probability >= threshold")
    ax[1].set_ylabel("cost per 100 ships")
    fig.tight_layout()
    fig.savefig(f"{FIG}/3_calibration_threshold.png", dpi=200)
    plt.close(fig)

    # (4) importance + confidence check
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1]))
    imp_s = imp.sort_values("auc_drop")
    ax[0].barh(imp_s["factor"], imp_s["auc_drop"], color=BLUE, height=0.6)
    ax[0].set_title("What matters overall (AUC drop if shuffled)", fontsize=9)
    ax[0].tick_params(axis="y", labelsize=8)
    ct = pd.DataFrame(conf_tab)
    ax[1].bar(ct["confidence"], ct["accuracy"] * 100, color=BLUE, width=0.55)
    for i, r in ct.iterrows():
        ax[1].text(i, r["accuracy"] * 100 + 2, f"{r['accuracy']:.0%}\n(n={r['n']})", ha="center",
                   color=INK2, fontsize=7.5)
    ax[1].set_ylim(0, 118)
    ax[1].set_title("Accuracy by confidence level (test)", fontsize=9)
    ax[1].set_ylabel("% correct alert decisions")
    fig.tight_layout()
    fig.savefig(f"{FIG}/4_importance_confidence.png", dpi=200)
    plt.close(fig)

    # (5) learning curve
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    l = pd.DataFrame(lc)
    ax.plot(l["n"], l["roc_auc"], "-o", color=BLUE, lw=2, ms=4)
    ax.set_title("How much data is needed?", fontsize=9)
    ax.set_xlabel("training rows")
    ax.set_ylabel("test ROC-AUC")
    fig.tight_layout()
    fig.savefig(f"{FIG}/5_learning_curve.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
