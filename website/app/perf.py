"""Cross-question comparison numbers for the "Baseline vs model" page.

Everything is derived from results/metrics.csv and the per-case parquet files written by scripts/run_pipeline.py.
Errors are oriented so that lower is better for every question (destination uses the top-1 miss rate), which lets the
four questions share one "x clean feed" scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from common import metrics, table

STRESS = {"stale": "Stale reports", "low_quality": "Unreliable source", "combined": "Stale + unreliable"}
SCHEDULE = [0.10, 0.20, 0.30, 0.45, 0.60]     # corruption share per test window, identical for all four questions

# fair = the metric that charges withheld answers (policy cost / penalised MAE / accuracy over all cases)
TASKS = {
    "position_15m": dict(short="Position in 15 min", color="#1565c0", model="hgb", fair="policy_cost_km",
                         raw="mean_km", unit="km", label="mean error (km)",
                         baselines={"cv_baseline": "constant velocity"}),
    "destination": dict(short="Destination zone", color="#8e24aa", model="hgb_ranker", fair="top1_over_all_cases",
                        raw="top1_acc", unit="", label="top-1 miss rate", accuracy=True,
                        baselines={"heading_nearest_baseline": "heading-aligned nearest zone"}),
    "traffic_density": dict(short="Area traffic density", color="#2e7d32", model="hgb", fair="policy_mae", raw="mae",
                            unit="vessels", label="MAE (vessels per cell)",
                            baselines={"persistence": "persistence", "rolling_mean": "3-h rolling mean",
                                       "hist_mean": "training-period mean"}),
    "port_occupancy": dict(short="Port occupancy proxy", color="#ef6c00", model="hgb", fair="policy_mae", raw="mae",
                           unit="vessels", label="MAE (vessels per zone)",
                           baselines={"persistence": "persistence", "rolling_mean": "3-h rolling mean",
                                      "hist_mean": "training-period mean"}),
}


def primary_baseline(task: str) -> str:
    return next(iter(TASKS[task]["baselines"]))


@st.cache_data
def errors(task: str, subgroup: str = "all") -> pd.DataFrame:
    """One row per (predictor, mitigation, condition, window, seed) with `err` (lower is better) and coverage."""
    spec = TASKS[task]
    m = metrics(task)
    m = m[m.subgroup == subgroup].copy()
    if m.empty:
        return m
    v = m[spec["fair"]].astype(float)
    m["err"] = 1.0 - v if spec.get("accuracy") else v
    m["window"] = m.window.astype(str)
    m["mitigation"] = m.mitigation.fillna("none")
    return m[["predictor", "mitigation", "condition", "window", "seed", "err", "coverage"]]


def pick(E: pd.DataFrame, predictor: str, mitigation: str = "none", window: str | None = "all") -> pd.DataFrame:
    s = E[(E.predictor == predictor) & (E.mitigation == mitigation)]
    return s if window is None else s[s.window == window]


@st.cache_data
def relative_by_window(task: str) -> pd.DataFrame:
    """Model error under each stress condition divided by its clean-feed error in the same window and seed."""
    E = errors(task)
    if E.empty:
        return E
    s = pick(E, TASKS[task]["model"], window=None)
    s = s[s.window != "all"]
    nom = s[s.condition == "nominal"].set_index(["window", "seed"]).err
    out = s[s.condition.isin(STRESS)].copy()
    out["ratio"] = out.err.to_numpy() / nom.reindex(pd.MultiIndex.from_frame(out[["window", "seed"]])).to_numpy()
    out["share"] = np.asarray(SCHEDULE)[out.window.astype(int)]
    return out


@st.cache_data
def headline(task: str) -> dict:
    """All-window, seed-averaged numbers used by the comparison charts and the rubric table."""
    E, spec = errors(task), TASKS[task]
    if E.empty:
        return {}
    mod, bl = spec["model"], primary_baseline(task)
    g = lambda pred, mit, cond: pick(E, pred, mit)[lambda d: d.condition == cond]
    r = {}
    for c in ["nominal", *STRESS, "holdout_unseen_bias"]:
        off, on, b = g(mod, "none", c), g(mod, "quality_aware", c), g(bl, "none", c)
        if off.empty:
            continue
        r[c] = dict(off=off.err.mean(), off_sd=off.err.std(), on=on.err.mean(), coverage_on=on.coverage.mean(),
                    base=b.err.mean() if len(b) else np.nan)
    nom = r["nominal"]["off"]
    for c in r:
        r[c]["x_clean"] = r[c]["off"] / nom
        loss = r[c]["off"] - nom
        r[c]["recovered"] = (r[c]["off"] - r[c]["on"]) / loss if loss > 0 else np.nan
    # signal-to-noise of the poisoning effect across seeds (effect / seed std)
    for c in STRESS:
        per_seed = g(mod, "none", c).set_index("seed").err / g(mod, "none", "nominal").set_index("seed").err
        r[c]["snr"] = (per_seed.mean() - 1) / max(per_seed.std(), 1e-9)
    W = relative_by_window(task)
    for c in STRESS:
        w = W[W.condition == c].groupby("share").ratio.mean()
        r[c]["trend_r"] = float(np.corrcoef(w.index, w.values)[0, 1]) if len(w) > 2 else np.nan
    return r


# ------------------------------------------------------------------------------------------- per-case
AGE_BINS = [-1, 1, 5, 10, 15, 20, 30, 45, 1e9]
AGE_MID = [0, 3, 7.5, 12.5, 17.5, 25, 37.5, 50]


@st.cache_data
def staleness_curve() -> pd.DataFrame:
    """Error x clean against how old the input was, matched case by case to the same case on the clean feed."""
    rows = []
    P = table("cases_default.parquet")
    if P is not None:
        n = P[P.condition == "nominal"][["seed", "case_id", "err_un_km", "err_cv_km"]]
        s = P[P.condition == "stale"].merge(n, on=["seed", "case_id"], suffixes=("", "_nom"))
        s["b"] = pd.cut(s.age_s / 60, AGE_BINS, labels=AGE_MID)
        for b, d in s.groupby("b", observed=True):
            rows.append(dict(task="position_15m", age_min=float(b), n=len(d),
                             model=d.err_un_km.mean() / d.err_un_km_nom.mean(),
                             baseline=d.err_cv_km.mean() / d.err_cv_km_nom.mean()))
    D = table("destination_cases_seed0.parquet")
    if D is not None:
        D = D.assign(miss=D.un_pred1 != D.true_zone, miss_bl=D.bl_pred1 != D.true_zone)
        s = D[D.condition == "stale"].merge(D[D.condition == "nominal"][["seed", "case_id", "miss", "miss_bl"]],
                                            on=["seed", "case_id"], suffixes=("", "_nom"))
        s["b"] = pd.cut(s.age_s / 60, AGE_BINS, labels=AGE_MID)
        for b, d in s.groupby("b", observed=True):
            rows.append(dict(task="destination", age_min=float(b), n=len(d), model=d.miss.mean() / d.miss_nom.mean(),
                             baseline=d.miss_bl.mean() / d.miss_bl_nom.mean()))
    for task, f in [("traffic_density", "density"), ("port_occupancy", "port")]:
        C = table(f"{f}_cases_seed0.parquet")
        if C is None:
            continue
        k = ["seed", "unit", "o"]
        s = C[C.condition == "stale"].merge(C[C.condition == "nominal"][k + ["pred_un", "persistence"]], on=k,
                                            suffixes=("", "_nom"))
        for b, d in s.groupby("delay_buckets"):
            rows.append(dict(task=task, age_min=15.0 * b, n=len(d),
                             model=(d.pred_un - d.y).abs().mean() / (d.pred_un_nom - d.y).abs().mean(),
                             baseline=(d.persistence - d.y).abs().mean() / (d.persistence_nom - d.y).abs().mean()))
    return pd.DataFrame(rows)


def _defect_class(s: str) -> str:
    s = s or ""
    for key, lab in [("position_outlier", "gross position outlier (3-15 km)"), ("sog_sentinel", "SOG sentinel 102.3"),
                     ("sog_scaled", "SOG scaled x0.3-2.5"), ("cog_random", "random COG"),
                     ("missing", "SOG or COG missing")]:
        if key in s:
            return lab
    return "position jitter only (~200 m)"


@st.cache_data
def defect_breakdown() -> pd.DataFrame:
    """Unreliable-source cases by their worst injected defect: model and baseline error x the same case on clean."""
    rows = []
    P = table("cases_default.parquet")
    if P is not None:
        n = P[P.condition == "nominal"][["seed", "case_id", "err_un_km", "err_cv_km"]]
        s = P[(P.condition == "low_quality") & P.is_lowq].merge(n, on=["seed", "case_id"], suffixes=("", "_nom"))
        s["defect"] = s.defects.map(_defect_class)
        for k, d in s.groupby("defect"):
            rows.append(dict(task="position_15m", defect=k, n=len(d), model=d.err_un_km.mean() / d.err_un_km_nom.mean(),
                             baseline=d.err_cv_km.mean() / d.err_cv_km_nom.mean(),
                             countermeasure=d.err_mi_km[d.answered].mean() / d.err_un_km_nom[d.answered].mean()))
    D = table("destination_cases_seed0.parquet")
    if D is not None:
        D = D.assign(miss=D.un_pred1 != D.true_zone, miss_bl=D.bl_pred1 != D.true_zone,
                     miss_mi=D.mi_pred1 != D.true_zone)
        s = D[(D.condition == "low_quality") & D.is_lowq].merge(
            D[D.condition == "nominal"][["seed", "case_id", "miss", "miss_bl"]], on=["seed", "case_id"],
            suffixes=("", "_nom"))
        s["defect"] = s.defects.map(_defect_class)
        for k, d in s.groupby("defect"):
            a = d[d.mi_answered]
            rows.append(dict(task="destination", defect=k, n=len(d), model=d.miss.mean() / d.miss_nom.mean(),
                             baseline=d.miss_bl.mean() / d.miss_bl_nom.mean(),
                             countermeasure=a.miss_mi.mean() / a.miss_nom.mean() if len(a) else np.nan))
    return pd.DataFrame(rows)


GHOST_BINS = [-0.5, 0.5, 1.5, 3.5, 10.5, 30.5, 1e9]
GHOST_LAB = ["0", "1", "2-3", "4-10", "11-30", "31+"]


@st.cache_data
def ghost_curve() -> pd.DataFrame:
    """Counts: error x clean against how many extra IDs the unreliable source added to the unit's last hour."""
    rows = []
    for task, f in [("traffic_density", "density"), ("port_occupancy", "port")]:
        C = table(f"{f}_cases_seed0.parquet")
        if C is None:
            continue
        k = ["seed", "unit", "o"]
        s = C[C.condition == "low_quality"].merge(C[C.condition == "nominal"][k + ["pred_un", "h0_delivered"]], on=k,
                                                  suffixes=("", "_nom"))
        s["b"] = pd.cut(s.h0_delivered - s.h0_delivered_nom, GHOST_BINS, labels=GHOST_LAB)
        for b, d in s.groupby("b", observed=True):
            rows.append(dict(task=task, extra_ids=str(b), n=len(d),
                             model=(d.pred_un - d.y).abs().mean() / max((d.pred_un_nom - d.y).abs().mean(), 1e-9)))
    return pd.DataFrame(rows)
