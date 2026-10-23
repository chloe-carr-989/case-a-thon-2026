"""When will this vessel arrive? Time to the next inferred arrival, along learned sea lanes.

Arrival proxy: the start of the vessel's next sustained dwell (>= 30 min below 1 kn) inside an inferred
arrival zone, exactly the destination task's label. It is *not* a verified berth call, and arriving is not
the same as starting or finishing unloading.

ETA is conditional on a destination: the destination ranker's top zone. Features come only from the
(possibly degraded) state at the decision time: distance along learned lanes to that zone, straight-line
distance, reported speed, the typical lane speed where the vessel is, course vs bearing to the zone,
vessel type, time of day and how busy the zone was in training. Three HistGradientBoostingRegressors give
the median and a 10-90% interval (quantile loss), trained once on clean training-period decisions. The
interval is widened by split-conformal calibration (conformalised quantile regression) on the chronologically
last 20% of training decisions, which the models do not see, so it targets 80% coverage.

Pipelines on identical cases (the destination task's stress conditions):
* unmitigated: delivered record taken at face value, as if current; ranker zone without input checks
* quality_aware: the destination task's sanitised, age-compensated state and its quality-aware zone,
  deferring when that pipeline defers
* baseline: straight-line distance / reported speed to the unmitigated zone (no answer below 1 kn)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .destination import bearing_deg
from .geo import haversine_m
from .lanes import POOLED, LaneModel

ETA_FEATURES = ["route_km", "gc_km", "route_ratio", "sog", "lane_sog", "bdiff", "vessel_type", "hour",
                "log_zone_arrivals"]


def features(lat, lon, sog, cog, vtype, decision_t, zone_ids, zones: pd.DataFrame, lanes: LaneModel) -> pd.DataFrame:
    z = zones.set_index("zone_id")
    zid = pd.Series(np.asarray(zone_ids, dtype=object))
    zl, zo = zid.map(z.lat).to_numpy(float), zid.map(z.lon).to_numpy(float)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    gc = haversine_m(lat, lon, zl, zo) / 1000
    route = lanes.route_km(lat, lon, zl, zo)
    st = lanes.cell_stats(lat, lon, np.full(len(lat), POOLED, dtype=object))
    brg = bearing_deg(lat, lon, zl, zo)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = route / np.maximum(gc, 0.5)
    return pd.DataFrame({
        "route_km": route, "gc_km": gc, "route_ratio": ratio, "sog": np.asarray(sog, float),
        "lane_sog": st.sog_med.to_numpy(float), "bdiff": np.abs((brg - np.asarray(cog, float) + 180) % 360 - 180),
        "vessel_type": np.asarray(vtype, float), "hour": (np.asarray(decision_t, float) % 86400) / 3600,
        "log_zone_arrivals": np.log1p(zid.map(z.train_arrivals).to_numpy(float))})[ETA_FEATURES]


def fit(train: pd.DataFrame, zones: pd.DataFrame, lanes: LaneModel) -> dict:
    """Train on clean training-period decisions with their observed (true) arrival zone."""
    X = features(train.lat, train.lon, train.sog, train.cog, train.vessel_type, train.decision_t, train.true_zone,
                 zones, lanes)
    y = (train.arrival_t.to_numpy(float) - train.decision_t.to_numpy(float)) / 3600
    kw = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
              random_state=0)
    cut = np.quantile(train.decision_t.to_numpy(float), 0.8)          # conformal calibration = latest 20%
    fit_m = train.decision_t.to_numpy(float) < cut
    Xf, yf, Xc, yc = X[fit_m], y[fit_m], X[~fit_m], y[~fit_m]
    models = {"mid": HistGradientBoostingRegressor(loss="absolute_error", **kw).fit(Xf, yf),
              "lo": HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **kw).fit(Xf, yf),
              "hi": HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **kw).fit(Xf, yf)}
    s = np.maximum(models["lo"].predict(Xc) - yc, yc - models["hi"].predict(Xc))
    n = len(s)
    models["conformal_h"] = float(np.quantile(s, min(1.0, np.ceil((n + 1) * 0.8) / n))) if n else 0.0
    raw = float(((yc >= models["lo"].predict(Xc)) & (yc <= models["hi"].predict(Xc))).mean()) if n else np.nan
    models["info"] = {"n_train": int(fit_m.sum()), "n_conformal_calibration": int(n),
                      "n_train_vessels": int(train.mmsi.nunique()), "conformal_widening_h": models["conformal_h"],
                      "raw_interval_coverage_on_calibration": raw,
                      "route_found_share": float(np.isfinite(X.route_km).mean()), "features": ETA_FEATURES}
    return models


def predict(models: dict, X: pd.DataFrame):
    mid = np.maximum(models["mid"].predict(X), 0)
    q = models.get("conformal_h", 0.0)
    lo = np.minimum(np.maximum(models["lo"].predict(X) - q, 0), mid)
    hi = np.maximum(models["hi"].predict(X) + q, mid)
    return mid, lo, hi


def add_eta(cases: pd.DataFrame, models: dict, zones: pd.DataFrame, lanes: LaneModel) -> pd.DataFrame:
    """ETA columns (hours after the decision) for the unmitigated, quality-aware and baseline pipelines."""
    c = cases.copy()
    c["true_eta_h"] = (c.arrival_t - c.decision_t) / 3600
    for tag, (la, lo, so, co) in {"un": ("in_lat", "in_lon", "in_sog", "in_cog"),
                                  "mi": ("mi_lat", "mi_lon", "mi_sog", "mi_cog")}.items():
        zone = c[f"{tag}_pred1"]
        has = zone.notna().to_numpy()
        for k in ["eta_h", "eta_lo_h", "eta_hi_h", "route_km"]:
            c[f"{tag}_{k}"] = np.nan
        if has.any():
            s = c[has]
            X = features(s[la], s[lo], s[so], s[co], s.vessel_type, s.decision_t, zone[has], zones, lanes)
            mid, lo_, hi_ = predict(models, X)
            c.loc[has, f"{tag}_eta_h"], c.loc[has, f"{tag}_eta_lo_h"], c.loc[has, f"{tag}_eta_hi_h"] = mid, lo_, hi_
            c.loc[has, f"{tag}_route_km"] = X.route_km.to_numpy()
    c["un_eta_answered"] = c.un_answered & c.un_eta_h.notna()
    c["mi_eta_answered"] = c.mi_answered & c.mi_eta_h.notna()
    z = zones.set_index("zone_id")
    gc = haversine_m(c.in_lat, c.in_lon, c.un_pred1.map(z.lat), c.un_pred1.map(z.lon)) / 1000
    v = c.in_sog.to_numpy(float) * 1.852
    c["bl_eta_h"] = np.where((v >= 1.852) & np.isfinite(gc), gc / np.maximum(v, 1e-9), np.nan)
    c["bl_eta_answered"] = c.un_answered & c.bl_eta_h.notna()
    for col in [x for x in c.columns if x.endswith(("_eta_h", "_lo_h", "_hi_h", "_route_km"))]:
        c[col] = c[col].astype("float32")
    return c


def _err(g: pd.DataFrame, tag: str, penalty: float) -> dict:
    ans = g[f"{tag}_eta_answered"].to_numpy(bool)
    e = (g[f"{tag}_eta_h"] - g.true_eta_h).to_numpy(float)
    ae = np.abs(e[ans])
    pred1 = g["un_pred1" if tag == "bl" else f"{tag}_pred1"].to_numpy(dtype=object)
    right = ans & (pred1 == g.true_zone.to_numpy(dtype=object))
    n, na = len(g), int(ans.sum())
    r = {"n_cases": n, "n_answered": na, "coverage": na / n if n else np.nan,
         "mae_h": ae.mean() if na else np.nan, "median_ae_h": np.median(ae) if na else np.nan,
         "p90_ae_h": np.quantile(ae, 0.9) if na else np.nan, "bias_h": e[ans].mean() if na else np.nan,
         "mae_h_zone_correct": np.abs(e[right]).mean() if right.any() else np.nan,
         "n_zone_correct": int(right.sum()),
         "policy_mae_h": (ae.sum() + penalty * (n - na)) / n if n else np.nan}
    if tag != "bl" and na:
        t = g.true_eta_h.to_numpy(float)[ans]
        r["interval_coverage"] = float(((t >= g[f"{tag}_eta_lo_h"].to_numpy(float)[ans]) &
                                        (t <= g[f"{tag}_eta_hi_h"].to_numpy(float)[ans])).mean())
    return r


def summarize(cases: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Metrics rows (task='eta'). Deferred cases are charged the p90 of nominal unmitigated error."""
    nom = cases[(cases.condition == "nominal") & cases.un_eta_answered]
    penalty = float(np.quantile(np.abs(nom.un_eta_h - nom.true_eta_h), 0.9)) if len(nom) else np.nan
    ct = cases.vessel_type.between(70, 89)
    rows = []
    for (cond, seed), g0 in cases.groupby(["condition", "seed"]):
        for sub, gs in [("all", g0), ("cargo_tanker", g0[ct.loc[g0.index]])]:
            for w, g in list(gs.groupby("window")) + [("all", gs)]:
                if g.empty:
                    continue
                base = {"task": "eta", "condition": cond, "seed": seed, "subgroup": sub, "window": str(w),
                        "mean_age_min": g.age_s.mean() / 60, "stale_frac": g.is_stale.mean(),
                        "lowq_frac": g.is_lowq.mean(), "synthetic_share": (g.source != "AIS-original").mean()}
                for mit, pred, tag in [("none", "hgb_eta", "un"), ("none", "naive_distance_speed", "bl"),
                                       ("quality_aware", "hgb_eta", "mi")]:
                    rows.append(dict(base, mitigation=mit, predictor=pred, **_err(g, tag, penalty)))
    return pd.DataFrame(rows), penalty
