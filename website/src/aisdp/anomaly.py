"""Which vessels are moving unusually relative to the learned sea lanes? (Government mode)

Unit: a vessel-window = one vessel's fixes (one per minute) in the 60 min up to a scoring time T, every
30 min. Only fixes observed at or before T are used, so a score is what the system could have known at T.

Movement components (each compared with the vessel's own lane group, see lanes.py):
* off_lane   current distance from the nearest lane cell (km): moving fixes of the last 10 min, to the
             vessel's own category lanes when one lies within 20 km, else to the pooled lanes. Only
             counted within 20 km of any learned (pooled) traffic: where nothing was ever observed there is
             no pattern to deviate from, and such windows are labelled 'no_history' instead of scored.
* wrong_way  share of moving fixes whose course is shared by < 10% of the cell's training traffic
* speed      |speed - cell median| / cell spread of moving fixes (robust z)
* stop       minutes stopped where training traffic rarely stops, or far from any lane

Normalisation: each component is turned into its percentile among *calibration* windows of the same
vessel category (training period, test-hash vessels the lanes never saw), the tails are combined
(weighted sum of -log(1 - p): off-lane 1, the others 0.25) and the sum is again ranked against calibration
windows. The score is that
percentile (0-100). Labels: >= 97.5 unusual, >= 99.5 high deviation. They describe movement relative to
historical traffic, not intent or threat; on clean calibration data 2.5% of windows are 'unusual' by
construction.

Data quality is reported separately from movement: newest-fix age, longest gap, spikes (fixes implying
> 60 kn in and out) and invalid speeds. Weather is joined in the app as context only and never changes
a label.

Pipelines on identical vessel-windows:
* naive: every delivered fix taken at face value; means
* quality_aware (countermeasure): spike and invalid-speed fixes dropped, medians, and no movement verdict
  when the newest fix is older than 20 min (reported as a data problem instead)

Stress tests (SIMULATED, seeded, labelled): stale = the newest 5-45 min of a window not yet delivered;
low quality = a faulty source adds one record per two real ones (jitter, 25% gross outliers 3-15 km,
wrong/missing/sentinel speed, wrong/missing course). Known-label check: synthetic route deviations
(2-8 km lateral drift over 30 min) injected into windows that were normal on the clean feed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import AnomalyConfig, ExperimentConfig, LaneConfig
from .features import vessel_key_sql
from .geo import KN_TO_MS, haversine_m, offset
from .lanes import LaneModel, direction_support, lane_group

COMPONENTS = ["off_lane", "wrong_way", "speed", "stop"]
CONDITIONS = ["nominal", "stale", "low_quality", "combined"]
PIPELINES = {"none": "naive", "quality_aware": "quality_aware"}


class InsufficientAnomaly(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------- points
def load_points(con, split_t: float, ac: AnomalyConfig, ecfg: ExperimentConfig) -> pd.DataFrame:
    test_sql = f"({vessel_key_sql('mmsi')} % {ecfg.test_vessel_mod}) < {ecfg.test_vessel_cut}"
    P = con.execute(f"""
        SELECT mmsi, t, lat, lon, sog, cog, category, (t >= {split_t}) AS test_period
        FROM vm WHERE t >= {split_t} OR (t >= {split_t - ac.calib_s} AND {test_sql})
        ORDER BY mmsi, t
    """).df()
    P["category"] = P.category.astype("category")
    for c in ["lat", "lon", "sog", "cog"]:          # float32 lat/lon: < 1 m rounding at these magnitudes
        P[c] = P[c].astype("float32")
    return P


def point_features(P: pd.DataFrame, lanes: LaneModel, lc: LaneConfig) -> pd.DataFrame:
    """Pipeline-independent per-fix features (masks for moving / valid fixes are applied later)."""
    grp = lane_group(P.category.astype(str).to_numpy())
    lat, lon = P.lat.to_numpy(float), P.lon.to_numpy(float)
    st = lanes.cell_stats(lat, lon, grp)
    sog, cog = P.sog.to_numpy(float), P.cog.to_numpy(float)
    dense = st.n_moving.to_numpy(float) >= lc.min_cell_moving
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.maximum((st.sog_q75 - st.sog_q25).to_numpy(float) / 1.349, 1.0)
        z = np.where(dense, np.abs(sog - st.sog_med.to_numpy(float)) / scale, np.nan)
    sup = np.where(dense, direction_support(st, cog), np.nan)
    pooled = lanes.lane_distance_km(lat, lon, np.full(len(lat), "commercial", dtype=object))
    own = lanes.lane_distance_km(lat, lon, grp)
    own = np.where(own <= lc.own_lane_max_km, own, pooled)        # sparse own-category lanes: use pooled ones
    return pd.DataFrame({"lane_km": own.astype("float32"),
                         "pooled_km": pooled.astype("float32"),
                         "dir_sup": sup.astype("float32"), "speed_z": z.astype("float32"),
                         "stop_share": np.where(st.n_all.to_numpy(float) >= 20, st.stop_share, np.nan).astype("float32")},
                        index=P.index)


def expand(t: np.ndarray, ac: AnomalyConfig, t_min: float, t_max: float):
    """(row index, window end T) for every window (T - window, T] that contains each fix."""
    k1 = np.ceil(t / ac.step_s).astype(np.int64)
    idx, Ts = [], []
    for o in range(ac.window_s // ac.step_s):
        T = (k1 + o) * ac.step_s
        m = (T >= t_min) & (T <= t_max) & (T - ac.window_s < t)
        idx.append(np.flatnonzero(m))
        Ts.append(T[m])
    return np.concatenate(idx), np.concatenate(Ts)


# ------------------------------------------------------------------------------------------- windows
def _spikes(key, t, lat, lon, ac: AnomalyConfig) -> np.ndarray:
    """A fix implying > spike speed both from the previous and to the next fix of the same window."""
    n = len(t)
    same_prev = np.r_[False, key[1:] == key[:-1]]
    same_next = np.r_[key[:-1] == key[1:], False]
    d = np.r_[haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]), np.nan] / 1000      # to next
    dt = np.r_[np.diff(t), np.nan]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = d / np.maximum(dt, 1.0) * 3600 / 1.852                                       # knots
    jump = (v > ac.spike_speed_kn) & (d > ac.spike_min_km)
    j_next = same_next & jump
    j_prev = same_prev & np.r_[False, jump[:-1]]
    has_n, has_p = same_next, same_prev
    out = (j_prev | ~has_p) & (j_next | ~has_n) & (has_p | has_n)
    return out if n else np.zeros(0, bool)


def aggregate(E: pd.DataFrame, ac: AnomalyConfig) -> dict[str, pd.DataFrame]:
    """Window features for both pipelines. E: one row per (fix, window), sorted by key then t."""
    key = E.key.to_numpy()
    t = E.t.to_numpy(float)
    sog, cog = E.sog.to_numpy(float), E.cog.to_numpy(float)
    spike = _spikes(key, t, E.lat.to_numpy(float), E.lon.to_numpy(float), ac)
    valid_sog = np.isfinite(sog) & (sog <= ac.max_valid_sog_kn)
    covered = np.nan_to_num(E.pooled_km.to_numpy(float), nan=99) <= ac.coverage_km
    stop_pt = covered & (np.nan_to_num(sog, nan=99) < 0.5) & (
        (np.nan_to_num(E.stop_share.to_numpy(float), nan=1) < ac.unusual_stop_share) |
        (E.lane_km.to_numpy(float) > ac.lane_far_km))
    recent = t > E["T"].to_numpy(float) - ac.recent_s
    gap = np.r_[np.nan, np.diff(t)]
    gap[np.r_[True, key[1:] != key[:-1]]] = np.nan
    base = pd.DataFrame({"key": key, "gap": gap, "spike": spike, "invalid": np.isfinite(sog) & ~valid_sog})
    dq = base.groupby("key", sort=True).agg(n_pts=("gap", "size"), max_gap_s=("gap", "max"), n_spike=("spike", "sum"),
                                            n_invalid=("invalid", "sum"))
    out = {}
    for mit, robust in [("none", False), ("quality_aware", True)]:
        keep = ~spike if robust else np.ones(len(E), bool)
        s_ok = valid_sog if robust else np.isfinite(sog)
        moving = keep & s_ok & (sog >= 2.0) & np.isfinite(cog)
        f = pd.DataFrame({"key": key,
                          "lane_km": np.where(moving & recent & covered, E.lane_km, np.nan),
                          "uncov": np.where(moving & recent, ~covered, np.nan),
                          "wrong": np.where(moving & np.isfinite(E.dir_sup), (E.dir_sup < ac.wrongway_support), np.nan),
                          "z": np.where(moving, E.speed_z, np.nan),
                          "stop": (stop_pt & keep & (s_ok | ~np.isfinite(sog))).astype(np.int16),
                          "t": np.where(keep, t, np.nan), "mv": moving.astype(np.int16)})
        g = f.groupby("key", sort=True)
        agg = "median" if robust else "mean"
        w = pd.DataFrame({"off_lane": g.lane_km.agg(agg), "wrong_way": g.wrong.mean(), "speed": g.z.agg(agg),
                          "stop": g.stop.sum().astype(float), "n_moving": g.mv.sum(), "n_recent": g.lane_km.count(),
                          "uncovered": g.uncov.mean(),
                          "n_dir": g.wrong.count(), "n_z": g.z.count()})
        w.loc[w.n_recent < 3, "off_lane"] = np.nan
        w.loc[w.n_dir < 3, "wrong_way"] = np.nan
        w.loc[w.n_z < 3, "speed"] = np.nan
        last = pd.DataFrame({"key": key, "t": t, "lat": E.lat.to_numpy(float), "lon": E.lon.to_numpy(float),
                             "sog": sog, "cog": cog})[keep]
        last = last[~last.key.duplicated(keep="last")].set_index("key")      # rows are sorted by key, t
        w = w.join(last.rename(columns={"t": "last_t"})).join(dq)
        out[mit] = w
    return out


# ------------------------------------------------------------------------------------------- scoring
class Calibration:
    """Per-(pipeline, category) empirical distributions from clean calibration windows."""

    def __init__(self, W: dict[str, pd.DataFrame], cats: pd.Series, ac: AnomalyConfig):
        self.ac, self.tab = ac, {}
        for mit, w in W.items():
            c = cats.reindex(w.index).astype(str).to_numpy()
            for cat in list(np.unique(c)) + ["__pooled__"]:
                m = np.ones(len(w), bool) if cat == "__pooled__" else (c == cat)
                if m.sum() < ac.min_calib_windows and cat != "__pooled__":
                    continue
                comp = {k: np.sort(w[k].to_numpy(float)[m & np.isfinite(w[k].to_numpy(float))]) for k in COMPONENTS}
                self.tab[(mit, cat)] = {"comp": comp, "n": int(m.sum())}
                raw = self._raw(w[m], comp)
                self.tab[(mit, cat)]["raw"] = np.sort(raw)

    @staticmethod
    def _p(sorted_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
        if len(sorted_vals) == 0:
            return np.full(len(x), np.nan)
        lo, hi = np.searchsorted(sorted_vals, x, side="left"), np.searchsorted(sorted_vals, x, side="right")
        p = (lo + hi) / 2 / (len(sorted_vals) + 1)                   # mid-rank: ties (e.g. capped values) share a rank
        return np.where(np.isfinite(x), p, np.nan)

    def _raw(self, w: pd.DataFrame, comp: dict) -> np.ndarray:
        tot = np.zeros(len(w))
        for k in COMPONENTS:
            p = self._p(comp[k], w[k].to_numpy(float))
            if k in ("stop", "wrong_way"):        # zero stop minutes / no wrong-way fixes is not evidence
                p = np.where(w[k].to_numpy(float) > 0, p, np.nan)
            tot += dict(self.ac.weights)[k] * np.nan_to_num(-np.log(1 - p), nan=0.0)
        return tot

    def score(self, mit: str, w: pd.DataFrame, cats: np.ndarray) -> pd.DataFrame:
        out = pd.DataFrame(index=w.index, columns=["score", *[f"p_{k}" for k in COMPONENTS]], dtype=float)
        for cat in np.unique(cats):
            m = cats == cat
            tab = self.tab.get((mit, cat)) or self.tab[(mit, "__pooled__")]
            sub = w[m]
            for k in COMPONENTS:
                p = self._p(tab["comp"][k], sub[k].to_numpy(float))
                if k in ("stop", "wrong_way"):
                    p = np.where(sub[k].to_numpy(float) > 0, p, np.nan)
                out.loc[m, f"p_{k}"] = p
            out.loc[m, "score"] = 100 * self._p(tab["raw"], self._raw(sub, tab["comp"]))
        return out

    def summary(self) -> dict:
        return {f"{m}|{c}": v["n"] for (m, c), v in self.tab.items()}


def label(score, age_s, robust: bool, ac: AnomalyConfig, uncovered=None) -> np.ndarray:
    s = np.asarray(score, float)
    lab = np.where(s >= ac.high_pct, "high", np.where(s >= ac.unusual_pct, "unusual", "normal")).astype(object)
    if uncovered is not None:
        lab[np.nan_to_num(np.asarray(uncovered, float)) > 0.5] = "no_history"
    if robust:
        lab[np.asarray(age_s, float) > ac.stale_hold_s] = "no_verdict"
    return lab


def reasons(sc: pd.DataFrame) -> np.ndarray:
    r = np.array([""] * len(sc), dtype=object)
    for k in COMPONENTS:
        m = np.nan_to_num(sc[f"p_{k}"].to_numpy(float)) >= 0.95
        r[m] = r[m] + k + ";"
    return r


# ------------------------------------------------------------------------------------------- degradation
def _corrupt_points(E: pd.DataFrame, rng: np.random.Generator, share: float) -> pd.DataFrame:
    """Synthetic records from the faulty source: copies of real fixes with injected defects."""
    m = rng.random(len(E)) < share
    S = E[m].copy()
    n = len(S)
    if n == 0:
        return S
    S["t"] = np.minimum(S.t.to_numpy() + 5 + 20 * rng.random(n), S["T"].to_numpy())
    lat, lon = offset(S.lat.to_numpy(), S.lon.to_numpy(), 200 * rng.normal(size=n), 200 * rng.normal(size=n))
    out = rng.random(n) < 0.25
    dist, brg = 3000 + 12000 * rng.random(n), rng.random(n) * 2 * np.pi
    lat2, lon2 = offset(lat, lon, dist * np.sin(brg), dist * np.cos(brg))
    S["lat"], S["lon"] = np.where(out, lat2, lat), np.where(out, lon2, lon)
    sog, cog = S.sog.to_numpy(float).copy(), S.cog.to_numpy(float).copy()
    us, uc = rng.random(n), rng.random(n)
    sog = np.where(us < 0.25, sog * (0.3 + 2.2 * rng.random(n)), sog)
    sog = np.where((us >= 0.25) & (us < 0.40), np.nan, sog)
    sog = np.where((us >= 0.40) & (us < 0.50), 102.3, sog)
    cog = np.where(uc < 0.30, rng.random(n) * 360, np.where(uc < 0.45, np.nan, (cog + 5 * rng.normal(size=n)) % 360))
    S["sog"], S["cog"] = sog, cog
    S["synthetic"] = True
    return S


def _with_features(S: pd.DataFrame, lanes: LaneModel, lc: LaneConfig) -> pd.DataFrame:
    if S.empty:
        return S
    f = point_features(S.reset_index(drop=True), lanes, lc)
    S = S.reset_index(drop=True)
    S[f.columns] = f.to_numpy()
    return S


def degrade(E: pd.DataFrame, win_meta: pd.DataFrame, cond: str, lanes: LaneModel, lc: LaneConfig,
            ac: AnomalyConfig, seed: int) -> pd.DataFrame:
    """Rows delivered under one condition. win_meta (per key): is_stale, delay_s, is_lowq."""
    if cond == "nominal":
        return E
    D = E
    if cond in ("stale", "combined"):
        st = win_meta.reindex(D.key.to_numpy())
        cut = D["T"].to_numpy() - np.where(st.is_stale.to_numpy(bool), st.delay_s.to_numpy(float), 0)
        D = D[D.t.to_numpy() <= cut]
    if cond in ("low_quality", "combined"):
        lq = win_meta.is_lowq.reindex(D.key.to_numpy()).to_numpy(bool)
        rng = np.random.default_rng(seed + 1000)
        S = _with_features(_corrupt_points(D[lq], rng, ac.lowq_point_share), lanes, lc)
        D = pd.concat([D, S], ignore_index=True).sort_values(["key", "t"], kind="stable")
    return D


# ------------------------------------------------------------------------------------------- run
def _score_all(W: dict, cal: Calibration, cats: pd.Series, ac: AnomalyConfig, T_of_key) -> dict:
    out = {}
    for mit, w in W.items():
        c = cats.reindex(w.index).astype(str).to_numpy()
        sc = cal.score(mit, w, c)
        age = T_of_key(w.index.to_numpy()) - w.last_t.to_numpy(float)
        r = w.join(sc)
        r["age_s"] = age
        r["label"] = label(sc.score, age, mit == "quality_aware", ac, w.uncovered)
        r["reasons"] = reasons(sc)
        out[mit] = r
    return out


def run(con, lanes: LaneModel, split_t: float, ecfg: ExperimentConfig, lc: LaneConfig, ac: AnomalyConfig,
        log=print) -> dict:
    t_all = time.perf_counter()
    P = load_points(con, split_t, ac, ecfg)
    if len(P) < 5000:
        raise InsufficientAnomaly(f"only {len(P)} vessel-minutes available for scoring")
    P = P.join(point_features(P, lanes, lc))
    t_first, t_last = float(P.t.min()), float(P.t.max())
    base_k = int(np.floor(t_first / ac.step_s))
    cats = P.groupby("mmsi").category.first()

    def build(mask, T_min, T_max):
        sub = P[mask]
        idx, Ts = expand(sub.t.to_numpy(float), ac, T_min, T_max)
        E = sub[["t", "lat", "lon", "sog", "cog", "category", "lane_km", "pooled_km", "dir_sup", "speed_z",
                 "stop_share"]].iloc[idx]
        E = E.reset_index(drop=True)
        E["T"] = Ts.astype(np.float64)
        E["key"] = sub.mmsi.to_numpy(np.int64)[idx] * 1_048_576 + (Ts // ac.step_s - base_k).astype(np.int64)
        E["synthetic"] = False
        return E.sort_values(["key", "t"], kind="stable").reset_index(drop=True)

    def T_of_key(k):
        return ((np.asarray(k, np.int64) % 1_048_576) + base_k) * float(ac.step_s)

    def mmsi_of_key(k):
        return np.asarray(k, np.int64) // 1_048_576

    Ecal = build(~P.test_period.to_numpy(), t_first + ac.window_s, split_t)
    Ete = build(P.test_period.to_numpy(), split_t + ac.window_s, t_last + ac.step_s)
    if Ecal.key.nunique() < ac.min_calib_windows or Ete.key.nunique() < 100:
        raise InsufficientAnomaly(f"calibration windows {Ecal.key.nunique()}, test windows {Ete.key.nunique()}")
    Wcal = aggregate(Ecal, ac)
    key_cats = pd.Series(cats.reindex(mmsi_of_key(Wcal["none"].index)).to_numpy(), index=Wcal["none"].index)
    cal = Calibration(Wcal, key_cats, ac)
    log(f"anomaly: {len(Ecal):,} calibration fix-windows / {len(Wcal['none']):,} windows; "
        f"{len(Ete):,} test fix-windows / {Ete.key.nunique():,} windows")

    # degradation draws per test window (chronological fifths of the test period)
    keys = np.sort(Ete.key.unique())
    Tk = T_of_key(keys)
    edges = np.linspace(Tk.min(), Tk.max() + 1e-6, ac.n_windows + 1)
    win = np.clip(np.searchsorted(edges, Tk, side="right") - 1, 0, ac.n_windows - 1)
    rng = np.random.default_rng(ac.seed)
    u_st, u_d, u_lq = rng.random(len(keys)), rng.random(len(keys)), rng.random(len(keys))
    meta = pd.DataFrame({"window": win, "is_stale": u_st < np.asarray(ac.stale_schedule)[win],
                         "delay_s": ac.stale_delay_min_s + u_d * (ac.stale_delay_max_s - ac.stale_delay_min_s),
                         "is_lowq": u_lq < np.asarray(ac.lowq_schedule)[win]}, index=keys)
    all_cats = pd.Series(cats.reindex(mmsi_of_key(keys)).to_numpy(), index=keys)
    frames = []
    for cond in CONDITIONS:
        D = degrade(Ete, meta, cond, lanes, lc, ac, ac.seed)
        R = _score_all(aggregate(D, ac), cal, all_cats, ac, T_of_key)
        for mit, r in R.items():
            r = r.reindex(keys)                           # a fully delayed window still exists, with no data
            r["condition"], r["mitigation"] = cond, mit
            r.loc[r.label.isna(), "label"] = "no_data"            # every fix of the window was withheld
            frames.append(r)
        log(f"anomaly: scored {cond}")
    S = pd.concat(frames)
    S["mmsi"] = mmsi_of_key(S.index.to_numpy())
    S["T"] = T_of_key(S.index.to_numpy())
    S = S.join(meta).reset_index(names="key")
    S["category"] = S.key.map(all_cats).astype(str)
    S["is_stale"] = S.is_stale & S.condition.isin(["stale", "combined"])
    S["is_lowq"] = S.is_lowq & S.condition.isin(["low_quality", "combined"])
    syn = synthetic_test(Ete, meta, S, cal, all_cats, lanes, lc, ac, T_of_key)
    metrics = summarize(S, syn, ac)
    return {"windows": S, "synthetic": syn, "metrics": metrics, "calibration": cal.summary(),
            "n_test_windows": int(len(keys)), "n_test_vessels": int(len(np.unique(mmsi_of_key(keys)))),
            "n_calibration_windows": int(len(Wcal["none"])), "window_edges_t": edges.tolist(),
            "seconds": round(time.perf_counter() - t_all, 1)}


def synthetic_test(Ete, meta, S, cal, cats, lanes, lc, ac, T_of_key) -> pd.DataFrame:
    """Known-label check: lateral drift injected into moving windows that were normal on the clean feed."""
    nom = S[(S.condition == "nominal")].pivot(index="key", columns="mitigation", values="label")
    nm = S[(S.condition == "nominal") & (S.mitigation == "quality_aware")].set_index("key").n_moving
    ok = nom.index[(nom["none"] == "normal") & (nom["quality_aware"] == "normal") & (nm.reindex(nom.index) >= 20)]
    rng = np.random.default_rng(ac.seed + 7)
    pick = np.sort(rng.choice(ok, size=min(ac.synth_windows, len(ok)), replace=False)) if len(ok) else ok
    E = Ete[Ete.key.isin(pick)].copy()
    D_km = pd.Series(ac.synth_offset_km[0] + rng.random(len(pick)) * (ac.synth_offset_km[1] - ac.synth_offset_km[0]),
                     index=pick)
    T = E["T"].to_numpy()
    frac = np.clip((E.t.to_numpy() - (T - ac.window_s / 2)) / (ac.window_s / 2), 0, 1)   # drift over the last 30 min
    d = frac * D_km.reindex(E.key).to_numpy() * 1000
    side = np.where(E.key.to_numpy() % 2 == 0, 1.0, -1.0)
    c = np.radians(np.nan_to_num(E.cog.to_numpy(float)) + 90 * side)
    E["lat"], E["lon"] = offset(E.lat.to_numpy(), E.lon.to_numpy(), d * np.sin(c), d * np.cos(c))
    E = _with_features(E, lanes, lc).sort_values(["key", "t"], kind="stable")
    rows = []
    for cond in CONDITIONS:
        D = degrade(E, meta, cond, lanes, lc, ac, ac.seed)
        R = _score_all(aggregate(D, ac), cal, cats, ac, T_of_key)
        for mit, r in R.items():
            r = r.reindex(pick)
            lab = r.label.fillna("no_data")
            rows.append(pd.DataFrame({"key": pick, "condition": cond, "mitigation": mit, "offset_km": D_km.to_numpy(),
                                      "label": lab.to_numpy(), "score": r.score.to_numpy(),
                                      **{k: r[k].to_numpy() for k in COMPONENTS}, "category": cats.reindex(pick).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def summarize(S: pd.DataFrame, syn: pd.DataFrame, ac: AnomalyConfig) -> pd.DataFrame:
    alert = S.label.isin(["unusual", "high"])
    S = S.assign(alert=alert)
    ref = S[S.condition == "nominal"].set_index(["key", "mitigation"])[["alert", "score"]]
    S = S.join(ref, on=["key", "mitigation"], rsuffix="_clean")
    rows = []
    for (cond, mit), g0 in S.groupby(["condition", "mitigation"]):
        for w, g in list(g0.groupby("window")) + [("all", g0)]:
            a, ac_ = g.alert.to_numpy(bool), g.alert_clean.to_numpy(bool)
            verdict = (~g.label.isin(["no_verdict", "no_data", "no_history"])).to_numpy()
            rows.append({"task": "anomaly", "condition": cond, "mitigation": mit, "predictor": "lane_anomaly",
                         "subgroup": "all", "window": str(w), "seed": ac.seed, "n_cases": len(g),
                         "coverage": verdict.mean(), "alert_rate": a.mean(),
                         "high_rate": (g.label == "high").mean(),
                         "spurious_alert_share": (a & ~ac_).sum() / max(a.sum(), 1),
                         "spurious_alerts_per_1000": 1000 * (a & ~ac_).mean(),
                         "missed_clean_alert_share": (ac_ & ~a).sum() / max(ac_.sum(), 1),
                         "mean_abs_score_shift": np.nanmean(np.abs(g.score - g.score_clean)),
                         "stale_frac": g.is_stale.mean(), "lowq_frac": g.is_lowq.mean(),
                         "data_flag_rate": ((g.age_s > ac.stale_flag_s) | (g.max_gap_s > ac.gap_flag_s) |
                                            (g.n_spike > 0) | (g.n_invalid > 0)).mean()})
    for (cond, mit), g in syn.groupby(["condition", "mitigation"]):
        rows.append({"task": "anomaly_synthetic", "condition": cond, "mitigation": mit, "predictor": "lane_anomaly",
                     "subgroup": "injected_drift_2_8km", "window": "all", "seed": ac.seed, "n_cases": len(g),
                     "recall": g.label.isin(["unusual", "high"]).mean(), "recall_high": (g.label == "high").mean(),
                     "coverage": (~g.label.isin(["no_verdict", "no_data", "no_history"])).mean()})
    return pd.DataFrame(rows)
