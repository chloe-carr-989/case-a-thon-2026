"""Shared count forecasting: unique MMSIs per spatial unit in the next 60 minutes.

Used by two tasks with distinct targets and metrics:
* traffic_density: unique vessels observed in a 0.25 deg grid cell (raw count and per-1000 km2 rate).
* port_occupancy: unique vessels dwelling (SOG < 1 kn) in an inferred port/anchorage study zone,
  an AIS **occupancy proxy**, not measured congestion, berth occupancy or waiting time.

Counting is on a (unit, 15-min bucket, mmsi) presence table, so one vessel counts once per window no
matter how often it reports. Windows are counted with DuckDB range joins (no unit x minute grid).
A feature for forecast origin o only reads buckets < o (and < o - delay when the feed is late);
the target reads buckets [o, o + 4) from the *unmodified* table.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import CountConfig
from .ingest import parquet_glob
from .zones import VESSEL_WHERE, cell_area_km2, clean_sog_sql

# No hour-of-day features: with one day of data the test hours never occur in training.
# unit_mean_train is a baseline only; as a feature it would be in-sample for training rows.
FEATURES = ["h0", "h1", "h2", "q0", "h3", "trend", "recent_share", "log_msgs_h0", "msgs_per_vessel",
            "lat", "lon", "log_area"]
SYN_SOURCE = "SYN-FASTFEED"
VALID_MMSI = (201_000_000, 775_999_999)


class InsufficientCounts(RuntimeError):
    pass


# ------------------------------------------------------------------------------ presence tables
def _src(info):
    return f"read_parquet('{parquet_glob(info)}', hive_partitioning=true)"


def feed_volume(con, info: dict, cc: CountConfig) -> pd.Series:
    """Messages per bucket in the raw source (all records). Zero = no source data, not zero traffic."""
    v = con.execute(f"SELECT floor(epoch(ts) / {cc.bucket_s})::BIGINT b, count(*) n FROM {_src(info)} GROUP BY 1").df()
    return v.set_index("b").n.sort_index()


def presence_density(con, info: dict, cc: CountConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = cc.cell_deg
    ub = con.execute(f"""
        SELECT floor(lat / {d})::BIGINT AS ci, floor(lon / {d})::BIGINT AS cj,
               floor(epoch(ts) / {cc.bucket_s})::BIGINT AS b, mmsi, count(*)::INT AS n_msgs
        FROM {_src(info)} WHERE {VESSEL_WHERE} GROUP BY ALL ORDER BY ALL
    """).df()
    units = ub[["ci", "cj"]].drop_duplicates().sort_values(["ci", "cj"]).reset_index(drop=True)
    units["unit"] = np.arange(len(units))
    units["lat"], units["lon"] = (units.ci + 0.5) * d, (units.cj + 0.5) * d
    units["area_km2"] = cell_area_km2(units.ci, d)
    units["unit_id"] = [f"C{d:g}_{a:+.3f}_{o:+.3f}" for a, o in zip(units.lat, units.lon)]
    ub = ub.merge(units[["ci", "cj", "unit"]], on=["ci", "cj"])[["unit", "b", "mmsi", "n_msgs"]]
    ub = ub.sort_values(["unit", "b", "mmsi"], kind="stable").reset_index(drop=True)
    return ub, units


def presence_zones(con, info: dict, zones: pd.DataFrame, cc: CountConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dwelling presence (a report inside the zone with SOG < 1 kn) per zone, bucket and vessel."""
    d = cc.port_zone_deg
    z = zones.reset_index(drop=True).copy()
    z["unit"] = np.arange(len(z))
    con.register("zone_cells", z[["zi", "zj", "unit"]])
    ub = con.execute(f"""
        SELECT z.unit, floor(epoch(p.ts) / {cc.bucket_s})::BIGINT AS b, p.mmsi, count(*)::INT AS n_msgs
        FROM {_src(info)} p JOIN zone_cells z
          ON z.zi = floor(p.lat / {d})::BIGINT AND z.zj = floor(p.lon / {d})::BIGINT
        WHERE {VESSEL_WHERE} AND {clean_sog_sql('p.sog')} < 1.0
        GROUP BY ALL ORDER BY ALL
    """).df()
    con.unregister("zone_cells")
    units = z.rename(columns={"zone_id": "unit_id"})
    units["lat"], units["lon"] = (units.lat0 + units.lat1) / 2, (units.lon0 + units.lon1) / 2
    return ub, units[["unit", "unit_id", "lat", "lon", "area_km2", "lat0", "lat1", "lon0", "lon1",
                      "train_vessels", "train_arrivals"]]


# ------------------------------------------------------------------------------------ windows
class Presence:
    """(unit, mmsi, bucket) presence rows sorted once by (unit, mmsi, b), for fast windowed distinct counts."""

    def __init__(self, ub: pd.DataFrame, n_units: int):
        x = ub.sort_values(["unit", "mmsi", "b"], kind="stable")
        self.unit = x.unit.to_numpy(np.int64)
        self.mmsi = x.mmsi.to_numpy(np.int64)
        self.b = x.b.to_numpy(np.int64)
        self.n_msgs = x.n_msgs.to_numpy(float)
        self.n_units = n_units
        self.same = np.r_[False, (self.unit[1:] == self.unit[:-1]) & (self.mmsi[1:] == self.mmsi[:-1])]
        self.db = np.r_[0, np.diff(self.b)]


def cover_counts(P: Presence, k1: int, k2: int, o_lo: int, o_hi: int, distinct: bool = True) -> np.ndarray:
    """counts[unit, o - o_lo] for o in [o_lo, o_hi]: unique MMSIs (or summed messages) with a report in a
    bucket b such that k1 <= o - b <= k2. A vessel is counted once however many buckets it appears in:
    each row covers origins [b + k1, b + k2]; overlapping covers of one (unit, mmsi) are merged first."""
    width = o_hi - o_lo + 2
    out = np.zeros((P.n_units, width - 1))
    if k1 > k2 or len(P.b) == 0:
        return out
    if distinct:
        brk = ~P.same | (P.db > (k2 - k1 + 1))
        first = np.flatnonzero(brk)
        last = np.r_[first[1:] - 1, len(P.b) - 1]
        gs, ge, gu, w = P.b[first] + k1, P.b[last] + k2, P.unit[first], np.ones(len(first))
    else:
        gs, ge, gu, w = P.b + k1, P.b + k2, P.unit, P.n_msgs
    keep = (ge >= o_lo) & (gs <= o_hi)
    gs, ge, gu, w = np.clip(gs[keep], o_lo, o_hi) - o_lo, np.clip(ge[keep], o_lo, o_hi) - o_lo + 1, gu[keep], w[keep]
    n = P.n_units * width
    diff = np.bincount(gu * width + gs, w, minlength=n) - np.bincount(gu * width + ge, w, minlength=n)
    return np.cumsum(diff.reshape(P.n_units, width), axis=1)[:, :-1]


def window_features(P: Presence, origins, delay, cc: CountConfig, delay_hat=None) -> pd.DataFrame:
    """Per (unit, origin), dense over all units: unique vessels in past windows using only buckets
    delivered by o. delay[i] = most-recent buckets not yet delivered at origin i (0 = timely feed).
    h0/h1/h2 = previous, 2nd and 3rd previous hour; q0 = last 15 min; h3 = last 3 h (distinct);
    h0_shift = last delivered complete hour, buckets (delay_hat, delay_hat + 4] before o.
    Units with no presence are confirmed zeros (callers restrict origins to covered source data)."""
    W, L = cc.window_buckets, cc.lookback_buckets
    o = np.asarray(origins, np.int64)
    d = np.asarray(delay, np.int64)
    dh = d if delay_hat is None else np.asarray(delay_hat, np.int64)
    lo, hi = int(o.min()), int(o.max())
    cols = {c: np.zeros((P.n_units, len(o))) for c in ["h0", "h1", "h2", "q0", "h3", "msgs_h0", "h0_shift"]}
    spec = {"h0": (1, W), "h1": (W + 1, 2 * W), "h2": (2 * W + 1, 3 * W), "q0": (1, 1), "h3": (1, L)}
    for dv in np.unique(d):
        sel = np.flatnonzero(d == dv)
        idx = o[sel] - lo
        for c, (k1, k2) in spec.items():
            cols[c][:, sel] = cover_counts(P, max(k1, dv + 1), k2, lo, hi)[:, idx]
        cols["msgs_h0"][:, sel] = cover_counts(P, max(1, dv + 1), W, lo, hi, distinct=False)[:, idx]
        for hv in np.unique(dh[sel]):
            s2 = sel[dh[sel] == hv]
            cols["h0_shift"][:, s2] = cover_counts(P, max(hv + 1, dv + 1), hv + W, lo, hi)[:, o[s2] - lo]
    out = pd.DataFrame({"unit": np.repeat(np.arange(P.n_units), len(o)), "o": np.tile(o, P.n_units)})
    for c, v in cols.items():
        out[c] = v.reshape(-1)
    return out


def targets(P: Presence, origins, cc: CountConfig) -> pd.DataFrame:
    """Unique vessels in [o, o + window) from the unmodified presence table."""
    W = cc.window_buckets
    o = np.asarray(origins, np.int64)
    t = window_features(P, o + W, np.zeros(len(o), np.int64), cc)
    t["o"] = t.o - W
    return t[["unit", "o", "h0"]].rename(columns={"h0": "y"})


def add_model_features(df: pd.DataFrame, units: pd.DataFrame, unit_mean: pd.Series, cc: CountConfig) -> pd.DataFrame:
    df = df.merge(units[["unit", "lat", "lon", "area_km2"]], on="unit", how="left")
    df["unit_mean_train"] = df.unit.map(unit_mean).fillna(0.0).to_numpy()
    df["log_msgs_h0"] = np.log1p(df.msgs_h0)
    df["msgs_per_vessel"] = np.where(df.h0 > 0, df.msgs_h0 / np.maximum(df.h0, 1), 0.0)
    df["trend"] = df.h0 - df.h1
    df["recent_share"] = np.where(df.h0 > 0, df.q0 / np.maximum(df.h0, 1), 0.0)
    df["log_area"] = np.log(df.area_km2)
    return df


def train_model(X: pd.DataFrame, y, seed: int = 0):
    """Learns the change from persistence (y - h0); trees cannot represent identity on large counts."""
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=seed).fit(X[FEATURES], np.asarray(y) - X["h0"].to_numpy())


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return np.maximum(X["h0"].to_numpy() + model.predict(X[FEATURES]), 0.0)


# --------------------------------------------------------------------------- feed degradation
def bucket_window(b, test_b0: int, edges_b) -> np.ndarray:
    """Chronological test window of a bucket (buckets before the test period use window 0's rates)."""
    w = np.searchsorted(np.asarray(edges_b), np.asarray(b), side="right") - 1
    return np.clip(w, 0, len(edges_b) - 2)


def draw_delays(origins, win, seed: int, cc: CountConfig, severity: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed + 101)
    p = np.minimum(np.asarray(cc.delay_schedule)[win] * severity, 1.0)
    late = rng.random(len(origins)) < p
    return np.where(late, rng.integers(1, cc.delay_max_buckets + 1, len(origins)), 0)


def corrupt_presence(ub: pd.DataFrame, b_from: int, edges_b, seed: int, cc: CountConfig,
                     severity: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """SYNTHETIC low-quality feed applied to buckets >= b_from (inputs of test origins only).

    * ghosts: synthetic vessel-buckets from source SYN-FASTFEED, share per real row = ghost_schedule[window].
      50% carry an invalid MMSI (MID 100-199), 35% are valid-looking one-message transients, 15% are
      valid-looking persistent ghosts that repeat within the hour (designed to evade the countermeasure);
    * duplicates: messages of some real vessel-buckets tripled (inflates volume, not unique counts);
    * omissions: real vessel-buckets lost, at omission_rate scaled by the window's ghost share.
    """
    rng = np.random.default_rng(seed + 202)
    x = ub.copy()
    x["source"] = "AIS-original"
    aff = x.b.to_numpy() >= b_from
    share = np.asarray(cc.ghost_schedule)[bucket_window(x.b.to_numpy(), b_from, edges_b)] * severity
    share = np.where(aff, np.minimum(share, 1.0), 0.0)
    u_om, u_dup, u_gh, u_kind = rng.random((4, len(x)))
    drop = u_om < cc.omission_rate * share / max(cc.ghost_schedule)
    dup = aff & (u_dup < cc.duplicate_rate)
    x.loc[dup, "n_msgs"] = x.loc[dup, "n_msgs"] * 3
    spawn = u_gh < share
    g = x.loc[spawn, ["unit", "b"]].copy()
    k = u_kind[spawn]
    rand_valid = rng.integers(VALID_MMSI[0], VALID_MMSI[1], len(g))
    rand_invalid = rng.integers(100_000_000, 199_999_999, len(g))
    persist = ((g.unit.to_numpy() * 1_000_003 + g.b.to_numpy() // cc.window_buckets) * 7919) % \
        (VALID_MMSI[1] - VALID_MMSI[0]) + VALID_MMSI[0]
    g["mmsi"] = np.where(k < 0.5, rand_invalid, np.where(k < 0.85, rand_valid, persist))
    g["n_msgs"] = np.where(k < 0.85, 1, 3)
    g["source"] = SYN_SOURCE
    g["ghost_kind"] = np.where(k < 0.5, "invalid_mmsi", np.where(k < 0.85, "transient", "persistent"))
    out = pd.concat([x[~drop].assign(ghost_kind=""), g], ignore_index=True)
    out = out.groupby(["unit", "b", "mmsi"], as_index=False).agg(n_msgs=("n_msgs", "sum"), source=("source", "first"),
                                                                 ghost_kind=("ghost_kind", "first"))
    log = {"real_rows_affected_period": int(aff.sum()), "omitted_rows": int(drop.sum()),
           "duplicated_rows": int(dup.sum()), "ghost_rows": int(len(g)),
           "ghost_kinds": pd.Series(g.ghost_kind).value_counts().to_dict(), "seed": seed, "severity": severity}
    return out, log


# ---------------------------------------------------------------------------- countermeasure
def mitigate_presence(ub: pd.DataFrame, cc: CountConfig) -> pd.DataFrame:
    """Drop non-vessel/invalid MMSIs and single-message transients (causal: trailing hour up to bucket b)."""
    x = ub[(ub.mmsi >= VALID_MMSI[0]) & (ub.mmsi <= VALID_MMSI[1])]
    per = x.groupby(["mmsi", "b"], as_index=False).n_msgs.sum().sort_values(["mmsi", "b"])
    # trailing-hour message total per vessel over all units, using only buckets (b - window, b]
    per["cum"] = per.groupby("mmsi").n_msgs.cumsum()
    look = per[["mmsi", "b"]].assign(bq=per.b - cc.window_buckets).sort_values("bq")
    ref = per[["mmsi", "b", "cum"]].rename(columns={"b": "bq", "cum": "cum_prev"}).sort_values("bq")
    m = pd.merge_asof(look, ref, on="bq", by="mmsi", direction="backward")
    per = per.merge(m[["mmsi", "b", "cum_prev"]], on=["mmsi", "b"])
    per["trail"] = per.cum - per.cum_prev.fillna(0)
    keep = per[per.trail >= cc.min_msgs_per_vessel][["mmsi", "b"]]
    return x.merge(keep, on=["mmsi", "b"])


def detect_delay(vol_delivered: pd.Series, origins, delay, cc: CountConfig) -> np.ndarray:
    """Most-recent buckets whose delivered volume is < feed_complete_ratio x trailing median (late feed)."""
    L = cc.lookback_buckets
    out = np.zeros(len(origins), np.int64)
    v = vol_delivered
    for i, (o, d) in enumerate(zip(origins, delay)):
        vals = np.array([v.get(b, 0) if b < o - d else 0 for b in range(o - L, o)], float)  # oldest .. newest
        ref = np.median(vals[vals > 0]) if (vals > 0).any() else 0.0
        k = 0
        while k < L and vals[L - 1 - k] < cc.feed_complete_ratio * ref:
            k += 1
        out[i] = k
    return out


# --------------------------------------------------------------------------------- experiment
def _metrics(y, p, ans, pen):
    y, p, ans = np.asarray(y, float), np.asarray(p, float), np.asarray(ans, bool)
    e = p[ans] - y[ans]
    n = len(y)
    ae = np.abs(np.where(ans, p - y, np.nan))
    return {"n_cases": n, "n_answered": int(ans.sum()), "coverage": ans.mean() if n else np.nan,
            "deferral_rate": 1 - ans.mean() if n else np.nan,
            "mae": np.abs(e).mean() if len(e) else np.nan, "rmse": np.sqrt((e ** 2).mean()) if len(e) else np.nan,
            "bias": e.mean() if len(e) else np.nan, "p90_abs_err": np.quantile(np.abs(e), 0.9) if len(e) else np.nan,
            "policy_mae": np.where(ans, ae, pen).mean() if n else np.nan}


def run(con, info: dict, task: str, cc: CountConfig, zones: pd.DataFrame | None = None, log=print) -> dict:
    t_all = time.perf_counter()
    if task == "traffic_density":
        ub, units = presence_density(con, info, cc)
    else:
        if zones is None or zones.empty:
            raise InsufficientCounts("no port/anchorage study zones could be derived from the training period")
        ub, units = presence_zones(con, info, zones, cc)
    vol_raw = feed_volume(con, info, cc)
    if ub.empty or vol_raw.empty:
        raise InsufficientCounts("no vessel presence rows")
    t_first, t_last = con.execute(f"SELECT min(epoch(ts)), max(epoch(ts)) FROM {_src(info)}").fetchall()[0]
    b_min, b_max = int(vol_raw.index.min()), int(np.floor(t_last / cc.bucket_s))
    L, W = cc.lookback_buckets, cc.window_buckets
    all_o = np.arange(b_min + L, b_max - W + 2)
    # source-data coverage: a bucket with zero raw messages is missing data, never "zero traffic"
    has = np.array([vol_raw.get(b, 0) > 0 for b in range(b_min, b_max + 1)])
    ok = np.array([has[o - L - b_min:o + W - b_min].all() for o in all_o], bool) if len(all_o) else np.array([], bool)
    all_o = all_o[ok]
    split_t = t_first + cc.train_time_frac * (t_last - t_first)
    tr_o = all_o[(all_o + W) * cc.bucket_s <= split_t]
    te_o = all_o[all_o * cc.bucket_s >= split_t]
    if len(tr_o) < 4 or len(te_o) < 2:
        raise InsufficientCounts(f"not enough complete 60-min forecast origins (train={len(tr_o)}, test={len(te_o)}; "
                                 f"need >=4 / >=2, each needs {L * cc.bucket_s // 3600} h lookback + 1 h target)")
    # training-period unit index and per-unit means (fit on train only)
    P = Presence(ub, len(units))
    active = units[units.unit.isin(ub[ub.b * cc.bucket_s < split_t].unit.unique())].reset_index(drop=True)
    act = set(active.unit)

    def frame(Pf, origins, d, y, dh=None):
        f = window_features(Pf, origins, d, cc, delay_hat=dh)
        return f[f.unit.isin(act)].merge(y, on=["unit", "o"])

    y_tr = targets(P, tr_o, cc)
    unit_mean = y_tr[y_tr.unit.isin(act)].groupby("unit").y.mean()
    Dtr = add_model_features(frame(P, tr_o, np.zeros(len(tr_o), np.int64), y_tr), active, unit_mean, cc)
    if len(Dtr) > cc.max_train_rows:
        Dtr = Dtr.sample(cc.max_train_rows, random_state=0)
    t0 = time.perf_counter()
    model = train_model(Dtr, Dtr.y)
    train_s = time.perf_counter() - t0
    log(f"{task}: {len(active)} units, train {len(tr_o)} origins x units = {len(Dtr)} rows "
        f"({train_s:.1f}s); test {len(te_o)} origins")
    y_te = targets(P, te_o, cc)
    edges = np.linspace(te_o.min(), te_o.max() + 1, cc.n_windows + 1)
    win_o = bucket_window(te_o, te_o.min(), edges)
    dense_units = set(unit_mean[unit_mean >= cc.dense_min_mean].index)
    b_from = int(te_o.min() - L)
    P_mit_nom = Presence(mitigate_presence(ub, cc), len(units))
    frames, logs = [], []
    for seed in cc.seeds:
        delay = draw_delays(te_o, win_o, seed, cc)
        ub_lq, clog = corrupt_presence(ub, b_from, edges, seed, cc)
        logs.append(clog)
        P_lq = Presence(ub_lq, len(units))
        zero = np.zeros(len(te_o), np.int64)
        vol_nom, vol_lq = ub.groupby("b").n_msgs.sum(), ub_lq.groupby("b").n_msgs.sum()
        for cond, Pf, feed, vol_del, d in [("nominal", P, ub, vol_nom, zero), ("stale", P, ub, vol_nom, delay),
                                           ("low_quality", P_lq, ub_lq, vol_lq, zero),
                                           ("combined", P_lq, ub_lq, vol_lq, delay)]:
            t1 = time.perf_counter()
            Du = add_model_features(frame(Pf, te_o, d, y_te), active, unit_mean, cc)
            pu = predict(model, Du)
            t2 = time.perf_counter()
            Pm = P_mit_nom if feed is ub else Presence(mitigate_presence(feed, cc), len(units))
            dh = detect_delay(vol_del, te_o, d, cc)
            Dm = add_model_features(frame(Pm, te_o, d, y_te, dh), active, unit_mean, cc)
            dh_row = pd.Series(dh, index=te_o).reindex(Dm.o).to_numpy()
            pm_model = predict(model, Dm)
            pm = np.where(dh_row == 0, pm_model, Dm.h0_shift.to_numpy(float))
            ans = dh_row < cc.max_delay_defer_buckets
            action = np.where(~ans, "defer_late_feed", np.where(dh_row > 0, "fallback_last_complete_hour", "model"))
            t3 = time.perf_counter()
            d_row = pd.Series(d, index=te_o).reindex(Du.o).to_numpy()
            frames.append(pd.DataFrame({
                "condition": cond, "seed": seed, "unit": Du.unit.to_numpy(), "o": Du.o.to_numpy(),
                "window": pd.Series(win_o, index=te_o).reindex(Du.o).to_numpy(), "y": Du.y.to_numpy(),
                "pred_un": pu, "persistence": Du.h0.to_numpy(float),
                "rolling_mean": (Du.h0 + Du.h1 + Du.h2).to_numpy(float) / 3, "hist_mean": Du.unit_mean_train.to_numpy(),
                "pred_mi": np.where(ans, pm, np.nan), "answered": ans, "action": action,
                "h0_delivered": Du.h0.to_numpy(), "h0_mitigated": Dm.h0.to_numpy(),
                "msgs_h0_delivered": Du.msgs_h0.to_numpy(), "delay_buckets": d_row, "delay_detected": dh_row,
                "dense": Du.unit.isin(dense_units).to_numpy(), "t_un_s": (t2 - t1) / len(Du), "t_mi_s": (t3 - t2) / len(Du),
            }))
    cases = pd.concat(frames, ignore_index=True)
    nom = cases[(cases.condition == "nominal") & (cases.seed == cc.seeds[0])]
    pen = float(np.quantile(np.abs(nom.pred_un - nom.y), 0.9))
    metrics = summarize(cases, task, pen, cc)
    return {"cases": cases, "metrics": metrics, "units": active, "model": model, "corruption_logs": logs,
            "defer_penalty": pen, "n_units": len(active), "n_train_origins": len(tr_o), "n_test_origins": len(te_o),
            "n_train_rows": len(Dtr), "train_seconds": round(train_s, 2), "split_t": split_t,
            "test_window_edges_t": (edges * cc.bucket_s).tolist(), "n_dense_units": len(dense_units & set(active.unit)),
            "seconds": round(time.perf_counter() - t_all, 2)}


def summarize(cases: pd.DataFrame, task: str, pen: float, cc: CountConfig) -> pd.DataFrame:
    rows = []
    for (cond, seed), g0 in cases.groupby(["condition", "seed"]):
        for sub, gs in [("all", g0), ("dense", g0[g0.dense]), ("sparse", g0[~g0.dense])]:
            for w, g in list(gs.groupby("window")) + [("all", gs)]:
                if g.empty:
                    continue
                base = {"task": task, "condition": cond, "seed": seed, "subgroup": sub, "window": str(w),
                        "mean_delay_min": g.delay_buckets.mean() * cc.bucket_s / 60,
                        "stale_frac": (g.delay_buckets > 0).mean(),
                        "delay_detect_rate": ((g.delay_detected > 0) & (g.delay_buckets > 0)).sum() /
                        max((g.delay_buckets > 0).sum(), 1),
                        "input_msgs": g.msgs_h0_delivered.sum(),
                        "suspect_ids_removed": (g.h0_delivered - g.h0_mitigated).clip(lower=0).sum()}
                y = g.y.to_numpy()
                allans = np.ones(len(g), bool)
                for mit, pred, col, a in [("none", "hgb", "pred_un", allans),
                                          ("none", "persistence", "persistence", allans),
                                          ("none", "rolling_mean", "rolling_mean", allans),
                                          ("none", "hist_mean", "hist_mean", allans),
                                          ("quality_aware", "hgb", "pred_mi", g.answered.to_numpy())]:
                    r = dict(base, mitigation=mit, predictor=pred, **_metrics(y, g[col].to_numpy(), a, pen))
                    r["latency_ms_per_case"] = (g.t_mi_s.mean() if mit != "none" else g.t_un_s.mean()) * 1000
                    rows.append(r)
    return pd.DataFrame(rows)
