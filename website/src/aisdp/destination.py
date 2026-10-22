"""Where is this vessel going? Next inferred arrival zone from observations through the decision time.

AIS has no trustworthy destination labels here, so the label is constructed from the *future* part of
historical tracks: the first sustained dwell (>= 30 min below 1 kn) that starts after the decision,
within 6 h, mapped to an inferred arrival zone (a 0.05 deg grid cell where >= 3 distinct vessels dwelt
during the training period). Zones are coordinates, never port names, and a dwell is not a verified
berth call or cargo delivery. The label is only ever a target (y), never an input feature.

Model: candidate-zone ranking. For each decision, zones within reach are candidates; a
HistGradientBoostingClassifier scores (decision, zone) pairs from features at the decision time.
Baseline: heading-aligned nearest zone. Abstains ("unknown") when no zone is in reach or the top
normalised score is low.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import corruption as C
from .config import REGIONS, DestinationConfig, ExperimentConfig
from .features import build_states, is_test_vessel, load_reports
from .geo import haversine_m
from .mitigation import sanitize_state
from .zones import build_vessel_buckets, cell_ij, dwell_events, fit_zones

CAND_FEATURES = ["dist_km", "bdiff", "along_km", "cross_km", "eta_h", "dist_rank", "bdiff_rank", "log_zone_vessels",
                 "log_zone_arrivals", "is_last_zone", "dist_last_zone_km", "sog", "vessel_type", "n_candidates"]


class InsufficientDestination(RuntimeError):
    def __init__(self, msg: str, info: dict | None = None):
        super().__init__(msg)
        self.info = info or {}


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(x, y)) % 360


def assign_zone(lat, lon, zones: pd.DataFrame, deg: float) -> np.ndarray:
    zi, zj = cell_ij(lat, lon, deg)
    key = pd.Series(zones.zone_id.to_numpy(), index=pd.MultiIndex.from_arrays([zones.zi, zones.zj]))
    idx = pd.MultiIndex.from_arrays([zi, zj])
    return key.reindex(idx).to_numpy(dtype=object)


def region_of(lat, lon) -> np.ndarray:
    out = np.full(len(lat), "other", dtype=object)
    for name in ["gulf", "us_east", "us_west"]:     # gulf first: it overlaps the us_east box in Florida
        x0, y0, x1, y1 = REGIONS[name]
        m = (out == "other") & (np.asarray(lon) >= x0) & (np.asarray(lon) <= x1) & \
            (np.asarray(lat) >= y0) & (np.asarray(lat) <= y1)
        out[m] = name
    return out


def label_decisions(dec: pd.DataFrame, dw: pd.DataFrame, zones: pd.DataFrame, dc: DestinationConfig,
                    data_end: float) -> pd.DataFrame:
    """Next dwell starting at/after the decision, observable (confirmed) by data_end.

    status: arrival_zone (y = zone id) | arrival_unzoned | no_arrival_in_window | censored.
    """
    d = dec.sort_values(["decision_t", "mmsi"], kind="stable").copy()
    ev = dw[dw.confirm_t <= data_end].sort_values(["start_t", "mmsi"], kind="stable")[["mmsi", "start_t", "confirm_t", "lat", "lon"]]
    ev = ev.rename(columns={"lat": "arr_lat", "lon": "arr_lon"})
    m = pd.merge_asof(d, ev, left_on="decision_t", right_on="start_t", by="mmsi", direction="forward")
    within = m.start_t.notna() & (m.start_t - m.decision_t <= dc.window_s)
    zone = np.where(within, assign_zone(m.arr_lat.fillna(0), m.arr_lon.fillna(0), zones, dc.zone_deg), None)
    full_follow = m.decision_t + dc.window_s + dc.dwell_min_buckets * dc.bucket_s <= data_end
    m["status"] = np.select([within & pd.notna(zone), within, full_follow],
                            ["arrival_zone", "arrival_unzoned", "no_arrival_in_window"], "censored")
    m["true_zone"] = np.where(m.status == "arrival_zone", zone, None)
    for c in ["start_t", "confirm_t", "arr_lat", "arr_lon"]:
        m.loc[~within, c] = np.nan
    return m.rename(columns={"start_t": "arrival_t", "confirm_t": "arrival_confirm_t"})


def last_zone_before(dec: pd.DataFrame, dw: pd.DataFrame, zones: pd.DataFrame, dc: DestinationConfig) -> pd.Series:
    """Zone of the vessel's most recent dwell that had ENDED by the decision time (causal history)."""
    ev = dw.assign(zone=assign_zone(dw.lat, dw.lon, zones, dc.zone_deg)).dropna(subset=["zone"])
    ev = ev.sort_values(["end_t", "mmsi"], kind="stable")[["mmsi", "end_t", "zone"]]
    m = pd.merge_asof(dec[["case_id", "mmsi", "decision_t"]].sort_values(["decision_t", "case_id"], kind="stable"), ev, left_on="decision_t",
                      right_on="end_t", by="mmsi", direction="backward")
    return m.set_index("case_id").zone


def candidates(lat, lon, sog, cog, case_ids, last_zone, vtype, zones: pd.DataFrame, dc: DestinationConfig,
               chunk: int = 4000) -> pd.DataFrame:
    """(case, zone) candidate rows with features available at decision time."""
    zl, zo = zones.lat.to_numpy(), zones.lon.to_numpy()
    zid = zones.zone_id.to_numpy()
    zv = np.log1p(zones.train_vessels.to_numpy(float))
    za = np.log1p(zones.train_arrivals.to_numpy(float))
    zpos = {z: i for i, z in enumerate(zid)}
    out = []
    sog = np.asarray(sog, float)
    radius = np.clip(np.nan_to_num(sog, nan=0.0) * 1.852 * dc.window_s / 3600, dc.min_radius_km, dc.max_radius_km)
    for s0 in range(0, len(lat), chunk):
        sl = slice(s0, s0 + chunk)
        la, lo = np.asarray(lat, float)[sl, None], np.asarray(lon, float)[sl, None]
        dist = haversine_m(la, lo, zl[None, :], zo[None, :]) / 1000
        ok = np.isfinite(dist) & (dist <= radius[sl, None])
        dist_m = np.where(ok, dist, np.inf)
        k = min(dc.max_candidates, dist.shape[1])
        top = np.argpartition(dist_m, k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(dist.shape[0]), k)
        cols = top.reshape(-1)
        d = dist_m[rows, cols]
        keep = np.isfinite(d)
        rows, cols, d = rows[keep], cols[keep], d[keep]
        g = rows + s0
        brg = bearing_deg(np.asarray(lat, float)[g], np.asarray(lon, float)[g], zl[cols], zo[cols])
        c = np.asarray(cog, float)[g]
        bdiff = np.abs((brg - c + 180) % 360 - 180)
        lz = np.asarray(last_zone, dtype=object)[g]
        lzi = np.array([zpos.get(z, -1) if isinstance(z, str) else -1 for z in lz])
        dl = np.where(lzi >= 0, haversine_m(zl[cols], zo[cols], zl[np.maximum(lzi, 0)], zo[np.maximum(lzi, 0)]) / 1000,
                      np.nan)
        out.append(pd.DataFrame({
            "case_id": np.asarray(case_ids)[g], "zone_id": zid[cols], "dist_km": d, "bdiff": bdiff,
            "along_km": d * np.cos(np.radians(bdiff)), "cross_km": d * np.abs(np.sin(np.radians(bdiff))),
            "eta_h": np.minimum(d / np.maximum(np.nan_to_num(sog[g], nan=0.0) * 1.852, 0.1), 99.0),
            "log_zone_vessels": zv[cols], "log_zone_arrivals": za[cols], "is_last_zone": (lzi == cols).astype(float),
            "dist_last_zone_km": dl, "sog": sog[g], "vessel_type": np.asarray(vtype, float)[g]}))
    cand = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["case_id", "zone_id"] + CAND_FEATURES)
    if len(cand):
        grp = cand.groupby("case_id")
        cand["dist_rank"] = grp.dist_km.rank(method="first")
        cand["bdiff_rank"] = grp.bdiff.rank(method="first", na_option="bottom")
        cand["n_candidates"] = grp.zone_id.transform("size").astype(float)
    return cand


def rank_predictions(cand: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Top-3 zones per case with normalised scores (model scores, not calibrated probabilities)."""
    c = cand[["case_id", "zone_id", score_col]].copy()
    tot = c.groupby("case_id")[score_col].transform("sum")
    c["norm"] = np.where(tot > 0, c[score_col] / tot, 0)
    c = c.sort_values(["case_id", score_col, "zone_id"], ascending=[True, False, True], kind="stable")
    c["r"] = c.groupby("case_id").cumcount()
    top = c[c.r < 3].pivot(index="case_id", columns="r", values="zone_id")
    sc = c[c.r == 0].set_index("case_id").norm
    out = pd.DataFrame({"pred1": top.get(0), "pred2": top.get(1), "pred3": top.get(2)})
    out["top_score"] = sc
    return out


def baseline_score(cand: pd.DataFrame) -> np.ndarray:
    """Heading-aligned nearest zone: aligned (|bearing - COG| <= 30 deg) first, then by distance."""
    aligned = np.nan_to_num(cand.bdiff.to_numpy(), nan=180.0) <= 30
    return aligned * 1e6 + 1.0 / (1.0 + cand.dist_km.to_numpy())


def _score(model, cand):
    if cand.empty:
        return cand.assign(p=[], pb=[])
    return cand.assign(p=model.predict_proba(cand[CAND_FEATURES])[:, 1], pb=baseline_score(cand))


def run(con, info: dict, dc: DestinationConfig, ecfg: ExperimentConfig, log=print, port_zone_args=None) -> dict:
    t_all = time.perf_counter()
    nb = build_vessel_buckets(con, info, dc.bucket_s)
    dw = dwell_events(con, dc)
    t_first, t_last = con.execute("SELECT min(b), max(b) + 1 FROM vb").fetchall()[0]
    t_first, t_last = t_first * dc.bucket_s, t_last * dc.bucket_s
    split_t = t_first + dc.train_time_frac * (t_last - t_first)
    zones = fit_zones(dw, split_t, dc.zone_deg, dc.zone_min_vessels)
    port_zones = fit_zones(dw, split_t, **port_zone_args) if port_zone_args else None
    base = {"n_vessel_buckets": int(nb), "n_dwell_events": int(len(dw)), "n_arrival_dwells": int(dw.arrival.sum()) if len(dw) else 0,
            "n_zones": int(len(zones)), "split_t": split_t, "zones": zones, "port_zones": port_zones,
            "data_t": [t_first, t_last]}
    if len(zones) < 3:
        raise InsufficientDestination(f"only {len(zones)} inferred arrival zones from the training period "
                                      f"({len(dw)} dwell events); need >= 3", base)
    # decision anchors: underway buckets, at most one per vessel per anchor spacing
    dec = con.execute(f"""
        SELECT mmsi, min((b + 1) * {dc.bucket_s})::DOUBLE AS decision_t
        FROM vb WHERE sog >= {dc.underway_min_sog_kn}
        GROUP BY mmsi, floor(b * {dc.bucket_s} / {dc.anchor_spacing_s})
        ORDER BY mmsi, decision_t
    """).df()
    test_v = is_test_vessel(dec.mmsi.to_numpy(), ecfg)
    tr = dec[~test_v & (dec.decision_t < split_t)]
    te = dec[test_v & (dec.decision_t >= split_t)]
    tr = label_decisions(tr, dw, zones, dc, data_end=split_t)            # train labels only use data < split_t
    te = label_decisions(te, dw, zones, dc, data_end=t_last)
    tr["split"], te["split"] = "train", "test"
    allc = pd.concat([tr, te], ignore_index=True)
    allc = allc.sort_values(["split", "decision_t", "mmsi"], kind="stable").reset_index(drop=True)  # stable case ids
    allc["case_id"] = np.arange(len(allc))
    status_counts = {s: allc[allc.split == s].status.value_counts().to_dict() for s in ["train", "test"]}
    log(f"destination: {len(zones)} zones, {len(dw)} dwells; label status {status_counts}")
    lab = allc[allc.status == "arrival_zone"].reset_index(drop=True)
    n_tr, n_te = (lab.split == "train").sum(), (lab.split == "test").sum()
    n_cls = lab[lab.split == "train"].true_zone.nunique()
    base.update(status_counts=status_counts, n_train_labeled=int(n_tr), n_test_labeled=int(n_te), n_train_classes=int(n_cls))
    if n_tr < 200 or n_te < 30 or n_cls < 3:
        raise InsufficientDestination(f"insufficient labeled arrivals (train={n_tr}, test={n_te}, train zones={n_cls}; "
                                      "need >=200 / >=30 / >=3)", base)
    # states: latest real report <= decision (all labeled vessels; reports table r)
    load_reports(con, info, ExperimentConfig(sample_vessels=10 ** 9))
    lz = last_zone_before(lab, dw[dw.end_t <= t_last], zones, dc)
    lab["last_zone"] = lab.case_id.map(lz)
    s_nom = build_states(con, lab.assign(query_t=lab.decision_t), ecfg)
    lab = lab.merge(s_nom[["case_id", "state_t", "lat", "lon", "sog", "cog", "vessel_type"]], on="case_id")
    trl = lab[lab.split == "train"]
    ctr = candidates(trl.lat, trl.lon, trl.sog, trl.cog, trl.case_id, trl.last_zone, trl.vessel_type, zones, dc)
    ctr["y"] = (ctr.zone_id.to_numpy() == ctr.case_id.map(trl.set_index("case_id").true_zone).to_numpy()).astype(int)
    t0 = time.perf_counter()
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                                           l2_regularization=1.0, random_state=0).fit(ctr[CAND_FEATURES], ctr.y)
    train_s = time.perf_counter() - t0
    log(f"destination: trained on {len(trl)} cases / {len(ctr)} candidate rows ({train_s:.1f}s)")
    te = lab[lab.split == "test"]
    # a stale (delayed) state must exist: same >= 45 min history rule as the position task
    first = con.execute("SELECT mmsi, min(t) AS first_t FROM r GROUP BY mmsi").df().set_index("mmsi").first_t
    te = te[(te.decision_t - te.mmsi.map(first)) >= ecfg.min_history_s]
    te = te.sort_values(["decision_t", "mmsi"], kind="stable").reset_index(drop=True)   # seeded draws follow this order
    edges = np.linspace(te.decision_t.min(), te.decision_t.max() + 1e-6, dc.n_windows + 1)
    te["window"] = np.clip(np.searchsorted(edges, te.decision_t, side="right") - 1, 0, dc.n_windows - 1)
    te["region"] = region_of(te.lat, te.lon)
    zt = zones.set_index("zone_id").train_arrivals
    te["rare_zone"] = te.true_zone.map(zt).fillna(0).to_numpy() < 5
    win = te.window.to_numpy()
    dec_t = te.decision_t.to_numpy(float)
    q_nom = te[["case_id", "mmsi"]].assign(query_t=dec_t)
    s_nom = build_states(con, q_nom, ecfg)
    frames = []
    for seed in dc.seeds:
        d = C.draw(len(te), seed)
        sm, lm = C.stale_mask(win, d, ecfg), C.lowq_mask(win, d, ecfg)
        s_stale = build_states(con, q_nom.assign(query_t=C.stale_query_t(dec_t, sm, d, ecfg)), ecfg)
        none = np.zeros(len(te), bool)
        for cond, s, smask, lmask in [("nominal", C.corrupt(s_nom, none, d), none, none),
                                      ("stale", C.corrupt(s_stale, none, d), sm, none),
                                      ("low_quality", C.corrupt(s_nom, lm, d, ecfg.severity), none, lm),
                                      ("combined", C.corrupt(s_stale, lm, d, ecfg.severity), sm, lm)]:
            t1 = time.perf_counter()
            cu = _score(model, candidates(s.lat, s.lon, s.sog, s.cog, te.case_id, te.last_zone, s.vessel_type, zones, dc))
            ru, rb = rank_predictions(cu, "p"), rank_predictions(cu, "pb")
            t2 = time.perf_counter()
            z = sanitize_state(s, dec_t, ecfg)
            cm = _score(model, candidates(z["lat"], z["lon"], z["sog"], z["cog"], te.case_id, te.last_zone,
                                          s.vessel_type, zones, dc))
            rm = rank_predictions(cm, "p")
            t3 = time.perf_counter()
            f = te[["case_id", "mmsi", "decision_t", "window", "region", "rare_zone", "true_zone", "arrival_t",
                    "arr_lat", "arr_lon", "last_zone"]].copy()
            f["condition"], f["seed"] = cond, seed
            f["is_stale"], f["is_lowq"] = smask, lmask
            f["in_lat"], f["in_lon"], f["in_sog"], f["in_cog"] = s.lat.to_numpy(), s.lon.to_numpy(), s.sog.to_numpy(), s.cog.to_numpy()
            f["age_s"] = dec_t - s.state_t.to_numpy(float)
            f["source"], f["defects"] = s.source.to_numpy(), s.defects.to_numpy()
            for tag, r in [("un", ru), ("bl", rb), ("mi", rm)]:
                r = r.reindex(f.case_id)
                for k in ["pred1", "pred2", "pred3", "top_score"]:
                    f[f"{tag}_{k}"] = r[k].to_numpy() if k in r else np.nan
            f["un_answered"] = f.un_pred1.notna().to_numpy()
            f["bl_answered"] = f.bl_pred1.notna().to_numpy()
            f["mi_answered"] = f.mi_pred1.notna().to_numpy() & ~z["defer"] & \
                (np.nan_to_num(f.mi_top_score.to_numpy(float)) >= dc.abstain_score)
            f["mi_action"] = np.where(z["defer"], "defer_quality", np.where(f.mi_pred1.isna(), "abstain_no_zone_in_reach",
                                      np.where(~f.mi_answered, "abstain_low_score", "model")))
            f["mi_flags"] = z["flags"]
            f["cand_recall"] = [tz in set(g) for tz, g in zip(f.true_zone, cu.groupby("case_id").zone_id.apply(list)
                                                              .reindex(f.case_id).apply(lambda v: v if isinstance(v, list) else []))]
            f["t_un_ms"], f["t_mi_ms"] = (t2 - t1) * 1000 / len(f), (t3 - t2) * 1000 / len(f)
            frames.append(f)
    cases = pd.concat(frames, ignore_index=True)
    metrics = summarize(cases, zones)
    conf = confusion(cases[(cases.condition == "nominal") & (cases.seed == dc.seeds[0])])
    base.update(cases=cases, metrics=metrics, confusion=conf, model=model, n_train_candidate_rows=int(len(ctr)),
                train_seconds=round(train_s, 2), n_test_scored=int(len(te)), n_test_vessels=int(te.mmsi.nunique()),
                n_train_vessels=int(trl.mmsi.nunique()), n_test_classes=int(te.true_zone.nunique()),
                test_window_edges_t=edges.tolist(), seconds=round(time.perf_counter() - t_all, 2))
    return base


def _acc(g: pd.DataFrame, tag: str, ans_col: str) -> dict:
    ans = g[ans_col].to_numpy(bool)
    t = g.true_zone.to_numpy(dtype=object)
    p1 = g[f"{tag}_pred1"].to_numpy(dtype=object)
    hit1 = ans & (p1 == t)
    hit3 = ans & ((p1 == t) | (g[f"{tag}_pred2"].to_numpy(dtype=object) == t) |
                  (g[f"{tag}_pred3"].to_numpy(dtype=object) == t))
    n, na = len(g), int(ans.sum())
    return {"n_cases": n, "n_answered": na, "coverage": na / n if n else np.nan, "deferral_rate": 1 - na / n if n else np.nan,
            "top1_acc": hit1.sum() / na if na else np.nan, "top3_acc": hit3.sum() / na if na else np.nan,
            "top1_over_all_cases": hit1.sum() / n if n else np.nan, "top3_over_all_cases": hit3.sum() / n if n else np.nan,
            "n_classes": int(pd.Series(t).nunique())}


def summarize(cases: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, seed), g0 in cases.groupby(["condition", "seed"]):
        subs = [("all", g0), ("rare_zone_lt5_train_arrivals", g0[g0.rare_zone]), ("common_zone", g0[~g0.rare_zone])]
        subs += [(f"region_{r}", g) for r, g in g0.groupby("region")]
        for sub, gs in subs:
            for w, g in list(gs.groupby("window")) + [("all", gs)]:
                if g.empty:
                    continue
                base = {"task": "destination", "condition": cond, "seed": seed, "subgroup": sub, "window": str(w),
                        "mean_age_min": g.age_s.mean() / 60, "stale_frac": g.is_stale.mean(), "lowq_frac": g.is_lowq.mean(),
                        "synthetic_share": (g.source != "AIS-original").mean(), "candidate_recall": g.cand_recall.mean()}
                for mit, pred, tag, ac, tcol in [("none", "hgb_ranker", "un", "un_answered", "t_un_ms"),
                                                 ("none", "heading_nearest_baseline", "bl", "bl_answered", "t_un_ms"),
                                                 ("quality_aware", "hgb_ranker", "mi", "mi_answered", "t_mi_ms")]:
                    rows.append(dict(base, mitigation=mit, predictor=pred, **_acc(g, tag, ac),
                                     latency_ms_per_case=g[tcol].mean()))
    return pd.DataFrame(rows)


def confusion(g: pd.DataFrame, min_support: int = 5) -> pd.DataFrame:
    """Long-format confusion (true zone x predicted zone) for zones with >= min_support test cases."""
    sup = g.true_zone.value_counts()
    keep = sup[sup >= min_support].index
    h = g[g.true_zone.isin(keep)]
    pred = h.un_pred1.where(h.un_answered, "UNKNOWN").where(h.un_pred1.isin(keep) | ~h.un_answered, "OTHER_ZONE")
    out = pd.DataFrame({"true_zone": h.true_zone, "pred_zone": pred}).value_counts().rename("n").reset_index()
    out["support"] = out.true_zone.map(sup)
    return out
