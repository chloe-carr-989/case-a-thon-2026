"""End-to-end experiment: ingest -> features -> train (once) -> 4 conditions x 2 treatments x windows x seeds."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import corruption as C
from .config import ExperimentConfig
from .features import FEATURES, build_cases, build_states, load_reports, make_features
from .geo import enu_displacement, haversine_m
from .mitigation import mitigated_predict, unmitigated_predict
from .model import train

CONDITIONS = ["nominal", "stale", "low_quality", "combined", "holdout_unseen_bias"]
REQUIRED = CONDITIONS[:4]


class InsufficientData(RuntimeError):
    pass


def fit_model(con, cases: pd.DataFrame, cfg: ExperimentConfig, seed: int = 0):
    tr = cases[cases["split"] == "train"]
    if len(tr) > cfg.max_train_rows:
        tr = tr.sample(cfg.max_train_rows, random_state=seed)
    st = build_states(con, tr.assign(query_t=tr["decision_t"]), cfg)
    st = st.merge(tr[["case_id", "decision_t", "target_t", "target_lat", "target_lon"]], on="case_id")
    X = make_features(st, st["decision_t"].to_numpy())
    de, dn = enu_displacement(st["lat"], st["lon"], st["target_lat"], st["target_lon"])
    dt = (st["target_t"] - st["state_t"]).to_numpy(float)
    t0 = time.perf_counter()
    model = train(X, de / dt, dn / dt, seed=seed)
    return model, {"train_rows": len(X), "train_vessels": int(st["mmsi"].nunique()),
                   "train_seconds": round(time.perf_counter() - t0, 2), "features": FEATURES}


def _case_frame(cond, seed, te, s, un, mi, stale_m, lq_m, nominal_sog):
    tlat, tlon = te["target_lat"].to_numpy(), te["target_lon"].to_numpy()
    err_un = haversine_m(un["pred_lat"], un["pred_lon"], tlat, tlon) / 1000
    err_cv = haversine_m(un["cv_lat"], un["cv_lon"], tlat, tlon) / 1000
    err_mi = haversine_m(mi["pred_lat"], mi["pred_lon"], tlat, tlon) / 1000
    return pd.DataFrame({
        "condition": cond, "seed": seed, "case_id": te["case_id"].to_numpy(), "mmsi": te["mmsi"].to_numpy(),
        "window": te["window"].to_numpy(), "decision_t": te["decision_t"].to_numpy(),
        "target_t": te["target_t"].to_numpy(), "target_lat": tlat, "target_lon": tlon,
        "state_t": s["state_t"].to_numpy(), "age_s": te["decision_t"].to_numpy() - s["state_t"].to_numpy(),
        "in_lat": s["lat"].to_numpy(), "in_lon": s["lon"].to_numpy(), "in_sog": s["sog"].to_numpy(),
        "in_cog": s["cog"].to_numpy(), "source": s.get("source", pd.Series("AIS-original", index=s.index)).to_numpy(),
        "defects": s.get("defects", pd.Series("", index=s.index)).to_numpy(),
        "is_stale": stale_m, "is_lowq": lq_m, "nominal_sog": nominal_sog,
        "un_lat": un["pred_lat"], "un_lon": un["pred_lon"], "err_un_km": err_un, "err_cv_km": err_cv,
        "mi_lat": mi["pred_lat"], "mi_lon": mi["pred_lon"], "err_mi_km": err_mi,
        "answered": mi["answered"], "action": mi["action"], "quality": mi["quality"], "flags": mi["flags"],
        "implied_speed_kn": mi["implied_speed_kn"],
    })


def _summ(g: pd.DataFrame, err_col: str, answered: np.ndarray, defer_pen: float) -> dict:
    e = g[err_col].to_numpy()
    a = answered
    ea = e[a]
    return {
        "n_cases": len(g), "n_answered": int(a.sum()), "coverage": a.mean() if len(g) else np.nan,
        "deferral_rate": 1 - a.mean() if len(g) else np.nan,
        "mean_km": ea.mean() if len(ea) else np.nan, "median_km": np.median(ea) if len(ea) else np.nan,
        "p90_km": np.quantile(ea, 0.9) if len(ea) else np.nan,
        "policy_cost_km": np.where(a, e, defer_pen).mean() if len(g) else np.nan,
    }


def summarize(cases: pd.DataFrame, defer_pen: float, vol: dict, runtimes: dict) -> pd.DataFrame:
    rows = []
    for (cond, seed), g0 in cases.groupby(["condition", "seed"]):
        for sub, gs in [("all", g0), ("underway_ge2kn", g0[g0["nominal_sog"] >= 2])]:
            for w, g in list(gs.groupby("window")) + [("all", gs)]:
                if g.empty:
                    continue
                ans = g["answered"].to_numpy()
                fast = g["nominal_sog"].to_numpy() >= 20
                flagged = (g["flags"].str.len() > 0).to_numpy() | ~ans
                base = {"condition": cond, "seed": seed, "subgroup": sub, "window": str(w),
                        "is_holdout_window": w == cases["window"].max(),
                        "mean_age_min": g["age_s"].mean() / 60, "p90_age_min": g["age_s"].quantile(0.9) / 60,
                        "stale_frac": g["is_stale"].mean(), "lowq_frac": g["is_lowq"].mean(),
                        "interaction_frac": (g["is_stale"] & g["is_lowq"]).mean(),
                        "synthetic_share": (g["source"] == C.SYN_SOURCE).mean(),
                        "n_flagged": int(flagged.sum()), "n_fast_ge20kn": int(fast.sum()),
                        "n_fast_flagged": int((fast & flagged).sum())}
                rv = vol.get(str(w), {}) if sub == "all" else {}
                base["reliable_msgs"] = rv.get("reliable", np.nan)
                share = base["synthetic_share"]
                base["synthetic_msgs_simulated"] = (rv.get("reliable", np.nan) * share / (1 - share)
                                                    if share < 1 else np.nan)
                rt = runtimes.get((cond, seed), {})
                un_ans_common = g["err_un_km"].to_numpy()[ans]
                for mit, pred, col, a in [("none", "hgb", "err_un_km", np.ones(len(g), bool)),
                                          ("none", "cv_baseline", "err_cv_km", np.ones(len(g), bool)),
                                          ("quality_aware", "hgb", "err_mi_km", ans)]:
                    r = dict(base, mitigation=mit, predictor=pred, **_summ(g, col, a, defer_pen))
                    r["median_km_on_mitigated_answered_cases"] = (np.median(un_ans_common) if mit == "none" and
                                                                  pred == "hgb" and len(un_ans_common) else
                                                                  r["median_km"] if mit != "none" else np.nan)
                    r["runtime_s"] = rt.get(mit if pred == "hgb" else "cv", np.nan)
                    r["latency_ms_per_case"] = r["runtime_s"] * 1000 / max(rt.get("n", 1), 1)
                    rows.append(r)
    return pd.DataFrame(rows)


def window_volumes(con, te: pd.DataFrame, edges) -> dict:
    """Real AIS message counts from test vessels per window (the reliable source volume)."""
    con.register("tv", pd.DataFrame({"mmsi": te["mmsi"].unique()}))
    out = {}
    for w in range(len(edges) - 1):
        n = con.execute(f"SELECT count(*) FROM r SEMI JOIN tv USING (mmsi) WHERE t >= {edges[w]} AND t < {edges[w+1]}").fetchall()[0][0]
        out[str(w)] = {"reliable": n}
    out["all"] = {"reliable": sum(v["reliable"] for v in out.values())}
    con.unregister("tv")
    return out


def run(con, info: dict, cfg: ExperimentConfig, log=print) -> dict:
    t_all = time.perf_counter()
    rep = load_reports(con, info, cfg)
    log(f"reports: {rep}")
    cases = build_cases(con, cfg)
    if cases.empty or (cases["split"] == "train").sum() < 200 or (cases["split"] == "test").sum() < 100:
        n_tr = 0 if cases.empty else int((cases["split"] == "train").sum())
        n_te = 0 if cases.empty else int((cases["split"] == "test").sum())
        raise InsufficientData(f"insufficient target pairs for a {cfg.horizon_s}s horizon "
                               f"(train={n_tr}, test={n_te}; need >=200 / >=100)")
    model, tinfo = fit_model(con, cases, cfg)
    log(f"model: {tinfo}")
    te = cases[cases["split"] == "test"].reset_index(drop=True)
    win = te["window"].to_numpy()
    dec, tgt = te["decision_t"].to_numpy(float), te["target_t"].to_numpy(float)
    s_nom = build_states(con, te.assign(query_t=te["decision_t"]), cfg)
    nominal_sog = s_nom["sog"].to_numpy(float)
    vol = window_volumes(con, te, cases.attrs["window_edges"])
    frames, runtimes = [], {}
    for seed in cfg.seeds:
        d = C.draw(len(te), seed)
        sm, lm = C.stale_mask(win, d, cfg), C.lowq_mask(win, d, cfg)
        s_stale = build_states(con, te.assign(query_t=C.stale_query_t(dec, sm, d, cfg)), cfg)
        none = np.zeros(len(te), bool)
        variants = {
            "nominal": (C.corrupt(s_nom, none, d), none, none),
            "stale": (C.corrupt(s_stale, none, d), sm, none),
            "low_quality": (C.corrupt(s_nom, lm, d, cfg.severity), none, lm),
            "combined": (C.corrupt(s_stale, lm, d, cfg.severity), sm, lm),
            "holdout_unseen_bias": (C.corrupt(s_nom, lm, d, cfg.severity, mode="bias"), none, lm),
        }
        for cond, (s, smask, lmask) in variants.items():
            t0 = time.perf_counter()
            un = unmitigated_predict(s, dec, tgt, model)
            t1 = time.perf_counter()
            mi = mitigated_predict(s, dec, tgt, model, cfg)
            t2 = time.perf_counter()
            runtimes[(cond, seed)] = {"none": t1 - t0, "quality_aware": t2 - t1, "cv": np.nan, "n": len(te)}
            frames.append(_case_frame(cond, seed, te, s, un, mi, smask, lmask, nominal_sog))
    allc = pd.concat(frames, ignore_index=True)
    nom = allc[(allc["condition"] == "nominal") & (allc["seed"] == cfg.seeds[0])]
    defer_pen = float(np.quantile(nom["err_un_km"], 0.9))
    metrics = summarize(allc, defer_pen, vol, runtimes)
    return {"metrics": metrics, "cases": allc, "test_cases": te, "model_info": tinfo, "reports": rep,
            "defer_penalty_km": defer_pen, "n_test_cases": len(te), "n_test_vessels": int(te["mmsi"].nunique()),
            "n_train_cases": int((cases["split"] == "train").sum()), "split_t": cases.attrs["split_t"],
            "window_edges": cases.attrs["window_edges"], "volumes": vol,
            "seconds": round(time.perf_counter() - t_all, 2)}
