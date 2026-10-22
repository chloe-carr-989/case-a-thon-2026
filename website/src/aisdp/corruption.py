"""Controlled, seeded inference-time defects. Everything here is SYNTHETIC and logged.

* Staleness: the feed delivers a vessel's state late / frozen. The delivered state is the
  latest *real* report observed at or before (decision time - delay); its true observation
  timestamp is kept so report age can be computed. No future record is ever used.
* Low quality: a synthetic high-rate, low-accuracy source ("SYN-FASTFEED", not a real
  source label) supplies the latest record: position jitter, gross position outliers,
  speed/course corruption, sentinels and missing fields.
* Unseen holdout defect: a constant position bias from that source (never used when the
  countermeasure was designed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .geo import offset

SYN_SOURCE = "SYN-FASTFEED"


def draw(n: int, seed: int) -> dict:
    """All random numbers for one seed, drawn up front so every condition reuses the same draws."""
    rng = np.random.default_rng(seed)
    return {
        "u_stale": rng.random(n), "u_delay": rng.random(n), "u_lq": rng.random(n),
        "jit": rng.normal(size=(n, 4)), "u_out": rng.random(n), "out_dist": rng.random(n),
        "out_bearing": rng.random(n) * 2 * np.pi, "u_sog": rng.random(n), "sog_scale": rng.random(n),
        "u_cog": rng.random(n), "cog_rand": rng.random(n) * 360, "cog_noise": rng.normal(size=n),
    }


def stale_mask(window: np.ndarray, d: dict, cfg: ExperimentConfig) -> np.ndarray:
    p = np.asarray(cfg.stale_schedule)[window]
    return d["u_stale"] < p


def lowq_mask(window: np.ndarray, d: dict, cfg: ExperimentConfig) -> np.ndarray:
    p = np.asarray(cfg.lowq_schedule)[window]
    return d["u_lq"] < p


def stale_query_t(decision_t: np.ndarray, mask: np.ndarray, d: dict, cfg: ExperimentConfig) -> np.ndarray:
    delay = cfg.stale_delay_min_s + d["u_delay"] * (cfg.stale_delay_max_s - cfg.stale_delay_min_s)
    return np.where(mask, decision_t - delay, decision_t)


def corrupt(states: pd.DataFrame, mask: np.ndarray, d: dict, severity: float = 1.0,
            mode: str = "mixed") -> pd.DataFrame:
    """Return a copy of `states` where rows in `mask` come from the synthetic unreliable source."""
    s = states.copy()
    s["source"] = np.where(mask, SYN_SOURCE, "AIS-original")
    s["defects"] = ""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return s
    lat, lon = s["lat"].to_numpy(float).copy(), s["lon"].to_numpy(float).copy()
    plat, plon = s["prev_lat"].to_numpy(float).copy(), s["prev_lon"].to_numpy(float).copy()
    sog, cog = s["sog"].to_numpy(float).copy(), s["cog"].to_numpy(float).copy()
    hdg = s["heading"].to_numpy(float).copy()
    defects = np.array([""] * len(s), dtype=object)

    def add(sel, tag):
        defects[sel] = defects[sel] + tag + ";"

    m = np.zeros(len(s), bool)
    m[idx] = True
    if mode == "bias":
        bias_m = 1500.0 * severity
        b = np.radians(60.0)
        lat[m], lon[m] = offset(lat[m], lon[m], bias_m * np.sin(b), bias_m * np.cos(b))
        add(m, "source_bias_1.5km")
    else:
        sigma = 200.0 * severity
        lat[m], lon[m] = offset(lat[m], lon[m], sigma * d["jit"][m, 0], sigma * d["jit"][m, 1])
        plat[m], plon[m] = offset(plat[m], plon[m], sigma * d["jit"][m, 2], sigma * d["jit"][m, 3])
        add(m, "jitter")
        out = m & (d["u_out"] < 0.25 * min(1.0, severity))
        dist = (3000 + 12000 * d["out_dist"][out]) * severity
        lat[out], lon[out] = offset(lat[out], lon[out], dist * np.sin(d["out_bearing"][out]),
                                    dist * np.cos(d["out_bearing"][out]))
        add(out, "position_outlier")
        us = d["u_sog"]
        sc = m & (us < 0.25)
        sog[sc] = sog[sc] * (0.3 + 2.2 * d["sog_scale"][sc])
        add(sc, "sog_scaled")
        sn = m & (us >= 0.25) & (us < 0.40)
        sog[sn] = np.nan
        add(sn, "sog_missing")
        ss = m & (us >= 0.40) & (us < 0.50)
        sog[ss] = 102.3   # AIS "not available" sentinel passed through raw by the faulty source
        add(ss, "sog_sentinel")
        uc = d["u_cog"]
        cr = m & (uc < 0.30)
        cog[cr] = d["cog_rand"][cr]
        add(cr, "cog_random")
        cn = m & (uc >= 0.30) & (uc < 0.45)
        cog[cn] = np.nan
        add(cn, "cog_missing")
        ok = m & (uc >= 0.45)
        cog[ok] = (cog[ok] + 5 * d["cog_noise"][ok]) % 360
        hdg[m] = np.nan
    s["lat"], s["lon"], s["prev_lat"], s["prev_lon"] = lat, lon, plat, plon
    s["sog"], s["cog"], s["heading"] = sog, cog, hdg
    s["defects"] = defects
    return s
