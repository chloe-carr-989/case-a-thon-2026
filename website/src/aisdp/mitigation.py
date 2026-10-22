"""Quality-aware inference pipeline (the countermeasure). Thresholds are fixed a priori.

Steps: (1) age gate + age compensation, (2) AIS validity checks (sentinels, implausible
SOG/COG), (3) track-consistency checks (implied speed between fixes, reported vs. observed
velocity), (4) robust imputation of bad kinematics from the observed track, (5) fallback to
the last credible fix, (6) quality score with explicit deferral to a human reviewer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .features import make_features
from .geo import KN_TO_MS, haversine_m, sog_cog_to_ve_vn
from .model import cv_baseline, predict_velocity, project


def mitigated_predict(s: pd.DataFrame, decision_t, target_t, model: dict, cfg: ExperimentConfig) -> dict:
    n = len(s)
    decision_t = np.asarray(decision_t, float)
    target_t = np.asarray(target_t, float)
    age = decision_t - s["state_t"].to_numpy(float)
    sog = s["sog"].to_numpy(float).copy()
    cog = s["cog"].to_numpy(float).copy()
    prev_sog = s["prev_sog"].to_numpy(float).copy()
    prev_cog = s["prev_cog"].to_numpy(float).copy()

    bad_sog = ~np.isfinite(sog) | (sog > cfg.max_valid_sog_kn)
    bad_cog = ~np.isfinite(cog) | (cog >= 360)
    sog[bad_sog] = np.nan
    cog[bad_cog] = np.nan
    prev_sog[~np.isfinite(prev_sog) | (prev_sog > cfg.max_valid_sog_kn)] = np.nan
    prev_cog[~np.isfinite(prev_cog) | (prev_cog >= 360)] = np.nan

    has_prev = np.isfinite(s["prev_t"].to_numpy(float))
    dt_prev = s["state_t"].to_numpy(float) - s["prev_t"].to_numpy(float)
    dist = haversine_m(s["prev_lat"].to_numpy(float), s["prev_lon"].to_numpy(float),
                       s["lat"].to_numpy(float), s["lon"].to_numpy(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        implied = dist / dt_prev
    rep_v = np.fmax(sog, prev_sog) * KN_TO_MS
    limit = np.where(np.isfinite(rep_v), np.maximum(2.0 * rep_v + 2.5, 2.5), cfg.max_valid_sog_kn * KN_TO_MS)
    pos_flag = has_prev & (implied > limit)

    clean = s.copy()
    clean["sog"], clean["cog"], clean["prev_sog"], clean["prev_cog"] = sog, cog, prev_sog, prev_cog
    X = make_features(clean, decision_t)
    ve_rep, vn_rep = X["ve_rep"].to_numpy(), X["vn_rep"].to_numpy()
    ve_obs, vn_obs = X["ve_obs"].to_numpy(), X["vn_obs"].to_numpy()
    vdiff = np.hypot(ve_rep - ve_obs, vn_rep - vn_obs)
    kin_flag = has_prev & ~pos_flag & np.isfinite(vdiff) & (vdiff > np.maximum(3.0, 0.5 * np.hypot(ve_rep, vn_rep)))
    missing_kin = bad_sog | bad_cog
    # Robust imputation: replace inconsistent or missing reported kinematics by the observed track.
    use_obs = (kin_flag | missing_kin) & has_prev & ~pos_flag
    sp = np.hypot(ve_obs, vn_obs)
    X.loc[use_obs, "ve_rep"] = ve_obs[use_obs]
    X.loc[use_obs, "vn_rep"] = vn_obs[use_obs]
    X.loc[use_obs, "sog"] = sp[use_obs] / KN_TO_MS
    X.loc[use_obs, "cog_sin"] = np.where(sp[use_obs] > 0, ve_obs[use_obs] / np.maximum(sp[use_obs], 1e-9), 0)
    X.loc[use_obs, "cog_cos"] = np.where(sp[use_obs] > 0, vn_obs[use_obs] / np.maximum(sp[use_obs], 1e-9), 0)

    ve, vn = predict_velocity(model, X)
    dt_eff = target_t - s["state_t"].to_numpy(float)          # age-compensated horizon
    plat, plon = project(s["lat"].to_numpy(float), s["lon"].to_numpy(float), ve, vn, dt_eff)

    # Fallback: last credible fix (previous report) dead-reckoned with its own SOG/COG.
    fb_ok = pos_flag & np.isfinite(prev_sog) & np.isfinite(prev_cog)
    fve, fvn = sog_cog_to_ve_vn(prev_sog, prev_cog)
    fdt = target_t - s["prev_t"].to_numpy(float)
    flat, flon = project(s["prev_lat"].to_numpy(float), s["prev_lon"].to_numpy(float),
                         np.nan_to_num(fve), np.nan_to_num(fvn), np.nan_to_num(fdt))
    plat = np.where(fb_ok, flat, plat)
    plon = np.where(fb_ok, flon, plon)

    quality = (1.0 - 0.5 * np.clip(age / cfg.max_age_defer_s, 0, 1) - 0.35 * pos_flag - 0.2 * kin_flag
               - 0.15 * (missing_kin & ~use_obs) - 0.1 * ~has_prev)
    defer = (age > cfg.max_age_defer_s) | (pos_flag & ~fb_ok) | (quality < cfg.defer_quality) | \
        (missing_kin & ~has_prev)
    action = np.where(defer, "defer", np.where(fb_ok, "fallback_last_credible",
                      np.where(age > cfg.age_comp_s, "age_compensated", "model")))
    flags = np.array([""] * n, dtype=object)
    for m, tag in [(age > cfg.age_comp_s, "stale_report"), (age > cfg.max_age_defer_s, "too_old"),
                   (pos_flag, "implausible_jump"), (kin_flag, "sog_cog_inconsistent"),
                   (missing_kin, "invalid_or_missing_sog_cog"), (~has_prev, "no_recent_track")]:
        flags[m] = flags[m] + tag + ";"
    return {"pred_lat": np.where(defer, np.nan, plat), "pred_lon": np.where(defer, np.nan, plon),
            "answered": ~defer, "action": action, "quality": quality, "flags": flags,
            "implied_speed_kn": implied / KN_TO_MS}


def unmitigated_predict(s: pd.DataFrame, decision_t, target_t, model: dict) -> dict:
    """Naive pipeline: takes the delivered record at face value and assumes it is current."""
    decision_t = np.asarray(decision_t, float)
    X = make_features(s, decision_t)
    ve, vn = predict_velocity(model, X)
    dt = np.asarray(target_t, float) - decision_t
    plat, plon = project(s["lat"].to_numpy(float), s["lon"].to_numpy(float), ve, vn, dt)
    cve, cvn = cv_baseline(s["sog"].to_numpy(float), s["cog"].to_numpy(float))
    clat, clon = project(s["lat"].to_numpy(float), s["lon"].to_numpy(float), cve, cvn, dt)
    return {"pred_lat": plat, "pred_lon": plon, "cv_lat": clat, "cv_lon": clon}
