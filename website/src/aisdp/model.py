"""Trained predictor (HistGradientBoosting) and the non-ML constant-velocity benchmark."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .geo import offset, sog_cog_to_ve_vn


def base_velocity(X):
    """Physics prior the model corrects: reported SOG/COG, else observed track velocity, else 0."""
    ve = np.where(np.isfinite(X["ve_rep"]), X["ve_rep"], np.where(np.isfinite(X["ve_obs"]), X["ve_obs"], 0.0))
    vn = np.where(np.isfinite(X["vn_rep"]), X["vn_rep"], np.where(np.isfinite(X["vn_obs"]), X["vn_obs"], 0.0))
    return ve, vn


def train(X, y_e, y_n, seed: int = 0) -> dict:
    """Learns the residual between the mean velocity over the horizon and the physics prior."""
    params = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, random_state=seed)
    be, bn = base_velocity(X)
    return {"ve": HistGradientBoostingRegressor(loss="absolute_error", **params).fit(X, np.asarray(y_e) - be),
            "vn": HistGradientBoostingRegressor(loss="absolute_error", **params).fit(X, np.asarray(y_n) - bn)}


def predict_velocity(model: dict, X):
    be, bn = base_velocity(X)
    return be + model["ve"].predict(X), bn + model["vn"].predict(X)


def project(lat, lon, ve, vn, dt):
    return offset(lat, lon, np.asarray(ve) * dt, np.asarray(vn) * dt)


def cv_baseline(sog, cog):
    """Constant velocity from reported SOG/COG; missing kinematics -> assume stationary."""
    ve, vn = sog_cog_to_ve_vn(sog, cog)
    return np.nan_to_num(ve), np.nan_to_num(vn)
