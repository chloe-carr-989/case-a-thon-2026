"""Geodesy helpers (spherical earth; adequate for <100 km displacements)."""
from __future__ import annotations

import numpy as np

R_EARTH = 6_371_008.8
KN_TO_MS = 0.514444


def wrap_lon(lon):
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def enu_displacement(lat1, lon1, lat2, lon2):
    """East/north displacement in metres from point 1 to point 2 (handles the antimeridian)."""
    dlon = wrap_lon(np.asarray(lon2) - np.asarray(lon1))
    mlat = np.radians((np.asarray(lat1) + np.asarray(lat2)) / 2)
    de = np.radians(dlon) * R_EARTH * np.cos(mlat)
    dn = np.radians(np.asarray(lat2) - np.asarray(lat1)) * R_EARTH
    return de, dn


def offset(lat, lon, de, dn):
    """Move a point by east/north metres."""
    lat = np.asarray(lat, dtype=float)
    lat2 = lat + np.degrees(np.asarray(dn) / R_EARTH)
    coslat = np.maximum(np.cos(np.radians(lat)), 1e-6)
    lon2 = wrap_lon(np.asarray(lon) + np.degrees(np.asarray(de) / (R_EARTH * coslat)))
    return lat2, lon2


def sog_cog_to_ve_vn(sog_kn, cog_deg):
    v = np.asarray(sog_kn, dtype=float) * KN_TO_MS
    c = np.radians(np.asarray(cog_deg, dtype=float))
    return v * np.sin(c), v * np.cos(c)
