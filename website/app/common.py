"""Shared helpers for the dashboard pages: result loading, study areas, review logging, summary tables.

All numbers shown in the app come from files written by scripts/run_pipeline.py (results/).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LOG = RES / "review_log.csv"

CONDITIONS = {"nominal": "Clean feed", "stale": "Stale reports", "low_quality": "Unreliable source",
              "combined": "Stale + unreliable"}
# (lon_min, lat_min, lon_max, lat_max); display filters only, not port boundaries
AREAS = {
    "Houston / Galveston": (-95.6, 28.9, -94.3, 30.0),
    "Lower Mississippi": (-91.5, 29.0, -89.0, 30.7),
    "New York Harbor": (-74.4, 40.4, -73.7, 40.95),
    "Chesapeake Bay": (-77.0, 36.7, -75.8, 39.4),
    "Puget Sound": (-123.2, 47.0, -122.2, 48.5),
    "San Francisco Bay": (-122.7, 37.4, -122.0, 38.2),
    "Whole U.S.": None,
}


@st.cache_data
def manifest() -> dict:
    p = RES / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def metrics(task: str) -> pd.DataFrame:
    p = RES / "metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    m = pd.read_csv(p, low_memory=False)
    if "task" not in m.columns:          # results written before the expansion: position only
        m["task"] = "position_15m"
    m["task"] = m["task"].fillna("position_15m")
    return m[m.task == task].reset_index(drop=True)


@st.cache_data
def table(name: str) -> pd.DataFrame | None:
    p = RES / name
    if not p.exists():
        return None
    return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)


def task_status(task: str) -> dict:
    return manifest().get("tasks", {}).get(task, {}).get("run") or {}


def unavailable(task: str, what: str):
    """Explicit, evidence-backed unavailability state instead of a made-up answer."""
    run = task_status(task)
    st.warning(f"{what} is not available for the default subset ({manifest().get('default_demo_subset', '?')}).")
    if run:
        st.write(f"Status: **{run.get('status')}**. {run.get('reason', '')}")
    st.caption("Run `.venv\\Scripts\\python scripts\\run_pipeline.py` to compute it. If the data cannot support "
               "the task, the pipeline records why instead of producing a result.")
    st.stop()


def hhmm(t) -> str:
    return pd.to_datetime(t, unit="s").strftime("%H:%M")


def in_area(lat, lon, area: str) -> np.ndarray:
    box = AREAS.get(area)
    if box is None:
        return np.ones(len(lat), bool)
    x0, y0, x1, y1 = box
    return (np.asarray(lon) >= x0) & (np.asarray(lon) <= x1) & (np.asarray(lat) >= y0) & (np.asarray(lat) <= y1)


def geo_layout(fig: go.Figure, area: str, height: int = 560, bounds=None):
    if bounds is not None:
        y0, y1, x0, x1 = bounds
        fig.update_geos(lataxis_range=[y0, y1], lonaxis_range=[x0, x1])
    elif AREAS.get(area):
        x0, y0, x1, y1 = AREAS[area]
        fig.update_geos(lataxis_range=[y0, y1], lonaxis_range=[x0, x1])
    else:
        fig.update_geos(fitbounds="locations")
    fig.update_geos(projection_type="mercator", showland=True, landcolor="#efece6", showocean=True,
                    oceancolor="#d7e6f2", showlakes=True, lakecolor="#d7e6f2", showrivers=True,
                    showcountries=True, showsubunits=True, resolution=50)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select",
                      legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=11)))


def log_review(row: dict):
    """Append a reviewer action. Older logs (position-only columns) are widened rather than misaligned."""
    row = {"logged_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
    new = pd.DataFrame([row])
    if LOG.exists():   # small file: rewrite with the union of columns so rows never misalign
        new = pd.concat([pd.read_csv(LOG), new], ignore_index=True)
    new.to_csv(LOG, index=False)


def review_form(key: str, task: str, case: dict, options: list[str], t0: float, correction_label: str | None = None):
    """Accept / correct / defer with a mandatory reason. Records measured seconds since the case opened."""
    import time
    with st.form(f"review_{key}", clear_on_submit=True, border=False):
        act = st.radio("Decision", options, label_visibility="collapsed", key=f"act_{key}")
        corr = st.text_input(correction_label, key=f"corr_{key}") if correction_label else ""
        reason = st.text_input("Reason", placeholder="e.g. feed 30 min late", key=f"reason_{key}")
        if st.form_submit_button("Record decision"):
            if not reason.strip():
                st.error("Add a reason first.")
            else:
                log_review({"task": task, **case, "operator_action": act, "correction": corr, "reason": reason,
                            "review_seconds_measured": round(time.time() - t0, 1)})
                st.success("Logged.")


def eight_cell(m: pd.DataFrame, value_cols: dict, predictor: str, extra_none: dict | None = None) -> pd.DataFrame:
    """4 conditions x mitigation off/on (mean over seeds), on identical cases, for one predictor."""
    g = m[(m.window.astype(str) == "all") & (m.subgroup == "all")]
    rows = []
    for c, lab in CONDITIONS.items():
        for mit, ml, pred in [("none", "off", predictor), ("quality_aware", "on", predictor)]:
            s = g[(g.condition == c) & (g.mitigation == mit) & (g.predictor == pred)]
            if s.empty:
                continue
            r = {"feed condition": lab, "countermeasure": ml}
            for col, name in value_cols.items():
                r[name] = s[col].mean()
            rows.append(r)
        for pred, lab2 in (extra_none or {}).items():
            s = g[(g.condition == c) & (g.mitigation == "none") & (g.predictor == pred)]
            if len(s):
                r = {"feed condition": lab, "countermeasure": f"off ({lab2})"}
                for col, name in value_cols.items():
                    r[name] = s[col].mean()
                rows.append(r)
    return pd.DataFrame(rows)
