"""Shared interface: top bar for every page; data access, toolbar, poisoning card and formatting for the two modes.

Everything shown comes from files written by scripts/run_pipeline.py (results/). Model computation lives in
src/aisdp; the only model call made here is drawing an expected route on the already-learned lane graph.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from aisdp.lanes import CATEGORY_LABEL, LaneModel  # noqa: E402

from common import AREAS, CONDITIONS, manifest  # noqa: E402

C = {"bg": "#0c0f12", "water": "#0b0f14", "land": "#1a1f25", "panel": "#12161b", "line": "#262d36",
     "line2": "#333c47", "fg": "#d6dce4", "dim": "#8a95a3", "faint": "#56606d", "accent": "#49b6dc",
     "warn": "#e9b44c", "crit": "#e2574c"}
CAT_COLOR = {"cargo": "#4fb3f0", "tanker": "#f0a94f", "passenger": "#c07be0", "tug_service": "#6fd0c5",
             "fishing": "#a8d46f", "government": "#f2e66d", "unclassified": "#8a95a3", "other": "#8a95a3"}
LABEL_COLOR = {"normal": "#5f6b78", "unusual": C["warn"], "high": C["crit"], "no_verdict": C["dim"], "no_data": C["dim"]}
FOCUS_DEFAULT = ["cargo", "tanker", "passenger", "tug_service", "government", "unclassified", "other"]
TRAIL_SAMPLE_S = 300     # trail resolution when zoomed in (30 min when zoomed out)

CSS = f"""
<style>
:root {{--livemap-chrome: 146px;}}
.block-container {{padding: .5rem 1.25rem .5rem; max-width: 100%;}}
[data-testid="stHeader"] {{display: none;}}
/* top bar: title on the left, modes and evaluation pages on the right */
.st-key-topbar {{border-bottom: 1px solid {C['line']}; padding: 2px 0 8px; margin-bottom: 2px; min-height: 52px;}}
.st-key-topbar [data-testid="stPageLink"] a {{padding: 7px 16px; border-radius: 3px; margin: 0;}}
.st-key-topbar [data-testid="stPageLink"] p {{font: 500 15px 'IBM Plex Sans', sans-serif; color: {C['dim']};
           white-space: nowrap;}}
.st-key-topbar [class*="st-key-navon"] a {{background: {C['panel']}; box-shadow: inset 0 -2px 0 {C['accent']};}}
.st-key-topbar [class*="st-key-navon"] p {{color: {C['fg']};}}
.st-key-navtabs {{border: 1px solid {C['line2']}; border-radius: 4px; padding: 2px;}}
[data-baseweb="tab"] p {{font-size: 15px !important;}}
.brand {{font: 600 19px 'IBM Plex Sans', sans-serif; letter-spacing: .01em; color: {C['fg']}; white-space: nowrap;}}
.brand span {{color: {C['accent']};}}
/* the data-poisoning controls: their own group, amber while a degraded feed is on */
[class*="st-key-poison"] {{border: 1px solid {C['line2']}; border-radius: 4px; padding: 3px 10px;}}
.st-key-poison_on {{border-color: {C['warn']}; background: rgba(233, 180, 76, .06);}}
.ptag {{font: 500 12px 'IBM Plex Mono', monospace; letter-spacing: .06em; text-transform: uppercase;
        color: {C['warn']}; white-space: nowrap;}}
h1, h2, h3, h4 {{letter-spacing: .005em;}}
.eyebrow {{font: 500 12px 'IBM Plex Mono', monospace; letter-spacing: .06em; text-transform: uppercase;
          color: {C['dim']}; margin: 0 0 4px;}}
.bar {{display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 18px; padding: 0 0 8px;
       border-bottom: 1px solid {C['line']}; margin-bottom: 4px;}}
.bar b {{font-weight: 600; font-size: 16px; letter-spacing: .01em; color: {C['fg']}}}
.bar span {{font: 13px 'IBM Plex Mono', monospace; color: {C['dim']}}}
.bar span i {{font-style: normal; color: {C['fg']}}}
.kv {{display: grid; grid-template-columns: auto 1fr; gap: 3px 14px; font: 13px 'IBM Plex Mono', monospace;
      margin: 4px 0 10px; font-variant-numeric: tabular-nums;}}
.kv dt {{color: {C['dim']};}} .kv dd {{margin: 0; color: {C['fg']}; overflow-wrap: anywhere;}}
.big {{font: 500 30px/1.1 'IBM Plex Sans', sans-serif; color: {C['fg']}; font-variant-numeric: tabular-nums;}}
.big small {{font-size: 14px; color: {C['dim']}; font-weight: 400; margin-left: 6px}}
.sub {{font: 13px 'IBM Plex Mono', monospace; color: {C['dim']}; margin-top: 2px}}
.chip {{display: inline-block; padding: 1px 6px; border: 1px solid var(--c); color: var(--c); border-radius: 2px;
        font: 500 12px 'IBM Plex Mono', monospace; letter-spacing: .04em; text-transform: uppercase; margin-right: 6px}}
.why {{margin: 2px 0 0; padding-left: 16px; font-size: 14px}} .why li {{margin: 2px 0}}
.note {{font: 12px 'IBM Plex Mono', monospace; color: {C['dim']}; line-height: 1.5}}
[data-testid="stMetricLabel"] p {{font: 500 12px 'IBM Plex Mono', monospace !important; letter-spacing: .06em;
           text-transform: uppercase; color: {C['dim']}}}
[data-testid="stVerticalBlockBorderWrapper"] {{background: {C['panel']};}}
/* no stale-element fade or "running" indicator when the side panel follows the playing map */
[data-stale="true"], .stale-element, [data-testid="stElementContainer"] {{opacity: 1 !important; transition: none !important;}}
[data-testid="stStatusWidget"], [data-testid="stSkeleton"] {{display: none !important;}}
/* one screen, no page scroll: map and side panel take the height the viewport leaves */
.block-container [data-testid="stVerticalBlock"] {{gap: .45rem;}}
.st-key-side_box {{height: max(420px, calc(100dvh - var(--livemap-chrome))); overflow-y: auto; overflow-x: hidden;
                    padding-right: 4px; flex-wrap: nowrap; flex: none !important;}}
.st-key-side_box > * {{flex-shrink: 0 !important;}}
.st-key-side_box [data-testid="stMarkdownContainer"] {{margin-bottom: 0 !important;}}
.st-key-side_box > [class*="_queue"] {{flex: 1 1 0 !important; min-height: 120px;}}
.st-key-side_box > [class*="_queue"] > div {{height: 100% !important;}}
@media (max-width: 640px) {{ .block-container {{padding: .5rem 1rem 1rem}} .big {{font-size: 24px}} .brand {{display: none}}
  .st-key-side_box {{height: auto; overflow: visible}} }}
</style>
"""


def style():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))


def kv(rows: list[tuple[str, str]]) -> str:
    return "<dl class='kv'>" + "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def reasons_list(s) -> list[str]:
    return [x for x in str(s).split(";") if x and x != "nan"]


def hhmm(t) -> str:
    return pd.to_datetime(t, unit="s").strftime("%H:%M")


def stamp(t) -> str:
    return pd.to_datetime(t, unit="s").strftime("%b %d %H:%M UTC")


def dur(h) -> str:
    if h is None or not np.isfinite(h):
        return "n/a"
    m = int(round(h * 60))
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def km(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else (f"{x * 1000:.0f} m" if x < 1 else f"{x:.1f} km")


def mid_flag(mmsi) -> str:
    """MMSI country prefix (MID). Claimed by the transmitter, never verified here."""
    s = str(int(mmsi))
    return f"MID {s[:3]} (from MMSI; claimed, unverified)" if s[0] in "234567" else "no MID prefix (not a ship station)"


# ------------------------------------------------------------------------------------------- persisted state
def keep(key: str, default):
    """Seed a widget from a page-independent value so it survives switching modes."""
    if key not in st.session_state:
        st.session_state[key] = default
    st.session_state[f"w_{key}"] = st.session_state[key]


def sync(key: str):
    st.session_state[key] = st.session_state[f"w_{key}"]


# ------------------------------------------------------------------------------------------- data
def available(name: str) -> bool:
    return (RES / name).exists()


@st.cache_data
def run_info() -> dict:
    m = manifest()
    p = RES / "runs" / m.get("default_demo_subset", "") / "run.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource
def lanes() -> LaneModel | None:
    return LaneModel.load(RES) if available("lanes_cells.parquet") else None


@st.cache_data
def lane_cells() -> pd.DataFrame:
    return pd.read_parquet(RES / "lanes_cells.parquet")


@st.cache_data
def vessels() -> pd.DataFrame:
    if not available("vessels.parquet"):
        return pd.DataFrame(columns=["name", "imo", "call_sign", "vessel_type", "length", "draft"])
    return pd.read_parquet(RES / "vessels.parquet").set_index("mmsi")


@st.cache_data
def windows(cond: str, mit: str) -> pd.DataFrame:
    return pd.read_parquet(RES / "anomaly_windows.parquet",
                           filters=[("condition", "==", cond), ("mitigation", "==", mit)])


@st.cache_data
def eta_cases() -> pd.DataFrame:
    d = pd.read_parquet(RES / "destination_cases_seed0.parquet")
    return d if "un_eta_h" in d.columns else pd.DataFrame()


@st.cache_data
def zones() -> pd.DataFrame:
    return pd.read_csv(RES / "destination_zones.csv").set_index("zone_id")


@st.cache_data
def weather() -> pd.DataFrame | None:
    return pd.read_parquet(RES / "weather_hourly.parquet") if available("weather_hourly.parquet") else None


@st.cache_data(max_entries=128)
def track(mmsi: int, t0: float, t1: float) -> pd.DataFrame:
    """Real AIS reports of one vessel between t0 and t1 (never later than the timeline time)."""
    d = run_info().get("ingest", {}).get("parquet_dir")
    if not d:
        return pd.DataFrame(columns=["t", "lat", "lon", "sog", "cog"])
    days = sorted({str(x.date()) for x in pd.to_datetime([t0, t1], unit="s")})
    pq = (Path(d) / "**" / "*.parquet").as_posix()
    try:
        return duckdb.sql(f"""SELECT epoch(ts) t, lat, lon, sog, cog FROM read_parquet('{pq}', hive_partitioning=true)
                              WHERE day IN ({", ".join(f"'{x}'" for x in days)}) AND mmsi = {int(mmsi)}
                                AND epoch(ts) BETWEEN {t0} AND {t1} ORDER BY ts""").df()
    except Exception:
        return pd.DataFrame(columns=["t", "lat", "lon", "sog", "cog"])


def trails(t0: float, t1: float, s: int) -> pd.DataFrame:
    """Real AIS positions between t0 and t1, one per vessel per `s` seconds, for the trail layer.

    Only vessels scored by the anomaly pipeline are kept, so trail colours use the same vessel categories."""
    d = run_info().get("ingest", {}).get("parquet_dir")
    empty = pd.DataFrame(columns=["mmsi", "t", "lat", "lon", "category"])
    if not d or not available("anomaly_windows.parquet"):
        return empty
    days = pd.date_range(pd.to_datetime(t0, unit="s").normalize(), pd.to_datetime(t1, unit="s").normalize())
    pq = (Path(d) / "**" / "*.parquet").as_posix()
    try:
        tr = duckdb.sql(f"""SELECT mmsi, floor(epoch(ts) / {s}) * {s} + {s} t, last(lat ORDER BY ts) lat,
                                   last(lon ORDER BY ts) lon
                            FROM read_parquet('{pq}', hive_partitioning=true)
                            WHERE day IN ({", ".join(f"'{x.date()}'" for x in days)})
                              AND epoch(ts) BETWEEN {t0} AND {t1} GROUP BY ALL""").df()
    except Exception:
        return empty
    cat = windows("nominal", "quality_aware")[["mmsi", "category"]].drop_duplicates("mmsi")
    tr = tr.merge(cat.assign(category=cat.category.astype(str)), on="mmsi")
    return tr.astype({"lat": "float32", "lon": "float32"}).sort_values(["mmsi", "t"], ignore_index=True)


def name_of(mmsi) -> str:
    v = vessels()
    n = v.name.get(int(mmsi)) if len(v) else None
    return n if isinstance(n, str) and n.strip() else f"MMSI {int(mmsi)}"


def category_of(vtype) -> str:
    t = float(vtype) if vtype is not None and np.isfinite(float(vtype)) else np.nan
    if not np.isfinite(t) or t == 0:
        return "unclassified"
    for lo, hi, c in [(70, 79, "cargo"), (80, 89, "tanker"), (60, 69, "passenger")]:
        if lo <= t <= hi:
            return c
    if t in (35, 55):
        return "government"
    if t in (31, 32, 33, 34, 50, 51, 52, 53, 54, 56, 57, 58, 59):
        return "tug_service"
    return {30: "fishing", 36: "pleasure", 37: "pleasure"}.get(int(t), "other")


# ------------------------------------------------------------------------------------------- top bar + toolbar
def topbar(modes: dict, evals: dict, current):
    """Every page: the title on the left; the two modes and the two evaluation pages, as one tab group, on the right."""
    style()
    with st.container(horizontal=True, vertical_alignment="center", key="topbar", gap=None):
        st.html("<div class='brand'>AIS Maritime <span>Intelligence</span></div>", width="content")
        st.space("stretch")
        with st.container(horizontal=True, vertical_alignment="center", key="navtabs", gap=None, width="content"):
            for i, (label, p) in enumerate({**modes, **evals}.items()):
                with st.container(key=f"nav{'on' if p.url_path == current.url_path else 'off'}_{i}", width="content"):
                    st.page_link(p, label=label)


def toolbar(mode: str, layers_default: list[str]):
    """Map toolbar: region and layers, then the data-poisoning test as its own group. Returns
    (area, layers, condition, countermeasure)."""
    keep("area", "Houston / Galveston")
    keep(f"{mode}_layers", layers_default)
    keep("cond", "nominal")
    keep("mitig", True)
    c1, c2, c3 = st.columns([1.7, 2.3, 5.6], vertical_alignment="center")
    c1.selectbox("Region", list(AREAS), key="w_area", on_change=sync, args=("area",), label_visibility="collapsed")
    c2.pills("Layers", ["Sea lanes", "Density", "Weather"], selection_mode="multi", key=f"w_{mode}_layers",
             on_change=sync, args=(f"{mode}_layers",), label_visibility="collapsed")
    poisoned = (st.session_state["cond"] or "nominal") != "nominal"
    with c3.container(horizontal=True, vertical_alignment="center", key=f"poison_{'on' if poisoned else 'off'}"):
        st.html("<div class='ptag'>Data poisoning test</div>", width="content")
        st.segmented_control("Feed", list(CONDITIONS), format_func=CONDITIONS.get, key="w_cond", on_change=sync,
                             args=("cond",), label_visibility="collapsed",
                             help="Clean feed = AIS as recorded. The other three inject SIMULATED defects into the "
                                  "same held-out day: reports delivered 5-45 min late (stale), a fast unreliable "
                                  "source with jitter, position outliers and bad speeds (unreliable), or both.")
        st.toggle("Countermeasure", key="w_mitig", on_change=sync, args=("mitig",),
                  help="Quality-aware processing: drops spikes and invalid speeds, compensates report age, and "
                       "withholds the answer (sends it to a reviewer) when the newest report is over 20 min old.")
    return (st.session_state["area"], st.session_state[f"{mode}_layers"] or [], st.session_state["cond"] or "nominal",
            st.session_state["mitig"])


# ------------------------------------------------------------------------------------------- poisoning
@st.cache_data
def degraded_share(cond: str) -> pd.Series | None:
    """Share of scored vessels whose reports carry a simulated defect, per 30-min step (None on the clean feed)."""
    if cond == "nominal" or not available("anomaly_windows.parquet"):
        return None
    w = windows(cond, "none")
    return (w.is_stale | w.is_lowq).groupby(w["T"]).mean()


# ------------------------------------------------------------------------------------------- map
def in_box(lat, lon, area: str, pad: float = 0.0) -> np.ndarray:
    box = AREAS.get(area)
    if box is None:
        return np.ones(len(lat), bool)
    x0, y0, x1, y1 = box
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (lon >= x0 - pad) & (lon <= x1 + pad) & (lat >= y0 - pad) & (lat <= y1 + pad)


def zoom_bounds(lat, lon, pad: float = 0.08):
    la, lo = np.asarray(lat, float), np.asarray(lon, float)
    m = np.isfinite(la) & np.isfinite(lo)
    if not m.any():
        return None
    return (la[m].min() - pad, la[m].max() + pad, lo[m].min() - pad, lo[m].max() + pad)


def category_chip(cat: str) -> str:
    return f"<span class='chip' style='--c:{CAT_COLOR.get(cat, C['dim'])}'>{esc(CATEGORY_LABEL.get(cat, cat))}</span>"


def unavailable(what: str, how: str = "run scripts\\run_pipeline.py --subsets jan01_04 --train-frac 0.75"):
    st.warning(f"{what} is not available in results/. Nothing is shown instead of an invented answer.")
    st.caption(f"To compute it: {how}")
    st.stop()
