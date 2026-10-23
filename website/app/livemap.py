"""Live map: a browser-side component (livemap.js) that plays the held-out day with colour-by-type trails.

Playback runs in the browser, so the map never re-renders on the server. Python sends the layers for the
current settings and hears back the 30-min step (`view`) and clicked vessels (`click`). Trails for the whole
coast are written once to app/static/trails/ (needs server.enableStaticServing) as a 30-min overview plus 5-min
tiles, and the learned lanes likewise as coarse cells plus fine tiles; the browser fetches the tiles around
wherever the view is, so any port can be panned or zoomed to.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import ui
from aisdp.lanes import CATEGORY_LABEL
from common import AREAS

HERE = Path(__file__).parent
STATIC = HERE / "static" / "trails"
US_BOX = (-128.0, 23.0, -64.0, 50.0)
TILE = 1                # degrees per side of a 5-min trail tile
COARSE_S = 1800         # overview trail resolution, used when zoomed out

_render = st.components.v2.component("aismap", js=(HERE / "livemap.js").read_text(encoding="utf-8"),
                                     css=(HERE / "livemap.css").read_text(encoding="utf-8"))


def box(area: str) -> tuple:
    return AREAS.get(area) or US_BOX


def _runs(tr: pd.DataFrame, t0: float):
    """Split each vessel's fixes into runs that stay inside one TILE-degree tile, ordered by tile.

    Every run also carries the fix just before and just after it (in the neighbouring tile), so a trail drawn from
    separate tiles stays joined at the tile edge. Returns float32 (lat, lon, seconds after t0) rows and, per run,
    (mmsi, tile x, tile y, start row, end row)."""
    mm, la, lo = tr.mmsi.to_numpy(), tr.lat.to_numpy(), tr.lon.to_numpy()
    tx, ty, n = np.floor(lo / TILE).astype(int), np.floor(la / TILE).astype(int), len(tr)
    brk = np.flatnonzero((mm[1:] != mm[:-1]) | (tx[1:] != tx[:-1]) | (ty[1:] != ty[:-1])) + 1
    s, e = np.r_[0, brk], np.r_[brk, n]
    es = np.where((s > 0) & (mm[np.maximum(s - 1, 0)] == mm[s]), s - 1, s)
    ee = np.where((e < n) & (mm[np.minimum(e, n - 1)] == mm[e - 1]), e + 1, e)
    o = np.lexsort((s, ty[s], tx[s]))
    s, es, ee = s[o], es[o], ee[o]
    off = np.r_[0, np.cumsum(ee - es)]
    rows = np.repeat(es - off[:-1], ee - es) + np.arange(off[-1])
    arr = np.column_stack([la, lo, tr.t.to_numpy() - t0]).astype("<f4")[rows]
    return arr, np.column_stack([mm[s], tx[s], ty[s], off[:-1], off[1:]])


@st.cache_data(show_spinner="Preparing AIS trails…")
def trail_url(t0: float, t1: float) -> str | None:
    """Write the trails for every region once: a 30-min overview plus 5-min tiles the browser fetches near the view.

    Layout under app/static/trails/<hash>/: index.json (tiles that have data, vessel names), overview.bin/.json, and
    t<x>_<y>.bin/.json per TILE-degree tile. Each .bin is float32 (lat, lon, seconds after t0) rows; each .json
    lists runs as [mmsi, tile x, tile y, start row, end row]."""
    name = hashlib.sha1(f"{t0}|{t1}|{ui.TRAIL_SAMPLE_S}|{COARSE_S}|{TILE}".encode()).hexdigest()[:12]
    out = STATIC / name
    if not (out / "index.json").exists():
        fine = ui.trails(t0, t1, ui.TRAIL_SAMPLE_S)
        if fine.empty:
            return None
        out.mkdir(parents=True, exist_ok=True)

        def write(stem: str, arr: np.ndarray, runs: np.ndarray):
            (out / f"{stem}.bin").write_bytes(arr.tobytes())
            (out / f"{stem}.json").write_text(json.dumps(runs.tolist()), encoding="utf-8")

        arr, runs = _runs(fine, t0)
        key = runs[:, 1:3]
        cut = np.r_[0, np.flatnonzero((key[1:] != key[:-1]).any(axis=1)) + 1, len(runs)]
        for a, b in zip(cut[:-1], cut[1:]):
            r = runs[a:b].copy()
            r[:, 3:] -= r[0, 3]
            write(f"t{r[0, 1]}_{r[0, 2]}", arr[runs[a, 3]:runs[b - 1, 4]], r)
        write("overview", *_runs(ui.trails(t0, t1, COARSE_S), t0))
        cats = fine.drop_duplicates("mmsi")
        index = {"t0": t0, "sample": ui.TRAIL_SAMPLE_S, "coarse": COARSE_S, "tile": TILE,
                 "tiles": key[cut[:-1]].tolist(),
                 "names": {int(m): [str(c), ui.name_of(m)] for m, c in zip(cats.mmsi, cats.category)}}
        (out / "index.json").write_text(json.dumps(index), encoding="utf-8")   # written last: marks the set complete
    return f"app/static/trails/{name}"


LANE_TIERS = [{"c": "rgba(73,182,220,0.22)", "w": 1.2}, {"c": "rgba(73,182,220,0.42)", "w": 1.6},
              {"c": "rgba(73,182,220,0.7)", "w": 2.0}]


@st.cache_data(show_spinner=False)
def lane_url() -> str | None:
    """Write the learned corridors once, laid out like the trails: coarse (0.1°) cells for the zoomed-out view in
    overview.json, fine (0.01°) cells per TILE-degree tile in t<x>_<y>.json, which the browser fetches near the view.

    Each cell is a short segment along its dominant axis. Rows are sorted busiest group first within a cell (k = 1
    marks a row in the same cell as the row before), so the browser draws one segment per cell for whichever vessel
    groups are selected. Traffic tier t (0-2) splits each resolution and group at its 50th and 85th percentile."""
    src = ui.RES / "lanes_cells.parquet"
    name = hashlib.sha1(f"{src.stat().st_mtime_ns}|{TILE}".encode()).hexdigest()[:12]
    out = HERE / "static" / "lanes" / name
    if (out / "index.json").exists():
        return f"app/static/lanes/{name}"
    cells = ui.lane_cells()
    d = cells[cells.lane].copy()
    if d.empty:
        return None
    groups = sorted(d.g.unique())
    n = d.groupby(["res", "g"]).n_vessels
    d["t"] = (d.n_vessels > n.transform("median")).astype(int) + (d.n_vessels > n.transform(lambda x: x.quantile(0.85)))

    def rows(x: pd.DataFrame, half: float) -> dict:
        x = x.sort_values(["i", "j", "n_vessels"], ascending=[True, True, False])
        la, lo, a = x.lat.to_numpy(), x.lon.to_numpy(), np.radians(x.axis_deg.to_numpy())
        dlat, dlon = half * np.cos(a), half * np.sin(a) / np.cos(np.radians(la))
        i, j = x.i.to_numpy(), x.j.to_numpy()
        return {"p": np.round(np.column_stack([la - dlat, lo - dlon, la + dlat, lo + dlon]), 5).ravel().tolist(),
                "g": [groups.index(g) for g in x.g], "t": x.t.astype(int).tolist(),
                "k": np.r_[0, (i[1:] == i[:-1]) & (j[1:] == j[:-1])].astype(int).tolist(),
                "tx": np.floor(lo / TILE).astype(int).tolist(), "ty": np.floor(la / TILE).astype(int).tolist()}

    out.mkdir(parents=True, exist_ok=True)
    fine = d[d.res == "fine"]
    tiles = fine.groupby([np.floor(fine.lon / TILE).astype(int), np.floor(fine.lat / TILE).astype(int)])
    for (tx, ty), x in tiles:
        (out / f"t{tx}_{ty}.json").write_text(json.dumps(rows(x, 0.01 * 0.55)), encoding="utf-8")
    (out / "overview.json").write_text(json.dumps(rows(d[d.res == "coarse"], 0.1 * 0.55)), encoding="utf-8")
    index = {"tile": TILE, "groups": groups, "tiers": LANE_TIERS, "tiles": [[int(a), int(b)] for a, b in tiles.groups]}
    (out / "index.json").write_text(json.dumps(index), encoding="utf-8")   # written last: marks the set complete
    return f"app/static/lanes/{name}"


@st.cache_data
def density(area: str) -> dict | None:
    cells = ui.lane_cells()
    res = "coarse" if AREAS.get(area) is None else "fine"
    d = cells[(cells.res == res) & (cells.g == "commercial") & (cells.n_vessels_any >= 2)]
    d = d[ui.in_box(d.lat, d.lon, area, 0.05)]
    if d.empty:
        return None
    v = np.log10(d.n_vessels_any.to_numpy(float))
    v = (v - v.min()) / max(np.ptp(v), 1e-9)
    return {"lat": d.lat.round(4).tolist(), "lon": d.lon.round(4).tolist(), "v": v.round(3).tolist(),
            "d": 0.1 if res == "coarse" else 0.01}


@st.cache_data
def weather(area: str, t0: float, t1: float) -> dict | None:
    W = ui.weather()
    if W is None:
        return None
    w = W[(W.t >= t0) & (W.t <= t1) & ui.in_box(W.lat, W.lon, area, 0.3)]
    if w.empty:
        return None
    return {"t": w.t.astype(float).tolist(), "lat": w.lat.round(3).tolist(), "lon": w.lon.round(3).tolist(),
            "kn": w.wind_kn.round(1).tolist(), "dir": w.wind_dir_deg.round(0).tolist()}


def markers(rows: pd.DataFrame, style_idx, styles: list[dict], text) -> dict:
    return {"T": rows["T"].astype(float).tolist(), "mmsi": rows.mmsi.astype(int).tolist(),
            "lat": rows.lat.round(5).tolist(), "lon": rows.lon.round(5).tolist(),
            "s": [int(x) for x in style_idx], "styles": styles, "txt": list(text)}


def path(lat, lon, color: str, width: float, dash: bool = False, alpha: float = 1.0) -> dict:
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    return {"lat": lat[ok].round(5).tolist(), "lon": lon[ok].round(5).tolist(), "c": color, "w": width,
            "dash": dash, "a": alpha}


def command(**kw):
    """Send a one-off instruction to the browser (e.g. playing=False when a vessel is picked from a list)."""
    st.session_state["map_cmd"] = {"id": st.session_state.get("map_cmd", {}).get("id", 0) + 1, **kw}
    if "playing" in kw:
        st.session_state["playing"] = kw["playing"]


def current_step(key: str, steps: np.ndarray) -> float:
    """The step the browser last reported for map `key`, readable before the map is drawn this run."""
    view = None
    try:
        v = st.session_state.get(key)
        view = (v.get("view") if hasattr(v, "get") else None) or None
    except Exception:
        pass
    t = view["T"] if view else st.session_state.get("T", float(steps[0]))
    return float(steps[max(0, np.searchsorted(steps, t, side="right") - 1)])


def render(key: str, *, area: str, steps: np.ndarray, counts: pd.Series, mk: dict, cats: list[str],
           layer_groups: tuple, layers: list[str], hud: list[str], legend: list, sel: dict | None,
           fit_box, click_heads: bool, version: str) -> tuple[float, int | None]:
    """Draw the live map. Returns (current 30-min step T, clicked MMSI or None)."""
    st.session_state.setdefault("playing", True)
    st.session_state.setdefault("trail_h", 3)
    T0 = st.session_state.get("T", float(steps[0]))
    t0, t1 = float(steps[0]) - 6 * 3600, float(steps[-1])
    data = {
        "v": version, "steps": [float(s) for s in steps],
        "counts": counts.reindex(steps).fillna(0).astype(int).tolist(), "trails": trail_url(t0, t1),
        "mk": mk, "cats": cats, "clickHeads": click_heads,
        "lanes": {"url": lane_url(), "groups": list(layer_groups)} if "Sea lanes" in layers else None,
        "density": density(area) if "Density" in layers else None,
        "weather": weather(area, t0, t1) if "Weather" in layers else None,
        "hud": hud, "legend": legend, "sel": sel,
        "fit": {"key": f"{area}|{fit_box}", "box": list(fit_box)},
        "colors": {k: ui.C[k] for k in ("bg", "water", "land", "line", "line2", "fg", "dim", "faint", "accent",
                                         "warn", "crit")},
        "catColor": ui.CAT_COLOR, "catLabel": CATEGORY_LABEL,
        "init": {"T": T0, "playing": st.session_state["playing"], "trail_h": st.session_state["trail_h"]},
        "cmd": st.session_state.get("map_cmd"),
    }
    res = _render(key=key, data=data, default={"view": data["init"]}, height="content",
                  on_view_change=lambda: None, on_click_change=lambda: None)
    view = res.get("view") or data["init"]
    T = float(steps[max(0, np.searchsorted(steps, view["T"], side="right") - 1)])
    st.session_state["T"], st.session_state["playing"] = T, bool(view["playing"])
    st.session_state["trail_h"] = view.get("trail_h", 3)
    click = res.get("click")
    return T, (int(click) if click is not None else None)
