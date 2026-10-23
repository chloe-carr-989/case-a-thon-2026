"""Streamlit dashboard, built around one map: where the model said vessels would be vs. where they were.

Run (from website/):  .venv\\Scripts\\streamlit run app\\dashboard.py
All numbers come from files written by scripts/run_pipeline.py (results/). Nothing is invented here.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
LOG = RES / "review_log.csv"
sys.path.insert(0, str(ROOT / "src"))
from aisdp.geo import haversine_m  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log_review, metrics  # noqa: E402


CONDITIONS = {"nominal": "Clean feed", "stale": "Stale reports", "low_quality": "Unreliable source",
              "combined": "Stale + unreliable"}
# (lon_min, lat_min, lon_max, lat_max); chosen for the number of moving test vessels
AREAS = {
    "Houston / Galveston": (-95.6, 28.9, -94.3, 30.0),
    "Lower Mississippi": (-91.5, 29.0, -89.0, 30.7),
    "New York Harbor": (-74.4, 40.4, -73.7, 40.95),
    "Chesapeake Bay": (-77.0, 36.7, -75.8, 39.4),
    "Puget Sound": (-123.2, 47.0, -122.2, 48.5),
    "San Francisco Bay": (-122.7, 37.4, -122.0, 38.2),
    "Whole U.S.": None,
}
DEFECT_TEXT = {
    "jitter": "position noise (about 200 m)",
    "position_outlier": "position jumped several km",
    "sog_scaled": "speed value wrong",
    "sog_missing": "speed missing",
    "sog_sentinel": "speed = 102.3, the AIS code for 'unknown'",
    "cog_random": "course value wrong",
    "cog_missing": "course missing",
    "source_bias_1.5km": "position shifted 1.5 km",
}
FLAG_TEXT = {
    "stale_report": "Report is more than 2 minutes old, so the estimate starts from where the vessel was then.",
    "too_old": "Report is more than 20 minutes old: too old to extrapolate.",
    "implausible_jump": "The vessel would have had to move impossibly fast between its last two reports.",
    "sog_cog_inconsistent": "Reported speed/course disagree with the vessel's actual track.",
    "invalid_or_missing_sog_cog": "Speed or course is missing or physically impossible.",
    "no_recent_track": "No earlier report in the last 15 minutes to cross-check against.",
}
MISS_BANDS = [("under 1 km", 0, 1, "#2e8b57"), ("1–5 km", 1, 5, "#e69500"), ("over 5 km", 5, np.inf, "#c62828")]


@st.cache_data
def load():
    if not (RES / "cases_default.parquet").exists():
        return None
    man = json.loads((RES / "manifest.json").read_text())
    rj = RES / "runs" / man["default_demo_subset"] / "run.json"
    return {"cases": pd.read_parquet(RES / "cases_default.parquet"), "metrics": metrics("position_15m"),
            "manifest": man, "run": json.loads(rj.read_text()) if rj.exists() else {},
            "bench": pd.read_csv(RES / "benchmarks.csv") if (RES / "benchmarks.csv").exists() else pd.DataFrame()}


@st.cache_data
def track(pq: str, mmsi: int, t_end: float) -> pd.DataFrame:
    """Real AIS reports for one vessel in the hour before the decision (what an operator could see)."""
    return duckdb.sql(f"""SELECT epoch(ts) t, lat, lon FROM read_parquet('{pq}', hive_partitioning=true)
                          WHERE mmsi = {mmsi} AND epoch(ts) BETWEEN {t_end - 3600} AND {t_end} ORDER BY ts""").df()


def hhmm(t):
    return pd.to_datetime(t, unit="s").strftime("%H:%M")


def km(x):
    return "n/a" if pd.isna(x) else (f"{x*1000:.0f} m" if x < 1 else f"{x:.1f} km")


D = load()
if D is None:
    st.error("No results yet. From the website folder run:  .venv\\Scripts\\python scripts\\run_pipeline.py")
    st.stop()
CASES, M, MAN = D["cases"], D["metrics"], D["manifest"]
edges = MAN["split"]["window_edges_utc"]
win_label = {w: f"{edges[w][11:16]}–{edges[w+1][11:16]}" for w in range(len(edges) - 1)}

# ------------------------------------------------------------------ header + controls
st.title("Where will this vessel be in 15 minutes?")
st.caption("Each dot is where the AI predicted a vessel would be; the line runs to where its next AIS report "
           "actually put it. Degrade the feed, then switch on the countermeasure.")

c1, c2, c3, c4 = st.columns([3, 2, 3, 2])
cond = c1.segmented_control("Feed condition", list(CONDITIONS), format_func=CONDITIONS.get,
                            default="combined") or "combined"
mitig = c2.toggle("Countermeasure on", value=False,
                  help="Quality checks on each report: age, plausibility, consistency. Unreliable estimates are "
                       "sent to a human reviewer instead of being shown.")
win = c3.select_slider("Time of day (UTC). Feed quality gets worse later in the day.", options=list(win_label),
                       value=max(win_label), format_func=win_label.get)
area = c4.selectbox("Area", list(AREAS))

C = CASES[(CASES.condition == cond) & (CASES.window == win) & (CASES.nominal_sog >= 2)]
if AREAS[area]:
    x0, y0, x1, y1 = AREAS[area]
    C = C[C.target_lon.between(x0, x1) & C.target_lat.between(y0, y1)]
C = C.copy()
C["err"] = C.err_mi_km if mitig else C.err_un_km
C["p_lat"], C["p_lon"] = (C.mi_lat, C.mi_lon) if mitig else (C.un_lat, C.un_lon)
C["shown"] = C.answered if mitig else True

# ------------------------------------------------------------------ headline numbers
if C.empty:
    st.info("No moving vessels in this area and time window. Try another area or time.")
    st.stop()


def headline(err, shown):
    e = err[shown]
    return np.median(e) if len(e) else np.nan, int((e > 5).sum()), int((~shown).sum())


med, big, deferred = headline(C.err.to_numpy(), C.shown.to_numpy(bool))
med0, big0, _ = headline(C.err_un_km.to_numpy(), np.ones(len(C), bool))
k1, k2, k3 = st.columns(3)
k1.metric("Typical miss (median)", km(med),
          delta=f"{km(med0)} without countermeasure" if mitig else None, delta_color="off")
k2.metric("Predictions off by more than 5 km", f"{big} of {int(C.shown.sum())}",
          delta=f"{big0} without countermeasure" if mitig else None, delta_color="off")
k3.metric("Sent to human reviewer", f"{deferred}" if mitig else "—",
          help="With the countermeasure off, every estimate is shown, however bad.")

# ------------------------------------------------------------------ map + vessel panel
map_col, side = st.columns([5, 2], gap="medium")
with map_col:
    zoom = st.session_state.get("zoom", False)
    sel_id = st.session_state.get("sel")
    if sel_id not in set(C.case_id):
        sel_id = int(C.sort_values("err_un_km", ascending=False).case_id.iloc[0])
    S = C[C.case_id == sel_id].iloc[0]
    fig = go.Figure()
    P = C[C.shown]
    lat_l, lon_l = [], []
    for a, b, c, d in zip(P.p_lat, P.p_lon, P.target_lat, P.target_lon):
        lat_l += [a, c, None]
        lon_l += [b, d, None]
    fig.add_scattergeo(lat=lat_l, lon=lon_l, mode="lines", line=dict(width=1, color="rgba(90,90,90,0.55)"),
                       hoverinfo="skip", showlegend=False)
    fig.add_scattergeo(lat=P.target_lat, lon=P.target_lon, mode="markers", name="actual position",
                       marker=dict(size=4, color="#d6dce4"), hoverinfo="skip")
    for name, lo, hi, col in MISS_BANDS:
        s = P[(P.err >= lo) & (P.err < hi)]
        fig.add_scattergeo(lat=s.p_lat, lon=s.p_lon, mode="markers", name=f"prediction, miss {name}",
                           marker=dict(size=8, color=col, line=dict(width=0.5, color="#0c0f12")),
                           customdata=s.case_id, text=[f"MMSI {m} · miss {km(e)}" for m, e in zip(s.mmsi, s.err)],
                           hovertemplate="%{text}<extra></extra>")
    R = C[~C.shown]
    if len(R):
        fig.add_scattergeo(lat=R.in_lat, lon=R.in_lon, mode="markers", name="sent to reviewer",
                           marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(width=1.5, color="#8a95a3")),
                           customdata=R.case_id, text=[f"MMSI {m} · sent to reviewer" for m in R.mmsi],
                           hovertemplate="%{text}<extra></extra>")
    # selected vessel
    pq = (Path(D["run"].get("ingest", {}).get("parquet_dir", "")) / "**" / "*.parquet").as_posix()
    try:
        trk = track(pq, int(S.mmsi), float(S.decision_t))
    except Exception:
        trk = pd.DataFrame()
    if len(trk):
        fig.add_scattergeo(lat=trk.lat, lon=trk.lon, mode="lines", name="selected: real track, past hour",
                           line=dict(width=3, color="#49b6dc"), hoverinfo="skip")
    fig.add_scattergeo(lat=[S.in_lat], lon=[S.in_lon], mode="markers", name="selected: report the AI received",
                       marker=dict(size=13, symbol="square", color="#49b6dc", line=dict(width=1, color="#0c0f12")),
                       hoverinfo="skip")
    if S.shown:
        fig.add_scattergeo(lat=[S.p_lat, S.target_lat], lon=[S.p_lon, S.target_lon], mode="lines+markers",
                           showlegend=False, line=dict(width=3, color="#49b6dc"),
                           marker=dict(size=[14, 9], color=["#49b6dc", "#d6dce4"], symbol=["circle", "circle"]),
                           hoverinfo="skip")
    if zoom:
        pts = pd.DataFrame({"lat": [S.in_lat, S.target_lat, S.p_lat] + list(trk.lat if len(trk) else []),
                            "lon": [S.in_lon, S.target_lon, S.p_lon] + list(trk.lon if len(trk) else [])}).dropna()
        pad = 0.05
        fig.update_geos(lataxis_range=[pts.lat.min() - pad, pts.lat.max() + pad],
                        lonaxis_range=[pts.lon.min() - pad, pts.lon.max() + pad])
    elif AREAS[area]:
        x0, y0, x1, y1 = AREAS[area]
        fig.update_geos(lataxis_range=[y0, y1], lonaxis_range=[x0, x1])
    else:
        fig.update_geos(fitbounds="locations")
    fig.update_geos(projection_type="mercator", showland=True, landcolor="#1a1f25", showocean=True, bgcolor="#0b0f14",
                    oceancolor="#0b0f14", showlakes=True, lakecolor="#0b0f14", showrivers=False,
                    coastlinecolor="#333c47", showcountries=True, countrycolor="#333c47", showsubunits=True,
                    subunitcolor="#262d36", resolution=50)
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select",
                      legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=11)))
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="map")
    pts = (ev or {}).get("selection", {}).get("points", []) if isinstance(ev, dict) else \
        getattr(getattr(ev, "selection", None), "points", [])
    clicked = [p.get("customdata") for p in pts if p.get("customdata") is not None]
    if clicked and int(np.ravel(clicked[0])[0]) != sel_id:
        st.session_state["sel"] = int(np.ravel(clicked[0])[0])
        st.rerun()
    st.caption(f"{len(C)} decisions for moving vessels (≥ 2 kn) · {area} · {win_label[win]} UTC · seed 0. "
               "Click a dot to inspect that vessel. Offline basemap with coarse coastlines.")

# ------------------------------------------------------------------ selected vessel: the full chain
with side:
    order = C.sort_values("err_un_km", ascending=False)
    pick = st.selectbox("Vessel (largest misses first)", order.case_id.tolist(),
                        index=order.case_id.tolist().index(sel_id),
                        format_func=lambda c: f"MMSI {order.set_index('case_id').at[c, 'mmsi']} · "
                                              f"{hhmm(order.set_index('case_id').at[c, 'decision_t'])}")
    if pick != sel_id:
        st.session_state["sel"] = int(pick)
        st.rerun()
    st.toggle("Zoom to this vessel", key="zoom")
    if st.session_state.get("rq_case") != sel_id:
        st.session_state["rq_case"], st.session_state["rq_t0"] = sel_id, time.time()

    true_pos = CASES[(CASES.condition == "nominal") & (CASES.case_id == sel_id)].iloc[0]
    off = haversine_m(S.in_lat, S.in_lon, true_pos.in_lat, true_pos.in_lon) / 1000
    st.markdown(f"#### Decision at {hhmm(S.decision_t)} UTC")
    st.markdown("**1 · What the AI received**")
    lines = [f"- Report sent at {hhmm(S.state_t)}, **{S.age_s/60:.0f} min old**" if S.age_s > 120
             else "- Fresh report (under 2 min old)"]
    if S.source != "AIS-original":
        defs = [DEFECT_TEXT.get(d, d) for d in str(S.defects).split(";") if d]
        lines.append(f"- From the **unreliable test source**: {', '.join(defs)}")
    else:
        lines.append("- From the real AIS feed")
    if off > 0.05:
        lines.append(f"- Reported position is **{km(off)}** from where the vessel really was at {hhmm(S.decision_t)}")
    st.markdown("\n".join(lines))

    st.markdown("**2 · What it predicted, 15 min ahead**")
    st.markdown(f"- Without countermeasure: off by **{km(S.err_un_km)}**\n"
                f"- With countermeasure: " + ("**sent to reviewer**" if not S.answered
                                              else f"off by **{km(S.err_mi_km)}**"))

    flags = [f for f in str(S["flags"]).split(";") if f]
    st.markdown("**3 · Why the countermeasure " + ("held it back**" if not S.answered else
                                                    "flagged it**" if flags else "trusted it**"))
    st.markdown("\n".join(f"- {FLAG_TEXT.get(f, f)}" for f in flags) or "- No quality problems detected.")
    st.caption(f"Quality score {S.quality:.2f} (below 0.35 → reviewer).")

    st.markdown("**4 · Reviewer decision**")
    with st.form("review", clear_on_submit=True, border=False):
        act = st.radio("Decision", ["Accept estimate", "Correct position", "Defer: request a fresh report"],
                       label_visibility="collapsed")
        reason = st.text_input("Reason", placeholder="e.g. report 30 min old")
        if st.form_submit_button("Record decision"):
            if not reason.strip():
                st.error("Add a reason first.")
            else:
                log_review({
                    "task": "position_15m", "case_id": sel_id,
                    "mmsi": int(S.mmsi), "condition": cond, "countermeasure_on": mitig, "system_action": S.action,
                    "quality": round(float(S.quality), 3), "operator_action": act, "reason": reason,
                    "review_seconds_measured": round(time.time() - st.session_state["rq_t0"], 1)})
                st.success("Logged.")

# ------------------------------------------------------------------ secondary
st.divider()
with st.expander("How the error grows as the feed degrades (all areas, moving vessels, 3 seeds)"):
    W = M[(M.condition == cond) & (M.subgroup == "underway_ge2kn") & (M.predictor == "hgb") & (M.window != "all")]
    W = W.assign(window=W.window.astype(int)).groupby(["window", "mitigation"]).mean(numeric_only=True).reset_index()
    fig2 = go.Figure()
    for mit, name, col in [("none", "countermeasure off", "#c62828"), ("quality_aware", "countermeasure on", "#49b6dc")]:
        s = W[W.mitigation == mit]
        fig2.add_scatter(x=s.window, y=s.p90_km, name=name, mode="lines+markers", line=dict(color=col, width=3),
                         text=[f"{c:.0%} answered" for c in s.coverage], hovertemplate="%{y:.1f} km · %{text}")
    lab = W[W.mitigation == "none"]
    fig2.update_xaxes(tickvals=lab.window, ticktext=[f"{win_label[w]}<br>reports {a:.0f} min old · {sh:.0%} unreliable"
                                                     for w, a, sh in zip(lab.window, lab.mean_age_min,
                                                                         lab.synthetic_share)])
    fig2.update_layout(height=320, yaxis_title="bad-case miss (90th percentile, km)", margin=dict(t=10, b=10),
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, width="stretch")
    on = W[W.mitigation == "quality_aware"]
    st.caption(f"With the countermeasure on, {on.coverage.min():.0%}–{on.coverage.max():.0%} of estimates are "
               "answered; the rest go to a reviewer. Degradation rates rise across the day by design (a simulated "
               "progression on real data).")

with st.expander("Known limits of the countermeasure"):
    st.caption("Figures below were measured on the earlier Jan 1-2 run (train Jan 1, test Jan 2); the default "
               "results are now the Jan 1-4 run, so exact values differ. See results/example_cases.csv.")
    st.markdown("""
- **Unseen defect not fixed:** a constant 1.5 km position shift (never seen while designing the checks) is not caught
  for moving vessels; their bad-case error gets slightly worse (2.5 → 2.9 km, 90th percentile).
- **Coverage cost:** in the worst window about half of the estimates go to a reviewer.
- **Valid fast movers flagged:** 12 of 116 unmodified vessels moving faster than 20 kn were flagged.
- **Age compensation can backfire** when a vessel turns or stops during a long reporting gap.
""")

with st.expander("Review log"):
    if LOG.exists():
        st.dataframe(pd.read_csv(LOG).tail(50), width="stretch", hide_index=True)
        st.caption("review_seconds_measured = time from opening the vessel to recording a decision in this demo; "
                   "not an operator study.")
    else:
        st.write("No decisions recorded yet.")

with st.expander("About the data and method"):
    prof = MAN["dataset"]["profile_observed"]
    st.markdown(f"""
- **Data:** NOAA MarineCadastre AIS, {prof['time_min'][:10]} to {prof['time_max'][:10]}: {prof['rows']:,} reports
  from {prof['vessels']:,} vessels.
- **AI:** gradient-boosted trees predicting each vessel's movement over the next 15 minutes. Trained once on clean
  data before {MAN['split']['split_t_utc'][:16]} UTC, then frozen. Tested on {MAN['split']['n_test_cases']:,} later decisions from
  {MAN['split']['n_test_vessels']:,} vessels it never saw in training.
- **Truth:** the vessel's next AIS report about 15 min later. A noisy proxy, not perfect ground truth.
- **Degradations:** injected at prediction time and labelled synthetic. Stale = delayed reports.
  Unreliable = a simulated high-volume source with noisy, wrong or missing fields.
- **Countermeasure:** checks report age, impossible jumps, and speed/course consistency. It falls back to the last
  credible report or sends the estimate to a reviewer.
""")
    st.dataframe(D["bench"][["subset", "rows", "vessels", "status", "total_s", "test_cases"]], width="stretch",
                 hide_index=True)
    if st.button("Rerun all tasks on Jan 1 + Jan 2 (overwrites results)"):
        with st.spinner("Running pipeline..."):
            p = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--subsets", "jan01_02",
                                "--train-frac", "0.5", "--budget-s", "3000"], cwd=ROOT, capture_output=True, text=True)
        st.code(p.stdout[-1500:] + p.stderr[-1500:])
        st.cache_data.clear()
    st.json(MAN, expanded=False)
