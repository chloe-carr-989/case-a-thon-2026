"""Government mode: which vessels are moving unusually relative to the learned sea lanes, and why?"""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import livemap
import ui
from aisdp.config import AnomalyConfig
from aisdp.lanes import CATEGORY_LABEL, lane_group
from common import CONDITIONS, log_review

if not ui.available("anomaly_windows.parquet") or ui.lanes() is None:
    ui.unavailable("Anomaly scoring")

REASON = {
    "off_lane": lambda r, g: f"{ui.km(r.off_lane)} from the nearest learned {g} lane (median of the last 10 min)",
    "wrong_way": lambda r, g: f"{r.wrong_way:.0%} of moving fixes on a course under 10% of {g} traffic takes here",
    "speed": lambda r, g: f"speed {r.speed:.1f}× the usual spread away from the typical lane speed",
    "stop": lambda r, g: f"{r.stop:.0f} min stopped where vessels rarely stop, or away from any lane",
}
LABEL_TEXT = {"normal": "Normal", "unusual": "Unusual", "high": "High deviation", "no_verdict": "No verdict: data too old",
              "no_data": "No data delivered"}


area, layers, cond, mitig = ui.toolbar("gov", ["Sea lanes", "Weather"])
mit = "quality_aware" if mitig else "none"
W = ui.windows(cond, mit)
# military and law-enforcement vessels do not move like commercial traffic, so lanes cannot judge them: not scored
W = W[W.category.astype(str) != "government"]
steps = np.sort(W["T"].unique())
cats = sorted(W.category.astype(str).unique(), key=lambda c: list(ui.CAT_COLOR).index(c) if c in ui.CAT_COLOR else 99)
ui.keep("gov_cats", [c for c in ui.FOCUS_DEFAULT if c in cats])
focus = st.session_state["gov_cats"] or cats
inreg = ui.in_box(W.lat, W.lon, area) & W.category.astype(str).isin(focus).to_numpy()
alerts = W[inreg & W.label.isin(["unusual", "high"]).to_numpy()]


# ------------------------------------------------------------------------------------------- panel
def explain(r) -> list[str]:
    g = CATEGORY_LABEL.get(str(lane_group([r.category])[0]), "commercial").lower()
    g = "pooled commercial" if g == "commercial" else g
    keys = ui.reasons_list(r.reasons) if isinstance(r.reasons, str) else []
    if keys or str(r.label) not in ("unusual", "high"):
        return [REASON[k](r, g) for k in keys]
    # flagged with no component past its own 95th percentile: name the ones that lifted the combined score, largest first
    w = dict(AnomalyConfig().weights)
    p = {k: float(r[f"p_{k}"]) for k in w if np.isfinite(r[f"p_{k}"]) and r[f"p_{k}"] >= 0.5}
    top = sorted(p, key=lambda k: np.log(1 - min(p[k], 0.999)) * w[k])[:2]
    return [f"{REASON[k](r, g)}: more extreme than {p[k]:.0%} of usual {g} windows" for k in top]


def wind_at(lat, lon, T):
    """Nearest hourly ERA5 point within ~50 km of the vessel, or None."""
    Wx = ui.weather()
    if Wx is None or not np.isfinite(lat):
        return None
    w = Wx[Wx.t == np.floor(T / 3600) * 3600]
    if w.empty:
        return None
    d = np.hypot(w.lat - lat, (w.lon - lon) * np.cos(np.radians(lat)))
    return None if d.min() > 0.5 else w.iloc[int(np.argmin(d))]


def weather_check(x) -> tuple[str, str]:
    """(verdict, colour). The usual reason ships leave their lanes is weather, so an alert is read against it."""
    if x is None:
        return "no weather data here", ui.C["dim"]
    if x.wind_kn >= 25:
        return "weather may explain this", ui.C["dim"]
    if x.wind_kn >= 15:
        return "moderate wind: may partly explain this", ui.C["fg"]
    return "not explained by weather", ui.C["warn"]


def wind_word(x) -> str:
    return "—" if x is None else "strong" if x.wind_kn >= 25 else "moderate" if x.wind_kn >= 15 else "calm"


def dossier(r, T):
    v = ui.vessels()
    info = v.loc[int(r.mmsi)] if int(r.mmsi) in v.index else None
    lab = str(r.label)
    h1, h2 = st.columns([3, 1], vertical_alignment="top")
    h1.markdown(f"<div class='eyebrow'>selected vessel</div><h3 style='margin:0'>{ui.esc(ui.name_of(r.mmsi))}</h3>"
                f"<div style='margin:4px 0 4px'>{ui.category_chip(str(r.category))}"
                f"<span class='chip' style='--c:{ui.LABEL_COLOR.get(lab)}'>{LABEL_TEXT.get(lab, lab)}</span></div>",
                unsafe_allow_html=True)
    if h2.button("✕ Clear", key="gov_clear", width="stretch"):
        st.session_state.pop("gov_sel", None)
        st.rerun()
    if lab in ("no_verdict", "no_data"):
        st.markdown(f"<div class='big'>—<small>no movement verdict</small></div><div class='sub'>"
                    f"newest report is {ui.dur(r.age_s / 3600) if np.isfinite(r.age_s) else 'missing'} old: a data "
                    "problem, not a movement anomaly</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='big'>{r.score:.0f}<small>/ 100 anomaly score</small></div><div class='sub'>"
                    f"more deviant than {r.score:.1f}% of {CATEGORY_LABEL.get(str(r.category), '').lower()} "
                    "windows on the calibration days</div>", unsafe_allow_html=True)
    t_why, t_vessel, t_review = st.tabs(["Why", "Data trust", "Review"])
    with t_why:
        x = wind_at(r.lat, r.lon, T)
        verdict, col = weather_check(x)
        st.markdown("<div class='eyebrow'>weather check</div>"
                    f"<div style='font-size:15px;font-weight:600;color:{col}'>{ui.esc(verdict.capitalize())}</div>"
                    "<div class='sub' style='margin-bottom:6px'>"
                    + (ui.esc(f"wind {x.wind_kn:.0f} kn from {x.wind_dir_deg:.0f}° · historical ERA5, a live feed in "
                              "deployment. ") if x is not None else "")
                    + "Weather is the most common reason ships leave their lanes. It never changes the score.</div>",
                    unsafe_allow_html=True)
        why = explain(r)
        if why:
            st.markdown("<div class='eyebrow'>why it was flagged</div><ul class='why'>"
                        + "".join(f"<li>{ui.esc(x)}</li>" for x in why) + "</ul>", unsafe_allow_html=True)
        h = W[(W.mmsi == r.mmsi) & (W["T"] <= T)].sort_values("T")
        if len(h) > 1:
            f = go.Figure(go.Scatter(x=pd.to_datetime(h["T"], unit="s"), y=h.score, mode="lines+markers",
                                     line=dict(color=ui.C["accent"], width=1.5), marker=dict(size=4),
                                     hovertemplate="%{x|%H:%M} · %{y:.0f}<extra></extra>"))
            f.add_hline(y=97.5, line=dict(color=ui.C["warn"], width=1, dash="dot"))
            f.update_layout(height=110, margin=dict(l=0, r=0, t=4, b=0), paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(range=[0, 101], showgrid=False, tickfont=dict(size=12)),
                            xaxis=dict(showgrid=False, tickformat="%H:%M", tickfont=dict(size=12)))
            st.markdown("<div class='eyebrow'>score so far (no future data)</div>", unsafe_allow_html=True)
            st.plotly_chart(f, width="stretch", config={"displayModeBar": False}, theme=None, key="gov_hist")
    with t_vessel:
        st.markdown("<div class='eyebrow'>vessel</div>" + ui.kv([
            ("MMSI", ui.esc(int(r.mmsi))),
            ("IMO / call", ui.esc(f"{(info.imo if info is not None and isinstance(info.imo, str) else '—')} / "
                                  f"{(info.call_sign if info is not None and isinstance(info.call_sign, str) else '—')}")),
            ("Flag", ui.esc(ui.mid_flag(r.mmsi))),
            ("Type", ui.esc(f"{CATEGORY_LABEL.get(str(r.category))} (self-reported AIS type)")),
            ("Last report", ui.esc(ui.stamp(r.last_t)) if np.isfinite(r.last_t) else "—"),
            ("Position", ui.esc(f"{r.lat:.4f}, {r.lon:.4f}")),
            ("Speed / course", ui.esc(f"{r.sog:.1f} kn / {r.cog:.0f}°" if np.isfinite(r.sog) and np.isfinite(r.cog)
                                      else "missing")),
        ]), unsafe_allow_html=True)
        dq = []
        if np.isfinite(r.age_s):
            dq.append(("Freshness", ui.esc(f"newest fix {ui.dur(r.age_s / 3600)} before {ui.hhmm(T)}")))
        dq += [("Longest gap", ui.esc(ui.dur(r.max_gap_s / 3600) if r.max_gap_s >= 0 else "—")),
               ("Fixes in hour", ui.esc(f"{int(r.n_pts)} ({int(r.n_moving)} moving)" if r.n_pts >= 0 else "0")),
               ("Spikes / bad speed", ui.esc(f"{int(max(r.n_spike, 0))} / {int(max(r.n_invalid, 0))}"))]
        if r.is_stale or r.is_lowq:
            dq.append(("Simulated defect", ui.esc(", ".join(x for x, f in [("stale feed", r.is_stale),
                                                                          ("unreliable source", r.is_lowq)] if f))))
        st.markdown("<div class='eyebrow'>AIS data quality</div>" + ui.kv(dq), unsafe_allow_html=True)
    with t_review:
        st.markdown("<div class='sub'>Reviewer: <b>watch officer</b>. Confirm, dismiss as a false positive, or "
                    "escalate. No verdict is given on data over 20 min old. Every decision and its review time is "
                    "logged.</div>", unsafe_allow_html=True)
        tkey = f"gov_t0_{int(r.mmsi)}_{T}"
        st.session_state.setdefault(tkey, time.time())
        with st.form("gov_review", clear_on_submit=True, border=False):
            act = st.radio("Decision", ["Escalate: needs follow-up", "Reviewed: no action",
                                        "False positive: bad data", "False positive: normal movement"],
                           label_visibility="collapsed")
            note = st.text_input("Reason", placeholder="e.g. course change matches weather routing")
            if st.form_submit_button("Record review"):
                log_review({"task": "anomaly", "mmsi": int(r.mmsi), "window_end_utc": ui.stamp(T), "condition": cond,
                            "countermeasure_on": mitig, "system_label": lab, "score": round(float(r.score), 1)
                            if np.isfinite(r.score) else None, "operator_action": act, "reason": note,
                            "review_seconds_measured": round(time.time() - st.session_state[tkey], 1)})
                st.success("Review logged to results/review_log.csv.")


def queue(now, T):
    st.markdown(f"<div class='eyebrow' style='margin-top:4px'>alert queue · {ui.hhmm(T)} UTC · click a row to open</div>",
                unsafe_allow_html=True)
    q = now[now.label.isin(["unusual", "high"])].sort_values("score", ascending=False)
    if q.empty:
        st.markdown("<div class='sub'>No unusual movement in view at this time.</div>", unsafe_allow_html=True)
    else:
        wind = [wind_word(wind_at(la, lo, T)) for la, lo in zip(q.lat, q.lon)]
        hit = ui.queue_rows("gov_queue", [18, 5, 8], ["vessel", "score", "wind", "why"], [
            [ui.name_of(m), f"{s:.0f}", w, ", ".join(x.replace("_", " ") for x in ui.reasons_list(r)) or "combined"]
            for m, s, w, r in zip(q.mmsi, q.score, wind, q.reasons)])
        if hit is not None:
            st.session_state["gov_sel"] = int(q.mmsi.iloc[hit])
            livemap.command(playing=False)
            st.rerun()
        windy = sum(w == "strong" for w in wind)
        st.markdown(f"<div class='sub'>{len(q) - windy} of {len(q)} alerts in calm or moderate wind. Many vessels "
                    "veering together in strong wind is almost always weather.</div>", unsafe_allow_html=True)
    held = now[now.label.isin(["no_verdict", "no_data"])]
    if len(held):
        st.markdown(f"<div class='sub'>{len(held)} vessels with no verdict: newest report over 20 min old "
                    "(grey rings on the map).</div>", unsafe_allow_html=True)
    st.markdown("<div class='note'>Score = how unusual the last hour of movement is vs. past traffic on these "
                "lanes (97.5+ flagged). A review prompt, not proof of intent. Military and law-enforcement vessels "
                "are not scored: they do not follow commercial routes.</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- live map + panel
MK_STYLE = {"unusual": 0, "high": 1, "no_verdict": 2, "no_data": 2}
mk_rows = W[inreg & W.label.isin(list(MK_STYLE)).to_numpy()]
MK = livemap.markers(mk_rows, mk_rows.label.astype(str).map(MK_STYLE),
                     [{"c": ui.C["warn"], "r": 6}, {"c": ui.C["crit"], "r": 7}, {"c": ui.C["dim"], "r": 5, "ring": True}],
                     [f"{ui.name_of(m)} · {LABEL_TEXT[str(lab)]}" + (f" · score {s:.0f}" if np.isfinite(s) else "")
                      for m, lab, s in zip(mk_rows.mmsi, mk_rows.label, mk_rows.score)])
LEGEND = ([[CATEGORY_LABEL.get(c, c), ui.CAT_COLOR.get(c, ui.C["dim"]), ""] for c in focus]
          + [["unusual", ui.C["warn"], "dot"], ["high deviation", ui.C["crit"], "dot"],
             ["no verdict (old data)", ui.C["dim"], "ring"]]
          + ([ui.LANE_LEGEND] if "Sea lanes" in layers else [])
          + ([ui.DENSITY_LEGEND] if "Density" in layers else [])
          + ([ui.WEATHER_LEGEND] if "Weather" in layers else []))
VERSION = f"gov|{area}|{cond}|{mit}|{','.join(focus)}|{','.join(layers)}"


@st.fragment
def live():
    map_col, side = st.columns([2.3, 1], gap="medium")
    sel = st.session_state.get("gov_sel")
    T_prev = livemap.current_step("gov_map", steps)
    now = W[(W["T"] == T_prev) & inreg]
    S = now[now.mmsi == sel].iloc[0] if sel is not None and (now.mmsi == sel).any() else None
    overlay, fit = None, ui.AREAS.get(area) or livemap.US_BOX
    if S is not None:
        trk = ui.track(int(S.mmsi), T_prev - 3 * 3600, T_prev)
        w0 = trk[trk.t > T_prev - 3600]
        paths = [livemap.path(trk.lat, trk.lon, ui.C["track"], 1.5, alpha=0.45),
                 livemap.path(w0.lat, w0.lon, ui.C["track"], 3)]
        if len(w0) >= 2:
            p = ui.lanes().path(w0.lat.iloc[0], w0.lon.iloc[0], w0.lat.iloc[-1], w0.lon.iloc[-1])
            paths.append(livemap.path(p.lat, p.lon, ui.C["fg"], 2, dash=True, alpha=0.85))
        overlay = {"mmsi": int(S.mmsi), "lat": float(S.lat), "lon": float(S.lon), "paths": paths, "points": [],
                   "c": ui.LABEL_COLOR.get(str(S.label), ui.C["fg"])}
        b = ui.zoom_bounds(np.r_[trk.lat, S.lat], np.r_[trk.lon, S.lon], 0.06)
        if b is not None:
            fit = (b[2], b[0], b[3], b[1])
        LEGEND_SEL = [["track, last 60 min", ui.C["track"], ""], ["expected route along lanes", ui.C["fg"], "dash"]]
    else:
        LEGEND_SEL = []
    grp = str(lane_group([S.category])[0]) if S is not None else "commercial"
    flagged = int(now.label.isin(["unusual", "high"]).sum())
    with map_col:
        T, hit = livemap.render(
            "gov_map", area=area, steps=steps, counts=alerts.groupby("T").size(), bars="flagged vessels in view",
            degr=ui.degraded_share(cond), mk=MK, cats=focus,
            layer_groups=(grp,), layers=layers, legend=LEGEND + LEGEND_SEL, sel=overlay, fit_box=fit,
            click_heads=True, version=VERSION,
            hud=[f"{len(now):,} vessels reporting · {flagged} flagged",
                 f"feed: {CONDITIONS[cond]}{' (SIMULATED)' if cond != 'nominal' else ''} · countermeasure "
                 f"{'on' if mitig else 'off'}"])
    if hit is not None and hit != sel:
        st.session_state["gov_sel"] = hit
        st.rerun()
    if T != T_prev:                                  # the map moved to another step: refresh the panel for it
        st.rerun(scope="fragment")
    with side, st.container(key="side_box"):
        if S is not None:
            dossier(S, T)
        else:
            queue(now, T)


live()
