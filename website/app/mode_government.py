"""Government mode: which vessels are moving unusually relative to the learned sea lanes, and why?"""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import livemap
import ui
from aisdp.lanes import CATEGORY_LABEL, lane_group
from common import CONDITIONS, log_review, metrics

ui.header("Government · movement anomalies")
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


# ------------------------------------------------------------------------------------------- evaluation
@st.dialog("Model evaluation", width="large")
def evaluation():
    m = metrics("anomaly")
    s = metrics("anomaly_synthetic")
    if m.empty:
        st.write("No anomaly metrics in results/metrics.csv.")
        return
    g = m[(m.window == "all")]
    rows = []
    for c, lab in CONDITIONS.items():
        for mi, ml in [("none", "naive"), ("quality_aware", "countermeasure")]:
            a = g[(g.condition == c) & (g.mitigation == mi)]
            b = s[(s.condition == c) & (s.mitigation == mi)]
            if a.empty:
                continue
            a = a.iloc[0]
            rows.append({"feed": lab, "pipeline": ml, "alerts per 1000 windows": 1000 * a.alert_rate,
                         "alerts caused by bad data / 1000": a.spurious_alerts_per_1000,
                         "clean-feed alerts lost": a.missed_clean_alert_share, "verdict coverage": a.coverage,
                         "injected drifts caught": b.recall.iloc[0] if len(b) else np.nan})
    t = pd.DataFrame(rows)
    st.markdown(f"<div class='sub'>Selected feed: <b>{CONDITIONS[cond]}</b>. "
                "Same held-out vessel-windows in every row; degraded feeds are SIMULATED.</div>",
                unsafe_allow_html=True)
    st.dataframe(t.style.format({"alerts per 1000 windows": "{:.1f}", "alerts caused by bad data / 1000": "{:.1f}",
                                 "clean-feed alerts lost": "{:.0%}", "verdict coverage": "{:.0%}",
                                 "injected drifts caught": "{:.0%}"})
                 .apply(lambda r: [f"background-color: {ui.C['panel']}" if r.feed == CONDITIONS[cond] else ""
                                   for _ in r], axis=1), hide_index=True, width="stretch")
    w = m[(m.window != "all") & (m.condition == cond)].copy()
    if cond != "nominal" and len(w):
        w["window"] = w.window.astype(int)
        f = go.Figure()
        for mi, name, col in [("none", "naive", ui.C["crit"]), ("quality_aware", "countermeasure", ui.C["accent"])]:
            d = w[w.mitigation == mi].sort_values("window")
            f.add_scatter(x=d.window, y=d.spurious_alerts_per_1000, name=name, mode="lines+markers",
                          line=dict(color=col, width=2))
        share = w[w.mitigation == "none"].sort_values("window")
        f.update_xaxes(tickvals=share.window, ticktext=[f"{max(a, b):.0%} of windows degraded" for a, b in
                                                        zip(share.stale_frac, share.lowq_frac)])
        f.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="bad-data alerts / 1000",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", y=1.15))
        st.plotly_chart(f, width="stretch", theme="streamlit")
    st.markdown("""<div class='note'>
No verified anomaly labels exist for this data, so real-world detection accuracy is not claimed.
<b>alerts caused by bad data</b> = alerts on a degraded feed that the same pipeline did not raise on the clean feed.
<b>injected drifts caught</b> = known-label check: 2-8 km lateral drifts added to normal windows.
On clean calibration data about 2.5% of windows are 'unusual' by construction (97.5th percentile).
The countermeasure trades coverage (no verdict on data older than 20 min) for fewer data-driven alerts.
</div>""", unsafe_allow_html=True)


area, layers, cond, mitig, panel = ui.toolbar("gov", ["Sea lanes"])
mit = "quality_aware" if mitig else "none"
W = ui.windows(cond, mit)
steps = np.sort(W["T"].unique())
cats = sorted(W.category.astype(str).unique(), key=lambda c: list(ui.CAT_COLOR).index(c) if c in ui.CAT_COLOR else 99)
ui.keep("gov_cats", [c for c in ui.FOCUS_DEFAULT if c in cats])
focus = st.session_state["gov_cats"] or cats
inreg = ui.in_box(W.lat, W.lon, area) & W.category.astype(str).isin(focus).to_numpy()
alerts = W[inreg & W.label.isin(["unusual", "high"]).to_numpy()]
if st.session_state.get("gov_eval"):
    livemap.command(playing=False)
    evaluation()


# ------------------------------------------------------------------------------------------- panel
def explain(r) -> list[str]:
    g = CATEGORY_LABEL.get(str(lane_group([r.category])[0]), "commercial").lower()
    g = "pooled commercial" if g == "commercial" else g
    return [REASON[k](r, g) for k in ui.reasons_list(r.reasons)] if isinstance(r.reasons, str) else []


def weather_line(r, T) -> str:
    Wx = ui.weather()
    if Wx is None:
        return "not available (no weather file loaded)"
    w = Wx[Wx.t == np.floor(T / 3600) * 3600]
    if w.empty:
        return "no observation for this hour"
    d = np.hypot(w.lat - r.lat, (w.lon - r.lon) * np.cos(np.radians(r.lat)))
    if d.min() > 0.5:
        return "no weather point within ~50 km"
    x = w.iloc[int(np.argmin(d))]
    tag = ("strong wind: a possible environmental explanation" if x.wind_kn >= 25 else
           "moderate wind" if x.wind_kn >= 15 else "calm: wind is unlikely to explain a deviation")
    return f"{x.wind_kn:.0f} kn from {x.wind_dir_deg:.0f}° ({tag}). Context only; it never changes the label."


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
    t_why, t_vessel, t_review = st.tabs(["Why", "Vessel & data", "Review"])
    with t_why:
        why = explain(r)
        if lab in ("unusual", "high") or why:
            st.markdown("<div class='eyebrow'>why it was flagged</div><ul class='why'>"
                        + "".join(f"<li>{ui.esc(x)}</li>" for x in why) + "</ul>"
                        if why else "<div class='sub'>no single factor stands out; the combination is unusual</div>",
                        unsafe_allow_html=True)
        st.markdown("<div class='eyebrow' style='margin-top:8px'>weather context</div><div class='sub' "
                    f"style='margin-bottom:6px'>{ui.esc(weather_line(r, T))}</div>", unsafe_allow_html=True)
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
        st.toggle("Zoom to vessel", value=True, key="gov_zoom")
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
        tkey = f"gov_t0_{int(r.mmsi)}_{T}"
        st.session_state.setdefault(tkey, time.time())
        with st.form("gov_review", clear_on_submit=True, border=False):
            act = st.radio("Decision", ["Reviewed: needs follow-up", "Reviewed: no action", "False positive"],
                           label_visibility="collapsed")
            note = st.text_input("Note", placeholder="e.g. course change matches weather routing")
            if st.form_submit_button("Record review"):
                log_review({"task": "anomaly", "mmsi": int(r.mmsi), "window_end_utc": ui.stamp(T), "condition": cond,
                            "countermeasure_on": mitig, "system_label": lab, "score": round(float(r.score), 1)
                            if np.isfinite(r.score) else None, "operator_action": act, "reason": note,
                            "review_seconds_measured": round(time.time() - st.session_state[tkey], 1)})
                st.success("Review logged to results/review_log.csv.")


def queue(now, T):
    st.markdown(f"<div class='eyebrow'>alert queue · {ui.hhmm(T)} UTC</div>", unsafe_allow_html=True)
    st.pills("Vessel types", cats, selection_mode="multi", format_func=lambda c: CATEGORY_LABEL.get(c, c),
             key="w_gov_cats", on_change=ui.sync, args=("gov_cats",), label_visibility="collapsed")
    q = now[now.label.isin(["unusual", "high"])].sort_values("score", ascending=False)
    if q.empty:
        st.markdown("<div class='sub'>No unusual movement in view at this time.</div>", unsafe_allow_html=True)
    else:
        tab = pd.DataFrame({"vessel": [ui.name_of(m) for m in q.mmsi],
                            "type": [CATEGORY_LABEL.get(str(c), c) for c in q.category],
                            "score": q.score.round(0).astype(int),
                            "why": [", ".join(x.replace("_", " ") for x in ui.reasons_list(s)) or "combined"
                                    for s in q.reasons]})
        ev = st.dataframe(tab, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                          height=min(35 * len(tab) + 38, 330), key="gov_queue",
                          column_config={"score": st.column_config.ProgressColumn(
                              "score", min_value=0, max_value=100, format="%d")})
        rows = ev.selection.rows if ev is not None else []
        if rows:
            st.session_state["gov_sel"] = int(q.mmsi.iloc[rows[0]])
            livemap.command(playing=False)
            st.rerun()
    held = now[now.label.isin(["no_verdict", "no_data"])]
    if len(held):
        st.markdown(f"<div class='sub' style='margin-top:4px'>{len(held)} vessels held for data review "
                    "(newest report older than 20 min): shown as rings, not scored.</div>",
                    unsafe_allow_html=True)
    st.markdown("<div class='note' style='margin-top:8px'>Labels rank movement against learned traffic "
                "patterns. They are prompts for human review, not evidence of identity, intent or threat. "
                "Click a vessel on the map or in the list.</div>", unsafe_allow_html=True)


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
          + ([["learned lanes", ui.C["accent"], ""]] if "Sea lanes" in layers else []))
VERSION = f"gov|{area}|{cond}|{mit}|{','.join(focus)}|{','.join(layers)}"


@st.fragment
def live():
    map_col, side = st.columns([2.3, 1], gap="medium") if panel else (st.container(), None)
    sel = st.session_state.get("gov_sel")
    T_prev = livemap.current_step("gov_map", steps)
    now = W[(W["T"] == T_prev) & inreg]
    S = now[now.mmsi == sel].iloc[0] if sel is not None and (now.mmsi == sel).any() else None
    overlay, fit = None, ui.AREAS.get(area) or livemap.US_BOX
    if S is not None:
        trk = ui.track(int(S.mmsi), T_prev - 3 * 3600, T_prev)
        w0 = trk[trk.t > T_prev - 3600]
        paths = [livemap.path(trk.lat, trk.lon, ui.C["accent"], 1.5, alpha=0.45),
                 livemap.path(w0.lat, w0.lon, ui.C["accent"], 3)]
        if len(w0) >= 2:
            p = ui.lanes().path(w0.lat.iloc[0], w0.lon.iloc[0], w0.lat.iloc[-1], w0.lon.iloc[-1])
            paths.append(livemap.path(p.lat, p.lon, ui.C["fg"], 2, dash=True, alpha=0.85))
        overlay = {"mmsi": int(S.mmsi), "lat": float(S.lat), "lon": float(S.lon), "paths": paths, "points": [],
                   "c": ui.LABEL_COLOR.get(str(S.label), ui.C["fg"])}
        b = ui.zoom_bounds(np.r_[trk.lat, S.lat], np.r_[trk.lon, S.lon], 0.06) \
            if st.session_state.get("gov_zoom", True) else None
        if b is not None:
            fit = (b[2], b[0], b[3], b[1])
        LEGEND_SEL = [["track, last 60 min", ui.C["accent"], ""], ["expected route along lanes", ui.C["fg"], "dash"]]
    else:
        LEGEND_SEL = []
    grp = str(lane_group([S.category])[0]) if S is not None else "commercial"
    flagged = int(now.label.isin(["unusual", "high"]).sum())
    with map_col:
        T, hit = livemap.render(
            "gov_map", area=area, steps=steps, counts=alerts.groupby("T").size(), mk=MK, cats=focus,
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
    if side is not None:
        with side, st.container(key="side_box"):
            if S is not None:
                dossier(S, T)
            else:
                queue(now, T)


live()
