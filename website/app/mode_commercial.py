"""Commercial mode: when is this vessel expected to arrive, and who else is arriving at the same place?"""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import livemap
import ui
from aisdp.lanes import CATEGORY_LABEL
from common import CONDITIONS, log_review

E0 = ui.eta_cases() if ui.available("destination_cases_seed0.parquet") else pd.DataFrame()
if E0.empty or ui.lanes() is None:
    ui.unavailable("ETA prediction")
Z = ui.zones()
HELD = {"defer_quality": "held: newest report too old or an implausible jump",
        "abstain_low_score": "held: no destination stood out",
        "abstain_no_zone_in_reach": "held: no known stop area within reach"}

area, layers, cond, mitig = ui.toolbar("com", ["Sea lanes", "Weather"])
tag = "mi" if mitig else "un"
E = E0[E0.condition == cond].copy()
E["category"] = [ui.category_of(v) for v in E.vessel_type]
E["lat"], E["lon"] = (E.mi_lat, E.mi_lon) if mitig else (E.in_lat, E.in_lon)
E["zone"], E["eta_h"] = E[f"{tag}_pred1"], E[f"{tag}_eta_h"]
E["lo_h"], E["hi_h"], E["ok"] = E[f"{tag}_eta_lo_h"], E[f"{tag}_eta_hi_h"], E[f"{tag}_eta_answered"]
E["eta_t"] = E.decision_t + E.eta_h * 3600
E["step"] = np.ceil(E.decision_t / 1800) * 1800
GROUPS = {"cargo": "Cargo", "tanker": "Tanker", "other": "Other commercial"}
E["grp"] = np.where(E.category.isin(["cargo", "tanker"]), E.category, "other")
ui.keep("com_groups", ["cargo", "tanker"])
groups = st.session_state["com_groups"] or list(GROUPS)
view = E[ui.in_box(E.lat.fillna(0), E.lon.fillna(0), area) & E.grp.isin(groups)]
steps = np.sort(ui.windows("nominal", "quality_aware")["T"].unique()) if ui.available("anomaly_windows.parquet") \
    else np.sort(E.step.unique())
trail_cats = [c for c in ui.CAT_COLOR if c in groups or ("other" in groups and c not in ("cargo", "tanker"))]


FEED_COLOR = {"nominal": ui.C["fg"], "stale": ui.C["warn"], "low_quality": ui.C["crit"], "combined": "#c07be0"}


def miss_min(x: pd.DataFrame, t: str) -> pd.Series:
    """Minutes each issued ETA was off the actual arrival (+ late, - early); held ETAs are left out."""
    ok = x[f"{t}_eta_answered"].astype(bool)
    return ((x.decision_t + x[f"{t}_eta_h"] * 3600 - x.arrival_t) / 60)[ok]


@st.cache_data
def demo_case() -> dict | None:
    """The poisoning example the demo opens: a cargo/tanker trip whose ETAs are good on the clean feed and go wrong on
    degraded feeds, where the countermeasure helps most. Picked from the results by this rule, not hand-made."""
    d = E0[E0.vessel_type.between(70, 89)]
    best, score = None, 0.0
    for (m, a), x in d.groupby(["mmsi", "arrival_t"]):
        c = {k: g for k, g in x.groupby("condition")}
        if len(c.get("nominal", [])) < 6 or miss_min(c["nominal"], "un").abs().mean() > 20:
            continue
        s = sum(miss_min(c[k], "un").abs().mean() - miss_min(c[k], "mi").abs().mean()
                for k in ("stale", "low_quality", "combined") if k in c)
        if s > score:
            best, score = c, s
    if best is None:
        return None
    lq = best["low_quality"]                         # open on the update the countermeasure fixes most
    gain = miss_min(lq, "un").abs() - miss_min(lq, "mi").abs().reindex(lq.index)
    r = lq.loc[gain.idxmax()]
    return {"mmsi": int(r.mmsi), "step": float(np.ceil(r.decision_t / 1800) * 1800),
            "grp": "tanker" if 80 <= r.vessel_type <= 89 else "cargo", "lat": float(r.in_lat), "lon": float(r.in_lon)}


def zone_name(z) -> str:
    if not isinstance(z, str) or z not in Z.index:
        return "—"
    r = Z.loc[z]
    return f"{abs(r.lat):.3f}°{'N' if r.lat >= 0 else 'S'} {abs(r.lon):.3f}°{'W' if r.lon < 0 else 'E'}"


# ------------------------------------------------------------------------------------------- side panel
def feed_test(r):
    """The poisoning test for one trip: the ETAs issued under each of the four feeds, as lines over time, with the
    countermeasure as set in the toolbar. Same ship, same frozen model, same moments: only the feed differs."""
    trip = E0[(E0.mmsi == r.mmsi) & (E0.arrival_t == r.arrival_t)].sort_values("decision_t")
    clock = lambda x, t: pd.to_datetime(x.decision_t + x[f"{t}_eta_h"] * 3600, unit="s")
    arr = pd.to_datetime(r.arrival_t, unit="s")
    # fixed range over both countermeasure settings, so flipping the toggle moves the lines, not the axes
    vals = pd.concat([clock(trip, "un"), clock(trip, "mi")]).dropna()
    lo, hi = min(vals.min(), arr), max(vals.max(), arr)
    pad = (hi - lo) * 0.08 + pd.Timedelta(minutes=10)
    held_y = lo - pad
    x0, x1 = pd.to_datetime(trip.decision_t.min(), unit="s"), pd.to_datetime(trip.decision_t.max(), unit="s")
    f = go.Figure()
    f.add_scatter(x=[x0, x1], y=[arr, arr], mode="lines", name=f"actual stop {ui.hhmm(r.arrival_t)}",
                  line=dict(color=ui.C["track"], width=2, dash="dash"), hoverinfo="skip")
    for i, (c, lab) in enumerate(CONDITIONS.items()):
        x = trip[trip.condition == c]
        ok, on = x[f"{tag}_eta_answered"].astype(bool), c == cond
        t = pd.to_datetime(x.decision_t, unit="s")
        f.add_scatter(x=t, y=clock(x, tag).where(ok), mode="lines+markers", name=lab, connectgaps=False,
                      opacity=1 if on else 0.5, line=dict(color=FEED_COLOR[c], width=3.5 if on else 1.5),
                      marker=dict(size=8 if on else 5), hovertemplate=f"{lab}<br>issued %{{x|%H:%M}} · ETA "
                                                                       "%{y|%H:%M}<extra></extra>")
        if (~ok).any():                              # held ETAs sit on their own row under the chart
            f.add_scatter(x=t[~ok], y=[held_y - pad * 0.22 * i] * int((~ok).sum()), mode="markers", showlegend=False,
                          opacity=1 if on else 0.5, marker=dict(symbol="circle-open", size=10 if on else 8,
                                                                color=FEED_COLOR[c], line=dict(width=2)),
                          hovertemplate=f"{lab}<br>issued %{{x|%H:%M}} · held, sent to the scheduler<extra></extra>")
    f.add_shape(type="line", x0=pd.to_datetime(r.decision_t, unit="s"), x1=pd.to_datetime(r.decision_t, unit="s"),
                y0=0, y1=1, yref="paper", line=dict(color=ui.C["dim"], width=1, dash="dot"))
    if (~trip.mi_eta_answered.astype(bool)).any() or (~trip.un_eta_answered.astype(bool)).any():
        f.add_annotation(x=1, xref="paper", y=held_y - pad * 0.33, text="held", showarrow=False, xanchor="left",
                         font=dict(size=12, color=ui.C["dim"]))
    f.update_layout(height=270, margin=dict(l=0, r=34, t=4, b=0), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=ui.C["fg"]),
                    legend=dict(orientation="h", y=-0.16, x=0, font=dict(size=12)),
                    xaxis=dict(tickformat="%H:%M", gridcolor=ui.C["line"], title=dict(text="ETA issued at", font=dict(size=12))),
                    yaxis=dict(tickformat="%H:%M", gridcolor=ui.C["line"], range=[held_y - pad * 0.9, hi + pad * 0.4],
                               title=dict(text="predicted arrival", font=dict(size=12))))
    st.markdown(f"<div class='eyebrow'>ETA under each feed · countermeasure {'ON' if mitig else 'OFF'}</div>",
                unsafe_allow_html=True)
    st.plotly_chart(f, width="stretch", config={"displayModeBar": False}, theme=None, key="com_feeds")

    def cell(x, t):
        m, held = miss_min(x, t).abs(), int((~x[f"{t}_eta_answered"].astype(bool)).sum())
        s = f"{m.mean():.0f} / {m.max():.0f}" if len(m) else "—"
        return s + (f" <span style='color:{ui.C['dim']}'>· {held} held</span>" if held else "")
    rows = "".join(
        f"<tr{' class=cur' if c == cond else ''}><td><i style='background:{FEED_COLOR[c]}'></i>{ui.esc(lab)}</td>"
        f"<td{' class=on' if not mitig else ''}>{cell(trip[trip.condition == c], 'un')}</td>"
        f"<td{' class=on' if mitig else ''}>{cell(trip[trip.condition == c], 'mi')}</td></tr>"
        for c, lab in CONDITIONS.items())
    st.markdown(f"<table class='ft'><tr><th>minutes off (avg / worst)</th><th>countermeasure off</th>"
                f"<th>on</th></tr>{rows}</table>", unsafe_allow_html=True)
    st.markdown("<div class='note'>Same ship, same frozen model, same moments: only the feed differs. Stale and "
                "unreliable feeds are SIMULATED faults on the real AIS. Hollow circles = ETA held and sent to the "
                "scheduler instead of issued. Dotted line = now.</div>", unsafe_allow_html=True)


def dossier(r, T):
    h1, h2 = st.columns([3, 1], vertical_alignment="top")
    h1.markdown(f"<div class='eyebrow'>selected vessel</div><h3 style='margin:0'>{ui.esc(ui.name_of(r.mmsi))}</h3>"
                f"<div style='margin:4px 0 4px'>{ui.category_chip(r.category)}</div>", unsafe_allow_html=True)
    if h2.button("✕ Clear", key="com_clear", width="stretch"):
        st.session_state.pop("com_sel", None)
        st.rerun()
    if r.ok:
        st.markdown(f"<div class='eyebrow'>estimated arrival</div><div class='big'>{ui.hhmm(r.eta_t)}"
                    f"<small>UTC · in {ui.dur(r.eta_h)}</small></div><div class='sub'>80% window "
                    f"{ui.hhmm(r.decision_t + r.lo_h * 3600)}–{ui.hhmm(r.decision_t + r.hi_h * 3600)} UTC</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='big'>—<small>no ETA</small></div><div class='sub'>"
                    f"{ui.esc(HELD.get(r.mi_action, 'held: no destination'))}. Sent to the scheduler to review "
                    "instead of guessing.</div>", unsafe_allow_html=True)
    t_feed, t_pred, t_data, t_arr, t_review = st.tabs(["Poisoning test", "Prediction", "Data trust", "Arrivals here",
                                                       "Review"])
    with t_feed:
        feed_test(r)
    with t_pred:
        st.markdown(ui.kv([
            ("Stops at", ui.esc(zone_name(r.zone)) + " <span style='color:#8a95a3'>(predicted stop area)</span>"),
            ("Route", ui.esc(f"{ui.km(r[f'{tag}_route_km'])} along sea lanes") if np.isfinite(r[f"{tag}_route_km"])
             else "no lane route; ETA from straight-line distance"),
            ("Baseline ETA", ui.esc(ui.dur(r.bl_eta_h)) + " <span style='color:#8a95a3'>(distance ÷ speed)</span>"),
            ("Issued", ui.esc(ui.stamp(r.decision_t))),
        ]), unsafe_allow_html=True)
        err = (r.eta_h - r.true_eta_h) * 60 if r.ok else np.nan
        st.markdown("<div class='eyebrow'>what actually happened (never shown to the model)</div>"
                    + ui.kv([("Stopped at", ui.esc(ui.stamp(r.arrival_t))),
                             ("Stop area", ui.esc(zone_name(r.true_zone)) + (" ✓" if r.zone == r.true_zone else "")),
                             ("ETA error", ui.esc(f"{err:+.0f} min" if np.isfinite(err) else "—"))]),
                    unsafe_allow_html=True)
        st.caption("Arrival = the vessel's next stop of 30+ min, seen in later AIS reports. Not a verified berth "
                   "call or unloading time.")
    with t_data:
        sim = r.source != "AIS-original" or r.is_stale
        st.markdown(ui.kv([
            ("Report age", ui.esc(ui.dur(r.age_s / 3600)) + (" <span style='color:#e9b44c'>over 20 min: held</span>"
                                                             if r.age_s > 1200 else "")),
            ("Source", ui.esc("unreliable test source (SIMULATED)" if r.source != "AIS-original" else "AIS feed")),
            ("Defects", ui.esc(r.defects if isinstance(r.defects, str) and r.defects else "none injected")),
            ("Speed / course", ui.esc(f"{r.in_sog:.1f} kn / {r.in_cog:.0f}°" if np.isfinite(r.in_sog) and
                                      np.isfinite(r.in_cog) else "missing or invalid")),
            ("Position", ui.esc(f"{r.lat:.4f}, {r.lon:.4f}" + (" (age-compensated)" if mitig and r.age_s > 120 else ""))),
            ("MMSI / flag", ui.esc(f"{int(r.mmsi)} · {ui.mid_flag(r.mmsi)}")),
        ]), unsafe_allow_html=True)
        st.caption(("This vessel's input was degraded in the simulation. " if sim else "")
                   + "The countermeasure drops impossible speeds and position jumps, moves an old position forward "
                     "along its course, and holds the ETA when the newest report is over 20 min old.")
    with t_arr:
        # port congestion context: other ETAs into the same or adjacent stop area (~10 km)
        near = pd.DataFrame()
        if isinstance(r.zone, str) and r.zone in Z.index:
            zl, zo = Z.at[r.zone, "lat"], Z.at[r.zone, "lon"]
            near = E[(E.step == T) & E.ok & E.zone.isin(Z.index)].copy()
            near = near[(np.abs(near.zone.map(Z.lat) - zl) < 0.1) & (np.abs(near.zone.map(Z.lon) - zo) < 0.1)]
            near = near.sort_values("eta_t")
            if len(near) > 1:
                lo, hi = r.decision_t + r.lo_h * 3600, r.decision_t + r.hi_h * 3600
                ov = ((near.decision_t + near.lo_h * 3600 <= hi) & (near.decision_t + near.hi_h * 3600 >= lo)).sum() - 1
                st.markdown(f"<div class='sub'>{len(near)} tracked vessels due within ~10 km; {ov} with an "
                            "overlapping arrival window. Berth capacity is not in the data.</div>",
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({"#": np.arange(1, len(near) + 1), "vessel": [ui.name_of(m) for m in near.mmsi],
                                           "ETA": [ui.hhmm(t) for t in near.eta_t],
                                           "window": [f"{ui.hhmm(d + a * 3600)}–{ui.hhmm(d + b * 3600)}" for d, a, b in
                                                      zip(near.decision_t, near.lo_h, near.hi_h)]}),
                             hide_index=True, width="stretch", height=min(35 * len(near) + 38, 300))
        if len(near) <= 1:
            st.markdown("<div class='sub'>No other tracked vessels due within ~10 km of this stop area now.</div>",
                        unsafe_allow_html=True)
    with t_review:
        st.markdown("<div class='sub'>Reviewer: <b>port scheduler</b>. Use the ETA, correct it, or ask for a fresh "
                    "position. Every decision and its review time is logged.</div>", unsafe_allow_html=True)
        tkey = f"com_t0_{int(r.mmsi)}_{T}"
        st.session_state.setdefault(tkey, time.time())
        with st.form("com_review", clear_on_submit=True, border=False):
            act = st.radio("Scheduler decision", ["Use ETA for scheduling", "Correct ETA", "Request fresh position"],
                           label_visibility="collapsed")
            fix = st.text_input("Corrected ETA (UTC)", placeholder="e.g. 14:30")
            note = st.text_input("Reason", placeholder="e.g. pilot reports 30 min delay")
            if st.form_submit_button("Record decision"):
                log_review({"task": "eta", "case_id": int(r.case_id), "mmsi": int(r.mmsi), "condition": cond,
                            "countermeasure_on": mitig, "eta_utc": ui.stamp(r.eta_t) if r.ok else "none",
                            "operator_action": act, "correction": fix, "reason": note,
                            "review_seconds_measured": round(time.time() - st.session_state[tkey], 1)})
                st.success("Decision logged to results/review_log.csv.")


def queue(now, T):
    D = demo_case()
    if D is not None and st.button(f"▶ Poisoning test: {ui.name_of(D['mmsi'])}, ETA under four feeds",
                                   key="com_demo", width="stretch",
                                   help="Opens one cargo ship on the unreliable feed with the countermeasure off. "
                                        "Then flip the countermeasure and the feeds."):
        area = next((a for a, b in ui.AREAS.items() if b and ui.in_box([D["lat"]], [D["lon"]], a)[0]), "Whole U.S.")
        st.session_state.update(area=area, cond="low_quality", mitig=False, com_sel=D["mmsi"],
                                com_groups=sorted(set(st.session_state["com_groups"] or []) | {D["grp"]}))
        livemap.command(playing=False, T=steps[min(np.searchsorted(steps, D["step"]), len(steps) - 1)])
        st.rerun()
    st.markdown(f"<div class='eyebrow' style='margin-top:4px'>inbound vessels · ETAs issued by {ui.hhmm(T)} UTC · click a row to open</div>",
                unsafe_allow_html=True)
    q = pd.concat([now[now.ok].sort_values("eta_t"), now[~now.ok]])
    if q.empty:
        st.markdown("<div class='sub'>No ETAs issued in view at this time. Try another region or time."
                    "</div>", unsafe_allow_html=True)
    else:
        hit = ui.queue_rows("com_queue", [18, 7, 5], ["vessel", "type", "ETA", "in"], [
            [ui.name_of(m), {"other": "Other"}.get(g, GROUPS.get(g, g)), ui.hhmm(t) if ok else "held",
             ui.dur(h) if ok else "review"] for m, g, t, h, ok in zip(q.mmsi, q.grp, q.eta_t, q.eta_h, q.ok)])
        if hit is not None:
            st.session_state["com_sel"] = int(q.mmsi.iloc[hit])
            livemap.command(playing=False)
            st.rerun()
    st.markdown("<div class='note'>ETA = time to the vessel's next 30+ min stop, routed on learned sea lanes. "
                "<b>held</b> = data too poor. Why it matters: just-in-time supply chains plan production around "
                "arrivals, margins are thin, and an idle ship or berth earns nothing.</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- live map + panel
STYLE_KEYS = [(g, ok) for g in GROUPS for ok in (True, False)]
mk_rows = view[view.step.isin(steps)]
MK = livemap.markers(mk_rows.assign(T=mk_rows.step),
                     [STYLE_KEYS.index((g, bool(ok))) for g, ok in zip(mk_rows.grp, mk_rows.ok)],
                     [{"c": ui.CAT_COLOR.get(g, ui.C["dim"]) if ok else "#ffffff", "r": 6, "ring": not ok, "edge": "#ffffff"}
                      for g, ok in STYLE_KEYS],
                     [f"{ui.name_of(m)} · " + (f"ETA {ui.hhmm(t)} ({ui.dur(h)})" if ok else "ETA held for review")
                      for m, t, h, ok in zip(mk_rows.mmsi, mk_rows.eta_t, mk_rows.eta_h, mk_rows.ok)])
LEGEND = ([[CATEGORY_LABEL.get(c, c), ui.CAT_COLOR.get(c, ui.C["dim"]), ""] for c in trail_cats]
          + [["ETA issued (white edge)", ui.C["fg"], "edge"], ["ETA held for review (hollow)", "#ffffff", "ring"]]
          + ([ui.LANE_LEGEND] if "Sea lanes" in layers else [])
          + ([ui.DENSITY_LEGEND] if "Density" in layers else [])
          + ([ui.WEATHER_LEGEND] if "Weather" in layers else []))
VERSION = f"com|{area}|{cond}|{mitig}|{','.join(groups)}|{','.join(layers)}"


@st.fragment
def live():
    map_col, side = st.columns([2.3, 1], gap="medium")
    T_prev = livemap.current_step("com_map", steps)
    now = view[view.step == T_prev]
    sel = st.session_state.get("com_sel")
    picked = E[(E.step == T_prev) & (E.mmsi == sel)]        # a picked vessel stays open outside the region box
    S = picked.iloc[0] if sel is not None and len(picked) else None
    overlay, fit, legend_sel = None, ui.AREAS.get(area) or livemap.US_BOX, []
    if S is not None:
        trk = ui.track(int(S.mmsi), S.decision_t - 2 * 3600, S.decision_t)
        route = pd.DataFrame(columns=["lat", "lon"])
        points = []
        if isinstance(S.zone, str) and S.zone in Z.index and np.isfinite(S.lat):
            route = ui.lanes().path(S.lat, S.lon, Z.at[S.zone, "lat"], Z.at[S.zone, "lon"])
            points = [{"lat": float(Z.at[S.zone, "lat"]), "lon": float(Z.at[S.zone, "lon"]), "c": ui.C["fg"],
                       "text": f"ETA {ui.hhmm(S.eta_t)}" if S.ok else "stop area"}]
        overlay = {"mmsi": int(S.mmsi), "lat": float(S.lat), "lon": float(S.lon), "c": ui.C["track"],
                   "paths": [livemap.path(trk.lat, trk.lon, ui.C["track"], 3),
                             livemap.path(route.lat, route.lon, ui.C["fg"], 2, dash=True, alpha=0.85)],
                   "points": points}
        legend_sel = [["track, last 2 h", ui.C["track"], ""], ["expected route to stop area", ui.C["fg"], "dash"]]
        b = ui.zoom_bounds(np.r_[trk.lat, route.lat, S.lat], np.r_[trk.lon, route.lon, S.lon], 0.06)
        if b is not None:
            fit = (b[2], b[0], b[3], b[1])
    with map_col:
        T, hit = livemap.render(
            "com_map", area=area, steps=steps, counts=view[view.ok].groupby("step").size(),
            bars="ETAs issued in view", degr=ui.degraded_share(cond), mk=MK, cats=trail_cats,
            layer_groups=tuple([c for c in groups if c != "other"] + (["commercial"] if "other" in groups else [])),
            layers=layers, legend=LEGEND + legend_sel, sel=overlay, fit_box=fit, click_heads=False, version=VERSION,
            hud=[f"{int(now.ok.sum())} ETAs issued in the last 30 min · {int((~now.ok).sum())} held",
                 f"feed: {CONDITIONS[cond]}{' (SIMULATED)' if cond != 'nominal' else ''} · countermeasure "
                 f"{'on' if mitig else 'off'}"])
    if hit is not None and hit != sel:
        st.session_state["com_sel"] = hit
        st.rerun()
    if T != T_prev:                                  # the map moved to another step: refresh the panel for it
        st.rerun(scope="fragment")
    with side, st.container(key="side_box"):
        if S is not None:
            dossier(S, T)
        else:
            queue(now, T)


live()
