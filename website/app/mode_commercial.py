"""Commercial mode: when is this vessel expected to arrive, and who else is arriving at the same place?"""
import time

import numpy as np
import pandas as pd
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

area, layers, cond, mitig = ui.toolbar("com", ["Sea lanes"])
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


def zone_name(z) -> str:
    if not isinstance(z, str) or z not in Z.index:
        return "—"
    r = Z.loc[z]
    return f"{abs(r.lat):.3f}°{'N' if r.lat >= 0 else 'S'} {abs(r.lon):.3f}°{'W' if r.lon < 0 else 'E'}"


# ------------------------------------------------------------------------------------------- side panel
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
    t_pred, t_data, t_arr, t_review = st.tabs(["Prediction", "Data trust", "Arrivals here", "Review"])
    with t_pred:
        st.markdown(ui.kv([
            ("Stops at", ui.esc(zone_name(r.zone)) + " <span style='color:#8a95a3'>(predicted stop area)</span>"),
            ("Route", ui.esc(f"{ui.km(r[f'{tag}_route_km'])} along sea lanes") if np.isfinite(r[f"{tag}_route_km"])
             else "no lane route; ETA from straight-line distance"),
            ("Baseline ETA", ui.esc(ui.dur(r.bl_eta_h)) + " <span style='color:#8a95a3'>(distance ÷ speed)</span>"),
            ("Issued", ui.esc(ui.stamp(r.decision_t))),
        ]), unsafe_allow_html=True)
        with st.expander("What actually happened (never shown to the model)"):
            err = (r.eta_h - r.true_eta_h) * 60 if r.ok else np.nan
            st.markdown(ui.kv([("Stopped at", ui.esc(ui.stamp(r.arrival_t))),
                               ("Stop area", ui.esc(zone_name(r.true_zone)) + (" ✓" if r.zone == r.true_zone else "")),
                               ("ETA error", ui.esc(f"{err:+.0f} min" if np.isfinite(err) else "—"))]),
                        unsafe_allow_html=True)
            st.caption("Arrival = the vessel's next stop of 30+ min, seen in later AIS reports. Not a verified berth "
                       "call or unloading time.")
        st.toggle("Zoom to vessel", value=True, key="com_zoom")
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
    st.markdown(f"<div class='eyebrow' style='margin-top:4px'>inbound vessels · ETAs issued by {ui.hhmm(T)} UTC</div>",
                unsafe_allow_html=True)
    q = pd.concat([now[now.ok].sort_values("eta_t"), now[~now.ok]])
    if q.empty:
        st.markdown("<div class='sub'>No ETAs issued in view at this time. Try another region or time."
                    "</div>", unsafe_allow_html=True)
    else:
        tab = pd.DataFrame({"vessel": [ui.name_of(m) for m in q.mmsi],
                            "type": [GROUPS.get(g, g) for g in q.grp],
                            "ETA": [ui.hhmm(t) if ok else "held" for t, ok in zip(q.eta_t, q.ok)],
                            "in": [ui.dur(h) if ok else "review" for h, ok in zip(q.eta_h, q.ok)]})
        ev = st.dataframe(tab, hide_index=True, width="stretch", height="stretch", on_select="rerun",
                          selection_mode="single-row", key="com_queue")
        rows = ev.selection.rows if ev is not None else []
        if rows:
            st.session_state["com_sel"] = int(q.mmsi.iloc[rows[0]])
            livemap.command(playing=False)
            st.rerun()
    st.markdown("<div class='note'>ETA = time to the vessel's next 30+ min stop, routed on learned sea lanes. "
                "<b>held</b> = data too poor.</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------- live map + panel
STYLE_KEYS = [(g, ok) for g in GROUPS for ok in (True, False)]
mk_rows = view[view.step.isin(steps)]
MK = livemap.markers(mk_rows.assign(T=mk_rows.step),
                     [STYLE_KEYS.index((g, bool(ok))) for g, ok in zip(mk_rows.grp, mk_rows.ok)],
                     [{"c": ui.CAT_COLOR.get(g, ui.C["dim"]), "r": 6 if ok else 5, "ring": not ok, "edge": ui.C["fg"]}
                      for g, ok in STYLE_KEYS],
                     [f"{ui.name_of(m)} · " + (f"ETA {ui.hhmm(t)} ({ui.dur(h)})" if ok else "ETA held for review")
                      for m, t, h, ok in zip(mk_rows.mmsi, mk_rows.eta_t, mk_rows.eta_h, mk_rows.ok)])
LEGEND = ([[CATEGORY_LABEL.get(c, c), ui.CAT_COLOR.get(c, ui.C["dim"]), ""] for c in trail_cats]
          + [["ETA issued (white edge)", ui.C["fg"], "dot"], ["ETA held for review", ui.C["dim"], "ring"]]
          + ([["learned lanes", ui.C["accent"], ""]] if "Sea lanes" in layers else []))
VERSION = f"com|{area}|{cond}|{mitig}|{','.join(groups)}|{','.join(layers)}"


@st.fragment
def live():
    map_col, side = st.columns([2.3, 1], gap="medium")
    T_prev = livemap.current_step("com_map", steps)
    now = view[view.step == T_prev]
    sel = st.session_state.get("com_sel")
    S = now[now.mmsi == sel].iloc[0] if sel is not None and (now.mmsi == sel).any() else None
    overlay, fit, legend_sel = None, ui.AREAS.get(area) or livemap.US_BOX, []
    if S is not None:
        trk = ui.track(int(S.mmsi), S.decision_t - 2 * 3600, S.decision_t)
        route = pd.DataFrame(columns=["lat", "lon"])
        points = []
        if isinstance(S.zone, str) and S.zone in Z.index and np.isfinite(S.lat):
            route = ui.lanes().path(S.lat, S.lon, Z.at[S.zone, "lat"], Z.at[S.zone, "lon"])
            points = [{"lat": float(Z.at[S.zone, "lat"]), "lon": float(Z.at[S.zone, "lon"]), "c": ui.C["fg"],
                       "text": f"ETA {ui.hhmm(S.eta_t)}" if S.ok else "stop area"}]
        overlay = {"mmsi": int(S.mmsi), "lat": float(S.lat), "lon": float(S.lon), "c": ui.C["accent"],
                   "paths": [livemap.path(trk.lat, trk.lon, ui.C["accent"], 3),
                             livemap.path(route.lat, route.lon, ui.C["fg"], 2, dash=True, alpha=0.85)],
                   "points": points}
        legend_sel = [["track, last 2 h", ui.C["accent"], ""], ["expected route to stop area", ui.C["fg"], "dash"]]
        if st.session_state.get("com_zoom", True):
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
