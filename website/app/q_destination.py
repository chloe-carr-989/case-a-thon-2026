"""Where is this vessel going? Next inferred arrival zone within 6 hours (candidate-zone ranker)."""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (AREAS, CONDITIONS, eight_cell, geo_layout, hhmm, in_area, manifest, metrics, review_form, table,
                    task_status, unavailable)

DEFECT_TEXT = {"jitter": "position noise (about 200 m)", "position_outlier": "position jumped several km",
               "sog_scaled": "speed value wrong", "sog_missing": "speed missing",
               "sog_sentinel": "speed = 102.3, the AIS code for 'unknown'", "cog_random": "course value wrong",
               "cog_missing": "course missing"}
ACTION_TEXT = {"model": "answered by the ranker", "defer_quality": "input failed the quality checks (too old or an "
               "implausible jump with no credible fallback): sent to a reviewer",
               "abstain_low_score": "no zone scored high enough: answered 'unknown'",
               "abstain_no_zone_in_reach": "no known zone within reach: answered 'unknown'"}

st.title("Where is this vessel going?")
st.caption("Predicts the next **inferred arrival zone**: the 0.05° cell where the vessel next stops for 30+ minutes "
           "within 6 hours. It uses only AIS reports up to the decision time. Zones are derived from where vessels dwelt "
           "during training. They are coordinates, not named ports.")
C0, Z = table("destination_cases_seed0.parquet"), table("destination_zones.csv")
if C0 is None or Z is None:
    unavailable("destination", "Destination prediction")
M = metrics("destination")
run = task_status("destination")
zpos = Z.set_index("zone_id")[["lat", "lon", "train_vessels", "train_arrivals"]]

c1, c2, c3, c4 = st.columns([3, 2, 3, 2])
cond = c1.segmented_control("Feed condition", list(CONDITIONS), format_func=CONDITIONS.get, default="combined",
                            key="dst_cond") or "combined"
mitig = c2.toggle("Countermeasure on", value=False, key="dst_mit",
                  help="Same input checks as the 15-minute task: age gate, implausible-jump fallback, speed/course "
                       "repair from the track. Also answers 'unknown' when no zone scores high enough.")
edges = run.get("test_window_edges_t") or []
wins = sorted(C0.window.unique())
wl = {w: (f"{hhmm(edges[w])}–{hhmm(edges[w + 1])}" if len(edges) > w + 1 else str(w)) for w in wins}
win = c3.select_slider("Decision time (UTC). Feed quality gets worse later in the day.", wins, value=max(wins),
                       format_func=wl.get, key="dst_win")
area = c4.selectbox("Area", list(AREAS), key="dst_area")

C = C0[(C0.condition == cond) & (C0.window == win)]
C = C[in_area(C.in_lat.fillna(0), C.in_lon.fillna(0), area)].copy()
if C.empty:
    st.info("No labelled decisions in this area and time window. Try another area or time.")
    st.stop()
tag = "mi" if mitig else "un"
C["p1"], C["ans"] = C[f"{tag}_pred1"], C[f"{tag}_answered"]
C["hit1"] = C.ans & (C.p1 == C.true_zone)
C["hit3"] = C.ans & ((C.p1 == C.true_zone) | (C[f"{tag}_pred2"] == C.true_zone) | (C[f"{tag}_pred3"] == C.true_zone))
bl1 = (C.bl_answered & (C.bl_pred1 == C.true_zone)).mean()

k1, k2, k3, k4 = st.columns(4)
na = int(C.ans.sum())
k1.metric("Top-1 correct (answered)", f"{C.hit1.sum() / na:.0%}" if na else "n/a",
          delta=f"baseline {bl1:.0%}", delta_color="off")
k2.metric("Right zone in top 3", f"{C.hit3.sum() / na:.0%}" if na else "n/a")
k3.metric("Answered", f"{na} of {len(C)}")
k4.metric("Top-1 correct over ALL cases", f"{C.hit1.mean():.0%}",
          help="Unknown/deferred count as wrong here, so abstaining cannot look accurate.")

map_col, side = st.columns([5, 2], gap="medium")
with map_col:
    sel = st.session_state.get("dst_sel")
    if sel not in set(C.case_id):
        cand = C[C.is_lowq | C.is_stale]
        sel = int((cand if len(cand) else C).sort_values("decision_t").case_id.iloc[0])
    S = C[C.case_id == sel].iloc[0]
    fig = go.Figure()
    zs = Z[in_area(Z.lat, Z.lon, area)]
    fig.add_scattergeo(lat=zs.lat, lon=zs.lon, mode="markers", name="inferred arrival zones",
                       marker=dict(size=4, color="#999", symbol="square"), text=zs.zone_id,
                       hovertemplate="%{text}<extra></extra>")
    for lab, m, col in [("top-1 correct", C.hit1, "#2e8b57"), ("top-1 wrong", C.ans & ~C.hit1, "#c62828")]:
        s = C[m]
        fig.add_scattergeo(lat=s.in_lat, lon=s.in_lon, mode="markers", name=f"vessel at decision: {lab}",
                           marker=dict(size=7, color=col, line=dict(width=0.5, color="white")), customdata=s.case_id,
                           text=[f"MMSI {a} · {hhmm(t)}" for a, t in zip(s.mmsi, s.decision_t)],
                           hovertemplate="%{text}<extra></extra>")
    R = C[~C.ans]
    if len(R):
        fig.add_scattergeo(lat=R.in_lat, lon=R.in_lon, mode="markers", name="unknown / sent to reviewer",
                           marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(width=1.5, color="#666")),
                           customdata=R.case_id, text=[f"MMSI {a}" for a in R.mmsi], hovertemplate="%{text}<extra></extra>")
    # selected case: input -> predictions, and the actual arrival
    preds = [S[f"{tag}_pred{k}"] for k in (1, 2, 3)] if S.ans else []
    for k, z in enumerate(preds):
        if isinstance(z, str) and z in zpos.index:
            fig.add_scattergeo(lat=[S.in_lat, zpos.at[z, "lat"]], lon=[S.in_lon, zpos.at[z, "lon"]], mode="lines+markers",
                               name=f"selected: prediction #{k + 1}" if k == 0 else None, showlegend=k == 0,
                               line=dict(width=3 if k == 0 else 1, color="#1565c0", dash="solid" if k == 0 else "dot"),
                               marker=dict(size=[0, 12 - 3 * k], color="#1565c0"), hoverinfo="skip")
    fig.add_scattergeo(lat=[S.arr_lat], lon=[S.arr_lon], mode="markers", name="selected: where it actually stopped",
                       marker=dict(size=15, symbol="star", color="#222"), hoverinfo="skip")
    fig.add_scattergeo(lat=[S.in_lat], lon=[S.in_lon], mode="markers", name="selected: report the AI received",
                       marker=dict(size=13, symbol="square", color="#1565c0", line=dict(width=1, color="white")),
                       hoverinfo="skip")
    zoom = st.toggle("Zoom to this vessel", key="dst_zoom")
    bounds = None
    if zoom:
        pts = [(S.in_lat, S.in_lon), (S.arr_lat, S.arr_lon)] + [(zpos.at[z, "lat"], zpos.at[z, "lon"]) for z in preds
                                                                  if isinstance(z, str) and z in zpos.index]
        la, lo = [p[0] for p in pts], [p[1] for p in pts]
        bounds = (min(la) - 0.1, max(la) + 0.1, min(lo) - 0.1, max(lo) + 0.1)
    geo_layout(fig, area, height=600, bounds=bounds)
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="dst_map")
    pts = getattr(getattr(ev, "selection", None), "points", []) if not isinstance(ev, dict) else \
        ev.get("selection", {}).get("points", [])
    clicked = [p.get("customdata") for p in pts if p.get("customdata") is not None]
    if clicked and int(np.ravel(clicked[0])[0]) != sel:
        st.session_state["dst_sel"] = int(np.ravel(clicked[0])[0])
        st.rerun()
    st.caption(f"{len(C)} held-out decisions by underway vessels with an observed arrival in a known zone · {area} · "
               f"{wl[win]} UTC · seed 0. Click a vessel to inspect it.")

with side:
    order = C.sort_values("decision_t")
    ids = order.case_id.tolist()
    pick = st.selectbox("Decision", ids, index=ids.index(sel), key="dst_pick",
                        format_func=lambda c: f"MMSI {order.set_index('case_id').at[c, 'mmsi']} · "
                                              f"{hhmm(order.set_index('case_id').at[c, 'decision_t'])}")
    if pick != sel:
        st.session_state["dst_sel"] = int(pick)
        st.rerun()
    tkey = f"dst_t0_{sel}_{cond}"
    st.session_state.setdefault(tkey, time.time())
    st.markdown(f"#### Decision at {hhmm(S.decision_t)} UTC (input cutoff)")
    lines = [f"- Report {S.age_s / 60:.0f} min old" if S.age_s > 120 else "- Fresh report (under 2 min old)",
             f"- Speed {S.in_sog:.1f} kn, course {S.in_cog:.0f}°" if np.isfinite(S.in_sog) and np.isfinite(S.in_cog)
             else "- Speed or course missing"]
    if S.source != "AIS-original":
        defs = [DEFECT_TEXT.get(d, d) for d in str(S.defects).split(";") if d]
        lines.append(f"- From the **unreliable test source**: {', '.join(defs)}")
    if isinstance(S.last_zone, str):
        lines.append(f"- Last stop before this (history): {S.last_zone}")
    st.markdown("**1 · What the AI received**\n" + "\n".join(lines))

    def zl(z):
        return z if isinstance(z, str) else "—"
    st.markdown("**2 · Predicted next arrival zone**\n" +
                ("\n".join(f"- #{k}: {zl(S[f'{tag}_pred{k}'])}" for k in (1, 2, 3)) +
                 f"\n- Top model score {S[f'{tag}_top_score']:.2f} (normalised score, not a probability)"
                 if S.ans else "- **unknown / sent to reviewer**") +
                f"\n- Baseline (heading-aligned nearest): {zl(S.bl_pred1)}")
    st.markdown("**3 · What actually happened (held-out reference)**\n"
                f"- Stopped at {hhmm(S.arrival_t)} UTC ({(S.arrival_t - S.decision_t) / 3600:.1f} h later) in "
                f"**{S.true_zone}**\n- Outcome: " +
                ("top-1 correct" if S.hit1 else "in top 3" if S.hit3 else "not in top 3" if S.ans else "no answer"))
    if mitig:
        flags = [f for f in str(S.mi_flags).split(";") if f]
        st.markdown(f"**Countermeasure:** {ACTION_TEXT.get(S.mi_action, S.mi_action)}" +
                    (f" · flags: {', '.join(flags)}" if flags else ""))
    st.markdown("**4 · Reviewer decision**")
    review_form("dst", "destination", {"case_id": int(sel), "mmsi": int(S.mmsi), "condition": cond,
                                       "countermeasure_on": mitig, "system_action": S.mi_action if mitig else "model",
                                       "predicted_zone": zl(S.p1) if S.ans else "unknown"},
                ["Accept prediction", "Correct destination", "Defer: request fresh reports"], st.session_state[tkey],
                correction_label="Corrected zone id (if correcting)")

st.divider()
with st.expander("Four feed conditions × countermeasure (all held-out decisions, mean of seeds)", expanded=True):
    cols = {"top1_acc": "top-1 (answered)", "top3_acc": "top-3 (answered)", "coverage": "answered",
            "top1_over_all_cases": "top-1 over all cases"}
    t = eight_cell(M, cols, "hgb_ranker", extra_none={"heading_nearest_baseline": "heading-nearest baseline"})
    st.dataframe(t.style.format({c: "{:.3f}" for c in cols.values()}), hide_index=True, width="stretch")
    sc = run.get("status_counts", {}).get("test", {})
    n_scored = int(M[(M.condition == "nominal") & (M.subgroup == "all") & (M.window.astype(str) == "all")].n_cases.max())
    st.caption(f"{n_scored} held-out decisions from {run.get('n_test_vessels', '?')} vessels "
               f"not seen in training, across {run.get('n_test_classes', '?')} distinct arrival zones "
               f"({run.get('n_zones', '?')} zones in total). Only decisions followed by an observed stop in a known zone are "
               f"scored; test-period decisions that were not scored: {sc.get('arrival_unzoned', 0)} stopped outside "
               f"known zones, {sc.get('no_arrival_in_window', 0)} had no stop within 6 h, {sc.get('censored', 0)} "
               "were censored by the end of the data.")
with st.expander("Rare zones, regions and confusion (clean feed)"):
    g = M[(M.condition == "nominal") & (M.window.astype(str) == "all") & (M.mitigation == "none")]
    st.dataframe(g.groupby(["subgroup", "predictor"])[["n_cases", "top1_acc", "top3_acc", "candidate_recall"]].mean()
                 .round(3).reset_index(), hide_index=True, width="stretch")
    cf = table("destination_confusion.csv")
    if cf is not None:
        st.caption("True zone × predicted zone, for zones with 5+ held-out arrivals (OTHER_ZONE = a zone outside this "
                   "set, UNKNOWN = abstained).")
        st.dataframe(cf.sort_values(["support", "n"], ascending=False).head(60), hide_index=True, width="stretch")
with st.expander("Definition and known limits"):
    st.markdown("""
- **Label:** the first stop (30+ min below 1 kn) that starts after the decision and within 6 h, taken from *later*
  AIS reports. It is used only as the target, never as an input. A stop is not a verified berth call or cargo delivery.
- **Zones:** 0.05° cells where at least 3 vessels stopped during the training half of the day. They are coordinates,
  not port names. Stops elsewhere are counted as `arrival_unzoned` and are not scored.
- **Conditional accuracy:** accuracy is measured only on decisions that ended in a zoned stop. The model is never
  asked "will it stop at all?"
- **Split:** vessels are disjoint from training and decisions come later in the day. One day of data means short
  trips are over-represented near the end (censoring).
- **Scores** are normalised model scores, not calibrated probabilities.
- **Stress tests** reuse the 15-minute task's simulated stale and unreliable-source defects on the same held-out
  decisions. The countermeasure mostly trades coverage for accuracy here: destinations are hours away, so a few km of
  position error matter less than they do for a 15-minute position.
""")
    st.json(manifest().get("tasks", {}).get("destination", {}), expanded=False)
