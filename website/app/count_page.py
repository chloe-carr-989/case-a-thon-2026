"""One page implementation for both count forecasts (traffic density, port occupancy proxy)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (AREAS, CONDITIONS, eight_cell, geo_layout, hhmm, in_area, manifest, metrics, review_form, table,
                    task_status, unavailable)

ACTION_TEXT = {"model": "trained model on quality-filtered counts",
               "fallback_last_complete_hour": "feed is late: fell back to the last complete hour of reports",
               "defer_late_feed": "feed is 45+ min late: forecast withheld for review"}


def render(task: str, prefix: str, title: str, caption: str, unit_word: str, count_word: str, limits_md: str):
    st.title(title)
    st.caption(caption)
    C0, U = table(f"{prefix}_cases_seed0.parquet"), table(f"{prefix}_units.csv")
    if C0 is None or U is None:
        unavailable(task, title)
    M = metrics(task)
    run = task_status(task)
    bucket_s = 900

    c1, c2, c3, c4 = st.columns([3, 2, 3, 2])
    cond = c1.segmented_control("Feed condition", list(CONDITIONS), format_func=CONDITIONS.get,
                                default="combined", key=f"{prefix}_cond") or "combined"
    mitig = c2.toggle("Countermeasure on", value=False, key=f"{prefix}_mit",
                      help="Drops invalid and one-message transient IDs, detects a late feed and falls back to the "
                           "last complete hour, and withholds the forecast when the feed is 45+ minutes late.")
    origins = sorted(C0.o.unique())
    o = c3.select_slider("Forecast made at (UTC); counts are for the next 60 minutes", origins,
                         value=origins[len(origins) * 3 // 4], format_func=lambda b: hhmm(b * bucket_s),
                         key=f"{prefix}_o")
    area = c4.selectbox("Area", list(AREAS), index=len(AREAS) - 1 if prefix == "density" else 0, key=f"{prefix}_area")

    C = C0[(C0.condition == cond) & (C0.o == o)].merge(U[["unit", "unit_id", "lat", "lon", "area_km2"]], on="unit")
    C = C[in_area(C.lat, C.lon, area)].copy()
    C["pred"] = C.pred_mi if mitig else C.pred_un
    C["shown"] = C.answered if mitig else True
    C["err"] = C.pred - C.y
    C = C[(C.y > 0) | (C.pred.fillna(0) >= 0.5) | (C.h0_delivered > 0)]
    if C.empty:
        st.info(f"No {unit_word}s with traffic in this area at this time. Try another area or time.")
        st.stop()

    delay = int(C.delay_buckets.iloc[0])
    k1, k2, k3 = st.columns(3)
    shown = C[C.shown]
    k1.metric(f"Mean miss per {unit_word}", f"{shown.err.abs().mean():.2f} {count_word}" if len(shown) else "n/a",
              delta=f"{(C.pred_un - C.y).abs().mean():.2f} without countermeasure"
              if mitig else None, delta_color="off")
    k2.metric(f"{unit_word.capitalize()}s off by 5+ {count_word}", f"{int((shown.err.abs() >= 5).sum())} of {len(C)}")
    k3.metric("Feed delay at forecast time (simulated)", f"{delay * 15} min" if delay else "on time",
              help="Stale condition: the most recent 15-60 minutes of reports had not arrived yet.")
    if mitig and not C.answered.any():
        st.warning("Countermeasure withheld every forecast at this time: " + ACTION_TEXT["defer_late_feed"])

    map_col, side = st.columns([5, 2], gap="medium")
    with map_col:
        P = C[C.shown]
        fig = go.Figure()
        vmax = max(float(np.nanquantile(np.r_[P.pred.to_numpy(float), P.y.to_numpy(float)], 0.98)) if len(P) else 1, 1)
        fig.add_scattergeo(lat=P.lat, lon=P.lon, mode="markers", name="forecast",
                           marker=dict(size=np.clip(4 + 14 * np.sqrt(P.pred / vmax), 4, 22), color=P.err,
                                       colorscale="RdBu_r", cmid=0, cmin=-10, cmax=10,
                                       colorbar=dict(title="forecast − observed", thickness=10, len=0.6),
                                       line=dict(width=0.5, color="white")),
                           customdata=P.unit,
                           text=[f"{u} · forecast {p:.1f}, observed later {y:.0f} {count_word} "
                                 f"({1000 * p / a:.2f} per 1000 km²)"
                                 for u, p, y, a in zip(P.unit_id, P.pred, P.y, P.area_km2)],
                           hovertemplate="%{text}<extra></extra>")
        R = C[~C.shown]
        if len(R):
            fig.add_scattergeo(lat=R.lat, lon=R.lon, mode="markers", name="withheld (late feed)",
                               marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(width=1.5, color="#666")),
                               customdata=R.unit, hoverinfo="skip")
        geo_layout(fig, area)
        ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key=f"{prefix}_map")
        pts = getattr(getattr(ev, "selection", None), "points", []) if not isinstance(ev, dict) else \
            ev.get("selection", {}).get("points", [])
        clicked = [p.get("customdata") for p in pts if p.get("customdata") is not None]
        if clicked:
            st.session_state[f"{prefix}_sel"] = int(np.ravel(clicked[0])[0])
        st.caption(f"{len(C)} {unit_word}s with traffic · {area} · forecast at {hhmm(o * bucket_s)} UTC for the next hour · "
                   "seed 0. Color = forecast minus the count later observed in AIS (held-out reference).")

    with side:
        order = C.assign(ae=(C.pred_un - C.y).abs()).sort_values("ae", ascending=False)
        ids = order.unit.tolist()
        sel = st.session_state.get(f"{prefix}_sel")
        if sel not in ids:
            sel = ids[0]
        pick = st.selectbox(f"{unit_word.capitalize()} (largest misses first)", ids, index=ids.index(sel),
                            format_func=lambda u: order.set_index("unit").at[u, "unit_id"], key=f"{prefix}_pick")
        if pick != sel:
            st.session_state[f"{prefix}_sel"] = pick
            st.rerun()
        S = C[C.unit == sel].iloc[0]
        tkey = f"{prefix}_t0_{sel}_{o}_{cond}"
        st.session_state.setdefault(tkey, time.time())
        st.markdown(f"#### {S.unit_id}")
        st.markdown(f"**1 · What the system received** (last hour before {hhmm(o * bucket_s)})\n"
                    f"- {int(S.h0_delivered)} distinct IDs, {int(S.msgs_h0_delivered)} messages\n"
                    f"- Feed delay: {'none' if not S.delay_buckets else f'{int(S.delay_buckets) * 15} min (simulated)'}\n"
                    + (f"- Countermeasure removed {int(max(S.h0_delivered - S.h0_mitigated, 0))} suspect IDs "
                       "(invalid MMSI or single-message transient)" if mitig else ""))
        st.markdown("**2 · Forecast for the next 60 min**\n"
                    f"- Without countermeasure: **{S.pred_un:.1f}** {count_word}\n"
                    f"- With countermeasure: " + (f"**{S.pred_mi:.1f}** {count_word}" if S.answered else "**withheld**") +
                    f"\n- Persistence baseline: {S.persistence:.0f}\n"
                    f"- Observed later in AIS (held-out reference): **{S.y:.0f}**")
        st.markdown(f"**3 · Countermeasure action:** {ACTION_TEXT.get(S.action, S.action)}")
        st.markdown("**4 · Reviewer decision**")
        review_form(prefix, task, {"unit_id": S.unit_id, "origin_utc": hhmm(o * bucket_s), "condition": cond,
                                   "countermeasure_on": mitig, "system_action": S.action,
                                   "forecast": round(float(S.pred if S.shown else np.nan), 2)},
                    ["Accept forecast", "Correct count", "Defer: wait for the late feed"],
                    st.session_state[tkey], correction_label="Corrected count (if correcting)")

        H = C0[(C0.condition == cond) & (C0.unit == sel)].sort_values("o")
        fig2 = go.Figure()
        fig2.add_scatter(x=[hhmm(b * bucket_s) for b in H.o], y=H.y, name="observed later", line=dict(color="#222"))
        fig2.add_scatter(x=[hhmm(b * bucket_s) for b in H.o], y=H.pred_un, name="countermeasure off",
                         line=dict(color="#c62828"))
        fig2.add_scatter(x=[hhmm(b * bucket_s) for b in H.o], y=H.pred_mi, name="countermeasure on",
                         line=dict(color="#1565c0"))
        fig2.update_layout(height=240, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.2),
                           yaxis_title=count_word)
        st.plotly_chart(fig2, width="stretch")

    st.divider()
    with st.expander("Four feed conditions × countermeasure (all test origins and units, mean of seeds)", expanded=True):
        cols = {"mae": "MAE", "rmse": "RMSE", "bias": "bias", "coverage": "answered", "policy_mae": "MAE with deferral penalty"}
        t = eight_cell(M, cols, "hgb", extra_none={"persistence": "persistence baseline",
                                                    "rolling_mean": "3-h rolling mean", "hist_mean": "training mean"})
        st.dataframe(t.style.format({c: "{:.3f}" for c in cols.values()}), hide_index=True, width="stretch")
        st.caption(f"Errors in {count_word} per {unit_word} per hour on the same held-out origins and reference counts. "
                   f"Deferred forecasts are charged {run.get('defer_penalty', float('nan')):.2f} {count_word} (nominal p90 error) "
                   "in the penalty column so that withholding everything cannot look accurate.")
    with st.expander("Dense vs sparse units, and error over the test period"):
        g = M[(M.window.astype(str) == "all") & (M.predictor == "hgb")]
        st.dataframe(g.groupby(["subgroup", "condition", "mitigation"])[["mae", "bias", "coverage"]].mean().round(3)
                     .reset_index(), hide_index=True, width="stretch")
        W = M[(M.condition == cond) & (M.subgroup == "all") & (M.predictor == "hgb") & (M.window.astype(str) != "all")]
        W = W.assign(window=W.window.astype(int)).groupby(["window", "mitigation"]).mean(numeric_only=True).reset_index()
        fig3 = go.Figure()
        for mit, name, col in [("none", "countermeasure off", "#c62828"), ("quality_aware", "countermeasure on", "#1565c0")]:
            s = W[W.mitigation == mit]
            fig3.add_scatter(x=s.window, y=s.mae, name=name, mode="lines+markers", line=dict(color=col, width=3),
                             text=[f"{c:.0%} answered" for c in s.coverage], hovertemplate="%{y:.2f} · %{text}")
        lab = W[W.mitigation == "none"]
        fig3.update_xaxes(tickvals=lab.window, ticktext=[f"window {w}<br>{sf:.0%} late origins"
                                                         for w, sf in zip(lab.window, lab.stale_frac)])
        fig3.update_layout(height=300, yaxis_title=f"MAE ({count_word})", margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, width="stretch")
        st.caption("Degradation rates rise across the test period by design (a SIMULATED progression on real data).")
    with st.expander("Definition and known limits"):
        st.markdown(limits_md)
        st.json(manifest().get("tasks", {}).get(task, {}), expanded=False)
