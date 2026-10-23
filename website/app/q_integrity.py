"""Shared supporting capability: AIS integrity flags and observation/reporting gaps (observed + synthetic test)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import hhmm, manifest, table, task_status

st.title("Data integrity and reporting gaps")
st.caption("Checks shared by all four predictions. Each record gets a list of named flags, not one opaque score. "
           "Raw records are never changed; flags are stored separately.")
F, V, G, SY = (table("quality_flags.csv"), table("quality_vessels.parquet"), table("gaps.parquet"),
               table("gap_synthetic_test.csv"))
if F is None:
    st.warning("No integrity results yet. Run `.venv\\Scripts\\python scripts\\run_pipeline.py`.")
    st.stop()
run = (manifest().get("tasks", {}).get("data_quality", {}) or {}).get("run") or {}
fl = run.get("flags", {})
gs = run.get("gaps", {})
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records checked", f"{fl.get('rows', 0):,}")
k2.metric("Records with at least one flag", f"{fl.get('rows_flagged', 0):,}")
k3.metric("Reporting gaps found", f"{gs.get('n_gaps', 0):,}")
k4.metric("Vessels with a gap", f"{gs.get('n_vessels_with_gaps', 0):,}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Observed record flags")
    st.dataframe(F[["flag_code", "records", "share_of_records", "description", "recommended_response"]]
                 .sort_values("records", ascending=False).style.format({"share_of_records": "{:.4%}"}),
                 hide_index=True, width="stretch")
    st.caption("These are problems in the raw source, separate from the synthetic defects injected in the stress "
               "tests. This MarineCadastre product is already cleaned and downsampled, so sentinel values are rare. "
               "Row order is not preserved, so out-of-order timestamps cannot be checked.")
with c2:
    st.subheader("Reporting gaps")
    if G is not None and len(G):
        fig = go.Figure()
        for cls, col in [("stationary_reporting_gap", "#7a9cc6"), ("underway_reporting_gap", "#c62828"),
                         ("possible_coverage_exit", "#56606d")]:
            s = G[G.gap_class == cls]
            fig.add_histogram(x=np.clip(s.gap_s / 60, 0, 240), name=f"{cls.replace('_', ' ')} ({len(s):,})",
                              marker_color=col, xbins=dict(start=0, end=240, size=10))
        fig.update_layout(barmode="stack", height=300, margin=dict(t=10, b=10), xaxis_title="gap length, min (capped at 240)",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width="stretch")
    st.caption("A gap is a pause between consecutive reports from one vessel that is longer than "
               "max(10 min, 5 × the vessel's usual reporting interval). It is an **observation gap**, not a proven "
               "transmitter failure: reception coverage, the product's downsampling, and vessels leaving the area "
               "look the same. A *possible coverage exit* is a gap that starts or ends in a sparsely received area.")

st.subheader("Does the gap detector work? Synthetic withheld and delayed records")
if SY is not None and len(SY):
    rows = []
    for kind, g in SY.groupby("kind"):
        pos = g[g.should_detect]
        rows.append({"injected": "records never delivered" if kind == "missing" else "records delivered 20 min late",
                     "events": len(g), "long enough to count as a gap": len(pos),
                     "seen as an open gap at decision time": pos.open_gap_at_decision.mean(),
                     "afterwards: still a gap": pos.retrospective_gap.mean(),
                     "false alarms on short events": int((~g.should_detect & g.open_gap_at_decision).sum())})
    st.dataframe(pd.DataFrame(rows).style.format({"seen as an open gap at decision time": "{:.1%}",
                                                  "afterwards: still a gap": "{:.1%}"}), hide_index=True, width="stretch")
    st.caption("Real clean tracks with one run of 5–50 min of records withheld or delayed (labels known, SYNTHETIC). "
               "**At decision time a delay and a loss look identical.** Only later, when the delayed records arrive, "
               "can the gap be relabelled as a delay. That is why the predictors gate on report age rather than "
               "trying to tell the two apart.")

st.subheader("Look up a vessel")
mmsi = st.text_input("MMSI", placeholder="e.g. 367000000")
if mmsi.strip().isdigit() and V is not None:
    m = int(mmsi)
    v = V[V.mmsi == m]
    if v.empty:
        st.write("No records for this MMSI in the default subset.")
    else:
        r = v.iloc[0]
        st.write(f"{int(r.n_records):,} records, {int(r.n_flagged):,} flagged; codes: `{r.codes or 'none'}`")
        if G is not None:
            g = G[G.mmsi == m]
            if len(g):
                st.dataframe(pd.DataFrame({"gap start": [hhmm(t) for t in g.gap_start_t], "gap end": [hhmm(t) for t in g.gap_end_t],
                                           "minutes": (g.gap_s / 60).round(1), "usual interval s": g.cadence_s,
                                           "moved km": g.moved_km.round(2), "class": g.gap_class}),
                             hide_index=True, width="stretch")
            else:
                st.write("No reporting gaps detected.")
st.caption(f"Default subset: {manifest().get('default_demo_subset', '?')}. Status of the integrity run: "
           f"{task_status('data_quality').get('status', 'n/a') if task_status('data_quality') else 'see manifest'}.")
