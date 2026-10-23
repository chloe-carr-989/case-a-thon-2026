"""Model Performance: baseline vs model, the poisoning experiment (stale, unreliable, combined), the countermeasure,
human oversight, and the study card, for the two tasks the modes show (Commercial ETA, Government anomalies)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ui
from common import CONDITIONS, LOG, manifest, metrics

STRESS = {"stale": "Stale only", "low_quality": "Unreliable only", "combined": "Both"}
SCOL = {"nominal": ui.C["dim"], "stale": "#6fd0c5", "low_quality": ui.C["warn"], "combined": ui.C["crit"]}
OFF, ON, BASE = ui.C["crit"], ui.C["accent"], ui.C["dim"]
M = manifest()
RUN_A = M.get("tasks", {}).get("anomaly", {}).get("run", {})
SPLIT = M.get("split", {}).get("split_t_utc", "")[:16]
SCHED = M.get("config", {}).get("stale_schedule", [0.1, 0.2, 0.3, 0.45, 0.6])


def style(f: go.Figure, h: int, **kw) -> go.Figure:
    f.update_layout(height=h, margin=dict(l=0, r=0, t=28, b=0), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13), legend=dict(orientation="h", y=-0.22), **kw)
    f.update_xaxes(gridcolor=ui.C["line"], zerolinecolor=ui.C["line2"])
    f.update_yaxes(gridcolor=ui.C["line"], zerolinecolor=ui.C["line2"])
    return f


def note(text: str):
    st.markdown(f"<div class='note' style='font-size:13px;margin-bottom:16px'>{text}</div>", unsafe_allow_html=True)


def win_labels() -> list[str]:
    e = RUN_A.get("window_edges_t") or []
    return [f"{i + 1} · {ui.hhmm(a)}–{ui.hhmm(b)}<br>{s:.0%} degraded" for i, (a, b, s) in
            enumerate(zip(e[:-1], e[1:], SCHED))] if len(e) > 1 else [str(i + 1) for i in range(5)]


# ------------------------------------------------------------------------------------------- data
@st.cache_data
def eta(window_level: bool) -> pd.DataFrame:
    m = metrics("eta")
    m = m[(m.subgroup == "all") & ((m.window != "all") if window_level else (m.window == "all"))].copy()
    keys = ["condition", "predictor", "mitigation"] + (["window"] if window_level else [])
    if window_level:
        m["window"] = m.window.astype(int)
    out = m.groupby(keys).mean(numeric_only=True)
    out["mae_sd"] = m.groupby(keys).mae_h.std()
    return out.reset_index()


@st.cache_data
def anomaly(window_level: bool) -> pd.DataFrame:
    m = metrics("anomaly")
    m = m[(m.window != "all") if window_level else (m.window == "all")].copy()
    if window_level:
        m["window"] = m.window.astype(int)
    return m.groupby(["condition", "mitigation"] + (["window"] if window_level else [])).mean(numeric_only=True) \
        .reset_index()


@st.cache_data
def age_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    """C1: outputs against the age of the newest delivered report (stale feed, no countermeasure, seed 0)."""
    bins, labs = [-1, 5, 10, 20, 30, 45, 90], ["0-5", "5-10", "10-20", "20-30", "30-45", "45+"]
    w = pd.read_parquet(ui.RES / "anomaly_windows.parquet", columns=["condition", "mitigation", "label", "age_s"],
                        filters=[("condition", "==", "stale"), ("mitigation", "==", "none")])
    w = w[w.label.isin(["normal", "unusual", "high"])]
    w["age"] = pd.cut(w.age_s / 60, bins, labels=labs)
    a = w.groupby("age", observed=True).agg(n=("label", "size"),
                                             alert=("label", lambda x: x.isin(["unusual", "high"]).mean()))
    d = ui.eta_cases()
    n = d[d.condition == "nominal"].set_index("case_id")
    s = d[(d.condition == "stale") & d.un_eta_answered].copy()
    s["err"] = (s.un_eta_h - s.true_eta_h).abs()
    s["err0"] = (n.un_eta_h - n.true_eta_h).abs().reindex(s.case_id).to_numpy()
    s["bl"] = (s.bl_eta_h - s.true_eta_h).abs()
    s["bl0"] = (n.bl_eta_h - n.true_eta_h).abs().reindex(s.case_id).to_numpy()
    s["age"] = pd.cut(s.age_s / 60, [-1, 2, 10, 20, 30, 45, 90], labels=["0-2", "2-10", "10-20", "20-30", "30-45", "45+"])
    e = s.dropna(subset=["err0"]).groupby("age", observed=True).agg(n=("err", "size"), err=("err", "mean"),
                                                                     err0=("err0", "mean"), bl=("bl", "mean"),
                                                                     bl0=("bl0", "mean"))
    return a.reset_index(), e.reset_index()


@st.cache_data
def eta_zone_split() -> dict:
    d = ui.eta_cases()
    c = d[(d.condition == "nominal") & d.un_eta_answered]
    right = (c.un_pred1 == c.true_zone).to_numpy()
    err = (c.un_eta_h - c.true_eta_h).abs().to_numpy()
    return {"share": right.mean(), "right": err[right].mean(), "wrong": err[~right].mean(), "n": len(c),
            "ms_un": float(d.t_un_ms.median()), "ms_mi": float(d.t_mi_ms.median())}


def g(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df.iloc[0] if len(df) == 1 else df


# ------------------------------------------------------------------------------------------- page
st.markdown("<h2 style='margin:.2rem 0 0'>Model performance</h2>", unsafe_allow_html=True)
prof = M.get("dataset", {}).get("profile_observed", {})
st.markdown(f"<div class='sub' style='font-size:14px'>AIS data {ui.esc(prof.get('time_min', '')[:10])} → "
            f"{ui.esc(prof.get('time_max', '')[:10])}. Models learned only from reports before <b>{SPLIT} UTC</b>; "
            "every number here is scored on the following day, which no model saw (the test day). Degraded feeds "
            "are SIMULATED on that same day, so clean and poisoned runs score identical cases.</div>",
            unsafe_allow_html=True)
EA, EW, AA, AW = eta(False), eta(True), anomaly(False), anomaly(True)
if EA.empty or AA.empty:
    st.warning("No ETA or anomaly results in results/metrics.csv. Run scripts\\run_pipeline.py --subsets jan01_04.")
    st.stop()
SYN = metrics("anomaly_synthetic")
XL = win_labels()

t_base, t_poison, t_fix, t_human, t_card = st.tabs(
    ["Baseline vs model", "Poisoning: cause → effect", "Countermeasure", "Human oversight", "Study card"])

# ===================================================================================== baseline vs model (E2, E3)
with t_base:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Commercial · ETA")
        rows = []
        for pred, lab in [("naive_distance_speed", "Baseline: distance ÷ speed"), ("hgb_eta", "Model: lanes + boosting")]:
            r = g(EA, condition="nominal", predictor=pred, mitigation="none")
            rows.append({"predictor": lab, "mean error": ui.dur(r.mae_h), "± seeds": f"{r.mae_sd * 60:.1f} min",
                         "median": ui.dur(r.median_ae_h), "worst 10% from": ui.dur(r.p90_ae_h),
                         "bias": f"{r.bias_h * 60:+.0f} min", "answered": f"{r.coverage:.0%}",
                         "in 80% window": f"{r.interval_coverage:.0%}" if np.isfinite(r.interval_coverage) else "—",
                         "cases": f"{r.n_cases:,.0f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        b, m = (g(EA, condition="nominal", predictor=p, mitigation="none").mae_h for p in ("naive_distance_speed", "hgb_eta"))
        z = eta_zone_split()
        note(f"Clean feed, 3 seeds. The model's mean error is <b>{1 - m / b:.0%} lower</b> than the baseline's "
             f"({ui.dur(m)} vs {ui.dur(b)}) and it answers more vessels. Both predict the time to the vessel's next "
             "stop of 30+ min (a proxy for arrival, seen in later AIS; not a berth record). Bias is negative: both "
             "predict arrival early. Its 80% window holds only ~2/3 of the time: <b>overconfident</b>.")
        note(f"<b>Failure analysis:</b> the ETA is conditional on the predicted stop area, which is right for "
             f"{z['share']:.0%} of cases. Right area: mean error {ui.dur(z['right'])}; wrong area: "
             f"{ui.dur(z['wrong'])}. Most large errors come from the wrong destination, not the timing.")
        w = EW[(EW.condition == "nominal") & (EW.mitigation == "none")]
        f = go.Figure()
        for pred, lab, col in [("naive_distance_speed", "baseline", BASE), ("hgb_eta", "model", ON)]:
            d = w[w.predictor == pred]
            f.add_scatter(x=XL, y=d.mae_h * 60, name=lab, mode="lines+markers", line=dict(color=col, width=2.5))
        style(f, 260, title=dict(text="Mean ETA error by test window, clean feed (min, lower is better)", font=dict(size=13)))
        f.update_xaxes(ticktext=[x.split("<br>")[0] for x in XL], tickvals=XL)
        st.plotly_chart(f, width="stretch")
        note("The model's line sits below the baseline's in every window, so it wins every one. Error swings by "
             "window with real traffic (window 4 is harder for both), which is why results are reported per window, "
             "not one aggregate.")
    with c2:
        st.subheader("Government · movement anomalies")
        n0 = g(AA, condition="nominal", mitigation="none")
        sy = SYN[SYN.condition == "nominal"].set_index("mitigation")
        st.dataframe(pd.DataFrame([
            {"check": "Alert rate, clean feed", "value": f"{n0.alert_rate:.1%} of vessel checks",
             "meaning": "≈ 2.5% by design: flags the top 2.5% vs the 48 h before the test day"},
            {"check": "Injected 2-8 km drift caught", "value": f"{sy.loc['none', 'recall']:.0%}",
             "meaning": "known-label test: real normal tracks shifted sideways (SYNTHETIC)"},
            {"check": "…flagged high deviation", "value": f"{sy.loc['none', 'recall_high']:.0%}",
             "meaning": "most drifts land just past the threshold"},
            {"check": "Vessel checks scored", "value": f"{RUN_A.get('n_test_windows', 0):,}",
             "meaning": f"{RUN_A.get('n_test_vessels', 0):,} vessels, a check every 30 min over the last 60 min"},
        ]), hide_index=True, width="stretch")
        note("<b>No verified anomaly labels exist</b> for this data, so real-world precision is not claimed. The "
             "comparator for poisoning is the same detector on the clean feed: an alert on a degraded feed that the "
             "clean run did not raise is counted as caused by bad data; a clean-run alert that disappears is "
             "counted as missed.")
        note("<b>Limit found:</b> about half of 2-8 km sideways drifts go unflagged. Shipping lanes are wide near "
             "ports, so a few km off a learned lane is often still normal there.")
        s = SYN.groupby(["condition", "mitigation"]).recall.mean().unstack()
        f = go.Figure()
        for mi, lab, col in [("none", "no countermeasure", OFF), ("quality_aware", "with countermeasure", ON)]:
            f.add_bar(x=[CONDITIONS[c] for c in CONDITIONS], y=[s.loc[c, mi] for c in CONDITIONS], name=lab,
                      marker_color=col, text=[f"{s.loc[c, mi]:.0%}" for c in CONDITIONS], textposition="outside")
        style(f, 260, title=dict(text="Injected drifts caught, by feed", font=dict(size=13)),
              yaxis=dict(tickformat=".0%", range=[0, 0.7]), barmode="group")
        st.plotly_chart(f, width="stretch")

# ===================================================================================== poisoning (C1, C2, C3)
with t_poison:
    st.markdown("<div class='sub' style='font-size:14px;color:#d6dce4'>Chain: <b>data age / quality / volume</b> → "
                "the model's <b>input at decision time</b> (the frozen models are never retrained on bad data: "
                "poisoning enters at inference) → <b>output</b> (ETA, anomaly score) → <b>decision</b> (berth "
                "schedule, which vessel a watch officer reviews).</div>", unsafe_allow_html=True)
    A_age, E_age = age_curves()
    st.subheader("C1 · Staleness: reports that arrive late")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        f = go.Figure(go.Bar(x=A_age.age, y=A_age.alert, marker_color=SCOL["stale"],
                             text=[f"{v:.1%}" for v in A_age.alert], textposition="outside", customdata=A_age.n,
                             hovertemplate="%{x} min old · %{y:.2%} flagged · n=%{customdata:,}<extra></extra>"))
        style(f, 280, title=dict(text="Government: share of vessels flagged vs age of newest report", font=dict(size=13)),
              yaxis=dict(tickformat=".0%"), xaxis_title="minutes since the newest delivered report")
        st.plotly_chart(f, width="stretch")
        note("<b>Where it enters:</b> the detector scores each vessel's last 60 min using only reports delivered by "
             "then. A late feed leaves an old, short track. <b>Why it changes the output:</b> a short, old track "
             "shows no movement to judge, so it scores as normal. Once the newest report is over 10 min old almost "
             "nothing is flagged: stale data <b>silently hides real alerts</b> (false negatives), it never adds any.")
    with c2:
        f = go.Figure()
        f.add_scatter(x=E_age.age, y=E_age.err / E_age.err0, name="model", mode="lines+markers",
                      line=dict(color=SCOL["stale"], width=3), customdata=E_age.n,
                      hovertemplate="%{x} min old · %{y:.2f}× clean · n=%{customdata:,}<extra></extra>")
        f.add_scatter(x=E_age.age, y=E_age.bl / E_age.bl0, name="baseline", mode="lines+markers",
                      line=dict(color=BASE, width=2, dash="dot"))
        f.add_hline(y=1, line=dict(color=ui.C["line2"], width=1))
        style(f, 280, title=dict(text="Commercial: ETA error ÷ same case on clean feed", font=dict(size=13)),
              xaxis_title="minutes since the newest delivered report")
        st.plotly_chart(f, width="stretch")
        r45 = E_age[E_age.age == "30-45"]
        note("<b>Where it enters:</b> the ETA starts from the last reported position and speed. A 30-45 min old "
             "report places the vessel kilometres behind where it is, so the ETA is late. The effect is real but "
             + (f"small ({float(r45.err.iloc[0] / r45.err0.iloc[0]) - 1:+.0%} at 30-45 min) " if len(r45) else "small ")
             + "because ETAs look 0-6 h ahead. Staleness matters in proportion to the prediction horizon.")

    st.subheader("C2 · Low quality: a fast, unreliable second source")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        f = go.Figure()
        for mi, lab, col in [("none", "no countermeasure", OFF), ("quality_aware", "with countermeasure", ON)]:
            d = AW[(AW.condition == "low_quality") & (AW.mitigation == mi)]
            f.add_scatter(x=d.lowq_frac, y=d.spurious_alerts_per_1000, name=lab, mode="lines+markers",
                          line=dict(color=col, width=3))
        style(f, 280, title=dict(text="False alerts caused by bad data, per 1000 vessel checks", font=dict(size=13)),
              xaxis=dict(title="share of vessels receiving unreliable-source reports", tickformat=".0%"))
        st.plotly_chart(f, width="stretch")
    with c2:
        dc = M.get("defect_configuration", {})
        note("<b>The failure:</b> a second feed ('SYN-FASTFEED') sends extra reports with position jitter "
             "(~200 m), gross position outliers 3-15 km away (25% of its reports), scaled or missing speeds and "
             "random courses. It is fast, so it adds volume. <b>Mechanism:</b> one outlier puts the vessel's "
             "newest position kilometres off its lane; the detector's strongest signal is off-lane distance, so "
             "the vessel is flagged. These are false positives that a watch officer must clear.")
        note("<b>Prevalence / volume test:</b> the unreliable source's share rises across the test day (10% → 60% "
             "of vessels, SIMULATED). False alerts rise with it, almost linearly: more data from the bad source "
             "means more harm. This is the case's core question: here more information makes the decision worse.")
        lq = g(AA, condition="low_quality", mitigation="none")
        note(f"Over the whole test day: {lq.spurious_alerts_per_1000:.0f} false alerts per 1000 checks and the "
             f"alert rate goes from {g(AA, condition='nominal', mitigation='none').alert_rate:.1%} to "
             f"{lq.alert_rate:.1%}; the ETA error rises only "
             f"{g(EA, condition='low_quality', predictor='hgb_eta', mitigation='none').mae_h / g(EA, condition='nominal', predictor='hgb_eta', mitigation='none').mae_h - 1:+.0%}, "
             "because the ETA uses one sanitised report, not a track.")

    st.subheader("C3 · Stale only, unreliable only, and both, across the ordered test day")
    f = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=[
        "Government: false alerts / 1000 checks", "Government: real alerts missed", "Commercial: ETA error ÷ clean"])
    nom = EW[(EW.condition == "nominal") & (EW.predictor == "hgb_eta") & (EW.mitigation == "none")].set_index("window")
    for c, lab in STRESS.items():
        a = AW[(AW.condition == c) & (AW.mitigation == "none")]
        e = EW[(EW.condition == c) & (EW.predictor == "hgb_eta") & (EW.mitigation == "none")].set_index("window")
        kw = dict(mode="lines+markers", line=dict(color=SCOL[c], width=3), legendgroup=c)
        f.add_scatter(x=XL, y=a.spurious_alerts_per_1000, name=lab, row=1, col=1, **kw)
        f.add_scatter(x=XL, y=a.missed_clean_alert_share, name=lab, showlegend=False, row=1, col=2, **kw)
        f.add_scatter(x=XL, y=(e.mae_h / nom.mae_h).to_numpy(), name=lab, showlegend=False, row=1, col=3, **kw)
    f.update_yaxes(tickformat=".0%", row=1, col=2)
    style(f, 330)
    f.update_xaxes(tickvals=XL, ticktext=[x.split(" · ")[0] for x in XL], title_text="test window")
    f.update_annotations(font_size=13)
    st.plotly_chart(f, width="stretch")
    note("Windows are the real chronological periods of the test day (labels: window · UTC hours · degraded share). "
         "The rising degradation schedule (10% → 60%) is SIMULATED. <b>The two failures do different damage:</b> "
         "unreliable data creates false alerts, stale data hides real ones. 'Both' keeps all of the stale misses "
         "but fewer false alerts than unreliable alone, because a late feed also delays the bad source's reports. "
         "For the ETA neither effect is large; window-to-window swings come from real traffic.")

# ===================================================================================== countermeasure (C4)
with t_fix:
    st.markdown("<div class='sub' style='font-size:14px;color:#d6dce4'>Countermeasure (fixed before testing, "
                "never tuned on the test day): drop reports with impossible speeds or position jumps, move an old "
                "position forward along its course, and <b>withhold the answer and send it to a reviewer</b> when "
                "the newest report is over 20 min old. Every row below scores the same cases.</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Government")
        rows = []
        for c, lab in CONDITIONS.items():
            for mi, ml in [("none", "off"), ("quality_aware", "on")]:
                r = g(AA, condition=c, mitigation=mi)
                rows.append({"feed": lab, "fix": ml, "false alerts / 1000": f"{r.spurious_alerts_per_1000:.1f}",
                             "real alerts missed": f"{r.missed_clean_alert_share:.0%}",
                             "given a verdict": f"{r.coverage:.0%}", "alerts / 1000": f"{1000 * r.alert_rate:.0f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * 9 + 3)
    with c2:
        st.subheader("Commercial")
        rows = []
        for c, lab in CONDITIONS.items():
            for mi, ml in [("none", "off"), ("quality_aware", "on")]:
                r = g(EA, condition=c, predictor="hgb_eta", mitigation=mi)
                rows.append({"feed": lab, "fix": ml, "mean error": ui.dur(r.mae_h), "held": f"{1 - r.coverage:.0%}",
                             "error, held ones charged": ui.dur(r.policy_mae_h),
                             "in 80% window": f"{r.interval_coverage:.0%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * 9 + 3)
    a_lq, a_lq1 = g(AA, condition="low_quality", mitigation="none"), g(AA, condition="low_quality", mitigation="quality_aware")
    a_c, a_c1 = g(AA, condition="combined", mitigation="none"), g(AA, condition="combined", mitigation="quality_aware")
    a_n, a_n1 = g(AA, condition="nominal", mitigation="none"), g(AA, condition="nominal", mitigation="quality_aware")
    e_s, e_s1 = (g(EA, condition="stale", predictor="hgb_eta", mitigation=m) for m in ("none", "quality_aware"))
    w4 = AW[(AW.window == AW.window.max()) & (AW.condition == "low_quality")].set_index("mitigation")
    sy = SYN.groupby(["condition", "mitigation"]).recall.mean()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**What it fixes**")
        note(f"• False alerts from the unreliable source: {a_lq.spurious_alerts_per_1000:.0f} → "
             f"{a_lq1.spurious_alerts_per_1000:.0f} per 1000 checks "
             f"(<b>{1 - a_lq1.spurious_alerts_per_1000 / a_lq.spurious_alerts_per_1000:.0%} removed</b>); "
             f"both failures together: {a_c.spurious_alerts_per_1000:.0f} → {a_c1.spurious_alerts_per_1000:.0f}.")
        note(f"• <b>New, unseen condition:</b> window 5, the latest and most degraded period, was held out when the "
             f"thresholds were set. It still removes {1 - w4.loc['quality_aware', 'spurious_alerts_per_1000'] / w4.loc['none', 'spurious_alerts_per_1000']:.0%} "
             "of false alerts there.")
        note(f"• <b>Normal data cost:</b> on the clean feed the alert rate is unchanged ({a_n.alert_rate:.1%} → "
             f"{a_n1.alert_rate:.1%}) but {a_n.coverage - a_n1.coverage:.0%} of vessels lose their verdict because "
             "their own reports are genuinely over 20 min apart.")
    with c2:
        st.markdown("**Where it fails**")
        note(f"• <b>Stale data: nothing recovered.</b> Real alerts missed stays at "
             f"{g(AA, condition='stale', mitigation='quality_aware').missed_clean_alert_share:.0%}. The fix cannot "
             "invent the missing reports; it can only say 'no verdict' (verdicts drop to "
             f"{g(AA, condition='stale', mitigation='quality_aware').coverage:.0%}), which at least tells the "
             "officer the picture is incomplete instead of showing a clean map.")
        note(f"• <b>ETA, adverse tradeoff:</b> on the stale feed it holds {1 - e_s1.coverage:.0%} of ETAs for "
             f"review, but most of them were still usable: counting each held ETA as a typical bad miss, "
             f"error goes from {ui.dur(e_s.policy_mae_h)} to {ui.dur(e_s1.policy_mae_h)}. For a 0-6 h ETA, a "
             "20-min age limit is too strict; the threshold should scale with the horizon.")
        note(f"• <b>Rare real events:</b> it suppresses some genuine deviations. Injected drifts caught on the "
             f"unreliable feed: {sy.get(('low_quality', 'none'), np.nan):.0%} → "
             f"{sy.get(('low_quality', 'quality_aware'), np.nan):.0%} (clean feed "
             f"{sy.get(('nominal', 'none'), np.nan):.0%} → {sy.get(('nominal', 'quality_aware'), np.nan):.0%}).")
    f = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                      subplot_titles=["Both failures: false alerts / 1000 checks, by window",
                                      "Both failures: vessels given a verdict, by window"])
    for mi, lab, col in [("none", "countermeasure off", OFF), ("quality_aware", "countermeasure on", ON)]:
        d = AW[(AW.condition == "combined") & (AW.mitigation == mi)]
        f.add_scatter(x=XL, y=d.spurious_alerts_per_1000, name=lab, mode="lines+markers", legendgroup=mi,
                      line=dict(color=col, width=3), row=1, col=1)
        f.add_scatter(x=XL, y=d.coverage, name=lab, mode="lines+markers", legendgroup=mi, showlegend=False,
                      line=dict(color=col, width=3), row=1, col=2)
    f.update_yaxes(tickformat=".0%", row=1, col=2)
    style(f, 300)
    f.update_xaxes(tickvals=XL, ticktext=[x.split(" · ")[0] for x in XL], title_text="test window")
    f.update_annotations(font_size=13)
    st.plotly_chart(f, width="stretch")
    cov5 = AW[(AW.condition == "combined") & (AW.mitigation == "quality_aware")].sort_values("window").coverage.iloc[-1]
    note(f"Recovery has a price paid in coverage: by window 5 the countermeasure gives a verdict on {cov5:.0%} of "
         "vessels. Rejecting everything would also remove every false alert, so coverage is always reported next to "
         "the error.")

# ===================================================================================== human oversight (E4)
with t_human:
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        st.subheader("Who does what")
        st.dataframe(pd.DataFrame([
            {"": "Uses the output", "Commercial": "Port scheduler: plans berths and pilots from ETAs",
             "Government": "Watch officer: chooses which vessels to look at"},
            {"": "System must defer", "Commercial": "Newest report > 20 min old, impossible jump, or no stop area "
                                                    "in reach: ETA held",
             "Government": "Newest report > 20 min old: 'no verdict' instead of a score"},
            {"": "Reviews", "Commercial": "Scheduler reviews held ETAs and any ETA whose window overlaps a berth "
                                          "conflict", "Government": "Officer reviews every flagged vessel; held ones "
                                                                    "go to the data-quality owner"},
            {"": "Sees as evidence", "Commercial": "Report age, source, injected defects, track, expected route, "
                                                   "80% window, baseline ETA",
             "Government": "Score, plain-language reasons, report age, gaps, spikes, track vs expected lane, weather"},
            {"": "Corrects by", "Commercial": "Use / correct ETA (typed) / request fresh position + reason",
             "Government": "Escalate / no action / false positive (bad data or normal movement) + reason"},
            {"": "Logged to", "Commercial": "results/review_log.csv with measured review seconds",
             "Government": "same file, same fields"},
        ]), hide_index=True, width="stretch")
        note("<b>Walkthrough of an unreliable output</b> (live demo): Government → feed 'Unreliable source', "
             "countermeasure off → click a red/amber vessel → 'Why' says it is km off its lane → 'Data trust' "
             "shows spikes from the simulated source → record 'False positive: bad data'. Turn the countermeasure on "
             "and the same vessel is no longer flagged.")
    with c2:
        st.subheader("Effect on human effort")
        rows = []
        for c, lab in CONDITIONS.items():
            for mi, ml in [("none", "off"), ("quality_aware", "on")]:
                r = g(AA, condition=c, mitigation=mi)
                rows.append({"feed": lab, "fix": ml, "alerts to review": f"{1000 * r.alert_rate:.0f}",
                             "of which bad data": f"{r.spurious_alerts_per_1000:.0f}",
                             "held for data review": f"{1000 * (1 - r.coverage):.0f}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * 9 + 3)
        n_, n1_ = g(AA, condition="nominal", mitigation="none"), g(AA, condition="nominal", mitigation="quality_aware")
        note("Per 1000 vessel checks (Government). On feeds with unreliable data the countermeasure cuts the "
             "officer's alert queue by more than half, but moves work to the data-quality queue (held vessels); on "
             f"the stale feed it only adds holds. On the clean feed it adds {1000 * (n_.coverage - n1_.coverage):.0f} "
             "holds per 1000 checks for no alert benefit: the workload cost of the fix.")
        z = eta_zone_split()
        log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
        secs = log.review_seconds_measured.dropna() if "review_seconds_measured" in log else pd.Series(dtype=float)
        note(f"<b>Machine time vs human time:</b> the model takes {z['ms_un']:.2f} ms per ETA "
             f"({z['ms_mi']:.2f} ms with the countermeasure). Human review time is measured from opening a vessel to "
             "recording a decision: "
             + (f"<b>{len(secs)} reviews logged, median {secs.median():.0f} s</b>." if len(secs) else
                "<b>no reviews logged yet</b> in results/review_log.csv.")
             + " These are team walkthroughs only; they say nothing about trust or speed of trained operators, "
               "which a pilot would have to measure.")

# ===================================================================================== study card (E1, E5, E6, E7)
with t_card:
    c1, c2 = st.columns(2, gap="large")
    ds = M.get("dataset", {})
    with c1:
        st.subheader("Decision and success")
        note("<b>Commercial:</b> a port scheduler decides berth and pilot order from ETAs. A late ETA idles a "
             "berth; an early one queues ships. Success = lower mean ETA error than distance ÷ speed, an 80% window "
             "that holds ~80% of the time, and on a degraded feed, error that stays near clean without withholding "
             "most ETAs.")
        note("<b>Government:</b> a watch officer decides which vessels to look at. A false alert wastes review time; "
             "a missed one leaves unusual movement unseen. Success = on a degraded feed, few alerts caused by bad "
             "data and few clean-feed alerts lost, at high coverage.")
        note("<b>AI component:</b> gradient-boosted ETA models and a lane-relative anomaly score learned from 3 days "
             "of AIS, both trained once and frozen (not retrained during the test). Poisoning is applied at "
             "inference, on the feed the models read.")
        note("<b>Expert check:</b> design reviewed in an interview (Oct 22, 2026) with a retired U.S. Navy captain "
             "who ran civil maritime intelligence analysis at the Office of Naval Intelligence. She confirmed the "
             "core choices: ships keep to known lanes because they are the fastest, cheapest, safest route, so a "
             "deviation is the signal; weather is the biggest variable and belongs in a separate layer you can switch "
             "on and off, not in the training data; score commercial vessels only, since warships behave "
             "differently; expect AIS to go dark, be spoofed, and have gaps. Implemented: weather layer and "
             "per-alert weather check, military vessels not scored, data integrity flags. Live weather is the "
             "deployment step (this demo replays historical ERA5).")
        st.subheader("Next step")
        note("<b>Recommendation: controlled pilot, shadow mode.</b> Government detector + countermeasure, one port "
             "region (New York / New Jersey), 4 weeks, read-only, no automated action. Owner: model owner (ML engineer) with the AIS feed "
             "steward. Approval: operations supervisor. Checkpoint at week 2: stop if bad-data alerts are not at "
             "least 50% below the unprotected run, or held vessels exceed 30% for two days. "
             "<b>Iterate first on the ETA:</b> its countermeasure costs more than it saves (age limit too strict).")
    with c2:
        st.subheader("Data, governance, reproducibility")
        st.dataframe(pd.DataFrame([
            ("Source", ds.get("name", "")),
            ("Product", ds.get("product", "")),
            ("Files", ", ".join(x.split("/")[-1] for x in ds.get("files_used", []))),
            ("Split", f"train before {SPLIT} UTC, test after; test vessels never seen in training"),
            ("Defects", "SIMULATED in memory, labelled per case; raw files unchanged"),
            ("Rights", "public U.S. government AIS (MarineCadastre); raw files are not redistributed here"),
            ("Privacy", "vessel IDs are public broadcasts; no attempt to identify people"),
            ("Bias risk", "lanes learned from busy commercial traffic: rare routes and small craft score as unusual"),
            ("Not for", "navigation, collision avoidance, or evidence of intent or identity"),
            ("Versions", ", ".join(f"{k} {v}" for k, v in M.get("environment", {}).get("packages", {}).items())),
            ("Reproduce", "scripts\\run_pipeline.py --subsets jan01_04 --train-frac 0.75 ; streamlit run app\\dashboard.py"),
            ("AI use", M.get("ai_disclosure", "")),
        ], columns=["", "detail"]), hide_index=True, width="stretch", height=35 * 13 + 3)
