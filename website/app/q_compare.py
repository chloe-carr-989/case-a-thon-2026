"""Baseline vs trained model for all four questions, and which questions show the feed poisoning most clearly."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import CONDITIONS, manifest
from perf import (SCHEDULE, STRESS, TASKS, defect_breakdown, errors, ghost_curve, headline, pick, primary_baseline,
                  relative_by_window, staleness_curve)

st.title("Baseline vs model performance")
st.caption(f"All four questions on the same feed conditions, held-out cases and reference values. Default run: "
           f"{manifest().get('default_demo_subset', '?')}; means over seeds {manifest().get('seeds', '')}, "
           "error bars are the seed standard deviation. Withheld answers are charged a penalty (position, counts) or "
           "counted as misses (destination), so the countermeasure cannot look better by refusing to answer.")

if all(errors(t).empty for t in TASKS):
    st.warning("No results yet. Run `.venv\\Scripts\\python scripts\\run_pipeline.py`.")
    st.stop()

OFF, ON = "#c62828", "#49b6dc"
BASE_COLORS = ["#616161", "#9e9e9e", "#c8c8c8"]
LOGY = dict(type="log", tickformat="~g")


def natural(task, v):
    """Back to the metric's own orientation for display (destination: accuracy, higher is better)."""
    return 1 - v if TASKS[task].get("accuracy") else v


def stat(E, pred, mit, cond, window="all"):
    s = pick(E, pred, mit, window)
    s = s[s.condition == cond]
    return s.err.mean(), s.err.std(), s.coverage.mean()


tab_cmp, tab_pick = st.tabs(["Baseline vs model", "Which questions show the poisoning best?"])

# ===================================================================================== tab 1
with tab_cmp:
    underway = st.toggle("Position: underway vessels only (SOG ≥ 2 kn)", value=False,
                         help="Most vessels are moored, so all-vessel position errors are tiny. "
                              "The underway subgroup shows movement skill.")
    sub = {t: ("underway_ge2kn" if underway and t == "position_15m" else "all") for t in TASKS}

    rows = []
    for t, spec in TASKS.items():
        E = errors(t, sub[t])
        if E.empty:
            continue
        m, _, _ = stat(E, spec["model"], "none", "nominal")
        bl = {b: stat(E, b, "none", "nominal")[0] for b in spec["baselines"]}
        pb = primary_baseline(t)
        best = min(bl, key=bl.get)
        rows.append({"question": spec["short"], "metric": spec["label"] if not spec.get("accuracy") else "top-1 accuracy",
                     "trained model": natural(t, m), f"main baseline": natural(t, bl[pb]), "main baseline is": spec["baselines"][pb],
                     "model vs main baseline": (1 - m / bl[pb]),
                     "best baseline": spec["baselines"][best], "model vs best baseline": (1 - m / bl[best])})
    st.subheader("Clean feed: does the trained model beat a simple comparator?")
    st.dataframe(pd.DataFrame(rows).style.format({"trained model": "{:.3f}", "main baseline": "{:.3f}",
                                                  "model vs main baseline": "{:+.1%}", "model vs best baseline": "{:+.1%}"})
                 .map(lambda v: f"color: {'#2e7d32' if v > 0 else '#c62828'}",
                      subset=["model vs main baseline", "model vs best baseline"]),
                 hide_index=True, width="stretch")
    st.caption("Positive = the model's error is that much lower than the baseline's (for destination, its miss rate). "
               "The port-occupancy model does not beat persistence on a clean feed.")

    st.subheader("All four feed conditions")
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s["short"] for s in TASKS.values()],
                        vertical_spacing=0.16, horizontal_spacing=0.08)
    shown = set()
    for i, (t, spec) in enumerate(TASKS.items()):
        E = errors(t, sub[t])
        if E.empty:
            continue
        r, c = divmod(i, 2)
        series = [(b, "none", lab, BASE_COLORS[j]) for j, (b, lab) in enumerate(spec["baselines"].items())]
        series += [(spec["model"], "none", "model, countermeasure off", OFF),
                   (spec["model"], "quality_aware", "model, countermeasure on", ON)]
        for pred, mit, lab, col in series:
            v = [stat(E, pred, mit, cond) for cond in CONDITIONS]
            legend = lab
            fig.add_bar(x=list(CONDITIONS.values()), y=[natural(t, a) for a, _, _ in v],
                        error_y=dict(type="data", array=[s for _, s, _ in v], thickness=1, width=2),
                        marker_color=col, name=legend, legendgroup=legend, showlegend=legend not in shown,
                        customdata=[cv for _, _, cv in v],
                        hovertemplate=f"{lab}<br>%{{y:.3f}}<br>%{{customdata:.0%}} answered<extra></extra>",
                        row=r + 1, col=c + 1)
            shown.add(legend)
        ytitle = "top-1 accuracy (all cases)" if spec.get("accuracy") else spec["label"]
        fig.update_yaxes(title_text=ytitle, row=r + 1, col=c + 1,
                         **({} if spec.get("accuracy") else LOGY))
    fig.update_layout(barmode="group", height=720, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.06))
    st.plotly_chart(fig, width="stretch")
    st.caption("Error axes are logarithmic because the unreliable source raises some errors 10-30×. "
               "Count questions share three baselines (persistence, 3-hour rolling mean, training-period mean). The "
               "training-period mean ignores the live feed, so poisoning cannot affect it. That is why it beats the "
               "unprotected model under corruption, although it is much worse on a clean feed.")

    st.subheader("Over the test period")
    cond = st.segmented_control("Feed condition", list(CONDITIONS), format_func=CONDITIONS.get, default="combined",
                                key="cmp_cond") or "combined"
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s["short"] for s in TASKS.values()],
                        vertical_spacing=0.16, horizontal_spacing=0.08)
    shown = set()
    xlab = [f"{w}<br>{s:.0%}" for w, s in enumerate(SCHEDULE)]
    for i, (t, spec) in enumerate(TASKS.items()):
        E = errors(t, sub[t])
        if E.empty:
            continue
        r, c = divmod(i, 2)
        series = [(b, "none", lab, BASE_COLORS[j], "dot") for j, (b, lab) in enumerate(spec["baselines"].items())]
        series += [(spec["model"], "none", "model, countermeasure off", OFF, "solid"),
                   (spec["model"], "quality_aware", "model, countermeasure on", ON, "solid")]
        for pred, mit, lab, col, dash in series:
            s = pick(E, pred, mit, None)
            s = s[(s.condition == cond) & (s.window != "all")].groupby("window").agg(err=("err", "mean"),
                                                                                     cov=("coverage", "mean"))
            fig.add_scatter(x=xlab, y=natural(t, s.err.to_numpy()), mode="lines+markers", name=lab, legendgroup=lab,
                            showlegend=lab not in shown, line=dict(color=col, dash=dash, width=2.5),
                            customdata=s["cov"], hovertemplate=f"{lab}<br>%{{y:.3f}} · %{{customdata:.0%}} answered"
                                                               "<extra></extra>", row=r + 1, col=c + 1)
            shown.add(lab)
        fig.update_yaxes(title_text="top-1 accuracy" if spec.get("accuracy") else spec["label"], row=r + 1, col=c + 1,
                         **({} if spec.get("accuracy") else LOGY))
    fig.update_xaxes(title_text="test window · corrupted share (simulated)", row=2)
    fig.update_layout(height=680, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, width="stretch")
    st.caption("Windows are real chronological periods of the test day. The corruption schedule rising across them is "
               "SIMULATED. Under the clean feed, any change across windows is real drift in traffic, not poisoning.")

# ===================================================================================== tab 2
with tab_pick:
    st.markdown("Each chart below maps to a judging criterion. The questions are compared on one scale: "
                "**error × clean feed**, the model's error under a degraded feed divided by its error on the same "
                "cases with a clean feed. 1× means no effect. For destination, error is the top-1 miss rate.")
    H = {t: headline(t) for t in TASKS}
    H = {t: h for t, h in H.items() if h}
    SC = staleness_curve()

    # ---------------------------------------------------------------- rubric fit
    st.subheader("Rubric fit at a glance")

    def age_effect(t):
        s = SC[(SC.task == t) & (SC.age_min > 0) & (SC.age_min <= 45)]
        return float(s.sort_values("age_min").model.iloc[-1]) if len(s) else np.nan

    def grade(v, strong, moderate, weak=None):
        if pd.isna(v):
            return 0.0
        if v >= strong:
            return 1.0
        if v >= moderate:
            return 0.6
        return 0.3 if weak is None or v >= weak else 0.0

    crit = [  # (criterion, row label, value fn, format, grade fn, rubric points for this row)
        ("C1", "Stale feed: error × clean", lambda t: H[t]["stale"]["x_clean"], "{:.1f}×",
         lambda v: grade(v, 3, 1.5, 1.1), 2.5),
        ("C1", "Input 30-45 min old: error × clean", age_effect, "{:.1f}×", lambda v: grade(v, 3, 1.5, 1.1), 2.5),
        ("C2", "Unreliable source: error × clean", lambda t: H[t]["low_quality"]["x_clean"], "{:.1f}×",
         lambda v: grade(v, 3, 1.5, 1.1), 5),
        ("C3", "Combined: error × clean", lambda t: H[t]["combined"]["x_clean"], "{:.1f}×",
         lambda v: grade(v, 3, 1.5, 1.1), 2.5),
        ("C3", "Effect grows with corruption share (lowest r of 3 conditions)",
         lambda t: min(H[t][c]["trend_r"] for c in STRESS) if H[t]["combined"]["x_clean"] > 1.2 else np.nan,
         "r = {:.2f}", lambda v: grade(v, 0.9, 0.7, 0), 2.5),
        ("C4", "Countermeasure recovers combined loss", lambda t: H[t]["combined"]["recovered"], "{:.0%}",
         lambda v: grade(v, 0.8, 0.5, 0), 2),
        ("C4", "Countermeasure cost on clean feed", lambda t: H[t]["nominal"]["on"] / H[t]["nominal"]["off"] - 1,
         "{:+.1%}", lambda v: grade(-v, -0.02, -0.08, -0.5), 1.5),
        ("C4", "Tested on an unseen defect", lambda t: 1.0 if "holdout_unseen_bias" in H[t] else 0.0,
         "{:.0f}", lambda v: v, 1.5),
        ("E2", "Model beats main baseline, clean feed",
         lambda t: 1 - H[t]["nominal"]["off"] / H[t]["nominal"]["base"], "{:+.0%}", lambda v: grade(v, 0.15, 0.05, 0), 0),
        ("E3", "Combined effect vs seed noise", lambda t: H[t]["combined"]["snr"], "{:.0f}σ",
         lambda v: grade(v, 10, 3, 1), 0),
    ]
    tasks = list(H)
    Z, T, ylab, pts = [], [], [], {t: 0.0 for t in tasks}
    for code, lab, fn, fmt, gr, w in crit:
        vals = [fn(t) for t in tasks]
        g = [gr(v) for v in vals]
        Z.append(g)
        T.append([("yes" if v else "no") if code == "C4" and "unseen" in lab else
                  ("n/a" if pd.isna(v) else fmt.format(v)) for v in vals])
        ylab.append(f"<b>{code}</b> {lab}")
        for t, gv in zip(tasks, g):
            pts[t] += gv * w
    fig = go.Figure(go.Heatmap(z=Z, x=[TASKS[t]["short"] for t in tasks], y=ylab, text=T, texttemplate="%{text}",
                               colorscale=[[0, "#f3e5e5"], [0.3, "#fbe3c4"], [0.6, "#d7ecd9"], [1, "#66bb6a"]],
                               zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
                               hovertemplate="%{x}<br>%{y}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    fig.update_layout(height=470, margin=dict(t=30, b=10, l=10, r=10))
    c1, c2 = st.columns([3, 1])
    c1.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Case-criteria evidence (of 20)**")
        for t in sorted(pts, key=pts.get, reverse=True):
            st.metric(TASKS[t]["short"], f"{pts[t]:.1f}")
    st.caption("Green = strong evidence, pale = weak or none. Thresholds: × clean ≥3 strong, ≥1.5 moderate, ≥1.1 weak; "
               "recovery ≥80% strong; clean-feed cost ≤2% strong. The score weights the C1-C4 rows by their rubric "
               "points (5 each). It shows how much *evidence* each question supplies for the case criteria, not a "
               "predicted judge score. E2/E3 rows are shown for context only. Destination recovery is negative "
               "because deferred stale cases count as misses, and many of them had been answered correctly.")

    # ---------------------------------------------------------------- C1
    st.subheader("C1 · Does accuracy fall as the input gets older?")
    fig = go.Figure()
    for t in tasks:
        s = SC[SC.task == t].sort_values("age_min")
        if s.empty:
            continue
        col = TASKS[t]["color"]
        fig.add_scatter(x=s.age_min, y=s.model, mode="lines+markers", name=f"{TASKS[t]['short']} · model",
                        line=dict(color=col, width=3), customdata=s.n,
                        hovertemplate="%{x:.0f} min old · %{y:.2f}× clean · n=%{customdata:,}<extra></extra>")
        fig.add_scatter(x=s.age_min, y=s.baseline, mode="lines", name=f"{TASKS[t]['short']} · baseline",
                        line=dict(color=col, width=1.5, dash="dot"), customdata=s.n,
                        hovertemplate="%{x:.0f} min old · %{y:.2f}× clean · n=%{customdata:,}<extra></extra>")
    fig.add_hline(y=1, line_color="#56606d", line_width=1)
    fig.update_layout(height=420, margin=dict(t=10, b=10), yaxis=dict(title="error × same case on clean feed", **LOGY),
                      xaxis_title="age of the newest delivered input (min)", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Stale condition, seed 0, matched case by case with the clean run. Position and destination: binned "
               "age of the latest report the system received. Counts: how late the feed was (the last 15-60 min of "
               "reports not delivered yet). At 60 min the count models receive an empty last hour and predict near "
               "zero. Dotted = the question's main baseline, given the same stale input.")

    # ---------------------------------------------------------------- C2
    st.subheader("C2 · Which quality failures do the damage, and does more of the unreliable source mean more harm?")
    c1, c2 = st.columns(2)
    DB = defect_breakdown()
    with c1:
        if len(DB):
            fig = go.Figure()
            for t, dash in [("position_15m", None), ("destination", "/")]:
                s = DB[DB.task == t].sort_values("model")
                for col, name, color in [("baseline", "baseline", "#9e9e9e"), ("model", "model", OFF),
                                         ("countermeasure", "countermeasure on", ON)]:
                    fig.add_bar(y=s.defect, x=s[col], orientation="h", marker_color=color,
                                marker_pattern_shape=dash, name=f"{TASKS[t]['short']} · {name}",
                                customdata=s.n, hovertemplate="%{y}<br>%{x:.2f}× clean · n=%{customdata:,}<extra></extra>")
            fig.add_vline(x=1, line_color="#56606d", line_width=1)
            fig.update_layout(barmode="group", height=520, margin=dict(t=10, b=10),
                              xaxis=dict(title="error × same case on clean feed", **LOGY),
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")
            st.caption("Cases that received a SYN-FASTFEED record, grouped by their worst injected defect. Hatched = "
                       "destination. A gross position outlier makes the position model's error about 160× worse, versus 57× for "
                       "constant velocity: the trained model extrapolates the corrupted track.")
    with c2:
        GC = ghost_curve()
        if len(GC):
            fig = go.Figure()
            for t in ["traffic_density", "port_occupancy"]:
                s = GC[GC.task == t]
                fig.add_scatter(x=s.extra_ids, y=s.model, mode="lines+markers", name=TASKS[t]["short"],
                                line=dict(color=TASKS[t]["color"], width=3), customdata=s.n,
                                hovertemplate="%{x} extra IDs · %{y:.1f}× clean · n=%{customdata:,}<extra></extra>")
            fig.add_hline(y=1, line_color="#56606d", line_width=1)
            fig.update_layout(height=520, margin=dict(t=10, b=10), yaxis=dict(title="error × clean feed", **LOGY),
                              xaxis_title="extra vessel IDs the unreliable source added to the last hour",
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")
            st.caption("Counts, unreliable-source condition, seed 0. This is a direct volume effect: the more ghost "
                       "IDs, duplicates and invalid MMSIs reach a unit, the larger the error. Port zones are busy, so "
                       "almost all of them receive 30+ ghosts.")

    # ---------------------------------------------------------------- C3
    st.subheader("C3 · Stale-only, unreliable-only and combined, over the ordered timeline")
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(STRESS.values()), shared_yaxes=True,
                        horizontal_spacing=0.04)
    for j, c in enumerate(STRESS):
        for t in tasks:
            W = relative_by_window(t)
            s = W[W.condition == c].groupby("share").ratio.agg(["mean", "std"]).reset_index()
            fig.add_scatter(x=s.share, y=s["mean"], error_y=dict(type="data", array=s["std"], thickness=1),
                            mode="lines+markers", name=TASKS[t]["short"], legendgroup=t, showlegend=j == 0,
                            line=dict(color=TASKS[t]["color"], width=3),
                            hovertemplate="%{x:.0%} corrupted · %{y:.2f}× clean<extra></extra>", row=1, col=j + 1)
        fig.add_hline(y=1, line_color="#56606d", line_width=1, row=1, col=j + 1)
    fig.update_yaxes(**LOGY)
    fig.update_yaxes(title_text="model error × clean feed (same window)", row=1, col=1)
    fig.update_xaxes(tickformat=".0%", title_text="corrupted share in window")
    fig.update_layout(height=420, margin=dict(t=40, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    rows = []
    for t in tasks:
        h = H[t]
        s_, l_, c_ = (h[c]["x_clean"] - 1 for c in STRESS)
        rows.append({"question": TASKS[t]["short"], "stale only": s_, "unreliable only": l_, "sum of the two": s_ + l_,
                     "combined (observed)": c_})
    I = pd.DataFrame(rows)
    fig = go.Figure()
    for col, color in [("stale only", "#90caf9"), ("unreliable only", "#ffcc80"), ("sum of the two", "#bdbdbd"),
                       ("combined (observed)", OFF)]:
        fig.add_bar(x=I.question, y=I[col].clip(lower=1e-3), name=col, marker_color=color,
                    hovertemplate="%{x}<br>" + col + ": +%{y:.2f}× extra error<extra></extra>")
    fig.update_layout(barmode="group", height=360, margin=dict(t=10, b=10),
                      yaxis=dict(title="extra error (× clean − 1)", **LOGY), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Is the combined damage more or less than the two failures added up? Position is roughly additive. "
               "For the counts, combined is *less* than the sum: when the feed is late, the ghost rows of the missing "
               "hour are not delivered either.")

    # ---------------------------------------------------------------- C4
    st.subheader("C4 · Countermeasure: recovery against the same baseline, coverage cost and limits")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        sym = {"stale": "circle", "low_quality": "square", "combined": "diamond", "holdout_unseen_bias": "star"}
        for t in tasks:
            for c, s in sym.items():
                if c not in H[t]:
                    continue
                h = H[t][c]
                fig.add_scatter(x=[h["coverage_on"]], y=[h["recovered"]], mode="markers",
                                marker=dict(symbol=s, size=15, color=TASKS[t]["color"], line=dict(width=1, color="#0c0f12")),
                                name=f"{TASKS[t]['short']} · {STRESS.get(c, 'unseen defect (holdout)')}",
                                hovertemplate=f"{TASKS[t]['short']} · {STRESS.get(c, 'unseen defect')}<br>"
                                              "%{x:.0%} answered · %{y:.0%} of the loss recovered<extra></extra>")
        fig.add_hline(y=0, line_color="#56606d", line_width=1)
        fig.update_layout(height=430, margin=dict(t=10, b=10), xaxis=dict(title="share of cases answered", tickformat=".0%"),
                          yaxis=dict(title="share of the poisoning loss recovered", tickformat=".0%"),
                          legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, width="stretch")
        st.caption("Recovered = (error off − error on) ÷ (error off − clean error), with withheld answers charged. "
                   "Top right is best: most of the loss recovered while still answering. Shapes: ● stale, ■ unreliable, "
                   "◆ combined, ★ unseen defect (position only).")
    with c2:
        cost = pd.DataFrame({"question": [TASKS[t]["short"] for t in tasks],
                             "cost": [H[t]["nominal"]["on"] / H[t]["nominal"]["off"] - 1 for t in tasks]})
        fig = go.Figure(go.Bar(x=cost.question, y=cost.cost, marker_color=[TASKS[t]["color"] for t in tasks],
                               text=[f"{v:+.1%}" for v in cost.cost], textposition="outside"))
        fig.update_layout(height=430, margin=dict(t=10, b=10), yaxis=dict(title="error change on a clean feed",
                                                                          tickformat=".0%"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Normal-data tradeoff: how much the countermeasure hurts when nothing is wrong. For counts, it "
                   "drops real one-report vessels.")

    # ---------------------------------------------------------------- E2/E3
    st.subheader("E2 / E3 · Is the trained model more fragile than its baseline?")
    Z, T = [], []
    for t in tasks:
        Z.append([np.log2(H[t][c]["off"] / H[t][c]["base"]) for c in CONDITIONS])
        T.append([f"{H[t][c]['off'] / H[t][c]['base']:.2f}" for c in CONDITIONS])
    fig = go.Figure(go.Heatmap(z=Z, x=list(CONDITIONS.values()), y=[TASKS[t]["short"] for t in tasks], text=T,
                               texttemplate="%{text}", colorscale="RdBu_r", zmid=0, zmin=-2, zmax=2, xgap=3, ygap=3,
                               colorbar=dict(title="model ÷ baseline", tickvals=[-2, -1, 0, 1, 2],
                                             ticktext=["¼", "½", "1", "2", "4"], thickness=10),
                               hovertemplate="%{y} · %{x}<br>model error ÷ baseline error = %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=280, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Model error ÷ main-baseline error, both without the countermeasure. Blue = the model is better, "
               "red = worse. Under the unreliable source, the position model is about twice as bad as constant velocity, "
               "a clear 'AI amplifies the poisoning' finding. Count models stay close to persistence because they learn "
               "a correction to it, so they inherit its exposure.")

    # ---------------------------------------------------------------- recommendation
    st.subheader("Reading the charts")
    ranked = sorted(pts, key=pts.get, reverse=True)
    st.markdown(
        f"* **Strongest poisoning evidence:** {', '.join(TASKS[t]['short'] for t in ranked[:2])} "
        f"(case-criteria evidence {pts[ranked[0]]:.1f} and {pts[ranked[1]]:.1f} of 20).\n"
        "* **Position in 15 min** is the only question with every C1-C4 element: a smooth age-to-error curve, "
        "a named defect that the model amplifies beyond its physics baseline, a roughly additive interaction, and an unseen-defect "
        "holdout that shows a limit. Keep it as the anchor.\n"
        "* **Area traffic density** gives the cleanest *source-volume* story (error rises steadily with the number of ghost IDs) "
        "and still beats persistence on a clean feed, which supports E2.\n"
        "* **Port occupancy proxy** shows the most dramatic numbers, but it comes from the same count pipeline as density "
        "and loses to persistence on a clean feed, so it adds little that density does not.\n"
        "* **Destination** has the best clean-feed model-vs-baseline gain, but it barely reacts to poisoning: a 6-hour "
        "horizon makes a 30-minute-old report nearly as good as a fresh one. That makes it a useful *contrast* "
        "(staleness matters in proportion to the horizon), not a showcase. Its countermeasure also does net harm.")
    st.caption("Notes reflect the default run's numbers. The scores and charts above recompute when the pipeline is rerun.")
