"""Streamlit entrypoint: AIS maritime intelligence in two modes plus two evaluation pages.

Commercial (arrival prediction) and Government (movement anomalies) share one map, the learned sea lanes,
the AIS data and the timeline. Data Integrity (observed problems and the injected defects) and Model Performance
(baseline vs model, the poisoning experiment, oversight) sit to their right in the top bar.

Run (from website/):  .venv\\Scripts\\streamlit run app\\dashboard.py
All numbers come from files written by scripts/run_pipeline.py (results/). Nothing is invented here.
"""
import streamlit as st

import ui

st.set_page_config(page_title="AIS Maritime Intelligence", layout="wide", initial_sidebar_state="collapsed")

MODES = {"Commercial": st.Page("mode_commercial.py", title="Commercial", default=True),
         "Government": st.Page("mode_government.py", title="Government")}
EVALS = {"Data Integrity": st.Page("q_integrity.py", title="Data Integrity"),
         "Model Performance": st.Page("q_compare.py", title="Model Performance")}
page = st.navigation([*MODES.values(), *EVALS.values()], position="hidden")
ui.topbar(MODES, EVALS, page)
page.run()
