"""Streamlit entrypoint: AIS maritime intelligence in two modes plus the evaluation pages.

Commercial (arrival prediction) and Government (movement anomalies) share one map, the learned sea lanes,
the AIS data and the timeline. The original question pages and the data-integrity / comparison views sit
under Evaluation.

Run (from website/):  .venv\\Scripts\\streamlit run app\\dashboard.py
All numbers come from files written by scripts/run_pipeline.py (results/). Nothing is invented here.
"""
import streamlit as st

st.set_page_config(page_title="AIS Maritime Intelligence", layout="wide", initial_sidebar_state="collapsed")

st.navigation({
    "": [st.Page("mode_commercial.py", title="Commercial", default=True),
         st.Page("mode_government.py", title="Government")],
    "Evaluation": [
        st.Page("q_position.py", title="Position in 15 minutes"),
        st.Page("q_destination.py", title="Destination zone"),
        st.Page("q_port.py", title="Port occupancy proxy"),
        st.Page("q_density.py", title="Area traffic density"),
        st.Page("q_integrity.py", title="Data integrity and reporting gaps"),
        st.Page("q_compare.py", title="Baseline vs model performance"),
    ],
}, position="top").run()
