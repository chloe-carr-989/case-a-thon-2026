"""Streamlit entrypoint: four maritime prediction questions plus the shared data-integrity view.

Run (from website/):  .venv\\Scripts\\streamlit run app\\dashboard.py
All numbers come from files written by scripts/run_pipeline.py (results/). Nothing is invented here.
"""
import streamlit as st

st.set_page_config(page_title="AIS Feed-Degradation Lab", layout="wide")

st.navigation([
    st.Page("q_destination.py", title="Where is this vessel going?"),
    st.Page("q_position.py", title="Where will it be in 15 minutes?", default=True),
    st.Page("q_port.py", title="How congested will this port be?"),
    st.Page("q_density.py", title="How busy will this area be?"),
    st.Page("q_integrity.py", title="Data integrity and reporting gaps"),
    st.Page("q_compare.py", title="Baseline vs model performance"),
]).run()
