# helpers/eda_feature_eng/expander_eda_feedback_loop.py

import streamlit as st
from helpers.feedback_loop import show_feedback_loop
import pandas as pd
import frontend_config as config
import requests
import io

api_base  = config.API_BASE

def _download_csv(file_id):
    """Download a CSV file from backend as a pandas DataFrame."""
    resp = requests.get(f"{config.API_BASE}/eda/download", params={"file_id": file_id})
    if resp.status_code == 200:
        return pd.read_csv(io.BytesIO(resp.content))
    else:
        st.error("Could not load the file from backend.")
        return None

def expander_eda_feedback_loop():
    """Streamlit expander: EDA-Driven Feedback Loop for Synthetic Data Generation."""
    st.divider()
    with st.expander("🔁 EDA-Driven Feedback Loop: Refine & Re-Generate Synthetic Data", expanded=False):
        if "show_feedback_loop" not in st.session_state:
            st.session_state["show_feedback_loop"] = False

        file_id = st.session_state.get("file_id")
        if not file_id:
            st.info("Upload and select a dataset first to enable feedback loop.")
            return
        df = _download_csv(file_id)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            st.info("No data available. Please upload a dataset.")
            return

        if not st.session_state["show_feedback_loop"]:
            # Show the button to enter the feedback loop
            if st.button("Run Feedback Loop"):
                st.session_state["show_feedback_loop"] = True
                st.rerun()
        else:
            # Show the feedback loop UI
            show_feedback_loop(file_id, api_base, df)
            # Optionally, add a button to exit back to main view
            if st.button("⬅️ Back to main"):
                st.session_state["show_feedback_loop"] = False
                st.rerun()


