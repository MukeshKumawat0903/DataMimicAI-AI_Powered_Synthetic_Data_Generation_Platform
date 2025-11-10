import streamlit as st
import requests
import pandas as pd
import io
import os
import numpy as np

API_BASE = os.getenv("API_URL", "http://localhost:8000")

def _call_detect_outliers(file_id):
    """Call backend API to detect outliers in the dataset."""
    resp = requests.post(f"{API_BASE}/eda/detect-outliers", params={"file_id": file_id})
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error("Outlier detection failed.")
        return None

def _show_outlier_stats(result):
    """Show a dataframe with outlier statistics."""
    stats_df = pd.DataFrame(result["stats"])
    st.dataframe(stats_df)
    return stats_df

def _select_and_plot_outlier(df, stats_df, outlier_indices):
    """Let user select column and visualize outliers as a boxplot."""
    col = st.selectbox("Select column", stats_df["column"].tolist(), key="outlier_col")
    if col:
        _plot_outlier_box(df, col, outlier_indices.get(col, []))

def _plot_outlier_box(df, col, outlier_idx=None):
    """Show a boxplot for the selected column, marking detected outliers."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.boxplot(df[col].dropna(), vert=False)
    ax.set_title(f"Boxplot: {col}")
    if outlier_idx is not None and len(outlier_idx) > 0:
        outliers = df[col].iloc[outlier_idx]
        ax.scatter(outliers, np.ones_like(outliers), color="r", label="Outliers")
    st.pyplot(fig)

def _remove_outliers(file_id, cols_to_remove):
    """Send API request to remove outliers for selected columns."""
    rm_resp = requests.post(
        f"{API_BASE}/eda/remove-outliers",
        params={
            "file_id": file_id,
            "columns": ",".join(cols_to_remove)
        }
    )
    if rm_resp.status_code == 200:
        st.success("Outliers removed for selected columns. Data updated!")
        return True
    else:
        st.error("Failed to remove outliers.")
        return False

def _call_detect_drift(real_file_id, synth_file_id):
    """Call backend API to detect distribution drift between real and synthetic datasets."""
    resp = requests.post(
        f"{API_BASE}/eda/detect-drift",
        params={"real_file_id": real_file_id, "synth_file_id": synth_file_id}
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error("Drift detection failed.")
        return None

def _show_drift_stats(result):
    """Display drift statistics as a DataFrame and return for further use."""
    drift_df = pd.DataFrame(result["drift_stats"])
    st.dataframe(drift_df)
    return drift_df

def _select_and_plot_drift(real, synth, drift_df):
    """Allow user to select column and visualize distribution drift."""
    col = st.selectbox("Select column for drift visualization", drift_df["column"].tolist(), key="drift_col")
    if col:
        _plot_drift(real, synth, col)

def _plot_drift(real, synth, col):
    """Plot histograms of real vs synthetic data for the given column."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if col in real.columns and col in synth.columns:
        if np.issubdtype(real[col].dtype, np.number) and np.issubdtype(synth[col].dtype, np.number):
            ax.hist(real[col].dropna(), bins=30, alpha=0.5, label="Real")
            ax.hist(synth[col].dropna(), bins=30, alpha=0.5, label="Synthetic")
            ax.legend()
            ax.set_title(f"Distribution Drift: {col}")
            st.pyplot(fig)
        else:
            st.info("Drift visualization only supported for numeric columns.")
    else:
        st.warning("Selected column missing from real or synthetic data.")

def _download_csv(file_id):
    """Download a CSV file from backend as a pandas DataFrame."""
    resp = requests.get(f"{API_BASE}/eda/download", params={"file_id": file_id})
    if resp.status_code == 200:
        return pd.read_csv(io.BytesIO(resp.content))
    else:
        st.error("Could not load the file from backend.")
        return None

def _generate_pdf_report(file_id, generated_file_id):
    """Request PDF report for outlier and drift analysis from backend."""
    params = {"file_id": file_id}
    if generated_file_id:
        params["synth_file_id"] = generated_file_id
    resp = requests.post(f"{API_BASE}/eda/report-outlier-drift", params=params)
    if resp.status_code == 200:
        st.download_button(
            label="Download Outlier/Drift Report (PDF)",
            data=resp.content,
            file_name="outlier_drift_report.pdf"
        )
    else:
        st.error("Could not generate report.")

def _explain_outlier_drift(file_id, generated_file_id):
    """Request LLM-based explanation for outlier and drift analysis results."""
    params = {"file_id": file_id}
    if generated_file_id:
        params["synth_file_id"] = generated_file_id
    resp = requests.post(f"{API_BASE}/eda/explain-outlier-drift", params=params)
    if resp.status_code == 200:
        st.info(resp.json()["explanation"])
    else:
        st.error("Could not generate explanation.")

def expander_outlier_and_drift():
    """Streamlit expander: Outlier & Drift Detection tools and visualizations."""
    st.divider()
    with st.expander("⚠️ Outlier & Drift Detection", expanded=False):
        st.header("Outlier Detection")

        # ---- Outlier Detection ----
        run_outlier = st.button("Detect Outliers")
        if run_outlier:
            result = _call_detect_outliers(st.session_state.file_id)
            if result:
                st.session_state.outlier_results = result
                st.success("Outlier detection complete!")
        result = st.session_state.get("outlier_results", None)
        if result:
            stats_df = _show_outlier_stats(result)
            df = _download_csv(st.session_state.file_id)
            if df is not None:
                _select_and_plot_outlier(df, stats_df, result["outlier_indices"])

            st.markdown("#### Remove outliers in selected columns")
            cols_to_remove = st.multiselect(
                "Columns:",
                stats_df["column"].tolist(),
                key="remove_outlier_cols"
            )
            if cols_to_remove and st.button("Remove Outliers"):
                if _remove_outliers(st.session_state.file_id, cols_to_remove):
                    del st.session_state.outlier_results
        elif not run_outlier:
            st.info("Click **Detect Outliers** to start.")

        # ---- Drift Detection ----
        st.header("Drift Detection (Real vs. Synthetic)")
        generated_file_id = getattr(st.session_state, "generated_file_id", None)
        if generated_file_id:
            run_drift = st.button("Detect Drift")
            if run_drift or "drift_results" in st.session_state:
                if run_drift:
                    drift_result = _call_detect_drift(
                        st.session_state.file_id, generated_file_id
                        )
                    if drift_result:
                        st.session_state.drift_results = drift_result
                        st.success("Drift detection complete!")
                result = st.session_state.get("drift_results", None)
                if result and result.get("drift_stats"):
                    drift_df = _show_drift_stats(result)
                    real = _download_csv(st.session_state.file_id)
                    synth = _download_csv(generated_file_id)
                    if real is not None and synth is not None:
                        _select_and_plot_drift(real, synth, drift_df)
                else:
                    st.warning("No drift stats found (no common numeric columns?)")
        else:
            st.info("Generate synthetic data first for drift comparison.")

        # ---- PDF Report ----
        st.markdown("### 📄 Download PDF Outlier/Drift Report")
        if st.button("Generate PDF Report"):
            _generate_pdf_report(
                st.session_state.file_id, 
                getattr(st.session_state, "generated_file_id", None)
                )

        # ---- LLM Explanation ----
        st.markdown("### 💡 LLM Explanation for Outlier & Drift Results")
        if st.button("Explain with LLM (Demo)"):
            _explain_outlier_drift(
                st.session_state.file_id, 
                getattr(st.session_state, "generated_file_id", None)
                )
