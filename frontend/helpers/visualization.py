"""
DataMimicAI Visualization Tab - Tabbed, Modern, Modular
"""
import streamlit as st
import requests
import os

API_BASE = os.getenv("API_URL", "http://localhost:8000")

def show_visualization():
    """
    Main Visualization UI with tabs. Handles column selection, triggers visualization API, and renders results.
    """
    if not st.session_state.get("generated_file_id"):
        st.warning("Generate synthetic data first to enable visualization.")
        return

    if not st.session_state.get("data_columns"):
        st.error("No columns available for visualization.")
        return

    # Try to get numeric columns for stat/corr tabs (optional, see note below)
    numeric_columns = st.session_state.get("numeric_columns", st.session_state.data_columns)

    tab_labels = [
        "Distribution", "Pair Plot", "Real vs. Synthetic",
        "Drift Detection", "Correlation"
    ]
    tabs = st.tabs(tab_labels)

    # 1. Distribution Tab
    with tabs[0]:
        col = st.selectbox(
            "Select feature for distribution plot",
            options=st.session_state.data_columns,
            key="dist_col"
        )
        plot_type = st.radio(
            "Plot Type", ["Histogram", "KDE", "Boxplot"],
            horizontal=True, key="dist_plot_type"
        )
        overlay = st.checkbox("Overlay Real vs. Synthetic", value=True, key="dist_overlay")
        if st.button("Show Distribution", key="dist_btn"):
            with st.spinner("Generating plot..."):
                _render_api_html(
                    tab="distribution",
                    params={
                        "file_id": st.session_state.generated_file_id,
                        "column": col,
                        "plot_type": plot_type,
                        "overlay": str(overlay),
                    }
                )

    # 2. Pair Plot Tab
    with tabs[1]:
        pair_cols = st.multiselect(
            "Select 2-3 features for pair plot",
            options=st.session_state.data_columns,
            default=st.session_state.data_columns[:2],
            key="pair_cols"
        )
        overlay = st.checkbox("Color by Real vs. Synthetic", value=True, key="pair_overlay")
        if st.button("Show Pair Plot", key="pair_btn"):
            if len(pair_cols) < 2:
                st.warning("Select at least two columns.")
            else:
                with st.spinner("Generating pair plot..."):
                    _render_api_html(
                        tab="pairplot",
                        params={
                            "file_id": st.session_state.generated_file_id,
                            "columns": ",".join(pair_cols),
                            "overlay": str(overlay),
                        }
                    )

    # 3. Real vs. Synthetic Tab
    with tabs[2]:
        compare_cols = st.multiselect(
            "Select columns to compare (metrics & plots)",
            options=numeric_columns,  # Prefer numeric columns for stats
            default=numeric_columns[:2],
            key="rvss_cols"
        )
        if st.button("Compare Real vs. Synthetic", key="rvss_btn"):
            if not compare_cols:
                st.warning("Select at least one column.")
            else:
                with st.spinner("Comparing..."):
                    _render_api_html(
                        tab="real_vs_synth",
                        params={
                            "file_id": st.session_state.generated_file_id,
                            "columns": ",".join(compare_cols),
                        }
                    )

                    # Fetch metrics from backend and show a summary
                    try:
                        resp = requests.get(f"{API_BASE.rstrip('/')}/metrics", params={
                            "file_id": st.session_state.generated_file_id,
                            "columns": ",".join(compare_cols)
                        }, timeout=20)
                        if resp.status_code == 200:
                            data = resp.json()
                            metrics = data.get('metrics', {})
                            with st.expander("Metrics Summary", expanded=True):
                                for col, m in metrics.items():
                                    st.subheader(col)
                                    for k, v in m.items():
                                        st.markdown(f"- **{k}**: {v}")
                        else:
                            st.info(f"Metrics not available: {resp.status_code}")
                    except Exception as e:
                        st.info(f"Metrics request failed: {str(e)}")

    # 4. Drift Detection Tab
    with tabs[3]:
        drift_cols = st.multiselect(
            "Select columns for drift analysis",
            options=numeric_columns,  # Prefer numeric columns for drift
            default=numeric_columns[:2],
            key="drift_cols"
        )
        if st.button("Run Drift Detection", key="drift_btn"):
            if not drift_cols:
                st.warning("Select at least one column.")
            else:
                with st.spinner("Running drift analysis..."):
                    _render_api_html(
                        tab="drift",
                        params={
                            "file_id": st.session_state.generated_file_id,
                            "columns": ",".join(drift_cols),
                        }
                    )

    # 5. Correlation Tab
    with tabs[4]:
        show_type = st.radio(
            "Show correlation for:",
            ["Real Data", "Synthetic Data", "Compare Both"],
            horizontal=True, key="corr_show_type"
        )
        if st.button("Show Correlation Heatmap", key="corr_btn"):
            with st.spinner("Generating correlation heatmap..."):
                _render_api_html(
                    tab="correlation",
                    params={
                        "file_id": st.session_state.generated_file_id,
                        "show_type": show_type,
                    }
                )


def _render_api_html(tab, params):
    """Calls backend and renders returned HTML for each visualization."""
    try:
        url = f"{API_BASE}/visualize"
        params["tab"] = tab
        response = requests.get(url, params=params)
        if response.status_code == 200:
            st.components.v1.html(
                response.text, height=800, scrolling=True
            )
        else:
            with st.expander("See full error", expanded=False):
                st.error(f"Visualization failed: {response.text}")
    except Exception as e:
        with st.expander("See full error", expanded=False):
            st.error(f"Visualization error: {str(e)}")
