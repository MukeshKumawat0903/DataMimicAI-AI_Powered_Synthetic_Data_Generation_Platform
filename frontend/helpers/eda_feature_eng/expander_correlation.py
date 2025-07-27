import streamlit as st
import requests
import pandas as pd
import os

API_BASE = os.getenv("API_URL", "http://localhost:8000")

def _call_correlation_api(file_id, top_k=10):
    """Call backend to get correlation and pattern discovery results."""
    with st.spinner("Calculating correlations..."):
        response = requests.post(
            f"{API_BASE}/eda/correlation",
            params={"file_id": file_id, "top_k": top_k}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Correlation analysis failed. Please try again.")
            return None

def _show_corr_heatmap(result):
    """Display the Pearson correlation heatmap if available."""
    st.subheader("📈 Pearson Correlation Heatmap")
    if result.get("corr_heatmap_base64"):
        st.markdown(
            f'<img src="data:image/png;base64,{result["corr_heatmap_base64"]}" style="max-width:100%;">',
            unsafe_allow_html=True
        )
    else:
        st.info("Not enough numeric features for correlation heatmap.")

def _show_top_corr_pairs(result):
    st.subheader("🔥 Top Correlated Feature Pairs")
    top_corrs = result.get("top_corrs", [])
    if top_corrs:
        top_pairs_df = pd.DataFrame(top_corrs, columns=["Feature 1", "Feature 2", "Correlation"])
        st.table(top_pairs_df)
    else:
        st.info("No strong linear correlations detected.")

def _show_nonlinear_corrs(result):
    with st.expander("Show Nonlinear Correlations (Spearman, Kendall)", expanded=False):
        spearman = result.get("spearman_corr_matrix", {})
        kendall = result.get("kendall_corr_matrix", {})
        if spearman:
            st.write("**Spearman:**")
            st.dataframe(pd.DataFrame(spearman))
        else:
            st.info("Not enough numeric features for Spearman.")
        if kendall:
            st.write("**Kendall:**")
            st.dataframe(pd.DataFrame(kendall))
        else:
            st.info("Not enough numeric features for Kendall.")

def _show_categorical_associations(result):
    st.subheader("🔄 Categorical Associations (Cramér's V)")
    cat_df = pd.DataFrame(result.get("categorical_assoc", []))
    if not cat_df.empty:
        colmap = {k: "Feature 1" for k in cat_df.columns if k.lower() in ["col1", "feature 1"]}
        colmap.update({k: "Feature 2" for k in cat_df.columns if k.lower() in ["col2", "feature 2"]})
        colmap.update({k: "CramersV" for k in cat_df.columns if "cramers" in k.lower()})
        cat_df = cat_df.rename(columns=colmap)
        cat_df = cat_df[["Feature 1", "Feature 2", "CramersV"]]
        cat_df = cat_df.sort_values("CramersV", ascending=False)
        st.dataframe(cat_df.head(10))
        st.markdown(
            """
            - **Cramér’s V ≈ 1.0:** Features are almost always found together (strong dependency)
            - **0.4–0.6:** Strong business link—reflects a real market or operational structure
            - **Below 0.2:** Features are mostly independent
            """
        )
    else:
        st.info("No categorical pairs detected for Cramér’s V.")

def _show_data_leakage(result):
    st.subheader("🚨 Potential Data Leakage or Duplicates")
    leakage = result.get("leakage_pairs", [])
    if leakage:
        st.warning("Features with almost perfect correlation detected!")
        leak_df = pd.DataFrame(leakage, columns=["Feature 1", "Feature 2", "Correlation"])
        st.table(leak_df)
    else:
        st.success("No strong data leakage detected.")

def expander_correlation():
    """Main Streamlit expander for Smart Correlation & Pattern Discovery."""
    st.divider()
    with st.expander("🔗 Smart Correlation & Pattern Discovery", expanded=False):
        run_corr = st.button("Run Correlation & Pattern Discovery")
        if run_corr:
            result = _call_correlation_api(st.session_state.file_id)
            if result:
                st.session_state['correlation_result'] = result
        result = st.session_state.get('correlation_result', None)
        if result:
            _show_corr_heatmap(result)
            _show_top_corr_pairs(result)
            _show_nonlinear_corrs(result)
            _show_categorical_associations(result)
            _show_data_leakage(result)


