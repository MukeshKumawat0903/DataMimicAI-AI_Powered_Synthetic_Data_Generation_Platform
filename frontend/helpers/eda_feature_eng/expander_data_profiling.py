import streamlit as st
import requests
import pandas as pd
import os
import frontend_config as config

API_BASE = os.getenv("API_URL", "http://localhost:8000")   # Will be set in Render

def _call_profile_api(file_id):
    """Call backend to profile the dataset."""
    st.info("Profiling your dataset. Please wait...")
    with st.spinner("Profiling data..."):
        response = requests.post(
            f"{API_BASE}/eda/profile",
            params={"file_id": file_id}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Profiling failed. Please try again.")
            return None

def _display_profile_summary(result):
    """Display dataset summary and profile dataframe."""
    st.subheader("📊 Dataset Overview")
    st.json(result["summary"])
    df_profile = pd.DataFrame(result["profile"])
    # Convert all list/dict columns to string for safe display
    for col in df_profile.columns:
        if df_profile[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_profile[col] = df_profile[col].apply(str)
    st.dataframe(df_profile)

def _handle_suggestions(result, file_id):
    """Show and process fix suggestions with checkboxes."""
    st.markdown("#### 🛠️ Suggested Fixes")
    to_impute, impute_methods = [], []
    to_drop_const, to_drop_dup, to_drop_highmiss = [], [], []
    to_encode, to_outlier = [], []

    for sug in result["suggestions"]:
        col = sug["column"]
        label = f"{sug['action']}"
        key = f"{sug['fix_type']}_{col}"

        # Imputation with method select
        if sug["fix_type"] == "impute":
            checked = st.checkbox(label, key=key)
            method = sug.get("method", "mean")
            method = st.selectbox(
                f"Impute method for {col}", ["mean", "median", "mode", "knn"], 
                index=["mean", "median", "mode", "knn"].index(method), key=f"impute_method_{col}"
            ) if checked else method
            if checked:
                to_impute.append(col)
                impute_methods.append(method)

        elif sug["fix_type"] == "drop_constant":
            checked = st.checkbox(label, key=key)
            if checked: to_drop_const.append(col)

        elif sug["fix_type"] == "drop_duplicate":
            checked = st.checkbox(label, key=key)
            if checked: to_drop_dup.append(col)

        elif sug["fix_type"] == "drop_high_missing":
            checked = st.checkbox(label, key=key)
            if checked: to_drop_highmiss.append(col)

        elif sug["fix_type"] == "encode":
            checked = st.checkbox(label, key=key)
            if checked: to_encode.append(col)

        elif sug["fix_type"] == "outlier":
            checked = st.checkbox(label, key=key)
            if checked: to_outlier.append(col)

    if (to_impute or to_drop_const or to_drop_dup or to_drop_highmiss or to_encode or to_outlier):
        if st.button("Apply Selected Fixes"):
            _apply_fixes(
                file_id,
                to_impute, impute_methods,
                to_drop_const, to_drop_dup, to_drop_highmiss,
                to_encode, to_outlier
            )

def _apply_fixes(
    file_id, 
    to_impute, impute_methods, 
    to_drop_const, to_drop_dup, to_drop_highmiss,
    to_encode, to_outlier
):
    """Call API to apply all selected fixes."""
    # Impute
    for col, method in zip(to_impute, impute_methods):
        fix_resp = requests.post(
            f"{API_BASE}/eda/fix-missing",
            params={
                "file_id": file_id,
                "columns": col,
                "method": method
            }
        )
        if fix_resp.status_code == 200:
            st.success(f"Imputed: {col} ({method})")
        else:
            st.error(f"Impute failed for {col}")

    # Drop columns (const, dup, highmiss)
    for group, group_name in [
        (to_drop_const, "constant"),
        (to_drop_dup, "duplicate"),
        (to_drop_highmiss, "high-missing"),
    ]:
        if group:
            resp = requests.post(
                f"{API_BASE}/eda/drop-columns",
                params={
                    "file_id": file_id,
                    "columns": ",".join(group)
                }
            )
            if resp.status_code == 200:
                st.success(f"Dropped {group_name} columns: {', '.join(group)}")
            else:
                st.error(f"Failed to drop {group_name} columns.")

    # Encode high-cardinality
    if to_encode:
        encode_resp = requests.post(
            f"{API_BASE}/eda/encode-high-cardinality",
            params={
                "file_id": file_id,
                "columns": ",".join(to_encode),
                "max_values": 10
            }
        )
        if encode_resp.status_code == 200:
            st.success(f"Encoded high-cardinality columns: {', '.join(to_encode)}")
        else:
            st.error("Failed to encode columns.")

    # Drop outliers
    if to_outlier:
        out_resp = requests.post(
            f"{API_BASE}/eda/drop-outliers",
            params={
                "file_id": file_id,
                "columns": ",".join(to_outlier),
                "z_thresh": 3.0
            }
        )
        if out_resp.status_code == 200:
            st.success(f"Dropped outliers in: {', '.join(to_outlier)}")
        else:
            st.error("Failed to drop outliers.")

    # Clear profile cache so user can rerun and see updated suggestions
    st.session_state.pop('profile_result', None)
    st.info("Rerun profiling for updated stats.")

def expander_data_profiling():
    """Main Streamlit expander for Data Profiling, using helpers above."""
    st.divider()
    with st.expander("🔍 Automated Data Profiling", expanded=False):
        run_profile = st.button("Run Data Profiling")

        # Always reload if user clicks, or if no results yet
        if run_profile or 'profile_result' not in st.session_state:
            result = _call_profile_api(st.session_state.file_id)
            if result:
                st.session_state['profile_result'] = result
        result = st.session_state.get('profile_result', None)
        if result:
            _display_profile_summary(result)
            _handle_suggestions(result, st.session_state.file_id)
