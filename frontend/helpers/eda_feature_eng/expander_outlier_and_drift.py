import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import os
import numpy as np

API_BASE = os.getenv("API_URL", "http://localhost:8000")

# ============================================================================
# OUTLIER DETECTION FUNCTIONS
# ============================================================================

def _call_detect_outliers_comprehensive(file_id, methods, columns=None):
    """Call comprehensive outlier detection API."""
    params = {
        "file_id": file_id,
        "methods": ",".join(methods)
    }
    if columns:
        params["columns"] = ",".join(columns)
    
    resp = requests.post(
        f"{API_BASE}/eda/outliers/detect-comprehensive",
        params=params,
        timeout=60
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Outlier detection failed: {resp.text}")
        return None


def _call_detect_timeseries_outliers(file_id, value_col, datetime_col, freq=None):
    """Call time-series outlier detection API."""
    params = {
        "file_id": file_id,
        "value_col": value_col,
        "datetime_col": datetime_col
    }
    if freq:
        params["freq"] = freq
    
    resp = requests.post(
        f"{API_BASE}/eda/outliers/detect-timeseries",
        params=params,
        timeout=60
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Time-series outlier detection failed: {resp.text}")
        return None


def _call_detect_cooks_distance(file_id, target_col, feature_cols=None):
    """Call Cook's Distance outlier detection API."""
    params = {
        "file_id": file_id,
        "target_col": target_col
    }
    if feature_cols:
        params["feature_cols"] = ",".join(feature_cols)
    
    resp = requests.post(
        f"{API_BASE}/eda/outliers/detect-cooks-distance",
        params=params,
        timeout=60
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Cook's Distance detection failed: {resp.text}")
        return None


# ============================================================================
# REMEDIATION FUNCTIONS
# ============================================================================

def _call_remediate_outliers(file_id, method, columns, **kwargs):
    """Call outlier remediation API."""
    params = {
        "file_id": file_id,
        "method": method,
        "columns": ",".join(columns)
    }
    
    resp = requests.post(
        f"{API_BASE}/eda/outliers/remediate",
        params=params,
        json=kwargs,
        timeout=60
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Remediation failed: {resp.text}")
        return None


def _call_winsorize(file_id, columns, lower_limit, upper_limit):
    """Call winsorization API."""
    resp = requests.post(
        f"{API_BASE}/eda/outliers/winsorize",
        params={
            "file_id": file_id,
            "columns": ",".join(columns),
            "lower_limit": lower_limit,
            "upper_limit": upper_limit
        },
        timeout=60
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Winsorization failed: {resp.text}")
        return None


def _call_cap_outliers(file_id, columns, cap_method, multiplier, lower_pct, upper_pct):
    """Call capping API."""
    resp = requests.post(
        f"{API_BASE}/eda/outliers/cap",
        params={
            "file_id": file_id,
            "columns": ",".join(columns),
            "cap_method": cap_method,
            "multiplier": multiplier,
            "lower_percentile": lower_pct,
            "upper_percentile": upper_pct
        },
        timeout=60
    )
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Capping failed: {resp.text}")
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def _plot_outlier_boxplot(df, column, outlier_idx=None):
    """Create interactive Plotly boxplot with outliers highlighted."""
    fig = go.Figure()
    
    # Main boxplot
    fig.add_trace(go.Box(
        y=df[column].dropna(),
        name=column,
        boxmean='sd',
        marker_color='lightblue'
    ))
    
    # Highlight outliers
    if outlier_idx and len(outlier_idx) > 0:
        outlier_values = df.loc[outlier_idx, column].dropna()
        fig.add_trace(go.Scatter(
            y=outlier_values,
            x=[column] * len(outlier_values),
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title=f"Boxplot: {column}",
        yaxis_title=column,
        showlegend=True,
        height=400
    )
    
    return fig


def _plot_outlier_kde(df, column, outlier_idx=None):
    """Create KDE plot with outliers highlighted."""
    from scipy.stats import gaussian_kde
    
    data = df[column].dropna()
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    density = kde(x_range)
    
    fig = go.Figure()
    
    # KDE curve
    fig.add_trace(go.Scatter(
        x=x_range,
        y=density,
        mode='lines',
        name='Density',
        fill='tozeroy',
        line=dict(color='blue')
    ))
    
    # Outliers
    if outlier_idx and len(outlier_idx) > 0:
        outlier_values = df.loc[outlier_idx, column].dropna()
        outlier_density = kde(outlier_values)
        fig.add_trace(go.Scatter(
            x=outlier_values,
            y=outlier_density,
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title=f"KDE Plot: {column}",
        xaxis_title=column,
        yaxis_title='Density',
        showlegend=True,
        height=400
    )
    
    return fig


def _download_csv(file_id):
    """Download CSV from backend."""
    resp = requests.get(f"{API_BASE}/eda/download", params={"file_id": file_id}, timeout=30)
    if resp.status_code == 200:
        return pd.read_csv(io.BytesIO(resp.content))
    else:
        st.error("Could not load file from backend.")
        return None


# ============================================================================
# MAIN EXPANDER FUNCTION
# ============================================================================

def expander_outlier_detection_remediation():
    """
    Comprehensive Outlier Detection and Remediation Workbench for EDA.
    """
    st.divider()
    with st.expander("⚠️ Outlier Detection & Remediation", expanded=False):
        
        # Load data
        df = _download_csv(st.session_state.file_id)
        if df is None:
            st.warning("Could not load dataset.")
            return
        
        # Method selection
        st.subheader("1️⃣ Select Detection Methods")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Univariate Methods**")
            use_zscore = st.checkbox("Z-Score", value=True, key="use_zscore")
            use_iqr = st.checkbox("IQR (Interquartile Range)", value=True, key="use_iqr")
            use_mad = st.checkbox("MAD (Median Absolute Deviation)", value=True, key="use_mad")
        
        with col2:
            st.markdown("**Multivariate Methods**")
            use_isolation = st.checkbox("Isolation Forest", key="use_isolation")
            use_lof = st.checkbox("LOF (Local Outlier Factor)", key="use_lof")
        
        # Column selection
        st.markdown("---")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Select columns for analysis (leave empty for all numeric columns)",
            numeric_cols,
            key="outlier_cols"
        )
        
        # Build methods list
        methods = []
        if use_zscore:
            methods.append("zscore")
        if use_iqr:
            methods.append("iqr")
        if use_mad:
            methods.append("mad")
        if use_isolation:
            methods.append("isolation_forest")
        if use_lof:
            methods.append("lof")
        
        # Detect button - prominent placement
        st.markdown("")  # Add some spacing
        if st.button("🔍 Detect Outliers", type="primary", use_container_width=True, key="detect_outliers_btn"):
            if not methods:
                st.warning("Please select at least one detection method.")
            else:
                with st.spinner("Detecting outliers..."):
                    result = _call_detect_outliers_comprehensive(
                        st.session_state.file_id,
                        methods,
                        selected_cols if selected_cols else None
                    )
                    if result:
                        st.session_state.outlier_results = result
                        st.success("✅ Outlier detection complete!")
        
        # Display results
        if "outlier_results" in st.session_state:
            result = st.session_state.outlier_results
            
            st.subheader("2️⃣ Detection Results")
            
            # Summary statistics
            if "stats" in result and result["stats"]:
                stats_df = pd.DataFrame(result["stats"])
                st.dataframe(stats_df, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_outliers = stats_df["outlier_count"].sum()
                    st.metric("Total Outliers Detected", f"{int(total_outliers)}")
                with col2:
                    avg_pct = stats_df["outlier_percent"].mean()
                    st.metric("Avg Outlier %", f"{avg_pct:.2f}%")
                with col3:
                    st.metric("Methods Used", len(methods))
            
            # Results list + Quick Actions
            st.subheader("3️⃣ Results & Quick Actions")
            
            if "results" in result:
                # Get all columns with outliers
                all_outlier_cols = set()
                for method_results in result["results"].values():
                    all_outlier_cols.update(method_results.keys())

                # Remove 'multivariate' placeholder if present
                all_outlier_cols = {c for c in all_outlier_cols if c != 'multivariate'}

                if all_outlier_cols:
                        # Render a flat table-like list of columns with quick actions
                        st.markdown("**Per-column Quick Actions**")
                        header_cols = st.columns([2, 1, 1, 1, 1, 1])
                        header_cols[0].markdown("**Column**")
                        header_cols[1].markdown("**Outliers**")
                        header_cols[2].markdown("**Viz**")
                        header_cols[3].markdown("**Winsorize**")
                        header_cols[4].markdown("**Cap (IQR)**")
                        header_cols[5].markdown("**Actions**")

                        for col_name in sorted(all_outlier_cols):
                            # Collect outlier indices from all methods for this column
                            outlier_idx = set()
                            for method_results in result["results"].values():
                                if col_name in method_results:
                                    outlier_idx.update(method_results[col_name])

                            n_out = len(outlier_idx)
                            row_cols = st.columns([2, 1, 1, 1, 1, 1])
                            # Column name
                            row_cols[0].write(col_name)
                            # Outlier count
                            row_cols[1].write(str(n_out))
                            # Visualization (small)
                            if n_out > 0:
                                with row_cols[2]:
                                    fig = _plot_outlier_boxplot(df, col_name, list(outlier_idx))
                                    st.plotly_chart(fig, use_container_width=True, height=200)
                            else:
                                row_cols[2].write("—")

                            # Quick Winsorize button
                            with row_cols[3]:
                                if st.button("Winsorize", key=f"winsor_row_{col_name}"):
                                    res = _call_winsorize(st.session_state.file_id, [col_name], 0.05, 0.05)
                                    if res:
                                        st.success(f"Winsorize applied to {col_name}.")
                                        st.session_state.remediation_preset = {"method": "winsorize", "columns": [col_name], "params": {"lower_limit": 0.05, "upper_limit": 0.05}}

                            # Quick Cap (IQR) button
                            with row_cols[4]:
                                if st.button("Cap", key=f"cap_row_{col_name}"):
                                    res = _call_cap_outliers(st.session_state.file_id, [col_name], "iqr", 1.5, 1.0, 99.0)
                                    if res:
                                        st.success(f"Capping (IQR) applied to {col_name}.")
                                        st.session_state.remediation_preset = {"method": "cap", "columns": [col_name], "params": {"cap_method": "iqr", "multiplier": 1.5}}

                            # Actions: Preview and Open Remediation
                            with row_cols[5]:
                                if st.button("Preview", key=f"preview_row_{col_name}"):
                                    preview = _call_remediate_outliers(st.session_state.file_id, "winsorize", [col_name], dry_run=True, lower_limit=0.05, upper_limit=0.05)
                                    if preview:
                                        if "preview_sample" in preview:
                                            st.write(preview.get("preview_sample"))
                                        else:
                                            st.json(preview)
                                if st.button("Open Remediation", key=f"open_row_{col_name}"):
                                    st.session_state.remediation_preset = {"method": "winsorize", "columns": [col_name], "params": {}}
                                    st.info(f"Scroll down to Remediation section to review/apply for {col_name}.")
        
        # ========== REMEDIATION SECTION ==========
        st.divider()
        st.header("🔧 Outlier Remediation")
        
        if "outlier_results" not in st.session_state:
            st.info("ℹ️ Run outlier detection first to enable remediation.")
        else:
            df = _download_csv(st.session_state.file_id)
            if df is None:
                st.warning("Could not load dataset.")
            else:
                # Get columns with outliers
                result = st.session_state.outlier_results
                outlier_cols = set()
                if "results" in result:
                    for method_results in result["results"].values():
                        outlier_cols.update([k for k in method_results.keys() if k != 'multivariate'])
                
                if not outlier_cols:
                    st.info("No outliers detected in individual columns.")
                else:
                    # Remediation method selection
                    st.subheader("1️⃣ Select Remediation Method")

                    # Prefill remediation controls from Detection quick-action if available
                    preset = st.session_state.get("remediation_preset")
                    if preset:
                        method_map = {"winsorize": "Winsorize", "cap": "Cap Outliers", "remove": "Remove Outliers", "bin": "Bin Values", "transform": "Transform"}
                        # Set session defaults to allow selectbox/multiselect to show prefilled values
                        st.session_state.remediation_method = method_map.get(preset.get("method"), st.session_state.get("remediation_method", "Winsorize"))
                        st.session_state.remediation_cols = preset.get("columns", st.session_state.get("remediation_cols", []))
                        params = preset.get("params", {})
                        if params.get("lower_limit") is not None:
                            st.session_state.winsor_lower = params.get("lower_limit")
                        if params.get("upper_limit") is not None:
                            st.session_state.winsor_upper = params.get("upper_limit")
                        if params.get("multiplier") is not None:
                            st.session_state.iqr_mult = params.get("multiplier")

                    remediation_method = st.selectbox(
                        "Method",
                        ["Winsorize", "Cap Outliers", "Remove Outliers", "Bin Values", "Transform"],
                        key="remediation_method"
                    )
                    
                    # Column selection
                    remediation_cols = st.multiselect(
                        "Select columns to remediate",
                        sorted(outlier_cols),
                        key="remediation_cols"
                    )
                    
                    # Method-specific parameters
                    st.subheader("2️⃣ Configure Parameters")
                    
                    if remediation_method == "Winsorize":
                        col1, col2 = st.columns(2)
                        with col1:
                            lower_limit = st.slider("Lower percentile", 0.0, 0.2, 0.05, 0.01, key="winsor_lower")
                        with col2:
                            upper_limit = st.slider("Upper percentile", 0.0, 0.2, 0.05, 0.01, key="winsor_upper")
                        
                        if st.button("Apply Winsorization", key="apply_winsor"):
                            if remediation_cols:
                                with st.spinner("Applying winsorization..."):
                                    result = _call_winsorize(
                                        st.session_state.file_id,
                                        remediation_cols,
                                        lower_limit,
                                        upper_limit
                                    )
                                    if result:
                                        st.success(f"✅ Winsorization applied to {len(remediation_cols)} columns!")
                                        st.json(result.get("history", []))
                            else:
                                st.warning("Please select at least one column.")
                    
                    elif remediation_method == "Cap Outliers":
                            cap_method = st.radio("Capping method", ["IQR", "Percentile"], horizontal=True, key="cap_method_radio")
                            
                            if cap_method == "IQR":
                                multiplier = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1, key="iqr_mult")
                                lower_pct, upper_pct = 1.0, 99.0
                            else:
                                col1, col2 = st.columns(2)
                                with col1:
                                    lower_pct = st.slider("Lower percentile", 0.0, 10.0, 1.0, 0.5, key="cap_lower")
                                with col2:
                                    upper_pct = st.slider("Upper percentile", 90.0, 100.0, 99.0, 0.5, key="cap_upper")
                                multiplier = 1.5
                            
                            if st.button("Apply Capping", key="apply_cap"):
                                if remediation_cols:
                                    with st.spinner("Applying capping..."):
                                        result = _call_cap_outliers(
                                            st.session_state.file_id,
                                            remediation_cols,
                                            cap_method.lower(),
                                            multiplier,
                                            lower_pct,
                                            upper_pct
                                        )
                                        if result:
                                            st.success(f"✅ Capping applied to {len(remediation_cols)} columns!")
                                            st.json(result.get("history", []))
                            else:
                                st.warning("Please select at least one column.")
                    
                    elif remediation_method == "Remove Outliers":
                            st.info("This will remove rows containing outliers in the selected columns.")
                            removal_strategy = st.radio(
                                "Removal strategy",
                                ["Union (remove if outlier in ANY column)", "Intersection (remove if outlier in ALL columns)"],
                                key="removal_strategy"
                            )
                            
                            if st.button("Remove Outliers", key="apply_remove"):
                                if remediation_cols:
                                    st.warning("⚠️ This operation will modify the dataset. Ensure you have a backup!")
                                    
                                    # Collect outlier indices for selected columns
                                    outlier_indices = {}
                                    for method_results in result["results"].values():
                                        for col in remediation_cols:
                                            if col in method_results:
                                                if col not in outlier_indices:
                                                    outlier_indices[col] = set()
                                                outlier_indices[col].update(method_results[col])
                                    
                                    # Convert to lists
                                    outlier_indices = {k: list(v) for k, v in outlier_indices.items()}
                                    
                                    strategy = "union" if "Union" in removal_strategy else "intersection"
                                    
                                    with st.spinner("Removing outliers..."):
                                        result = _call_remediate_outliers(
                                            st.session_state.file_id,
                                            "remove",
                                            remediation_cols,
                                            outlier_indices=outlier_indices,
                                            removal_method=strategy
                                        )
                                        if result:
                                            st.success(f"✅ Outliers removed! Rows remaining: {result.get('n_rows', 'N/A')}")
                                            st.json(result.get("metadata", {}).get("history", []))
                            else:
                                st.warning("Please select at least one column.")
                    
                    elif remediation_method == "Bin Values":
                            bin_strategy = st.selectbox("Binning strategy", ["quantile", "uniform", "kmeans"], key="bin_strategy")
                            n_bins = st.slider("Number of bins", 2, 20, 5, 1, key="n_bins")
                            
                            if st.button("Apply Binning", key="apply_bin"):
                                if remediation_cols:
                                    with st.spinner("Applying binning..."):
                                        result = _call_remediate_outliers(
                                            st.session_state.file_id,
                                            "bin",
                                            remediation_cols,
                                            strategy=bin_strategy,
                                            n_bins=n_bins
                                        )
                                        if result:
                                            st.success(f"✅ Binning applied to {len(remediation_cols)} columns!")
                                            st.info("New binned columns created with '_binned' suffix.")
                                            st.json(result.get("metadata", {}).get("history", []))
                            else:
                                st.warning("Please select at least one column.")
                    
                    elif remediation_method == "Transform":
                            transform_method = st.selectbox(
                                "Transformation method",
                                ["log", "sqrt", "boxcox", "zscore"],
                                key="transform_method"
                            )
                            
                            if st.button("Apply Transformation", key="apply_transform"):
                                if remediation_cols:
                                    with st.spinner("Applying transformation..."):
                                        result = _call_remediate_outliers(
                                            st.session_state.file_id,
                                            "transform",
                                            remediation_cols,
                                            transform_method=transform_method
                                        )
                                        if result:
                                            st.success(f"✅ Transformation applied to {len(remediation_cols)} columns!")
                                            st.info(f"New transformed columns created with '_{transform_method}' suffix.")
                                            st.json(result.get("metadata", {}).get("history", []))
                                else:
                                    st.warning("Please select at least one column.")
