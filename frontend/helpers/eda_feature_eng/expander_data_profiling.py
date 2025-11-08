import streamlit as st
import requests
import pandas as pd
import os
import frontend_config as config
from .visual_profiling import (
    render_kpis, 
    filter_dataframe, 
    plot_feature_distribution,
    create_column_selector,
    load_data_from_api,
    prepare_data_for_viz
)

API_BASE = os.getenv("API_URL", "http://localhost:8000")   # Will be set in Render

def _get_pii_columns():
    """Get list of columns identified as containing PII from session state."""
    pii_scan = st.session_state.get('pii_scan_results', {})
    results = pii_scan.get('results', {})
    detections = results.get('detections', []) if isinstance(results, dict) else []
    
    pii_columns = set()
    for detection in detections:
        pii_columns.add(detection.get('column'))
    
    return pii_columns

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
    
    # Get PII columns for highlighting
    pii_columns = _get_pii_columns()
    
    # Add PII indicator column
    if pii_columns:
        df_profile['PII'] = df_profile['column'].apply(
            lambda x: '🔒 PII' if x in pii_columns else ''
        )
        # Reorder to show PII column first after column name
        cols = df_profile.columns.tolist()
        if 'PII' in cols:
            cols.remove('PII')
            cols.insert(1, 'PII')
            df_profile = df_profile[cols]
    
    # Convert all list/dict columns to string for safe display
    for col in df_profile.columns:
        if df_profile[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_profile[col] = df_profile[col].apply(str)
    
    st.dataframe(df_profile, use_container_width=True)

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
    """Enhanced Streamlit expander for Data Profiling with visual summaries and interactivity."""
    st.divider()
    with st.expander("🔍 Automated Data Profiling", expanded=False):
        
        # Load the actual DataFrame for visual profiling using API
        df = load_data_from_api(st.session_state.file_id)
        
        if df is None:
            st.error("Unable to load dataset. Please upload a file first.")
            return
        
        # Apply sampling for large datasets
        df_viz = prepare_data_for_viz(df, max_rows=50000)
        
        # Apply interactive filters
        filtered_df = filter_dataframe(df_viz, key_prefix="profile")
        
        # === STEP 1: Top-Level KPI Summary ===
        render_kpis(filtered_df)
        
        # === STEP 2: Feature-Wise Visual Exploration ===
        st.markdown("### 🎨 Visual Feature Exploration")
        st.markdown("Explore individual features with interactive charts and statistics.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_column = create_column_selector(
                filtered_df,
                key="profile_feature_selector",
                label="Select a feature to visualize"
            )
        
        with col2:
            # Optional group-by selector
            all_cols = ["None"] + filtered_df.columns.tolist()
            group_by_col = st.selectbox(
                "Group by (optional)",
                options=all_cols,
                key="profile_groupby_selector",
                help="Group distributions by another categorical column"
            )
            group_by = None if group_by_col == "None" else group_by_col
        
        if selected_column:
            st.markdown("---")
            plot_feature_distribution(filtered_df, selected_column, group_by=group_by)
            st.markdown("---")
        
        # === STEP 3: Backend Profiling API (existing functionality) ===
        st.markdown("### 🔬 Automated Profiling & Quality Checks")
        st.markdown("Run automated analysis to detect data quality issues and get fix suggestions.")
        
        run_profile = st.button("Run Data Profiling & Get Suggestions")

        # Always reload if user clicks, or if no results yet
        if run_profile or 'profile_result' not in st.session_state:
            result = _call_profile_api(st.session_state.file_id)
            if result:
                st.session_state['profile_result'] = result
        
        result = st.session_state.get('profile_result', None)
        if result:
            _display_profile_summary(result)
            _handle_suggestions(result, st.session_state.file_id)
        
        # === STEP 4: Statistical Distribution Fingerprinting ===
        st.markdown("---")
        st.markdown("### 📊 Statistical Distribution Analysis")
        st.markdown("Identify the best-fit probability distributions for numeric columns to understand data generation patterns.")
        
        _show_distribution_analysis_section(st.session_state.file_id)


def _show_distribution_analysis_section(file_id):
    """Display statistical distribution fingerprinting section."""
    
    if st.button("🎲 Analyze Distributions", use_container_width=True,
                help="Fit probability distributions to numeric columns to identify data generation patterns"):
        with st.spinner("Analyzing distributions (this may take a moment)..."):
            try:
                response = requests.post(f"{API_BASE}/eda/compute-distributions/{file_id}")
                
                if response.status_code == 200:
                    dist_data = response.json()
                    st.session_state['distribution_results'] = dist_data
                    st.success("✅ Distribution analysis completed!")
                else:
                    st.error(f"Distribution analysis failed: {response.text}")
            except Exception as e:
                st.error(f"Error during distribution analysis: {str(e)}")
    
    # Display results if available
    if 'distribution_results' in st.session_state:
        _display_distribution_results(st.session_state['distribution_results'])


def _display_distribution_results(dist_data):
    """Display distribution analysis results with summary table and visualizations."""
    import plotly.graph_objects as go
    from scipy import stats
    
    st.markdown("#### 📈 Distribution Analysis Results")
    
    distributions = dist_data.get('distributions', {})
    summary = dist_data.get('summary', {})
    
    if not distributions:
        st.info("No numeric columns found for distribution analysis.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Columns Analyzed", summary.get('total_columns_analyzed', 0))
    
    with col2:
        skewed_cols = len(summary.get('highly_skewed_columns', []))
        st.metric("Highly Skewed", skewed_cols,
                 delta="⚠️ Transform recommended" if skewed_cols > 0 else "✅ Good")
    
    with col3:
        heavy_tailed = len(summary.get('heavy_tailed_columns', []))
        st.metric("Heavy-Tailed", heavy_tailed)
    
    # Distribution counts pie chart
    dist_counts = summary.get('distribution_counts', {})
    if dist_counts:
        st.markdown("#### 📊 Distribution Type Breakdown")
        
        fig = go.Figure(data=[go.Pie(
            labels=list(dist_counts.keys()),
            values=list(dist_counts.values()),
            hole=0.3,
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
        )])
        
        fig.update_layout(
            title="Distribution Types Detected",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.markdown("#### 📋 Column-Level Distribution Details")
    
    dist_table = []
    for col, meta in distributions.items():
        dist_name = meta.get('best_distribution', 'unknown')
        params = meta.get('params', {})
        score = meta.get('score')
        skewness = meta.get('skewness')
        kurtosis = meta.get('kurtosis')
        error = meta.get('error')
        
        # Format parameters - handle NaN/None values
        if params and isinstance(params, dict):
            try:
                param_str = ', '.join([
                    f"{k}={v:.3f}" if isinstance(v, (int, float)) and not pd.isna(v) else f"{k}={v}" 
                    for k, v in params.items()
                ])
            except:
                param_str = str(params)
        else:
            param_str = 'N/A'
        
        # Handle None/NaN values in all fields
        dist_table.append({
            'Column': col,
            'Best Distribution': dist_name if dist_name not in ['error', 'unknown', 'insufficient_data'] else f"⚠️ {dist_name}",
            'RSS Score': f"{score:.4f}" if score is not None and not pd.isna(score) else 'N/A',
            'Parameters': param_str[:50] + '...' if len(param_str) > 50 else param_str,  # Truncate long params
            'Skewness': f"{skewness:.3f}" if skewness is not None and not pd.isna(skewness) else 'N/A',
            'Kurtosis': f"{kurtosis:.3f}" if kurtosis is not None and not pd.isna(kurtosis) else 'N/A',
            'Samples': meta.get('n_samples', 0),
            'Status': '✅ OK' if error is None else f"⚠️ {error[:30]}"
        })
    
    if dist_table:
        df_dist = pd.DataFrame(dist_table)
        
        # Color code by distribution type with proper text contrast
        def highlight_distribution(row):
            status = row.get('Status', '✅ OK')
            if '⚠️' in status or '⚠️' in str(row.get('Best Distribution', '')):
                # Red background with dark text for errors
                return ['background-color: #ffcccc; color: #721c24'] * len(row)
            
            dist = str(row.get('Best Distribution', '')).lower()
            if 'norm' in dist and 'lognorm' not in dist:
                # Green background with dark text for normal
                return ['background-color: #d4edda; color: #155724'] * len(row)
            elif any(x in dist for x in ['lognorm', 'expon', 'gamma', 'weibull']):
                # Yellow background with dark text for skewed
                return ['background-color: #fff3cd; color: #856404'] * len(row)
            else:
                # Default (no special coloring)
                return ['color: inherit'] * len(row)
        
        styled_df = df_dist.style.apply(highlight_distribution, axis=1)
        
        # Use custom column configuration for better visibility
        st.dataframe(
            styled_df, 
            use_container_width=True,
            column_config={
                "Column": st.column_config.TextColumn("Column", width="medium"),
                "Best Distribution": st.column_config.TextColumn("Best Distribution", width="medium"),
                "RSS Score": st.column_config.TextColumn("RSS Score", width="small"),
                "Parameters": st.column_config.TextColumn("Parameters", width="large"),
                "Skewness": st.column_config.TextColumn("Skewness", width="small"),
                "Kurtosis": st.column_config.TextColumn("Kurtosis", width="small"),
                "Samples": st.column_config.NumberColumn("Samples", width="small"),
                "Status": st.column_config.TextColumn("Status", width="medium")
            },
            height=400
        )
        
        # Download button
        csv = df_dist.to_csv(index=False)
        st.download_button(
            label="📥 Download Distribution Report (CSV)",
            data=csv,
            file_name=f"distribution_analysis_{dist_data.get('file_id', 'unknown')}.csv",
            mime="text/csv"
        )
    
    # Recommendations
    recommendations = summary.get('recommendations', [])
    if recommendations:
        st.markdown("#### 💡 Transformation Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.info(f"{i}. {rec}")
    
    # Interactive visualization selector
    st.markdown("#### 🎨 Visualize Distribution Fit")
    
    # Load actual data for visualization
    df = load_data_from_api(dist_data.get('file_id'))
    
    if df is not None:
        # Column selector for visualization
        numeric_cols = [col for col in distributions.keys() if col in df.columns]
        
        if numeric_cols:
            selected_col = st.selectbox(
                "Select column to visualize",
                options=numeric_cols,
                key="dist_viz_selector"
            )
            
            if selected_col:
                _plot_distribution_fit(df, selected_col, distributions[selected_col])
    
    # Additional info
    with st.expander("ℹ️ Distribution Analysis Details"):
        st.markdown(f"""
        **Analysis Configuration:**
        - Method: RSS (Residual Sum of Squares)
        - Distributions Tested: Normal, Log-Normal, Exponential, Gamma, Beta, Uniform, Chi-Square, Weibull, Student's t
        
        **Interpretation:**
        - **Normal**: Symmetric bell curve, most common in nature
        - **Log-Normal**: Right-skewed, common for positive-only data
        - **Exponential**: Memoryless distribution, common for time-between-events
        - **Gamma**: Flexible shape, generalizes exponential
        - **Beta**: Bounded [0,1], common for proportions/probabilities
        
        **RSS Score:** Lower is better (measures fit quality)
        
        **Skewness:** 
        - Near 0: Symmetric
        - > 1: Right-skewed (consider log transform)
        - < -1: Left-skewed
        
        **Kurtosis:**
        - Near 0: Normal tails
        - > 3: Heavy tails (outliers)
        - < -3: Light tails
        """)


def _plot_distribution_fit(df, column, dist_meta):
    """Plot histogram with overlaid fitted distribution PDF."""
    import plotly.graph_objects as go
    from scipy import stats
    import numpy as np
    
    # Get column data
    data = df[column].dropna().values
    
    if len(data) == 0:
        st.warning(f"No data available for column '{column}'")
        return
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data,
        name='Observed Data',
        nbinsx=30,
        opacity=0.7,
        marker_color='#4ECDC4',
        histnorm='probability density'
    ))
    
    # Get distribution parameters
    dist_name = dist_meta.get('best_distribution', 'unknown')
    params = dist_meta.get('params', {})
    
    # Plot fitted distribution if available
    if dist_name != 'unknown' and dist_name != 'error' and params:
        try:
            # Generate x values for PDF
            x_min, x_max = data.min(), data.max()
            x_range = x_max - x_min
            x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)
            
            # Get distribution from scipy.stats
            if hasattr(stats, dist_name):
                dist = getattr(stats, dist_name)
                
                # Reconstruct parameters
                loc = params.get('loc', 0)
                scale = params.get('scale', 1)
                
                # Shape parameters (if any)
                shape_params = [v for k, v in params.items() if k.startswith('shape_')]
                
                # Calculate PDF
                if shape_params:
                    pdf = dist.pdf(x, *shape_params, loc=loc, scale=scale)
                else:
                    pdf = dist.pdf(x, loc=loc, scale=scale)
                
                # Add PDF curve
                fig.add_trace(go.Scatter(
                    x=x,
                    y=pdf,
                    name=f'Fitted {dist_name}',
                    line=dict(color='#FF6B6B', width=3),
                    mode='lines'
                ))
        except Exception as e:
            st.warning(f"Could not plot fitted distribution: {str(e)}")
    
    # Update layout
    score = dist_meta.get('score')
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) and score is not None else "N/A"
    
    fig.update_layout(
        title=f"Distribution Fit for '{column}'<br><sub>Best Fit: {dist_name} | RSS: {score_str}</sub>",
        xaxis_title=column,
        yaxis_title="Density",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


