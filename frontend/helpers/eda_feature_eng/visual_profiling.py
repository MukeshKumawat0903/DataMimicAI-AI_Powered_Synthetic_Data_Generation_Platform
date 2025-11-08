"""
Visual Profiling Helper Functions for Enhanced EDA
Provides interactive visualizations and data filtering for Data Profiling and Correlation tabs
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import os
from typing import Dict, List, Tuple, Optional

# API base URL
API_BASE = os.getenv("API_URL", "http://localhost:8000")

# Fallback: Direct file access (for local development if API fails)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", None)
if not UPLOAD_DIR:
    # Try to resolve relative to this file
    try:
        frontend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        parent_dir = os.path.dirname(frontend_dir)
        UPLOAD_DIR = os.path.join(parent_dir, "backend", "uploads")
    except:
        UPLOAD_DIR = None


def load_data_from_file(file_id: str) -> Optional[pd.DataFrame]:
    """
    Fallback: Load data directly from file (for local development).
    
    Args:
        file_id: File identifier
    
    Returns:
        DataFrame or None if error
    """
    if not UPLOAD_DIR:
        return None
    
    try:
        path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load from file: {str(e)}")
    
    return None


def load_data_from_api(file_id: str, sample_size: int = None) -> Optional[pd.DataFrame]:
    """
    Load data from backend API endpoint (deployment-safe).
    Falls back to direct file access if API fails (local development).
    
    Args:
        file_id: File identifier
        sample_size: Optional sample size for performance
    
    Returns:
        DataFrame or None if error
    """
    try:
        params = {}
        if sample_size:
            params['sample_size'] = sample_size
        
        response = requests.get(
            f"{API_BASE}/eda/get-data/{file_id}",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            return df
        else:
            error_detail = response.text if response.text else "Unknown error"
            st.error(f"Failed to load data from API: {response.status_code}")
            with st.expander("Show error details"):
                st.code(error_detail)
            
            # Try fallback to file access
            st.info("Attempting to load data directly from file...")
            df = load_data_from_file(file_id)
            if df is not None:
                st.success("✅ Loaded data from file (fallback mode)")
                return df
            
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info(f"API URL: {API_BASE}/eda/get-data/{file_id}")
        
        # Try fallback to file access
        st.info("Attempting to load data directly from file...")
        df = load_data_from_file(file_id)
        if df is not None:
            st.success("✅ Loaded data from file (fallback mode)")
            return df
        
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def prepare_data_for_viz(df: pd.DataFrame, max_rows: int = 50000) -> pd.DataFrame:
    """
    Sample large datasets for visualization performance.
    
    Args:
        df: Input DataFrame
        max_rows: Maximum rows to use for visualization
    
    Returns:
        Sampled DataFrame if too large, otherwise original
    """
    if df is None or len(df) == 0:
        return df
    
    if len(df) > max_rows:
        st.info(
            f"⚡ **Large dataset detected** ({len(df):,} rows). "
            f"Using {max_rows:,} random sample for faster visualization. "
            f"All data is still used for backend analysis."
        )
        return df.sample(n=max_rows, random_state=42)
    
    return df


def render_kpis(df: pd.DataFrame) -> None:
    """
    Display top-level KPI summary metrics in a clean card layout.
    
    Args:
        df: Input DataFrame to analyze
    """
    st.markdown("### 📊 Dataset Overview")
    
    # Calculate metrics
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Missing value calculations
    total_cells = total_rows * total_cols
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    cols_with_missing = (df.isnull().sum() > 0).sum()
    
    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    duplicate_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
    
    # Display in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Rows",
            value=f"{total_rows:,}",
            help="Number of records in the dataset"
        )
    
    with col2:
        st.metric(
            label="Total Columns",
            value=f"{total_cols:,}",
            help="Number of features in the dataset"
        )
    
    with col3:
        st.metric(
            label="% Missing",
            value=f"{missing_pct:.2f}%",
            delta=f"{missing_cells:,} cells" if missing_cells > 0 else "No missing data",
            delta_color="inverse",
            help="Percentage of missing values across entire dataset"
        )
    
    with col4:
        st.metric(
            label="Cols with Missing",
            value=f"{cols_with_missing}",
            help="Number of columns containing missing values"
        )
    
    with col5:
        st.metric(
            label="% Duplicate Rows",
            value=f"{duplicate_pct:.2f}%",
            delta=f"{duplicate_rows:,} rows" if duplicate_rows > 0 else "No duplicates",
            delta_color="inverse",
            help="Percentage of duplicate rows in the dataset"
        )
    
    st.divider()


def filter_dataframe(df: pd.DataFrame, key_prefix: str = "filter") -> pd.DataFrame:
    """
    Create interactive sidebar filters for categorical and numeric columns.
    Returns filtered DataFrame based on user selections.
    
    Args:
        df: Input DataFrame to filter
        key_prefix: Unique prefix for widget keys to avoid conflicts
    
    Returns:
        Filtered DataFrame
    """
    st.sidebar.markdown("### 🔍 Interactive Filters")
    st.sidebar.markdown("Filter the dataset to focus on specific subsets:")
    
    filtered_df = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Add filter toggle
    enable_filters = st.sidebar.checkbox(
        "Enable Filters",
        value=False,
        key=f"{key_prefix}_enable",
        help="Toggle to activate/deactivate all filters"
    )
    
    if not enable_filters:
        st.sidebar.info("Filters are currently disabled. Check the box above to activate.")
        return filtered_df
    
    # === IMPROVED: Column Selection UI ===
    with st.sidebar.expander("⚙️ Configure Filter Columns", expanded=False):
        st.markdown("**Choose which columns to filter:**")
        
        # Categorical column selection
        default_cat = categorical_cols[:min(3, len(categorical_cols))]
        selected_cat_cols = st.multiselect(
            "Categorical columns",
            options=categorical_cols,
            default=default_cat,
            key=f"{key_prefix}_select_cat",
            help="Select which categorical columns to show filters for"
        )
        
        # Numeric column selection
        default_num = numeric_cols[:min(3, len(numeric_cols))]
        selected_num_cols = st.multiselect(
            "Numeric columns",
            options=numeric_cols,
            default=default_num,
            key=f"{key_prefix}_select_num",
            help="Select which numeric columns to show filters for"
        )
    
    # Categorical filters with smart high-cardinality handling
    if selected_cat_cols:
        st.sidebar.markdown("#### Categorical Filters")
        for col in selected_cat_cols:
            unique_values = df[col].dropna().unique()
            n_unique = len(unique_values)
            
            # === IMPROVED: Smart filtering based on cardinality ===
            if n_unique <= 50:
                # Standard multi-select for low cardinality
                selected = st.sidebar.multiselect(
                    f"{col} ({n_unique} values)",
                    options=sorted(unique_values.astype(str)),
                    default=None,
                    key=f"{key_prefix}_cat_{col}",
                    help=f"Filter by {col} values"
                )
                if selected:
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]
            
            elif n_unique <= 200:
                # High cardinality: Show top-N + "Other" option
                st.sidebar.markdown(f"**{col}** ({n_unique} unique values)")
                
                filter_mode = st.sidebar.radio(
                    f"Filter mode for {col}",
                    options=["Top values", "Search & select"],
                    key=f"{key_prefix}_mode_{col}",
                    horizontal=True
                )
                
                if filter_mode == "Top values":
                    top_n = st.sidebar.slider(
                        f"Show top N values",
                        min_value=5,
                        max_value=min(50, n_unique),
                        value=20,
                        key=f"{key_prefix}_topn_{col}"
                    )
                    value_counts = df[col].value_counts()
                    top_values = value_counts.head(top_n).index.tolist()
                    
                    selected = st.sidebar.multiselect(
                        f"Select from top {top_n}",
                        options=[str(v) for v in top_values],
                        default=None,
                        key=f"{key_prefix}_cat_{col}"
                    )
                    if selected:
                        filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]
                
                else:  # Search & select
                    search_term = st.sidebar.text_input(
                        f"Search {col} values",
                        key=f"{key_prefix}_search_{col}",
                        help="Type to search, then select from matching values"
                    )
                    
                    if search_term:
                        matching = [v for v in unique_values if search_term.lower() in str(v).lower()]
                        if matching:
                            selected = st.sidebar.multiselect(
                                f"Select from {len(matching)} matches",
                                options=[str(v) for v in matching[:100]],  # Limit to 100 for performance
                                key=f"{key_prefix}_cat_{col}"
                            )
                            if selected:
                                filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected)]
                        else:
                            st.sidebar.info(f"No matches for '{search_term}'")
            
            else:
                # Very high cardinality (>200): Text input for exact values
                st.sidebar.markdown(f"**{col}** ({n_unique} unique values - too many to list)")
                filter_text = st.sidebar.text_area(
                    f"Enter {col} values (one per line)",
                    key=f"{key_prefix}_text_{col}",
                    height=100,
                    help="Enter exact values to filter, one per line"
                )
                if filter_text.strip():
                    filter_values = [v.strip() for v in filter_text.split('\n') if v.strip()]
                    filtered_df = filtered_df[filtered_df[col].astype(str).isin(filter_values)]
    
    # Numeric filters
    if selected_num_cols:
        st.sidebar.markdown("#### Numeric Range Filters")
        for col in selected_num_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            
            if col_min != col_max:  # Only show slider if there's a range
                selected_range = st.sidebar.slider(
                    f"{col}",
                    min_value=col_min,
                    max_value=col_max,
                    value=(col_min, col_max),
                    key=f"{key_prefix}_num_{col}",
                    help=f"Filter {col} within range"
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= selected_range[0]) & 
                    (filtered_df[col] <= selected_range[1])
                ]
    
    # Show filter results
    rows_filtered = len(df) - len(filtered_df)
    if rows_filtered > 0:
        st.sidebar.success(f"✅ {rows_filtered:,} rows filtered out")
        st.sidebar.info(f"📊 {len(filtered_df):,} rows remaining")
    else:
        st.sidebar.info("No filters applied - showing all data")
    
    # Reset button
    if st.sidebar.button("🔄 Reset All Filters", key=f"{key_prefix}_reset"):
        st.rerun()
    
    return filtered_df


def plot_feature_distribution(
    df: pd.DataFrame,
    column: str,
    group_by: Optional[str] = None,
    chart_height: int = 500
) -> None:
    """
    Create interactive distribution plots for numeric and categorical features.
    
    Args:
        df: Input DataFrame
        column: Column name to visualize
        group_by: Optional column to group/color by
        chart_height: Height of the chart in pixels
    """
    if column not in df.columns:
        st.error(f"Column '{column}' not found in dataset")
        return
    
    col_data = df[column].dropna()
    
    if len(col_data) == 0:
        st.warning(f"No data available for column '{column}' (all values are missing)")
        return
    
    # Determine if numeric or categorical
    is_numeric = pd.api.types.is_numeric_dtype(df[column])
    
    if is_numeric:
        _plot_numeric_distribution(df, column, group_by, chart_height)
    else:
        _plot_categorical_distribution(df, column, group_by, chart_height)


def _plot_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    group_by: Optional[str],
    chart_height: int
) -> None:
    """Plot histogram and boxplot for numeric columns."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**📊 Histogram: {column}**")
        
        # Histogram
        if group_by and group_by in df.columns:
            fig = px.histogram(
                df,
                x=column,
                color=group_by,
                marginal="box",
                hover_data=df.columns,
                title=f"Distribution of {column} by {group_by}",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        else:
            fig = px.histogram(
                df,
                x=column,
                marginal="box",
                title=f"Distribution of {column}",
                color_discrete_sequence=['#636EFA']
            )
        
        fig.update_layout(
            height=chart_height,
            showlegend=True,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f"**📦 Boxplot: {column}**")
        
        # Boxplot with outlier detection
        if group_by and group_by in df.columns:
            fig = px.box(
                df,
                y=column,
                x=group_by,
                points="outliers",
                title=f"Boxplot of {column} by {group_by}",
                color=group_by,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        else:
            fig = px.box(
                df,
                y=column,
                points="outliers",
                title=f"Boxplot of {column}",
                color_discrete_sequence=['#636EFA']
            )
        
        fig.update_layout(
            height=chart_height,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown(f"**📈 Statistical Summary: {column}**")
    stats_df = df[column].describe().to_frame().T
    
    # Add additional statistics
    stats_df['missing'] = df[column].isnull().sum()
    stats_df['missing_pct'] = (df[column].isnull().sum() / len(df) * 100).round(2)
    stats_df['skewness'] = df[column].skew()
    stats_df['kurtosis'] = df[column].kurtosis()
    
    st.dataframe(stats_df, use_container_width=True)


def _plot_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    group_by: Optional[str],
    chart_height: int
) -> None:
    """Plot bar chart for categorical columns."""
    
    # Get value counts
    value_counts = df[column].value_counts().head(10)  # Top 10 categories
    
    st.markdown(f"**📊 Distribution: {column}** (Top 10 values)")
    
    if group_by and group_by in df.columns:
        # Grouped bar chart
        grouped_df = df.groupby([column, group_by]).size().reset_index(name='count')
        grouped_df = grouped_df[grouped_df[column].isin(value_counts.index)]
        
        fig = px.bar(
            grouped_df,
            x=column,
            y='count',
            color=group_by,
            title=f"Distribution of {column} by {group_by}",
            color_discrete_sequence=px.colors.qualitative.Set2,
            barmode='group'
        )
    else:
        # Simple bar chart
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={'x': column, 'y': 'Count'},
            title=f"Distribution of {column}",
            color_discrete_sequence=['#636EFA']
        )
    
    fig.update_layout(
        height=chart_height,
        showlegend=True,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown(f"**📈 Summary Statistics: {column}**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Unique Values", df[column].nunique())
    
    with col2:
        st.metric("Most Common", value_counts.index[0] if len(value_counts) > 0 else "N/A")
    
    with col3:
        st.metric("Most Common Count", value_counts.values[0] if len(value_counts) > 0 else 0)
    
    with col4:
        missing_pct = (df[column].isnull().sum() / len(df) * 100)
        st.metric("Missing %", f"{missing_pct:.2f}%")
    
    # Top 10 table
    if len(value_counts) > 0:
        st.markdown("**Top 10 Values:**")
        value_counts_df = value_counts.reset_index()
        value_counts_df.columns = [column, 'Count']
        value_counts_df['Percentage'] = (value_counts_df['Count'] / len(df) * 100).round(2)
        st.dataframe(value_counts_df, use_container_width=True)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    show_upper_triangle_only: bool = False,
    min_correlation: float = 0.0
) -> None:
    """
    Create an interactive correlation heatmap using Plotly.
    
    Args:
        df: Input DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')
        show_upper_triangle_only: If True, mask the lower triangle
        min_correlation: Minimum absolute correlation to display (for filtering)
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis")
        return
    
    # Compute correlation matrix
    try:
        corr_matrix = numeric_df.corr(method=method)
    except Exception as e:
        st.error(f"Error computing correlation: {str(e)}")
        return
    
    # Apply minimum correlation filter
    if min_correlation > 0:
        # Keep only correlations above threshold (in absolute value)
        mask = np.abs(corr_matrix) >= min_correlation
        # Always keep diagonal
        np.fill_diagonal(mask.values, True)
        corr_matrix = corr_matrix.where(mask)
    
    # Optional: Mask upper or lower triangle
    if show_upper_triangle_only:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix = corr_matrix.where(~mask)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(
            title=dict(
                text=f"{method.capitalize()}<br>Correlation",
                side="right"
            )
        ),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{method.capitalize()} Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        height=max(500, len(corr_matrix) * 30),  # Dynamic height based on number of features
        width=max(700, len(corr_matrix) * 30),
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show top correlated pairs
    st.markdown("**🔥 Strongest Correlations**")
    
    # Get upper triangle of correlation matrix
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    if corr_pairs:
        corr_pairs_df = pd.DataFrame(corr_pairs)
        corr_pairs_df = corr_pairs_df.dropna()
        corr_pairs_df['Abs_Correlation'] = corr_pairs_df['Correlation'].abs()
        corr_pairs_df = corr_pairs_df.sort_values('Abs_Correlation', ascending=False)
        
        # Display top 10
        top_pairs = corr_pairs_df.head(10)[['Feature 1', 'Feature 2', 'Correlation']]
        st.dataframe(top_pairs, use_container_width=True)
        
        # Warning for potential multicollinearity
        high_corr = corr_pairs_df[corr_pairs_df['Abs_Correlation'] > 0.8]
        if len(high_corr) > 0:
            st.warning(
                f"⚠️ Found {len(high_corr)} feature pair(s) with correlation > 0.8. "
                "This may indicate multicollinearity or data leakage."
            )


def create_column_selector(
    df: pd.DataFrame,
    key: str,
    label: str = "Select a feature to visualize",
    include_types: Optional[List[str]] = None
) -> str:
    """
    Create a dropdown selector for choosing columns with smart defaults.
    
    Args:
        df: Input DataFrame
        key: Unique key for the selector widget
        label: Label for the selector
        include_types: List of data types to include (e.g., ['numeric', 'categorical'])
    
    Returns:
        Selected column name
    """
    if include_types:
        if 'numeric' in include_types:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = []
        
        if 'categorical' in include_types:
            cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        else:
            cat_cols = []
        
        available_cols = list(set(numeric_cols + cat_cols))
    else:
        available_cols = df.columns.tolist()
    
    if not available_cols:
        st.error("No suitable columns found in the dataset")
        return None
    
    selected_column = st.selectbox(
        label,
        options=available_cols,
        key=key,
        help="Choose a column to view detailed distribution and statistics"
    )
    
    return selected_column


def visualize_k_risk_chart(risk_by_k: List[Dict], title: str = "K-Anonymity Risk Assessment") -> None:
    """
    Visualize k-anonymity risk with interactive chart for k=1 to 10.
    Shows percentage of records at risk for each k-threshold.
    
    Args:
        risk_by_k: List of dicts with 'k', 'at_risk_count', 'at_risk_percentage'
        title: Chart title
    
    Example hover: "At k=5, 18.3% of records are at risk."
    """
    if not risk_by_k:
        st.info("No k-anonymity data available.")
        return
    
    # Filter to k=1..10 range as per spec
    df_risk = pd.DataFrame(risk_by_k)
    df_risk = df_risk[df_risk['k'] <= 10].copy()
    
    if df_risk.empty:
        st.info("No data available for k-values 1-10.")
        return
    
    # Create interactive line chart with area fill
    fig = go.Figure()
    
    # Determine color based on risk severity
    colors = []
    for pct in df_risk['at_risk_percentage']:
        if pct > 10:
            colors.append('#FF4444')  # Red - Critical
        elif pct > 5:
            colors.append('#FFA500')  # Orange - High
        elif pct > 1:
            colors.append('#FFEB3B')  # Yellow - Medium
        else:
            colors.append('#4CAF50')  # Green - Low
    
    # Add scatter trace with markers
    fig.add_trace(go.Scatter(
        x=df_risk['k'],
        y=df_risk['at_risk_percentage'],
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(
            size=12,
            color=colors,
            line=dict(width=2, color='white')
        ),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
        hovertemplate=(
            "<b>At k=%{x}</b><br>"
            "%{y:.1f}% of records are at risk<br>"
            "<extra></extra>"
        ),
        name='Risk %'
    ))
    
    # Add reference lines for risk thresholds
    fig.add_hline(
        y=1, line_dash="dot", line_color="green",
        annotation_text="Low Risk (1%)", annotation_position="right"
    )
    fig.add_hline(
        y=5, line_dash="dot", line_color="orange",
        annotation_text="Medium Risk (5%)", annotation_position="right"
    )
    fig.add_hline(
        y=10, line_dash="dot", line_color="red",
        annotation_text="High Risk (10%)", annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sub>Percentage of records with k-anonymity below threshold</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="<b>k-anonymity threshold</b>",
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 10.5],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title="<b>% of records at risk</b>",
            ticksuffix='%',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        hovermode='closest',
        height=500,
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation guide
    with st.expander("📘 How to interpret this chart"):
        st.markdown("""
        **What does this chart show?**
        - The X-axis shows k-anonymity thresholds from 1 to 10
        - The Y-axis shows the percentage of records that fall below each threshold
        - **Example:** "At k=5, 18.3% of records are at risk" means 18.3% of your dataset has fewer than 5 similar records
        
        **Risk Levels:**
        - 🟢 **Low Risk** (< 1%): Minimal re-identification risk
        - 🟡 **Medium Risk** (1-5%): Moderate concern, consider generalization
        - 🟠 **High Risk** (5-10%): Significant risk, apply anonymization techniques
        - 🔴 **Critical Risk** (> 10%): High re-identification risk, strong anonymization needed
        
        **What is k-anonymity?**
        - k-anonymity = 5 means each record is identical to at least 4 other records based on quasi-identifiers
        - Higher k-values = Lower re-identification risk
        - Industry standard: k ≥ 5 for moderate privacy, k ≥ 10 for strong privacy
        """)
