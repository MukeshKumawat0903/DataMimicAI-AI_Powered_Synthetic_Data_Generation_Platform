"""
Decision Report UI Component - Display Validation Results.

This module provides the Streamlit UI component for displaying validation
results in the Decision Report tab. It shows before/after metric comparisons
without making judgments or recommendations.

Author: DataMimicAI Team
Date: February 6, 2026
"""

import streamlit as st
import pandas as pd
import requests
from typing import Dict, Any, Optional
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

logger = logging.getLogger(__name__)

# Backend API configuration
BACKEND_URL = "http://localhost:8000"


def fetch_validation_results(plan_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch validation results from the backend API.
    
    Parameters
    ----------
    plan_id : str
        Plan identifier to fetch results for
    
    Returns
    -------
    Dict | None
        Validation report dictionary if successful, None otherwise
    """
    try:
        url = f"{BACKEND_URL}/api/validation/decision-report/{plan_id}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            logger.info(f"No validation results found for plan_id={plan_id}")
            return None
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch validation results: {e}")
        return None


def fetch_available_plans() -> list:
    """
    Fetch list of plan IDs with validation results.
    
    Returns
    -------
    list
        List of plan_id strings, or empty list if error
    """
    try:
        url = f"{BACKEND_URL}/api/validation/plans"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("plan_ids", [])
        else:
            logger.error(f"Failed to fetch plan list: {response.status_code}")
            return []
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch plan list: {e}")
        return []


def create_impact_gauge(validation_results: list) -> go.Figure:
    """
    Create an interactive gauge showing overall transformation impact.
    
    Parameters
    ----------
    validation_results : list
        List of metric comparison dictionaries
    
    Returns
    -------
    go.Figure
        Plotly gauge chart
    """
    # Calculate impact score based on changes
    successful_results = [r for r in validation_results if r.get("status") == "SUCCESS"]
    
    if not successful_results:
        # Create empty gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            title={'text': "Transformation Impact"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "gray"},
                   'steps': [
                       {'range': [0, 30], 'color': "lightgray"},
                       {'range': [30, 70], 'color': "gray"},
                       {'range': [70, 100], 'color': "darkgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}
        ))
        fig.update_layout(height=250)
        return fig
    
    # Calculate percentage of metrics that changed
    metrics_changed = sum(1 for r in successful_results if abs(r.get('delta', 0)) > 0.0001)
    total_metrics = len(successful_results)
    impact_score = (metrics_changed / total_metrics * 100) if total_metrics > 0 else 0
    
    # Determine color based on impact
    if impact_score < 20:
        bar_color = "#90EE90"  # Light green - minimal impact
    elif impact_score < 50:
        bar_color = "#FFD700"  # Gold - moderate impact
    else:
        bar_color = "#FF6B6B"  # Red - high impact
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=impact_score,
        title={'text': "Transformation Impact Score", 'font': {'size': 18}},
        delta={'reference': 0, 'increasing': {'color': bar_color}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#E8F5E9'},
                {'range': [20, 50], 'color': '#FFF9C4'},
                {'range': [50, 100], 'color': '#FFEBEE'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'size': 12}
    )
    
    return fig


def create_delta_distribution_chart(validation_results: list) -> go.Figure:
    """
    Create an interactive chart showing distribution of metric changes.
    
    Parameters
    ----------
    validation_results : list
        List of metric comparison dictionaries
    
    Returns
    -------
    go.Figure
        Plotly bar chart
    """
    successful_results = [r for r in validation_results if r.get("status") == "SUCCESS"]
    
    if not successful_results:
        return go.Figure()
    
    # Categorize changes
    large_decrease = sum(1 for r in successful_results if r.get('delta', 0) < -0.1)
    small_decrease = sum(1 for r in successful_results if -0.1 <= r.get('delta', 0) < -0.001)
    no_change = sum(1 for r in successful_results if abs(r.get('delta', 0)) <= 0.001)
    small_increase = sum(1 for r in successful_results if 0.001 < r.get('delta', 0) <= 0.1)
    large_increase = sum(1 for r in successful_results if r.get('delta', 0) > 0.1)
    
    categories = ['Large\nDecrease', 'Small\nDecrease', 'No\nChange', 'Small\nIncrease', 'Large\nIncrease']
    counts = [large_decrease, small_decrease, no_change, small_increase, large_increase]
    colors = ['#2ECC71', '#A9DFBF', '#95A5A6', '#F8B195', '#E74C3C']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Metrics: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Distribution of Metric Changes",
        xaxis_title="Change Type",
        yaxis_title="Number of Metrics",
        height=350,
        margin=dict(l=40, r=20, t=60, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font={'size': 12}
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    return fig


def create_before_after_comparison(validation_results: list, top_n: int = 10) -> go.Figure:
    """
    Create interactive before/after comparison for top changed metrics.
    
    Parameters
    ----------
    validation_results : list
        List of metric comparison dictionaries
    top_n : int
        Number of top changed metrics to show
    
    Returns
    -------
    go.Figure
        Plotly grouped bar chart
    """
    successful_results = [r for r in validation_results if r.get("status") == "SUCCESS"]
    
    if not successful_results:
        return go.Figure()
    
    # Sort by absolute delta and take top N
    sorted_results = sorted(successful_results, key=lambda x: abs(x.get('delta', 0)), reverse=True)[:top_n]
    
    if not sorted_results:
        return go.Figure()
    
    # Prepare data
    labels = [f"{r.get('column', 'N/A')}\n{r.get('metric', 'N/A')}" for r in sorted_results]
    before_values = [r.get('before', 0) for r in sorted_results]
    after_values = [r.get('after', 0) for r in sorted_results]
    
    fig = go.Figure(data=[
        go.Bar(
            name='Before',
            x=labels,
            y=before_values,
            marker_color='#3498DB',
            text=[f"{v:.3f}" for v in before_values],
            textposition='auto',
            hovertemplate='<b>Before</b>: %{y:.4f}<extra></extra>'
        ),
        go.Bar(
            name='After',
            x=labels,
            y=after_values,
            marker_color='#E74C3C',
            text=[f"{v:.3f}" for v in after_values],
            textposition='auto',
            hovertemplate='<b>After</b>: %{y:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f"Top {len(sorted_results)} Changed Metrics: Before vs After",
        xaxis_title="Column & Metric",
        yaxis_title="Value",
        barmode='group',
        height=400,
        margin=dict(l=40, r=20, t=60, b=100),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'size': 11}
    )
    
    fig.update_xaxes(showgrid=False, tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    return fig


def create_column_impact_heatmap(validation_results: list) -> go.Figure:
    """
    Create heatmap showing impact per column across different metrics.
    
    Parameters
    ----------
    validation_results : list
        List of metric comparison dictionaries
    
    Returns
    -------
    go.Figure
        Plotly heatmap
    """
    successful_results = [r for r in validation_results if r.get("status") == "SUCCESS"]
    
    if not successful_results:
        return go.Figure()
    
    # Create matrix of columns x metrics
    columns = sorted(list(set(r.get('column', 'N/A') for r in successful_results)))
    metrics = sorted(list(set(r.get('metric', 'N/A') for r in successful_results)))
    
    # Build matrix
    matrix = np.zeros((len(columns), len(metrics)))
    for i, col in enumerate(columns):
        for j, metric in enumerate(metrics):
            matching = [r for r in successful_results 
                       if r.get('column') == col and r.get('metric') == metric]
            if matching:
                matrix[i, j] = abs(matching[0].get('delta', 0))
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=metrics,
        y=columns,
        colorscale='RdYlGn_r',
        hovertemplate='Column: %{y}<br>Metric: %{x}<br>|Delta|: %{z:.4f}<extra></extra>',
        colorbar=dict(title="|Delta|")
    ))
    
    fig.update_layout(
        title="Column-Level Impact Heatmap",
        xaxis_title="Metrics",
        yaxis_title="Columns",
        height=max(300, len(columns) * 30),
        margin=dict(l=100, r=20, t=60, b=100),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'size': 11}
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig


def render_metric_table(validation_results: list, change_filter: str = "All Changes", column_filter: list = None) -> None:
    """
    Render validation results as a filterable table.
    
    Parameters
    ----------
    validation_results : list
        List of metric comparison dictionaries
    change_filter : str
        Filter type for changes
    column_filter : list
        List of columns to include
    """
    # Filter successful comparisons
    successful_results = [
        r for r in validation_results
        if r.get("status") == "SUCCESS"
    ]
    
    if not successful_results:
        st.warning("No successfully computed metrics to display.")
        return
    
    # Apply filters
    filtered_results = successful_results.copy()
    
    # Filter by column
    if column_filter:
        filtered_results = [r for r in filtered_results if r.get('column') in column_filter]
    
    # Filter by change type
    if change_filter == "Only Changed":
        filtered_results = [r for r in filtered_results if abs(r.get('delta', 0)) > 0.001]
    elif change_filter == "Only Unchanged":
        filtered_results = [r for r in filtered_results if abs(r.get('delta', 0)) <= 0.001]
    elif change_filter == "Increased":
        filtered_results = [r for r in filtered_results if r.get('delta', 0) > 0.001]
    elif change_filter == "Decreased":
        filtered_results = [r for r in filtered_results if r.get('delta', 0) < -0.001]
    
    if not filtered_results:
        st.info("No metrics match the selected filters.")
        return
    
    # Prepare data for table
    table_data = []
    for result in filtered_results:
        delta_val = result.get('delta', 0)
        
        # Add text indicators for delta
        if abs(delta_val) < 0.001:
            delta_indicator = "="
        elif delta_val < 0:
            delta_indicator = "--" if delta_val < -0.1 else "-"
        else:
            delta_indicator = "++" if delta_val > 0.1 else "+"
        
        table_data.append({
            "Change": delta_indicator,
            "Column": result.get("column", "N/A"),
            "Metric": result.get("metric", "N/A"),
            "Before": f"{result.get('before', 0):.4f}" if result.get('before') is not None else "N/A",
            "After": f"{result.get('after', 0):.4f}" if result.get('after') is not None else "N/A",
            "Delta": f"{result.get('delta', 0):+.4f}" if result.get('delta') is not None else "N/A"
        })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Display with color coding for delta
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def render_summary_cards(summary: Dict[str, Any]) -> None:
    """
    Render summary statistics as metric cards.
    
    Parameters
    ----------
    summary : Dict
        Summary statistics dictionary
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Metrics Compared",
            value=summary.get("metrics_compared", 0)
        )
    
    with col2:
        st.metric(
            label="Columns Affected",
            value=summary.get("columns_affected", 0)
        )
    
    with col3:
        original_shape = summary.get("original_shape", {})
        st.metric(
            label="Original Rows",
            value=original_shape.get("rows", 0)
        )
    
    with col4:
        transformed_shape = summary.get("transformed_shape", {})
        delta_rows = transformed_shape.get("rows", 0) - original_shape.get("rows", 0)
        st.metric(
            label="Transformed Rows",
            value=transformed_shape.get("rows", 0),
            delta=delta_rows if delta_rows != 0 else None
        )


def render_unavailable_metrics(validation_results: list) -> None:
    """
    Render unavailable/error metrics as an expandable section.
    
    Parameters
    ----------
    validation_results : list
        List of metric comparison dictionaries
    """
    unavailable_results = [
        r for r in validation_results
        if r.get("status") in ["UNAVAILABLE", "ERROR"]
    ]
    
    if not unavailable_results:
        return
    
    with st.expander(f"⚠️ Unavailable Metrics ({len(unavailable_results)})", expanded=False):
        for result in unavailable_results:
            status_icon = "WARNING" if result.get("status") == "UNAVAILABLE" else "ERROR"
            st.write(f"{status_icon} **{result.get('column')}.{result.get('metric')}**: {result.get('error', 'Unknown error')}")


def render_conclusion_summary(validation_results: list, summary: Dict[str, Any]) -> None:
    """
    Render a visual conclusion summary with key insights.
    
    Parameters
    ----------
    validation_results : list
        List of metric comparison dictionaries
    summary : Dict
        Summary statistics dictionary
    """
    st.markdown("##### Transformation Conclusion")
    
    successful_results = [r for r in validation_results if r.get("status") == "SUCCESS"]
    
    if not successful_results:
        st.info("No conclusion data available.")
        return
    
    # Calculate key metrics
    total_metrics = len(successful_results)
    metrics_changed = sum(1 for r in successful_results if abs(r.get('delta', 0)) > 0.001)
    metrics_unchanged = total_metrics - metrics_changed
    
    # Get most changed metrics
    top_changes = sorted(successful_results, key=lambda x: abs(x.get('delta', 0)), reverse=True)[:3]
    
    # Create conclusion box
    conclusion_col1, conclusion_col2 = st.columns([2, 1])
    
    with conclusion_col1:
        st.markdown("**Key Findings:**")
        
        # Change percentage
        change_pct = (metrics_changed / total_metrics * 100) if total_metrics > 0 else 0
        if change_pct < 10:
            impact_level = "Minimal"
            impact_color = "green"
        elif change_pct < 40:
            impact_level = "Low"
            impact_color = "blue"
        elif change_pct < 70:
            impact_level = "Moderate"
            impact_color = "orange"
        else:
            impact_level = "High"
            impact_color = "red"
        
        st.markdown(f"- **Transformation Impact:** :{impact_color}[{impact_level}] ({metrics_changed}/{total_metrics} metrics changed = {change_pct:.1f}%)")
        st.markdown(f"- **Columns Affected:** {summary.get('columns_affected', 0)} out of {summary.get('transformed_shape', {}).get('columns', 0)}")
        
        # Data integrity
        original_rows = summary.get('original_shape', {}).get('rows', 0)
        transformed_rows = summary.get('transformed_shape', {}).get('rows', 0)
        row_change = transformed_rows - original_rows
        
        if row_change == 0:
            data_integrity = "Preserved (no rows added/removed)"
        elif row_change > 0:
            data_integrity = f"Expanded (+{row_change} rows added)"
        else:
            data_integrity = f"Reduced ({row_change} rows removed)"
        
        st.markdown(f"- **Data Integrity:** {data_integrity}")
        
        # Top changes
        if top_changes:
            st.markdown("\\n**Most Significant Changes:**")
            for i, change in enumerate(top_changes, 1):
                arrow = "+" if change.get('delta', 0) > 0 else "-"
                st.markdown(
                    f"  {i}. {arrow} **{change.get('column')}.{change.get('metric')}**: "
                    f"{change.get('before', 0):.3f} -> {change.get('after', 0):.3f} "
                    f"({change.get('delta', 0):+.3f})"
                )
    
    with conclusion_col2:
        # Create a simple donut chart for change distribution
        fig = go.Figure(data=[go.Pie(
            labels=['Changed', 'Unchanged'],
            values=[metrics_changed, metrics_unchanged],
            hole=.6,
            marker_colors=['#E74C3C', '#95A5A6'],
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            showlegend=False,
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f'{change_pct:.1f}%<br>Changed', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_decision_report_tab():
    """
    Render the Decision Report tab with validation results.
    
    This function is called from the main Streamlit app to display
    validation results in the Decision Report tab.
    """
    st.markdown("### 📄 Decision Report - Transformation Impact Analysis")
    
    st.info(
        "This report shows **factual before/after metric comparisons** for executed transformation plans. "
        "No judgments or recommendations are provided - only the measured changes."
    )
    
    # Fetch available plans
    available_plans = fetch_available_plans()
    
    if not available_plans:
        st.warning(
            "⚠️ **No validation results available yet.**\n\n"
            "Validation results are generated after transformation plans are executed. "
            "Execute a transformation plan to see its impact here."
        )
        return
    
    # Plan selection
    st.markdown("#### Select Plan")
    selected_plan = st.selectbox(
        "Choose a plan to view validation results:",
        options=available_plans,
        index=0,
        help="Plans are listed in alphabetical order"
    )
    
    if not selected_plan:
        return
    
    # Fetch validation results
    with st.spinner(f"Loading validation results for {selected_plan}..."):
        validation_report = fetch_validation_results(selected_plan)
    
    if validation_report is None:
        st.error(
            f"❌ **Failed to load validation results for {selected_plan}**\n\n"
            "This could mean:\n"
            "- The backend API is not running\n"
            "- The validation results were not stored\n"
            "- Network connectivity issues"
        )
        return
    
    # Extract data
    plan_id = validation_report.get("plan_id", selected_plan)
    validation_results = validation_report.get("validation_results", [])
    summary = validation_report.get("summary", {})
    
    # Display plan ID
    st.markdown(f"#### Validation Results: `{plan_id}`")
    
    # Display summary cards
    st.markdown("##### Summary")
    render_summary_cards(summary)
    
    st.markdown("---")
    
    # ========== INTERACTIVE VISUALIZATIONS ==========
    st.markdown("##### Visual Impact Analysis")
    
    # Row 1: Impact Gauge + Delta Distribution
    vis_col1, vis_col2 = st.columns([1, 1])
    
    with vis_col1:
        try:
            impact_gauge = create_impact_gauge(validation_results)
            st.plotly_chart(impact_gauge, use_container_width=True)
            st.caption("White: No change | Green: Decrease | Red: Increase")
        except Exception as e:
            st.error(f"Failed to render impact gauge: {e}")
    
    with vis_col2:
        try:
            delta_dist = create_delta_distribution_chart(validation_results)
            st.plotly_chart(delta_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render delta distribution: {e}")
    
    st.markdown("---")
    
    # Row 2: Before/After Comparison Chart
    try:
        comparison_chart = create_before_after_comparison(validation_results, top_n=10)
        if comparison_chart.data:
            st.plotly_chart(comparison_chart, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render comparison chart: {e}")
    
    st.markdown("---")
    
    # Row 3: Column Impact Heatmap (expandable for space)
    with st.expander("Column-Level Impact Heatmap", expanded=False):
        try:
            heatmap = create_column_impact_heatmap(validation_results)
            if heatmap.data:
                st.plotly_chart(heatmap, use_container_width=True)
                st.caption("Darker colors indicate larger changes in metrics")
        except Exception as e:
            st.error(f"Failed to render heatmap: {e}")
    
    st.markdown("---")
    
    # ========== DATA TABLE ==========
    # Display metric comparison table
    st.markdown("##### Detailed Metrics Table")
    
    # Add interactive filtering controls
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
    
    with filter_col1:
        change_filter = st.selectbox(
            "Filter by change:",
            ["All Changes", "Only Changed", "Only Unchanged", "Increased", "Decreased"],
            help="Filter metrics by type of change"
        )
    
    with filter_col2:
        # Get unique columns for filtering
        all_columns = sorted(list(set(r.get('column', 'N/A') for r in validation_results if r.get('status') == 'SUCCESS')))
        column_filter = st.multiselect(
            "Filter by columns:",
            options=all_columns,
            default=all_columns,
            help="Select specific columns to display"
        )
    
    with filter_col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        # Add download button for CSV export
        successful_results = [r for r in validation_results if r.get('status') == 'SUCCESS']
        if successful_results:
            csv_data = pd.DataFrame(successful_results).to_csv(index=False)
            st.download_button(
                label="Export CSV",
                data=csv_data,
                file_name=f"{plan_id}_metrics.csv",
                mime="text/csv",
                help="Download detailed metrics as CSV"
            )
    
    st.caption("Shows the measured change for each metric. Use filters above to explore specific changes.")
    
    # Apply filtering and render table
    render_metric_table(validation_results, change_filter, column_filter)
    
    # Display unavailable metrics (if any)
    render_unavailable_metrics(validation_results)
    
    st.markdown("---")
    
    # ========== CONCLUSION SUMMARY ==========
    render_conclusion_summary(validation_results, summary)
    
    st.markdown("---")
    
    # Help text
    with st.expander("ℹ️ Understanding the Metrics", expanded=False):
        st.markdown("""
        **Column-Level Metrics** (per column):
        - **skewness**: Distribution symmetry (-∞ to +∞, 0 = symmetric)
        - **outlier_pct**: Percentage of outliers using IQR method (0-100%)
        - **missing_pct**: Percentage of missing values (0-100%)
        
        **Dataset-Level Metrics**:
        - **row_count**: Total number of rows
        - **column_count**: Total number of columns
        - **overall_missing_pct**: Overall missing percentage across entire dataset
        - **avg_correlation**: Average absolute correlation between numeric columns (0-1)
        
        **Delta Interpretation**:
        - **Negative delta**: Metric decreased (e.g., skewness reduced, outliers removed)
        - **Positive delta**: Metric increased (e.g., rows added, columns added)
        - **Zero delta**: No change in metric
        
        **Status**:
        - **SUCCESS**: Metric computed successfully
        - **UNAVAILABLE**: Metric cannot be computed (e.g., insufficient data)
        - **ERROR**: Computation failed with error
        """)
