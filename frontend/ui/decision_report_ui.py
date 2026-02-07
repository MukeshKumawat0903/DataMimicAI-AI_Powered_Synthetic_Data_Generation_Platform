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


def render_metric_table(validation_results: list) -> None:
    """
    Render validation results as a table.
    
    Parameters
    ----------
    validation_results : list
        List of metric comparison dictionaries
    """
    # Filter successful comparisons
    successful_results = [
        r for r in validation_results
        if r.get("status") == "SUCCESS"
    ]
    
    if not successful_results:
        st.warning("No successfully computed metrics to display.")
        return
    
    # Prepare data for table
    table_data = []
    for result in successful_results:
        table_data.append({
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
            status_icon = "⚠️" if result.get("status") == "UNAVAILABLE" else "❌"
            st.write(f"{status_icon} **{result.get('column')}.{result.get('metric')}**: {result.get('error', 'Unknown error')}")


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
    
    # Display metric comparison table
    st.markdown("##### Before vs After Metrics")
    st.caption("Shows the measured change for each metric. Negative deltas indicate reduction, positive indicate increase.")
    
    render_metric_table(validation_results)
    
    # Display unavailable metrics (if any)
    render_unavailable_metrics(validation_results)
    
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
