"""
Enhanced Smart Preview with Visual Diffs and Metric Deltas
Displays before/after comparisons for accepted transformations.
"""

import streamlit as st
import requests
import pandas as pd
import os
import json
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from datetime import datetime

API_BASE = os.getenv("API_URL", "http://localhost:8000")


def calculate_kpi_summary(comparisons: List[Dict[str, Any]], total_columns: int) -> Dict[str, Any]:
    """
    Calculate KPI summary metrics from transformation comparisons.
    
    Args:
        comparisons: List of comparison dictionaries from backend
        total_columns: Total number of columns in the dataset
        
    Returns:
        Dictionary with KPI metrics:
        - pct_transformed: % of columns transformed
        - avg_skewness_change: Average change in skewness (negative is better)
        - avg_outlier_change: Average change in outlier % (negative is better)
        - risky_columns: List of columns with unresolved issues
        - improved_count: Number of columns improved
        - worsened_count: Number of columns that got worse
    """
    if not comparisons:
        return {
            "pct_transformed": 0.0,
            "avg_skewness_change": 0.0,
            "avg_outlier_change": 0.0,
            "risky_columns": [],
            "improved_count": 0,
            "worsened_count": 0,
            "total_transformed": 0
        }
    
    # Track columns (avoid duplicates for same column)
    transformed_columns = set()
    skewness_changes = []
    outlier_changes = []
    risky_columns = []
    improved = 0
    worsened = 0
    
    for comp in comparisons:
        column = comp.get("column")
        if column:
            transformed_columns.add(column)
        
        metrics = comp.get("metrics", {})
        if not metrics.get("is_numeric", False):
            continue
        
        # Get original and transformed metrics
        orig = metrics.get("original", {})
        trans = metrics.get("transformed", {})
        deltas = metrics.get("deltas", {})
        
        # Skip if we have errors
        if orig.get("error") or trans.get("error"):
            continue
        
        # Skewness change (negative is improvement)
        # Delta can be a dict with "absolute" key or a direct float
        skew_delta_data = deltas.get("skewness")
        skew_delta = None
        if skew_delta_data is not None:
            if isinstance(skew_delta_data, dict):
                skew_delta = skew_delta_data.get("absolute")
            else:
                skew_delta = skew_delta_data
        
        if skew_delta is not None:
            skewness_changes.append(skew_delta)
        
        # Outlier change (negative is improvement)
        # Delta can be a dict with "absolute" key or a direct float
        outlier_delta_data = deltas.get("outlier_percentage")
        outlier_delta = None
        if outlier_delta_data is not None:
            if isinstance(outlier_delta_data, dict):
                outlier_delta = outlier_delta_data.get("absolute")
            else:
                outlier_delta = outlier_delta_data
        
        if outlier_delta is not None:
            outlier_changes.append(outlier_delta)
        
        # Check if column improved or worsened
        is_improvement = False
        is_worse = False
        
        # Consider improved if skewness or outliers decreased
        if skew_delta is not None and skew_delta < -0.1:  # Threshold: reduced by 0.1
            is_improvement = True
        if outlier_delta is not None and outlier_delta < -1.0:  # Threshold: reduced by 1%
            is_improvement = True
        
        # Consider worse if both increased
        if skew_delta is not None and skew_delta > 0.1:
            is_worse = True
        if outlier_delta is not None and outlier_delta > 1.0:
            is_worse = True
        
        if is_improvement:
            improved += 1
        elif is_worse:
            worsened += 1
        
        # Check for risky columns (still have high skewness or outliers after transformation)
        # Ensure we safely extract numeric values
        trans_skew = trans.get("skewness")
        trans_outliers = trans.get("outlier_percentage")
        
        # Handle potential None or non-numeric values
        if trans_skew is not None and not isinstance(trans_skew, (int, float)):
            trans_skew = None
        if trans_outliers is not None and not isinstance(trans_outliers, (int, float)):
            trans_outliers = None
        
        is_risky = False
        risk_reasons = []
        
        if trans_skew is not None and abs(trans_skew) > 2.0:
            is_risky = True
            risk_reasons.append(f"High skewness: {trans_skew:.2f}")
        
        if trans_outliers is not None and trans_outliers > 10.0:
            is_risky = True
            risk_reasons.append(f"High outliers: {trans_outliers:.1f}%")
        
        if is_risky:
            risky_columns.append({
                "column": column,
                "reasons": risk_reasons,
                "skewness": trans_skew,
                "outlier_pct": trans_outliers
            })
    
    # Calculate aggregates
    pct_transformed = (len(transformed_columns) / total_columns * 100) if total_columns > 0 else 0.0
    avg_skewness_change = sum(skewness_changes) / len(skewness_changes) if skewness_changes else 0.0
    avg_outlier_change = sum(outlier_changes) / len(outlier_changes) if outlier_changes else 0.0
    
    return {
        "pct_transformed": pct_transformed,
        "avg_skewness_change": avg_skewness_change,
        "avg_outlier_change": avg_outlier_change,
        "risky_columns": risky_columns,
        "improved_count": improved,
        "worsened_count": worsened,
        "total_transformed": len(transformed_columns)
    }


def display_metric_delta(label: str, original: float, transformed: float, format_str: str = ".2f"):
    """Display a metric with before → after comparison."""
    delta = transformed - original
    delta_pct = (delta / original * 100) if original != 0 else 0
    
    # Choose color based on improvement (lower is better for skewness, outliers)
    if label.lower() in ["skewness", "outliers", "outlier_percentage"]:
        is_improvement = delta < 0
    else:
        is_improvement = delta > 0
    
    delta_color = "🟢" if is_improvement else "🔴"
    
    st.markdown(
        f"""
        <div style="padding: 0.5rem; margin: 0.25rem 0;">
            <strong>{label}:</strong> {original:{format_str}} → {transformed:{format_str}} 
            <span style="color: {'green' if is_improvement else 'red'};">
                ({delta:+{format_str}}, {delta_pct:+.1f}%) {delta_color}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_kpi_banner(kpi_summary: Dict[str, Any], total_columns: int):
    """
    Display KPI summary banner at the top of Smart Preview.
    Shows overall transformation impact with color-coded metrics.
    
    Args:
        kpi_summary: KPI metrics from calculate_kpi_summary
        total_columns: Total number of columns in dataset
    """
    st.markdown("### 📊 Transformation Impact Summary")
    
    # Main KPI metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pct = kpi_summary["pct_transformed"]
        # Color based on coverage: green >50%, yellow 25-50%, red <25%
        if pct >= 50:
            delta_color = "normal"
        elif pct >= 25:
            delta_color = "off"
        else:
            delta_color = "off"
        
        st.metric(
            label="📈 Coverage",
            value=f"{pct:.1f}%",
            delta=f"{kpi_summary['total_transformed']}/{total_columns} columns",
            help="Percentage of columns with transformations applied"
        )
    
    with col2:
        skew_change = kpi_summary["avg_skewness_change"]
        # Negative is good (reduced skewness)
        delta_display = f"{skew_change:+.2f}"
        
        st.metric(
            label="📐 Avg Skewness Δ",
            value=f"{abs(skew_change):.2f}",
            delta=delta_display + (" ✓" if skew_change < 0 else " ⚠️"),
            delta_color="inverse" if skew_change < 0 else "normal",
            help="Average change in skewness (negative = improvement)"
        )
    
    with col3:
        outlier_change = kpi_summary["avg_outlier_change"]
        # Negative is good (reduced outliers)
        delta_display = f"{outlier_change:+.1f}%"
        
        st.metric(
            label="🎯 Avg Outlier Δ",
            value=f"{abs(outlier_change):.1f}%",
            delta=delta_display + (" ✓" if outlier_change < 0 else " ⚠️"),
            delta_color="inverse" if outlier_change < 0 else "normal",
            help="Average change in outlier percentage (negative = improvement)"
        )
    
    with col4:
        improved = kpi_summary["improved_count"]
        worsened = kpi_summary["worsened_count"]
        
        # Show improvement score
        if improved > worsened:
            emoji = "🟢"
            label = "Quality Score"
            value = "Good"
            delta_val = f"+{improved - worsened}"
            delta_color = "normal"
        elif improved < worsened:
            emoji = "🔴"
            label = "Quality Score"
            value = "Needs Work"
            delta_val = f"{improved - worsened}"
            delta_color = "inverse"
        else:
            emoji = "🟡"
            label = "Quality Score"
            value = "Mixed"
            delta_val = "±0"
            delta_color = "off"
        
        st.metric(
            label=f"{emoji} {label}",
            value=value,
            delta=f"{improved} improved, {worsened} worsened",
            delta_color=delta_color,
            help="Overall transformation quality assessment"
        )
    
    # Risky columns warning (if any)
    risky_cols = kpi_summary["risky_columns"]
    if risky_cols:
        st.markdown("---")
        st.warning(f"⚠️ **{len(risky_cols)} column(s) still have issues after transformation:**")
        
        # Display risky columns in a compact format
        risky_display = []
        for risky in risky_cols[:5]:  # Show max 5
            col_name = risky["column"]
            reasons = ", ".join(risky["reasons"])
            risky_display.append(f"• **{col_name}**: {reasons}")
        
        st.markdown("\n".join(risky_display))
        
        if len(risky_cols) > 5:
            st.caption(f"...and {len(risky_cols) - 5} more. See individual comparisons below.")
    else:
        st.success("✅ All transformed columns are within acceptable ranges!")
    
    st.markdown("---")


def fetch_pet_mapping(file_id: str, accepted_decisions: List[Dict]) -> Optional[Dict]:
    """
    Fetch PET (Privacy-Enhanced Technology) mapping from backend.
    
    Args:
        file_id: Dataset identifier
        accepted_decisions: List of accepted transformation decisions
        
    Returns:
        Dictionary with pet_mapping and summary, or None on error
    """
    try:
        response = requests.post(
            f"{API_BASE}/eda/get-pet-mapping",
            params={"file_id": file_id},
            json={"decisions": accepted_decisions},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch PET mapping: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching PET mapping: {str(e)}")
        return None


def export_pet_mapping_markdown(pet_mapping: List[Dict], summary: Dict) -> str:
    """
    Export PET mapping as markdown format.
    
    Args:
        pet_mapping: List of column mapping dictionaries
        summary: Summary statistics
        
    Returns:
        Markdown formatted string
    """
    lines = []
    lines.append("# Privacy-Enhanced Technology (PET) Mapping")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary
    lines.append("## Summary")
    lines.append(f"- **Total Columns:** {summary['total_columns']}")
    lines.append(f"- **Risk Columns:** {summary['risk_columns']}")
    lines.append(f"- **Protected Columns:** {summary['protected_columns']}")
    lines.append(f"- **Unprotected Columns:** {summary['unprotected_columns']}")
    lines.append(f"- **Unprotected High-Risk:** {summary['unprotected_high_risk']}")
    lines.append(f"- **Protection Rate:** {summary['protection_rate']:.1f}%\n")
    
    # Mapping table
    lines.append("## Column Mappings\n")
    lines.append("| Column | Risk Level | PII Types | Suggested PET | Applied PET | Status |")
    lines.append("|--------|-----------|-----------|---------------|-------------|--------|")
    
    for mapping in pet_mapping:
        col = mapping["column"]
        risk = mapping["risk_level"]
        pii = ", ".join(mapping["pii_types"]) if mapping["pii_types"] else "-"
        suggested = mapping["suggested_pet"] or "-"
        applied = mapping["applied_pet"] or "-"
        status = f"{mapping['status_icon']} {mapping['status_text']}"
        
        lines.append(f"| {col} | {risk} | {pii} | {suggested} | {applied} | {status} |")
    
    return "\n".join(lines)


def export_pet_mapping_json(pet_mapping: List[Dict], summary: Dict) -> str:
    """
    Export PET mapping as JSON format.
    
    Args:
        pet_mapping: List of column mapping dictionaries
        summary: Summary statistics
        
    Returns:
        JSON formatted string
    """
    export_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "mappings": pet_mapping
    }
    return json.dumps(export_data, indent=2)


def display_pet_mapping_grid(pet_mapping: List[Dict], summary: Dict):
    """
    Display PET mapping grid with visual indicators.
    
    Args:
        pet_mapping: List of column mapping dictionaries
        summary: Summary statistics
    """
    st.markdown("### 🔒 Privacy Protection Mapping")
    st.caption("Shows privacy risks detected and protections applied per column")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Risk Columns",
            summary["risk_columns"],
            help="Columns with detected privacy risks"
        )
    
    with col2:
        protection_rate = summary["protection_rate"]
        delta_color = "normal" if protection_rate >= 80 else "inverse"
        
        # Format delta text to avoid "0 / 0" display
        if summary["risk_columns"] > 0:
            delta_text = f"{summary['protected_columns']} / {summary['risk_columns']}"
        else:
            delta_text = "No risks detected"
        
        st.metric(
            "Protection Rate",
            f"{protection_rate:.0f}%",
            delta=delta_text,
            delta_color=delta_color,
            help="Percentage of risk columns with applied protections"
        )
    
    with col3:
        unprotected = summary["unprotected_columns"]
        st.metric(
            "Unprotected",
            unprotected,
            delta="Needs attention" if unprotected > 0 else "None",
            delta_color="inverse" if unprotected > 0 else "off",
            help="Risk columns without protection"
        )
    
    with col4:
        high_risk = summary["unprotected_high_risk"]
        st.metric(
            "⚠️ High Risk",
            high_risk,
            delta="CRITICAL" if high_risk > 0 else "Safe",
            delta_color="inverse" if high_risk > 0 else "normal",
            help="Direct identifiers without protection"
        )
    
    # Warning for unprotected high-risk columns
    if summary["unprotected_high_risk"] > 0:
        st.error(
            f"🚨 **CRITICAL:** {summary['unprotected_high_risk']} Direct Identifier column(s) "
            f"are unprotected! These pose significant privacy risks."
        )
    
    # Filter options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        filter_option = st.selectbox(
            "Filter by",
            ["All Columns", "Risk Columns Only", "Unprotected Only", "Direct Identifiers"],
            help="Filter which columns to display"
        )
    
    with col2:
        show_details = st.checkbox(
            "Show PII Types & Reasons",
            value=True,
            help="Show detailed PII types and suggestion reasons"
        )
    
    # Filter mappings
    filtered_mapping = pet_mapping
    
    if filter_option == "Risk Columns Only":
        filtered_mapping = [m for m in pet_mapping if m["risk_level"] != "None"]
    elif filter_option == "Unprotected Only":
        filtered_mapping = [m for m in pet_mapping if m["risk_level"] != "None" and not m["is_protected"]]
    elif filter_option == "Direct Identifiers":
        filtered_mapping = [m for m in pet_mapping if m["risk_level"] == "Direct Identifier"]
    
    st.caption(f"Showing {len(filtered_mapping)} of {len(pet_mapping)} columns")
    
    # Build display dataframe
    display_data = []
    for mapping in filtered_mapping:
        row = {
            "Column": mapping["column"],
            "Risk Level": mapping["risk_level"],
            "Status": f"{mapping['status_icon']} {mapping['status_text']}",
            "Suggested PET": mapping["suggested_pet"] or "-",
            "Applied PET": mapping["applied_pet"] or "-"
        }
        
        if show_details:
            row["PII Types"] = ", ".join(mapping["pii_types"]) if mapping["pii_types"] else "-"
            if mapping["suggestion_reason"]:
                row["Reason"] = mapping["suggestion_reason"]
        
        display_data.append(row)
    
    # Display as dataframe
    if display_data:
        df_display = pd.DataFrame(display_data)
        
        # Color code by risk level using custom CSS
        def highlight_risk(row):
            risk = row["Risk Level"]
            if risk == "Direct Identifier":
                return ["background-color: #ffebee"] * len(row)  # Light red
            elif risk == "Quasi-Identifier":
                return ["background-color: #fff3e0"] * len(row)  # Light orange
            elif risk == "Sensitive":
                return ["background-color: #fff9c4"] * len(row)  # Light yellow
            else:
                return [""] * len(row)
        
        styled_df = df_display.style.apply(highlight_risk, axis=1)
        # Streamlit renders pandas Styler objects via st.write; st.dataframe expects a DataFrame
        # Use st.write to preserve the styling and provide an interactive table fallback
        st.write(styled_df)
        # Also provide an interactive DataFrame view below the styled table
        st.dataframe(df_display, use_container_width=True, height=400)
    else:
        st.info("No columns match the current filter")
    
    # Export options
    st.markdown("---")
    st.markdown("#### 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Generate markdown content and provide download button directly
        markdown_content = export_pet_mapping_markdown(pet_mapping, summary)
        st.download_button(
            label="📄 Export as Markdown",
            data=markdown_content,
            file_name=f"pet_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            key="pet_mapping_export_markdown"
        )
    
    with col2:
        # Generate JSON content and provide download button directly
        json_content = export_pet_mapping_json(pet_mapping, summary)
        st.download_button(
            label="📊 Export as JSON",
            data=json_content,
            file_name=f"pet_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            key="pet_mapping_export_json"
        )


def display_transformation_comparison(comparison: Dict[str, Any], expanded: bool = False):
    """Display a single transformation comparison with metrics and plots."""
    column = comparison.get("column")
    action = comparison.get("action")
    metrics = comparison.get("metrics", {})
    plot_data = comparison.get("plot", {})
    
    # Create expander for this transformation
    with st.expander(f"📊 {column} - {action}", expanded=expanded):
        if not metrics.get("is_numeric", False):
            st.info(f"ℹ️ Non-numeric column. Visual comparison not available.")
            return
        
        # Check for errors in metrics
        orig_metrics = metrics.get("original", {})
        trans_metrics = metrics.get("transformed", {})
        
        if orig_metrics.get("error") or trans_metrics.get("error"):
            st.warning(f"⚠️ {orig_metrics.get('error') or trans_metrics.get('error')}")
            return
        
        # Display metrics side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Original Metrics")
            if orig_metrics:
                st.metric("Mean", f"{orig_metrics.get('mean', 0):.2f}" if orig_metrics.get('mean') is not None else "N/A")
                st.metric("Median", f"{orig_metrics.get('median', 0):.2f}" if orig_metrics.get('median') is not None else "N/A")
                st.metric("Std Dev", f"{orig_metrics.get('std', 0):.2f}" if orig_metrics.get('std') is not None else "N/A")
                st.metric("Skewness", f"{orig_metrics.get('skewness', 0):.2f}" if orig_metrics.get('skewness') is not None else "N/A")
                st.metric("Outliers", f"{orig_metrics.get('outlier_percentage', 0):.1f}%" if orig_metrics.get('outlier_percentage') is not None else "N/A")
        
        with col2:
            st.markdown("#### 📉 Transformed Metrics")
            if trans_metrics:
                st.metric("Mean", f"{trans_metrics.get('mean', 0):.2f}" if trans_metrics.get('mean') is not None else "N/A")
                st.metric("Median", f"{trans_metrics.get('median', 0):.2f}" if trans_metrics.get('median') is not None else "N/A")
                st.metric("Std Dev", f"{trans_metrics.get('std', 0):.2f}" if trans_metrics.get('std') is not None else "N/A")
                st.metric("Skewness", f"{trans_metrics.get('skewness', 0):.2f}" if trans_metrics.get('skewness') is not None else "N/A")
                st.metric("Outliers", f"{trans_metrics.get('outlier_percentage', 0):.1f}%" if trans_metrics.get('outlier_percentage') is not None else "N/A")
        
        # Display deltas
        st.markdown("---")
        st.markdown("#### 🔄 Changes (Deltas)")
        
        deltas = metrics.get("deltas", {})
        if deltas:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "skewness" in deltas and orig_metrics.get("skewness") is not None:
                    display_metric_delta(
                        "Skewness",
                        orig_metrics.get("skewness", 0),
                        trans_metrics.get("skewness", 0)
                    )
            
            with col2:
                if "outlier_percentage" in deltas and orig_metrics.get("outlier_percentage") is not None:
                    display_metric_delta(
                        "Outliers %",
                        orig_metrics.get("outlier_percentage", 0),
                        trans_metrics.get("outlier_percentage", 0),
                        format_str=".1f"
                    )
            
            with col3:
                if "std" in deltas and orig_metrics.get("std") is not None:
                    display_metric_delta(
                        "Std Dev",
                        orig_metrics.get("std", 0),
                        trans_metrics.get("std", 0)
                    )
        
        # Display plot if available
        if plot_data and plot_data.get("is_numeric", False):
            st.markdown("---")
            st.markdown("#### 📊 Visual Comparison")
            
            # Check for plot errors
            if plot_data.get("error"):
                st.warning(f"⚠️ Plot unavailable: {plot_data.get('error')}")
            else:
                try:
                    # Load the plot from JSON
                    plot_json = plot_data.get("plot_json")
                    if plot_json:
                        # Show sampling info if applicable
                        if plot_data.get("sampled", False):
                            st.info(f"ℹ️ Large dataset: showing {plot_data.get('plotted_count', 0):,} of {plot_data.get('original_count', 0):,} points")
                        
                        fig = go.Figure(json.loads(plot_json))
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying plot: {str(e)}")


def smart_preview_with_comparisons(df, file_id: str):
    """Enhanced Smart Preview with transformation comparisons."""
    
    if not file_id or df is None:
        st.info("Please upload your dataset in the **Data Upload** tab.")
        return
    
    # Initialize dry_run mode in session state if not present
    if 'dry_run_mode' not in st.session_state:
        st.session_state.dry_run_mode = False
    
    # Initialize sampling settings in session state
    if 'preview_row_sampling' not in st.session_state:
        st.session_state.preview_row_sampling = "random"
    if 'preview_row_sample_size' not in st.session_state:
        st.session_state.preview_row_sample_size = 1000
    if 'preview_column_filter' not in st.session_state:
        st.session_state.preview_column_filter = "all"
    if 'preview_sampling_seed' not in st.session_state:
        st.session_state.preview_sampling_seed = 42
    
    # Enhanced Quick Data Overview with modern card design
    st.markdown("""
    <style>
    .overview-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .overview-card h4 {
        color: #8b5cf6;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .overview-metric {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0.5rem 0;
    }
    .overview-label {
        font-size: 0.85rem;
        color: #6b7280;
        font-weight: 500;
    }
    .type-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
        color: #2563eb;
    }
    @media (prefers-color-scheme: dark) {
        .overview-metric {
            color: #f9fafb;
        }
        .overview-label {
            color: #d1d5db;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Quick Data Overview")
    
    # Top row - Key metrics in card format
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
        <div class="overview-card">
            <h4>📐 Dataset Shape</h4>
            <div class="overview-metric">{df.shape[0]:,}</div>
            <div class="overview-label">Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="overview-card">
            <h4>📋 Columns</h4>
            <div class="overview-metric">{df.shape[1]}</div>
            <div class="overview-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        miss = df.isnull().sum().sum()
        miss_pct = (miss / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] * df.shape[1] > 0 else 0
        miss_color = "#ef4444" if miss > 0 else "#10b981"
        st.markdown(f"""
        <div class="overview-card">
            <h4>⚠️ Missing Values</h4>
            <div class="overview-metric" style="color: {miss_color};">{miss:,}</div>
            <div class="overview-label">{miss_pct:.2f}% of total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        dups = df.duplicated().sum()
        dup_pct = (dups / df.shape[0] * 100) if df.shape[0] > 0 else 0
        dup_color = "#ef4444" if dups > 0 else "#10b981"
        st.markdown(f"""
        <div class="overview-card">
            <h4>🔄 Duplicates</h4>
            <div class="overview-metric" style="color: {dup_color};">{dups:,}</div>
            <div class="overview-label">{dup_pct:.2f}% of rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second row - Detailed information
    detail_col1, detail_col2 = st.columns([1, 1])
    
    with detail_col1:
        st.markdown("#### 🏷️ Column Names")
        column_list = list(df.columns[:10])
        if len(df.columns) > 10:
            column_list.append(f"... and {len(df.columns) - 10} more")
        
        columns_html = "".join([f'<span class="type-badge">📌 {col}</span>' for col in column_list[:10]])
        if len(df.columns) > 10:
            columns_html += f'<span class="type-badge">➕ {len(df.columns) - 10} more</span>'
        
        st.markdown(f'<div style="line-height: 2.2;">{columns_html}</div>', unsafe_allow_html=True)
    
    with detail_col2:
        st.markdown("#### 🔍 Data Types Distribution")
        try:
            types_df = df.dtypes.astype(str).rename('Type').to_frame()
        except Exception:
            types_df = df.dtypes.rename('Type').to_frame()
            types_df['Type'] = types_df['Type'].map(lambda v: str(v))
        
        # Count types
        type_counts = types_df['Type'].value_counts()
        
        # Create visual type summary
        type_icons = {
            'object': '📝',
            'int64': '🔢',
            'float64': '📊',
            'bool': '✓',
            'datetime64': '📅'
        }
        
        type_summary = []
        for dtype, count in type_counts.items():
            icon = type_icons.get(dtype, '📌')
            type_summary.append(f'<span class="type-badge">{icon} {dtype}: {count}</span>')
        
        st.markdown(f'<div style="line-height: 2.2;">{"".join(type_summary)}</div>', unsafe_allow_html=True)
        st.caption("*Full type details available in EDA section*")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced Quality Assessment Section
    st.markdown("#### 🎯 Data Quality Assessment")
    
    feedback = []
    feedback_details = []
    
    # Check for issues
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    
    # Calculate quality score
    quality_score = 100
    
    if missing_count > 0:
        missing_pct = (missing_count / (df.shape[0] * df.shape[1]) * 100)
        quality_score -= min(missing_pct * 2, 30)  # Max 30 points deduction
        if missing_pct < 5:
            feedback.append(("warning", "⚠️ Minor missing values detected", f"{missing_count:,} cells ({missing_pct:.2f}%)"))
        elif missing_pct < 20:
            feedback.append(("warning", "⚠️ Moderate missing values detected", f"{missing_count:,} cells ({missing_pct:.2f}%)"))
        else:
            feedback.append(("error", "❌ Significant missing values detected", f"{missing_count:,} cells ({missing_pct:.2f}%)"))
    
    if duplicate_count > 0:
        dup_pct = (duplicate_count / df.shape[0] * 100)
        quality_score -= min(dup_pct, 20)  # Max 20 points deduction
        if dup_pct < 5:
            feedback.append(("warning", "⚠️ Few duplicate rows found", f"{duplicate_count:,} rows ({dup_pct:.2f}%)"))
        else:
            feedback.append(("error", "❌ Many duplicate rows found", f"{duplicate_count:,} rows ({dup_pct:.2f}%)"))
    
    # Check for constant columns
    try:
        constant_cols = [col for col in df.select_dtypes(include='object').columns 
                        if df[col].nunique() == 1]
        if constant_cols:
            quality_score -= len(constant_cols) * 5
            feedback.append(("info", "ℹ️ Constant columns detected", f"{len(constant_cols)} column(s) with single value"))
    except:
        pass
    
    quality_score = max(0, min(100, quality_score))
    
    # Display quality score with visual indicator
    if quality_score >= 90:
        score_color = "#10b981"
        score_emoji = "🌟"
        score_text = "Excellent"
    elif quality_score >= 70:
        score_color = "#3b82f6"
        score_emoji = "✅"
        score_text = "Good"
    elif quality_score >= 50:
        score_color = "#f59e0b"
        score_emoji = "⚠️"
        score_text = "Fair"
    else:
        score_color = "#ef4444"
        score_emoji = "❌"
        score_text = "Needs Attention"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {score_color}15 0%, {score_color}08 100%);
        border: 2px solid {score_color}40;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    ">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <div style="font-size: 3rem;">{score_emoji}</div>
            <div>
                <div style="font-size: 1.1rem; font-weight: 600; color: {score_color};">
                    Quality Score: {quality_score:.0f}/100
                </div>
                <div style="font-size: 0.9rem; opacity: 0.8;">
                    {score_text} quality dataset
                </div>
            </div>
        </div>
        <div style="
            background: {score_color}20;
            border-radius: 8px;
            height: 12px;
            overflow: hidden;
        ">
            <div style="
                background: {score_color};
                height: 100%;
                width: {quality_score}%;
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display specific feedback items
    if feedback:
        feedback_html = '<div style="margin-top: 1rem;">'
        for ftype, title, detail in feedback:
            if ftype == "error":
                bg_color = "#fef2f2"
                border_color = "#ef4444"
                text_color = "#991b1b"
            elif ftype == "warning":
                bg_color = "#fffbeb"
                border_color = "#f59e0b"
                text_color = "#92400e"
            else:
                bg_color = "#eff6ff"
                border_color = "#3b82f6"
                text_color = "#1e40af"
            
            feedback_html += f"""
            <div style="
                background: {bg_color};
                border-left: 4px solid {border_color};
                border-radius: 6px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
            ">
                <div style="font-weight: 600; color: {text_color}; margin-bottom: 0.25rem;">
                    {title}
                </div>
                <div style="font-size: 0.85rem; opacity: 0.8;">
                    {detail}
                </div>
            </div>
            """
        feedback_html += '</div>'
        st.markdown(feedback_html, unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="
                margin-top: 1rem;
                padding: 1.5rem;
                border-radius: 12px;
                border: 2px solid #10b981;
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
            ">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 2rem;">🎉</span>
                    <div>
                        <div style="font-weight: 700; font-size: 1.2rem; color: #10b981; margin-bottom: 0.25rem;">
                            Perfect Data Quality!
                        </div>
                        <div style="font-size: 0.95rem; opacity: 0.9;">
                            Your dataset is clean and ready for synthetic data generation.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    with st.expander("🔎 Show Data Sample", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # ===== PET MAPPING GRID (Always show, even without transformations) =====
    st.markdown("---")
    with st.expander("🔒 **Privacy Protection Mapping**", expanded=False):
        st.caption("View privacy risks and applied protections for each column")
        
        try:
            # Fetch PET mapping using accepted_decisions from session state
            accepted_decisions = st.session_state.get("accepted_decisions", [])
            pet_data = fetch_pet_mapping(file_id, accepted_decisions)
            
            if pet_data:
                pet_mapping = pet_data.get("pet_mapping", [])
                summary = pet_data.get("summary", {})
                
                if pet_mapping:
                    display_pet_mapping_grid(pet_mapping, summary)
                else:
                    st.info("No columns found in the dataset for privacy analysis.")
            else:
                st.warning("Unable to load privacy protection mapping. Privacy analysis may not be available for this dataset.")
        except Exception as e:
            st.error(f"Error loading PET mapping: {str(e)}")
    
    # ===== NEW: Transformation Comparisons Section =====
    if "accepted_decisions" in st.session_state and st.session_state.accepted_decisions:
        st.markdown("---")
        
        # Header with controls
        col_header, col_controls = st.columns([3, 1])
        
        with col_header:
            st.markdown("### 🔍 Transformation Impact Analysis")
            st.caption(f"Showing visual comparisons for {len(st.session_state.accepted_decisions)} accepted transformations")
        
        with col_controls:
            # Dry-run toggle
            dry_run = st.checkbox(
                "🔬 Dry Run Mode",
                value=st.session_state.dry_run_mode,
                help="Preview transformations without persisting changes",
                key="dry_run_toggle"
            )
            st.session_state.dry_run_mode = dry_run
            
            # Rollback button
            if st.button("🔄 Rollback All", type="secondary", use_container_width=True, help="Reset to original dataset state"):
                # Show confirmation dialog
                st.session_state.show_rollback_confirm = True
        
        # ===== SAMPLING CONTROLS SECTION =====
        with st.expander("⚙️ Preview Sampling Settings", expanded=False):
            st.markdown("**Configure sampling to improve performance with large datasets**")
            
            col_row, col_col = st.columns(2)
            
            with col_row:
                st.markdown("##### 📊 Row Sampling")
                
                # Row sampling method
                row_method = st.selectbox(
                    "Sampling Method",
                    options=["random", "first", "stratified"],
                    index=["random", "first", "stratified"].index(st.session_state.preview_row_sampling),
                    help="How to select rows for preview",
                    key="row_sampling_method_select"
                )
                st.session_state.preview_row_sampling = row_method
                
                # Row sample size
                max_rows = len(df)
                default_sample = min(1000, max_rows)
                
                sample_size = st.number_input(
                    "Sample Size (rows)",
                    min_value=0,
                    max_value=max_rows,
                    value=min(st.session_state.preview_row_sample_size, max_rows),
                    step=100,
                    help=f"Number of rows to use for preview (0 = all {max_rows} rows)",
                    key="row_sample_size_input"
                )
                st.session_state.preview_row_sample_size = sample_size
                
                if sample_size == 0:
                    st.info(f"📌 Using all {max_rows:,} rows")
                else:
                    pct = (sample_size / max_rows) * 100
                    st.info(f"📌 Using {sample_size:,} / {max_rows:,} rows ({pct:.1f}%)")
            
            with col_col:
                st.markdown("##### 🗂️ Column Filter")
                
                # Column filter
                col_filter = st.radio(
                    "Show Columns",
                    options=["all", "transformed_only"],
                    index=["all", "transformed_only"].index(st.session_state.preview_column_filter),
                    format_func=lambda x: "All Columns" if x == "all" else "Transformed Only",
                    help="Which columns to include in comparisons",
                    key="column_filter_radio"
                )
                st.session_state.preview_column_filter = col_filter
                
                # Random seed
                seed = st.number_input(
                    "Random Seed",
                    min_value=0,
                    max_value=9999,
                    value=st.session_state.preview_sampling_seed,
                    step=1,
                    help="Seed for reproducible random sampling",
                    key="sampling_seed_input"
                )
                st.session_state.preview_sampling_seed = seed
                
                st.markdown("---")
                
                # Method descriptions
                if row_method == "first":
                    st.caption("🔹 **First N:** Uses first N rows in order")
                elif row_method == "random":
                    st.caption("🔹 **Random:** Randomly samples N rows")
                elif row_method == "stratified":
                    st.caption("🔹 **Stratified:** Samples maintaining class distribution")
        
        # Show dry run status
        if st.session_state.dry_run_mode:
            st.info("🔬 **Dry Run Mode Active:** Transformations are previewed only. Changes are not persisted.")
        
        # Rollback confirmation dialog
        if st.session_state.get('show_rollback_confirm', False):
            st.warning("⚠️ **Confirm Rollback**")
            st.markdown("This will:")
            st.markdown("- Clear all accepted transformation decisions")
            st.markdown("- Archive transformation configs")
            st.markdown("- Reset preview to original dataset state")
            
            col_confirm, col_cancel = st.columns(2)
            
            with col_confirm:
                if st.button("✅ Confirm Rollback", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Rolling back transformations..."):
                            response = requests.post(
                                f"{API_BASE}/eda/rollback-transforms",
                                params={"file_id": file_id},
                                json={"confirmed": True}
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ {result.get('message', 'Rollback complete!')}")
                                
                                # Clear session state
                                st.session_state.accepted_decisions = []
                                st.session_state.show_rollback_confirm = False
                                st.session_state.dry_run_mode = False
                                
                                # Show archived configs
                                if result.get('archived_count', 0) > 0:
                                    st.info(f"📦 Archived {result['archived_count']} configuration file(s)")
                                
                                st.rerun()
                            else:
                                st.error(f"Rollback failed: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error during rollback: {str(e)}")
            
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_rollback_confirm = False
                    st.rerun()
        
        # Add refresh button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("📊 Visual diffs show before/after metrics and distributions for each transformation.")
        with col2:
            if st.button("🔄 Refresh Comparisons", use_container_width=True):
                st.rerun()
        
        # Fetch comparisons from backend
        try:
            with st.spinner("Loading transformation comparisons..."):
                # Prepare request body with sampling parameters
                request_body = {
                    "decisions": st.session_state.accepted_decisions,
                    "row_sampling_method": st.session_state.preview_row_sampling,
                    "row_sample_size": st.session_state.preview_row_sample_size,
                    "column_filter": st.session_state.preview_column_filter,
                    "sampling_seed": st.session_state.preview_sampling_seed
                }
                
                response = requests.post(
                    f"{API_BASE}/eda/get-transform-comparisons",
                    params={"file_id": file_id},
                    json=request_body
                )
                
                if response.status_code == 200:
                    result = response.json()
                    comparisons = result.get("comparisons", [])
                    sampling_info = result.get("sampling_info", {})
                    
                    if comparisons:
                        # Display sampling info if sampling was applied
                        if sampling_info.get("sampling_applied", False):
                            st.info(
                                f"📊 **Sampling Active:** Using {sampling_info.get('sampled_rows', 0):,} / "
                                f"{sampling_info.get('original_rows', 0):,} rows "
                                f"({sampling_info.get('sampling_method', 'unknown')} method, seed: {sampling_info.get('sampling_seed', 42)})"
                            )
                        
                        # ===== KPI SUMMARY BANNER =====
                        kpi_summary = calculate_kpi_summary(comparisons, df.shape[1])
                        display_kpi_banner(kpi_summary, df.shape[1])
                        
                        st.markdown("---")
                        
                        # Filter options
                        st.markdown("#### 🎛️ Filter Comparisons")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            show_all = st.checkbox("Expand All Comparisons", value=False)
                        
                        with col2:
                            filter_numeric = st.checkbox("Show Only Numeric Columns", value=False)
                        
                        # Display each comparison
                        filtered_comparisons = comparisons
                        if filter_numeric:
                            filtered_comparisons = [
                                c for c in comparisons 
                                if c.get("metrics", {}).get("is_numeric", False)
                            ]
                        
                        st.markdown(f"**Displaying {len(filtered_comparisons)} of {len(comparisons)} transformations**")
                        
                        for comparison in filtered_comparisons:
                            display_transformation_comparison(comparison, expanded=show_all)
                    else:
                        st.info("ℹ️ No numeric transformations to compare. Accept utility transformations in the EDA section to see comparisons here.")
                else:
                    st.error(f"Failed to load comparisons: {response.text}")
        
        except Exception as e:
            st.error(f"Error loading transformation comparisons: {str(e)}")
    else:
        st.markdown("---")
        st.info("💡 **Tip:** Accept transformation suggestions in the EDA section to see visual impact analysis here!")
    
    st.info("➡️ Switch to **Generation** to create synthetic data.")
