"""
Privacy Audit Expander
K-anonymity risk assessment and re-identification analysis.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from .visual_profiling import visualize_k_risk_chart

API_BASE = os.getenv("API_URL", "http://localhost:8000")


"""
Privacy Audit Expander
PII Detection and K-anonymity risk assessment.
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from .visual_profiling import visualize_k_risk_chart

API_BASE = os.getenv("API_URL", "http://localhost:8000")


def _show_pii_audit_section(file_id):
    """Display PII audit section with fast and deep scan options."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        fast_scan_button = st.button(
            "🚀 Fast Scan (Regex Patterns)",
            help="Quick scan using regex patterns for common PII types. Analyzes 1000 rows.",
            use_container_width=True,
            key="privacy_fast_scan"
        )
    
    with col2:
        deep_scan_button = st.button(
            "🔍 Deep Scan (AI-Powered)",
            help="Comprehensive AI-based scan using Microsoft Presidio. More accurate but slower.",
            use_container_width=True,
            key="privacy_deep_scan"
        )
    
    # Fast scan
    if fast_scan_button:
        with st.spinner("Running fast PII scan..."):
            try:
                response = requests.post(
                    f"{API_BASE}/eda/pii-scan-fast/{file_id}",
                    params={"sample_size": 1000}
                )
                
                if response.status_code == 200:
                    pii_data = response.json()
                    st.session_state['pii_scan_results'] = pii_data
                    st.success("✅ Fast PII scan completed!")
                else:
                    st.error(f"Fast scan failed: {response.text}")
            except Exception as e:
                st.error(f"Error during fast scan: {str(e)}")
    
    # Deep scan
    if deep_scan_button:
        with st.spinner("Running deep PII scan (this may take a minute)..."):
            try:
                response = requests.post(
                    f"{API_BASE}/eda/pii-scan-deep",
                    json={
                        "file_id": file_id,
                        "columns": None,
                        "sample_size": 100
                    }
                )
                
                if response.status_code == 200:
                    pii_data = response.json()
                    st.session_state['pii_scan_results'] = pii_data
                    
                    # Check if fallback occurred
                    if pii_data.get('fallback', False):
                        st.warning(pii_data.get('message', 'Deep scan not available, used fast scan instead.'))
                    else:
                        st.success("✅ Deep PII scan completed!")
                else:
                    st.error(f"Deep scan failed: {response.text}")
            except Exception as e:
                st.error(f"Error during deep scan: {str(e)}")
    
    # Display results if available
    if 'pii_scan_results' in st.session_state:
        _display_pii_results(st.session_state['pii_scan_results'])


def _display_pii_results(pii_data):
    """Display PII scan results with summary and detailed table."""
    
    st.markdown("#### 📋 PII Scan Results")
    
    # Summary metrics
    summary = pii_data.get('summary', {})
    scan_type = pii_data.get('scan_type', 'unknown')
    
    st.info(f"**Scan Type:** {scan_type.upper()}")
    
    # KPI metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_columns = summary.get('total_columns', 0)
        st.metric("Total Columns Scanned", total_columns)
    
    with col2:
        pii_columns = summary.get('columns_with_pii', 0)
        st.metric("Columns with PII", pii_columns, 
                 delta=f"{(pii_columns/total_columns*100):.1f}%" if total_columns > 0 else "0%")
    
    with col3:
        high_risk = summary.get('high_risk_columns', 0)
        st.metric("High Risk Columns", high_risk,
                 delta="⚠️ Action Required" if high_risk > 0 else "✅ Good")
    
    # Detailed results table
    results = pii_data.get('results', {})
    detections = results.get('detections', []) if isinstance(results, dict) else []
    
    if detections:
        st.markdown("#### 🔍 Detailed PII Findings")
        
        # Convert backend detection list to DataFrame for display
        pii_table = []
        for detection in detections:
            col_name = detection.get('column', 'Unknown')
            pii_types = detection.get('pii_types', [])
            confidence = detection.get('confidence', 0)
            match_count = detection.get('match_count', 0)
            detection_methods = detection.get('detection_method', [])
            
            # Create row for each PII type found in column
            for pii_type in pii_types:
                pii_table.append({
                    'Column': col_name,
                    'Entity Type': pii_type,
                    'Confidence': f"{confidence:.2%}",
                    'Match Count': match_count,
                    'Detection Method': ', '.join(detection_methods)
                })
        
        if pii_table:
            df_pii = pd.DataFrame(pii_table)
            
            # Color code by confidence
            def highlight_risk(row):
                conf = float(row['Confidence'].rstrip('%')) / 100
                if conf > 0.8:
                    return ['background-color: #ffcccc'] * len(row)  # High risk - red
                elif conf > 0.5:
                    return ['background-color: #fff3cd'] * len(row)  # Medium risk - yellow
                else:
                    return ['background-color: #d4edda'] * len(row)  # Low risk - green
            
            styled_df = df_pii.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Download button
            csv = df_pii.to_csv(index=False)
            st.download_button(
                label="📥 Download PII Report (CSV)",
                data=csv,
                file_name=f"pii_scan_report_{pii_data.get('file_id', 'unknown')}.csv",
                mime="text/csv",
                key="download_pii_report"
            )
        else:
            st.success("✅ No PII detected in the dataset!")
    
    # Recommendations
    recommendations = summary.get('recommendations', [])
    if recommendations:
        st.markdown("#### 💡 Privacy Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.warning(f"{i}. {rec}")
    
    # Additional info
    with st.expander("ℹ️ PII Detection Details"):
        st.markdown(f"""
        **Scan Configuration:**
        - Sample Size: {pii_data.get('sample_size', 'N/A')} rows
        - Scan Type: {scan_type}
        - Columns Analyzed: {summary.get('total_columns', 0)}
        
        **Detected PII Types:**
        - EMAIL: Email addresses
        - PHONE: Phone numbers
        - SSN: Social Security Numbers
        - CREDIT_CARD: Credit card numbers
        - IP_ADDRESS: IP addresses
        - URL: Web URLs
        - DOB: Dates of birth
        - ZIP_CODE: ZIP/Postal codes
        
        **Risk Levels:**
        - 🔴 High (> 80% confidence): Immediate action recommended
        - 🟡 Medium (50-80% confidence): Review and verify
        - 🟢 Low (< 50% confidence): Low priority
        """)


def _plot_k_anonymity_distribution(distribution):
    """
    Plot distribution of k-values across records.
    
    Args:
        distribution: List of dicts with k, count, percentage
    """
    if not distribution:
        st.info("No distribution data available.")
        return
    
    df = pd.DataFrame(distribution)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["k"],
        y=df["count"],
        text=df["percentage"].apply(lambda x: f"{x:.1f}%"),
        textposition='auto',
        marker=dict(
            color=df["k"],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="k-value")
        ),
        hovertemplate=(
            "<b>k=%{x}</b><br>"
            "Records: %{y}<br>"
            "Percentage: %{text}<br>"
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title="Distribution of K-Anonymity Values",
        xaxis_title="k-value (group size)",
        yaxis_title="Number of records",
        height=400,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _plot_risk_by_k_threshold(risk_by_k):
    """
    Plot cumulative risk chart showing % of records below each k-threshold.
    Uses k=1 to 10 range with proper hover text as per spec.
    
    Args:
        risk_by_k: List of dicts with k, at_risk_count, at_risk_percentage
    """
    # Use the standardized visualize_k_risk_chart function
    visualize_k_risk_chart(
        risk_by_k,
        title="Re-identification Risk by K-Threshold"
    )


def _display_risk_summary(risk_report):
    """
    Display risk summary with KPI cards.
    
    Args:
        risk_report: Dict with summary, statistics, etc.
    """
    summary = risk_report.get("summary", {})
    stats = risk_report.get("statistics", {})
    
    # Severity badge
    severity = summary.get("severity", "UNKNOWN")
    severity_colors = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }
    severity_icon = severity_colors.get(severity, "⚪")
    
    st.markdown(f"### {severity_icon} Overall Risk: **{severity}**")
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{summary.get('total_records', 0):,}"
        )
    
    with col2:
        unique_pct = stats.get("unique_percentage", 0)
        st.metric(
            label="Unique Records (k=1)",
            value=f"{stats.get('unique_records', 0):,}",
            delta=f"{unique_pct:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Average k-anonymity",
            value=f"{stats.get('mean_k', 0):.1f}",
            delta="Higher is better" if stats.get('mean_k', 0) >= 5 else "Too low"
        )
    
    with col4:
        st.metric(
            label="Minimum k-value",
            value=stats.get("min_k", 0),
            delta="Critical" if stats.get('min_k', 0) == 1 else "OK",
            delta_color="inverse" if stats.get('min_k', 0) == 1 else "normal"
        )


def _display_recommendations(recommendations):
    """
    Display privacy recommendations.
    
    Args:
        recommendations: List of recommendation dicts
    """
    if not recommendations:
        st.success("✅ No critical privacy issues detected!")
        return
    
    st.markdown("### 💡 Recommendations")
    
    priority_icons = {
        "HIGH": "🔴",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }
    
    for rec in recommendations:
        priority = rec.get("priority", "MEDIUM")
        icon = priority_icons.get(priority, "⚪")
        
        with st.expander(f"{icon} {rec.get('issue', 'Privacy Issue')}", expanded=(priority == "HIGH")):
            st.markdown(f"**Issue:** {rec.get('issue', 'N/A')}")
            st.markdown(f"**Recommendation:** {rec.get('recommendation', 'N/A')}")
            st.markdown(f"**Priority:** {priority}")


def expander_privacy_audit():
    """
    Privacy Audit expander with PII detection and k-anonymity analysis.
    """
    st.divider()
    with st.expander("🔒 Privacy Audit & Risk Assessment", expanded=False):
        st.markdown("""
        Comprehensive privacy and security risk assessment including:
        - **PII Detection**: Identify sensitive personal information
        - **K-anonymity Analysis**: Measure re-identification risk
        
        Protect your data and ensure compliance with privacy regulations.
        """)
        
        file_id = st.session_state.get("file_id")
        if not file_id:
            st.error("No file uploaded. Please upload a dataset first.")
            return
        
        # === STEP 1: PII Detection ===
        st.markdown("### 🔒 Step 1: PII Detection")
        st.markdown("Scan for Personally Identifiable Information (PII) such as emails, phone numbers, SSNs, and credit cards.")
        
        _show_pii_audit_section(file_id)
        
        st.markdown("---")
        
        # === STEP 2: K-Anonymity Risk Assessment ===
        st.markdown("### 🎯 Step 2: Identify Quasi-Identifiers")
        st.markdown("""
        **K-anonymity** measures re-identification risk by grouping records based on **quasi-identifiers** (QIs).
        A dataset has k-anonymity if each record is indistinguishable from at least k-1 other records.
        
        **Higher k = Lower re-identification risk**
        """)
        st.markdown(
            "Quasi-identifiers (QIs) are columns that could be used to re-identify individuals "
            "(e.g., Age, Gender, ZIP Code, Job Title)."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_card = st.number_input(
                "Max unique values for QI",
                min_value=2,
                max_value=1000,
                value=100,
                step=10,
                key="qi_max_card",
                help="Columns with more unique values are less likely to be good QIs"
            )
        
        with col2:
            min_card = st.number_input(
                "Min unique values for QI",
                min_value=2,
                max_value=50,
                value=2,
                step=1,
                key="qi_min_card",
                help="Columns with too few values (e.g., binary) may not be useful QIs"
            )
        
        if st.button("🔍 Suggest QI Candidates", key="suggest_qi_btn"):
            with st.spinner("Analyzing potential quasi-identifiers..."):
                try:
                    response = requests.get(
                        f"{API_BASE}/eda/suggest-quasi-identifiers/{file_id}",
                        params={
                            "max_cardinality": max_card,
                            "min_cardinality": min_card
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['qi_suggestions'] = result
                    else:
                        st.error(f"QI suggestion failed: {response.text}")
                        return
                        
                except Exception as e:
                    st.error(f"Error calling API: {str(e)}")
                    return
        
        # Display QI suggestions
        qi_result = st.session_state.get('qi_suggestions', None)
        if qi_result:
            potential_qis = qi_result.get("potential_qis", [])
            
            if potential_qis:
                st.success(f"Found {len(potential_qis)} potential quasi-identifiers!")
                
                qi_df = pd.DataFrame(potential_qis)
                
                display_df = qi_df[[
                    "column", "unique_values", "uniqueness_ratio", "risk_level"
                ]].copy()
                
                # Color-code risk levels (before renaming columns)
                def highlight_risk(row):
                    if row["risk_level"] == "High":
                        return ['background-color: #FFE5E5'] * len(row)
                    elif row["risk_level"] == "Medium":
                        return ['background-color: #FFF9E5'] * len(row)
                    else:
                        return ['background-color: #E5F9E5'] * len(row)
                
                styled_df = display_df.style.apply(highlight_risk, axis=1)
                
                # Rename columns for display after styling
                display_df.columns = ["Column", "Unique Values", "Uniqueness Ratio", "Risk Level"]
                display_df["Uniqueness Ratio"] = display_df["Uniqueness Ratio"].apply(lambda x: f"{x:.2%}")
                
                # Reapply styling with explicit text color for contrast
                def _style_row(row):
                    rl = row.get("Risk Level", "Low")
                    if rl == "High":
                        # light red background, dark red text
                        return ['background-color: #FFE5E5; color: #721c24'] * len(row)
                    elif rl == "Medium":
                        # light yellow background, dark yellow/brown text
                        return ['background-color: #FFF9E5; color: #856404'] * len(row)
                    else:
                        # light green background, dark green text
                        return ['background-color: #E5F9E5; color: #155724'] * len(row)

                styled_df = display_df.style.apply(_style_row, axis=1)

                # Display with improved column widths for readability
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=240,
                    column_config={
                        "Column": st.column_config.TextColumn("Column", width="medium"),
                        "Unique Values": st.column_config.NumberColumn("Unique Values", width="small"),
                        "Uniqueness Ratio": st.column_config.TextColumn("Uniqueness Ratio", width="small"),
                        "Risk Level": st.column_config.TextColumn("Risk Level", width="small")
                    }
                )
                
                st.markdown("""
                **Risk Levels:**
                - 🔴 **High**: Uniqueness > 10% (many unique values, high re-identification risk)
                - 🟡 **Medium**: Uniqueness 1-10% (moderate risk)
                - 🟢 **Low**: Uniqueness < 1% (low risk)
                """)
            else:
                st.info("No potential quasi-identifiers found with current settings.")
        
        st.markdown("---")
        
        # === STEP 3: Compute K-Anonymity ===
        st.markdown("### 📊 Step 3: Compute K-Anonymity")
        st.markdown("Select quasi-identifiers to analyze re-identification risk.")
        
        # Get available columns for selection
        if qi_result and qi_result.get("potential_qis"):
            available_qis = [qi["column"] for qi in qi_result["potential_qis"]]
        else:
            st.warning("Run QI suggestion first to see candidate columns.")
            available_qis = []
        
        selected_qis = st.multiselect(
            "Select Quasi-Identifiers",
            options=available_qis,
            default=available_qis[:3] if len(available_qis) >= 3 else available_qis,
            key="selected_qis",
            help="Choose columns that could be combined to re-identify individuals"
        )
        
        k_threshold = st.slider(
            "K-threshold for vulnerability",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            key="k_threshold",
            help="Records with k below this threshold are considered vulnerable"
        )
        
        if st.button("🔐 Compute K-Anonymity", key="compute_k_btn", disabled=len(selected_qis) == 0):
            with st.spinner("Computing k-anonymity..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/eda/compute-k-anonymity",
                        json={
                            "file_id": file_id,
                            "quasi_identifiers": selected_qis,
                            "k_threshold": k_threshold
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['k_anonymity_result'] = result
                    else:
                        st.error(f"K-anonymity computation failed: {response.text}")
                        return
                        
                except Exception as e:
                    st.error(f"Error calling API: {str(e)}")
                    return
        
        # Display k-anonymity results
        k_result = st.session_state.get('k_anonymity_result', None)
        if k_result:
            risk_report = k_result.get("risk_report", {})
            
            st.markdown("---")
            
            # Display risk summary
            _display_risk_summary(risk_report)
            
            st.markdown("---")
            
            # Display risk visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### K-Value Distribution")
                _plot_k_anonymity_distribution(risk_report.get("distribution", []))
            
            with col2:
                st.markdown("#### Cumulative Risk Analysis")
                _plot_risk_by_k_threshold(risk_report.get("risk_by_k", []))
            
            st.markdown("---")
            
            # Display recommendations
            _display_recommendations(risk_report.get("recommendations", []))
            
            # Detailed statistics
            with st.expander("📈 Detailed Statistics", expanded=False):
                stats = risk_report.get("statistics", {})
                risk_by_k = risk_report.get("risk_by_k", [])
                
                st.markdown("**K-Value Statistics:**")
                stats_df = pd.DataFrame([stats]).T
                stats_df.columns = ["Value"]
                st.dataframe(stats_df, use_container_width=True)
                
                st.markdown("**Risk by K-Threshold:**")
                risk_df = pd.DataFrame(risk_by_k)
                risk_df.columns = ["K-Threshold", "At Risk (Count)", "At Risk (%)"]
                st.dataframe(risk_df, use_container_width=True)
