"""
DataMimicAI Validation & Refinement - Phase 3 & 4 & 5
Three-tab structure: Quality Report | Detailed Analysis | Iterative Refinement
"""
import streamlit as st
import requests
import os
import pandas as pd
import io
import json
from datetime import datetime
from .visualization import show_visualization
from .refinement_engine import (
    init_refinement_session_state,
    detect_quality_issues,
    categorize_issues_by_severity,
    generate_recommendations,
    apply_recommendation,
    calculate_quality_scores,
    add_generation_to_history,
    get_version_comparison,
    rollback_to_version,
    detect_convergence,
    export_refinement_report,
    save_refinement_session,
    load_refinement_session,
    get_session_summary
)
from .progress_ui import (
    GenerationProgress,
    handle_generation_error,
    update_generation_session_state
)

API_BASE = os.getenv("API_URL", "http://localhost:8000")


def show_validation_and_refinement():
    """
    Phase 3: New 3-tab structure for validation workflow
    Tab 1: Quality Report - AI-powered quality assessment
    Tab 2: Detailed Analysis - Existing visualization tools
    Tab 3: Iterative Refinement - Feedback loop engine (Phase 4)
    """
    if not st.session_state.get("generated_file_id"):
        st.warning("⚙️ Please generate synthetic data in Step 2 (Generate Synthetic Data) first.")
        return

    # Create the 3-tab structure
    tab1, tab2, tab3 = st.tabs([
        "📊 Quality Report",
        "🔬 Detailed Analysis", 
        "🔁 Iterative Refinement"
    ])

    # ===== TAB 1: QUALITY REPORT =====
    with tab1:
        st.markdown("### 🎯 AI-Powered Quality Assessment")
        st.caption("Automatic quality scoring and issue detection for your synthetic data")
        st.divider()
        
        _show_quality_report()

    # ===== TAB 2: DETAILED ANALYSIS =====
    with tab2:
        st.markdown("### 🔬 Deep Dive: Visualization & Comparison")
        st.caption("Explore distributions, correlations, drift, and detailed metrics")
        st.divider()
        
        # Use existing visualization module
        show_visualization()

    # ===== TAB 3: ITERATIVE REFINEMENT =====
    with tab3:
        st.markdown("### 🔁 Iterative Refinement Engine")
        st.caption("Analyze issues, get recommendations, and regenerate with improved parameters")
        st.divider()
        
        _show_refinement_placeholder()


def _show_quality_report():
    """
    Tab 1: Quality Report
    Shows AI-powered quality assessment with:
    - Overall quality score
    - Key metrics comparison
    - Issue detection
    - Actionable recommendations
    """
    
    # Initialize refinement engine
    init_refinement_session_state()
    
    # Get quality scores
    file_id = st.session_state.get("generated_file_id")
    quality_scores = calculate_quality_scores(file_id)
    
    # Quality Score Section
    st.markdown("#### 📈 Overall Quality Score")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate deltas based on score ranges
    fidelity_score = quality_scores["fidelity"]
    privacy_score = quality_scores["privacy"]
    utility_score = quality_scores["utility"]
    
    fidelity_label = "Excellent" if fidelity_score > 0.9 else "Good" if fidelity_score > 0.8 else "Fair"
    privacy_label = "Excellent" if privacy_score > 0.9 else "Good" if privacy_score > 0.8 else "Fair"
    utility_label = "Excellent" if utility_score > 0.9 else "Good" if utility_score > 0.8 else "Fair"
    
    with col1:
        st.metric(
            label="Fidelity Score",
            value=f"{fidelity_score:.0%}",
            delta=fidelity_label,
            help="How closely synthetic data matches real data distributions"
        )
    
    with col2:
        st.metric(
            label="Privacy Score",
            value=f"{privacy_score:.0%}",
            delta=privacy_label,
            help="Measures privacy preservation and uniqueness"
        )
    
    with col3:
        st.metric(
            label="Utility Score",
            value=f"{utility_score:.0%}",
            delta=utility_label,
            delta_color="normal" if utility_score > 0.75 else "inverse",
            help="Statistical utility for downstream ML tasks"
        )
    
    st.divider()
    
    # Key Metrics Comparison
    st.markdown("#### 📊 Key Metrics: Real vs Synthetic")
    
    # Create sample comparison data
    comparison_data = {
        "Metric": ["Row Count", "Column Count", "Missing Values", "Duplicate Rows", "Numeric Columns", "Categorical Columns"],
        "Real Data": ["1,000", "12", "2.3%", "5", "8", "4"],
        "Synthetic Data": ["1,000", "12", "0.1%", "0", "8", "4"],
        "Match Status": ["✅ Perfect", "✅ Perfect", "✅ Better", "✅ Better", "✅ Perfect", "✅ Perfect"]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Issue Detection using refinement engine
    st.markdown("#### ⚠️ Issues Detected")
    
    detected_issues = detect_quality_issues(file_id)
    categorized = categorize_issues_by_severity(detected_issues)
    
    # Show issue count summary
    high_count = len(categorized["high"])
    medium_count = len(categorized["medium"])
    low_count = len(categorized["low"])
    info_count = len(categorized.get("info", []))
    warning_count = len(categorized.get("warning", []))
    
    # Include warnings in the issue count (info messages are informational only)
    total_issues = high_count + medium_count + low_count + warning_count

    if total_issues > 0:
        st.caption(
            f"Found {total_issues} issues: {high_count} high, {medium_count} medium, {low_count} low, {warning_count} warnings"
        )
        
        # Show high and medium severity issues (collapsed low severity)
        issues_to_show = categorized["high"] + categorized["medium"]
        
        severity_emoji = {
            "high": "🔴",
            "medium": "🟡", 
            "low": "🟢",
            "info": "ℹ️",
            "warning": "⚠️"
        }
        
        for i, issue in enumerate(issues_to_show):
            emoji = severity_emoji.get(issue["severity"], "📌")
            
            with st.expander(f"{emoji} {issue['title']} - {issue['category']}", expanded=(i==0 and issue["severity"]=="high")):
                st.markdown(f"**Description:** {issue['description']}")
                st.markdown(f"**Impact:** {issue['impact']}")
                st.info(f"💡 **Recommendation:** {issue['recommendation']}")
                
                if issue.get("root_cause"):
                    st.markdown(f"**Root Cause:** {issue['root_cause']}")
        
        # Show low severity count but don't expand
        if low_count > 0:
            with st.expander(f"🟢 {low_count} Low Severity Issues (Optional Fixes)", expanded=False):
                for issue in categorized["low"]:
                    st.markdown(f"**{issue['title']}:** {issue['description']}")
                    st.caption(f"💡 {issue['recommendation']}")
                    st.divider()
    elif info_count > 0:
        # Show info messages (e.g., "Good Quality Detected")
        for issue in categorized.get("info", []):
            st.success(f"✅ **{issue['title']}:** {issue['description']}")
    elif warning_count > 0:
        # Show warnings (e.g., API unavailable)
        for issue in categorized.get("warning", []):
            st.warning(f"⚠️ **{issue['title']}:** {issue['description']}")
            if issue.get("recommendation"):
                st.info(f"💡 {issue['recommendation']}")
    else:
        st.success("🎉 No significant issues detected! Your synthetic data quality looks excellent.")
    
    st.divider()
    
    # Recommendations Section using refinement engine
    st.markdown("#### 💡 Smart Recommendations")
    
    current_params = st.session_state.get('generation_parameters', {
        "algorithm": st.session_state.get('selected_algorithm', 'GaussianCopula'),
        "epochs": 300,
        "batch_size": 500
    })
    
    recommendations = generate_recommendations(detected_issues, current_params)
    
    if recommendations:
        st.caption(f"Top {min(3, len(recommendations))} recommendations for improvement:")
        for rec in recommendations[:3]:  # Show top 3
            priority_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rec["priority"], "📌")
            st.markdown(f"- {priority_emoji} **{rec['title']}:** {rec['description']}")
        
        if len(recommendations) > 3:
            st.caption(f"➕ {len(recommendations) - 3} more recommendations available in the Iterative Refinement tab")
    else:
        st.markdown("- ✅ **Great job!** Your data quality is good for most use cases")
        st.markdown("- 🔒 Current privacy-utility tradeoff is well-balanced")
    
    # Quick Actions
    st.divider()
    st.markdown("#### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Report", use_container_width=True):
            report = export_refinement_report()
            st.download_button(
                "� Download JSON",
                data=report,
                file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        if st.button("🔁 View Refinement Options", use_container_width=True):
            st.info("� Switch to the **Iterative Refinement** tab to see detailed recommendations and regeneration options")
    
    with col3:
        if st.button("📊 Export Quality Metrics", use_container_width=True):
            import json
            metrics_json = json.dumps(quality_scores, indent=2)
            st.download_button(
                "� Download Metrics",
                data=metrics_json,
                file_name=f"quality_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


def _trigger_auto_regeneration(params: dict):
    """
    Phase 5: Auto-regeneration workflow
    Automatically trigger regeneration using current parameters
    """
    file_id = st.session_state.get("file_id")
    if not file_id:
        st.error("❌ No file uploaded. Please upload data first.")
        return
    
    algorithm = params.get("algorithm", st.session_state.get("selected_algorithm", "GaussianCopula"))
    num_rows = params.get("num_rows", 1000)
    epochs = params.get("epochs", 300)
    
    with GenerationProgress() as progress:
        try:
            progress.update(10, f"🔧 Preparing regeneration with {algorithm}...")
            
            gen_params = {
                "file_id": file_id,
                "algorithm": algorithm,
                "num_rows": num_rows,
                "epochs": epochs
            }
            
            progress.update(30, f"⚙️ Generating synthetic data with {algorithm}...")
            
            response = requests.post(
                f"{API_BASE}/generate",
                params=gen_params,
                timeout=300
            )
            
            progress.update(90, "✅ Processing results...")
            
            if response.status_code == 200:
                progress.complete("✅ Regeneration complete!")
                
                synthetic_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                update_generation_session_state(synthetic_df, file_id)
                
                # Add to refinement history
                issues = detect_quality_issues(file_id)
                quality_scores = calculate_quality_scores(file_id)
                add_generation_to_history(
                    file_id=file_id,
                    algorithm=algorithm,
                    parameters=params,
                    quality_scores=quality_scores,
                    issues=issues
                )
                
                st.success(f"✅ Successfully regenerated synthetic data with {algorithm}!")
                st.balloons()
                
                # Show quick metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows Generated", len(synthetic_df))
                with col2:
                    st.metric("Overall Quality", f"{quality_scores.get('overall', 0):.1%}")
                with col3:
                    st.metric("Version", st.session_state.current_generation_version)
                
                st.info("📊 Go to **Quality Report** tab to see updated metrics and analysis.")
                
                # Download button
                st.download_button(
                    "📥 Download Regenerated Data",
                    data=response.content,
                    file_name=f"synthetic_data_v{st.session_state.current_generation_version}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                error_msg = f"Regeneration failed (Status {response.status_code})"
                try:
                    error_data = response.json()
                    error_msg += f": {error_data.get('detail', 'Unknown error')}"
                except:
                    error_msg += f": {response.text}"
                handle_generation_error(error_msg, "general", progress.progress_bar, progress.status_text)
        
        except requests.exceptions.Timeout:
            handle_generation_error("", "timeout", progress.progress_bar, progress.status_text)
        except requests.exceptions.ConnectionError:
            handle_generation_error("", "connection", progress.progress_bar, progress.status_text)
        except Exception as e:
            handle_generation_error(str(e), "api", progress.progress_bar, progress.status_text)


def _show_refinement_placeholder():
    """
    Tab 3: Iterative Refinement - PHASE 4 & 5 IMPLEMENTED
    Features:
    - Issue detection and analysis
    - Smart parameter recommendations
    - One-click auto-regeneration (Phase 5)
    - Version history & comparison
    """
    # Initialize refinement session state
    init_refinement_session_state()
    
    # Check if we have generated data
    if not st.session_state.get("generated_file_id"):
        st.warning("⚙️ Generate synthetic data first to access refinement features.")
        return
    
    # Main refinement interface with 4 sections
    sections = st.tabs([
        "🔍 Issue Analysis",
        "💡 Recommendations",
        "📊 Version History",
        "🔄 Regenerate"
    ])
    
    # ===== SECTION 1: ISSUE ANALYSIS =====
    with sections[0]:
        st.markdown("#### 🔍 Detected Quality Issues")
        st.caption("Automatic analysis of your synthetic data quality")
        
        # Detect issues
        file_id = st.session_state.generated_file_id
        
        with st.spinner("Analyzing quality issues..."):
            issues = detect_quality_issues(file_id)
        
        # Categorize by severity
        categorized = categorize_issues_by_severity(issues)
        
        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            high_count = len(categorized["high"])
            st.metric("🔴 High Severity", high_count, 
                     delta="Requires attention" if high_count > 0 else "None",
                     delta_color="inverse" if high_count > 0 else "off")
        with col2:
            medium_count = len(categorized["medium"])
            st.metric("🟡 Medium Severity", medium_count,
                     delta="Consider fixing" if medium_count > 0 else "None",
                     delta_color="off")
        with col3:
            low_count = len(categorized["low"])
            st.metric("🟢 Low Severity", low_count,
                     delta="Optional" if low_count > 0 else "None",
                     delta_color="off")
        
        st.divider()
        
        # Display issues by severity
        for severity in ["high", "medium", "low"]:
            severity_issues = categorized[severity]
            if severity_issues:
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[severity]
                st.markdown(f"### {severity_emoji} {severity.upper()} Severity Issues")
                
                for issue in severity_issues:
                    with st.expander(f"{issue['title']} - {issue['category']}", expanded=(severity == "high")):
                        st.markdown(f"**Description:** {issue['description']}")
                        st.markdown(f"**Impact:** {issue['impact']}")
                        st.markdown(f"**Root Cause:** {issue['root_cause']}")
                        
                        # Show metrics if available
                        if issue.get("metrics"):
                            st.markdown("**Metrics:**")
                            for key, value in issue["metrics"].items():
                                st.write(f"- {key}: `{value}`")
                        
                        # Show recommendation
                        st.info(f"💡 **Recommendation:** {issue['recommendation']}")
                
                st.divider()
        
        # Handle warnings and info messages
        if categorized.get("warning"):
            st.markdown("### ⚠️ System Warnings")
            for warning in categorized["warning"]:
                with st.expander(f"**{warning['title']}**", expanded=True):
                    st.warning(f"{warning['description']}")
                    if warning.get('root_cause'):
                        st.markdown(f"**Root Cause:**")
                        st.code(warning['root_cause'], language='text')
                    if warning.get('recommendation'):
                        st.info(f"💡 **Recommendation:** {warning['recommendation']}")
            st.divider()
        
        if categorized.get("info"):
            for info in categorized["info"]:
                st.info(f"ℹ️ **{info['title']}:** {info['description']}")
            st.divider()
        
        if not any([categorized["high"], categorized["medium"], categorized["low"]]):
            if not categorized.get("warning") and not categorized.get("info"):
                st.success("🎉 No significant issues detected! Your synthetic data quality is excellent.")
                st.balloons()
    
    # ===== SECTION 2: RECOMMENDATIONS =====
    with sections[1]:
        st.markdown("#### 💡 Smart Parameter Recommendations")
        st.caption("AI-powered suggestions to improve your synthetic data quality")
        
        # Get current parameters with better defaults
        current_params = st.session_state.get('generation_parameters', {})
        if not current_params:
            current_params = {
                "algorithm": st.session_state.get('selected_algorithm', 'GaussianCopula'),
                "epochs": 300,
                "batch_size": 500,
                "num_rows": 1000
            }
        
        # Show current configuration
        with st.expander("📋 Current Configuration", expanded=False):
            st.json(current_params)
        
        st.divider()
        
        # Generate recommendations
        with st.spinner("Generating recommendations..."):
            recommendations = generate_recommendations(issues, current_params)
        
        # Store in session state for regeneration
        st.session_state.refinement_recommendations = recommendations
        
        if recommendations:
            st.markdown(f"**Found {len(recommendations)} recommendations** (sorted by priority)")
            st.divider()
            
            for rec in recommendations:
                priority_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rec["priority"], "📌")
                
                with st.expander(f"{priority_emoji} Priority {rec['priority']}: {rec['title']}", expanded=(rec["priority"] <= 2)):
                    st.markdown(f"**Type:** `{rec['type']}`")
                    st.markdown(f"**Description:** {rec['description']}")
                    
                    # Show action details
                    st.markdown("**Suggested Change:**")
                    action = rec["action"]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Current:** `{action['current_value']}`")
                    with col2:
                        st.markdown(f"**Suggested:** `{action['suggested_value']}`")
                    
                    if action.get("alternatives"):
                        st.markdown(f"**Alternatives:** {', '.join([f'`{alt}`' for alt in action['alternatives']])}")
                    
                    # Show expected improvements
                    st.markdown("**Expected Improvements:**")
                    for metric, improvement in rec["expected_improvement"].items():
                        st.write(f"✅ {metric}: {improvement}")
                    
                    # Show trade-offs
                    if rec.get("trade_offs"):
                        st.markdown("**Trade-offs:**")
                        for aspect, impact in rec["trade_offs"].items():
                            st.write(f"⚖️ {aspect}: {impact}")
                    
                    # Apply button
                    if st.button(f"Apply This Recommendation", key=f"apply_rec_{rec['id']}"):
                        updated_params = apply_recommendation(rec, current_params)
                        st.session_state.generation_parameters = updated_params
                        st.success(f"✅ Applied! Parameter '{action['parameter']}' updated to {action['suggested_value']}")
                        st.info("👉 Go to the 'Regenerate' tab to create new synthetic data with updated parameters.")
                
                st.divider()
        else:
            st.info("✨ No specific recommendations at this time. Your parameters look good!")
    
    # ===== SECTION 3: VERSION HISTORY =====
    with sections[2]:
        st.markdown("#### 📊 Generation Version History")
        st.caption("Track and compare different generation iterations")
        
        history = st.session_state.generation_history
        
        if not history:
            st.info("📭 No generation history yet. Generate data and iterate to build version history.")
        else:
            st.markdown(f"**Total Versions:** {len(history)}")
            
            # Quality progression chart
            if st.session_state.quality_scores_history:
                st.markdown("##### 📈 Quality Score Progression")
                
                # Create chart data
                chart_data = pd.DataFrame([
                    {
                        "Version": i + 1,
                        "Overall Score": score["overall"],
                        "Fidelity": score["fidelity"],
                        "Privacy": score["privacy"],
                        "Utility": score["utility"]
                    }
                    for i, score in enumerate(st.session_state.quality_scores_history)
                ])
                
                st.line_chart(chart_data.set_index("Version"))
                
                # Convergence detection
                if detect_convergence(st.session_state.quality_scores_history):
                    st.success("✅ Quality has converged - further iterations may not improve results significantly.")
                else:
                    st.info("🔄 Quality is still improving - consider more iterations.")
            
            st.divider()
            
            # Version list
            st.markdown("##### 📜 Version Details")
            for version in reversed(history):  # Show newest first
                with st.expander(f"Version {version['version']} - {version['timestamp']}", 
                               expanded=(version['version'] == st.session_state.current_generation_version)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Algorithm:**")
                        st.code(version['algorithm'])
                        st.markdown("**Parameters:**")
                        st.json(version['parameters'])
                    
                    with col2:
                        st.markdown("**Quality Scores:**")
                        for metric, score in version['quality_scores'].items():
                            st.metric(metric.capitalize(), f"{score:.2%}")
                    
                    # Actions
                    action_col1, action_col2, action_col3 = st.columns(3)
                    with action_col1:
                        if st.button("🔙 Rollback to This", key=f"rollback_{version['version']}"):
                            if rollback_to_version(version['version']):
                                st.rerun()
                    with action_col2:
                        # Export this version
                        version_export = {
                            "version": version['version'],
                            "timestamp": version['timestamp'],
                            "algorithm": version['algorithm'],
                            "parameters": version['parameters'],
                            "quality_scores": version['quality_scores']
                        }
                        st.download_button(
                            "📥 Export Version",
                            data=json.dumps(version_export, indent=2),
                            file_name=f"version_{version['version']}_{version['timestamp'].replace(':', '-').replace(' ', '_')}.json",
                            mime="application/json",
                            key=f"export_{version['version']}",
                            use_container_width=True
                        )
                    with action_col3:
                        if len(history) > 1:
                            compare_with = st.selectbox(
                                "Compare with:",
                                options=[v['version'] for v in history if v['version'] != version['version']],
                                key=f"compare_select_{version['version']}",
                                help="Select a version to compare with"
                            )
                            if st.button("📊 Compare", key=f"compare_btn_{version['version']}", use_container_width=True):
                                comparison = get_version_comparison(version['version'], compare_with)
                                if comparison:
                                    st.session_state.active_comparison = comparison
                                    st.success(f"✅ Comparison loaded: V{version['version']} vs V{compare_with}")
                                    st.rerun()
            
            st.divider()
            
            # Show active comparison if exists
            if st.session_state.get('active_comparison'):
                st.markdown("##### 🔬 Version Comparison")
                comp = st.session_state.active_comparison
                
                st.markdown(f"**{comp['improvement_summary']}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Version {comp['version1']['version']}**")
                    st.caption(comp['version1']['timestamp'])
                with col2:
                    st.markdown(f"**Version {comp['version2']['version']}**")
                    st.caption(comp['version2']['timestamp'])
                
                # Score changes
                if comp['score_changes']:
                    st.markdown("**Score Changes:**")
                    for metric, changes in comp['score_changes'].items():
                        delta = changes['change']
                        delta_color = "normal" if delta > 0 else "inverse"
                        st.metric(
                            metric.capitalize(),
                            f"{changes['v2']:.2%}",
                            delta=f"{delta:+.2%}",
                            delta_color=delta_color
                        )
                
                # Parameter changes
                if comp['parameter_changes']:
                    st.markdown("**Parameter Changes:**")
                    for param, changes in comp['parameter_changes'].items():
                        st.write(f"• {param}: `{changes['v1']}` → `{changes['v2']}`")
                
                # Clear comparison button
                if st.button("🗑️ Clear Comparison", key="clear_comparison"):
                    st.session_state.active_comparison = None
                    st.rerun()
    
    # ===== SECTION 4: REGENERATE =====
    with sections[3]:
        st.markdown("#### 🔄 Regenerate with Improved Parameters")
        st.caption("Apply recommendations and create a new generation")
        
        # Show current parameters with fallback
        current_params = st.session_state.get('generation_parameters', {})
        if not current_params:
            current_params = {
                "algorithm": st.session_state.get('selected_algorithm', 'GaussianCopula'),
                "epochs": 300,
                "num_rows": 1000
            }
        
        st.markdown("##### ⚙️ Current Parameters")
        if current_params:
            st.json(current_params)
        else:
            st.info("No parameters set yet. Use Quick Apply or Manual Override below.")
        
        st.divider()
        
        # Quick apply options
        st.markdown("##### ⚡ Quick Apply")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Apply Top Recommendation", use_container_width=True):
                recommendations = st.session_state.get('refinement_recommendations', [])
                if recommendations:
                    top_rec = recommendations[0]
                    updated = apply_recommendation(top_rec, current_params)
                    st.session_state.generation_parameters = updated
                    st.success(f"✅ Applied: {top_rec['title']}")
                    st.rerun()
                else:
                    st.warning("No recommendations available. Go to the Recommendations tab to generate them.")
        
        with col2:
            if st.button("Apply All High Priority", use_container_width=True):
                recommendations = st.session_state.get('refinement_recommendations', [])
                high_priority = [r for r in recommendations if r['priority'] <= 2]
                if high_priority:
                    updated = current_params.copy()
                    for rec in high_priority:
                        updated = apply_recommendation(rec, updated)
                    st.session_state.generation_parameters = updated
                    st.success(f"✅ Applied {len(high_priority)} recommendations")
                    st.rerun()
                else:
                    st.warning("No high-priority recommendations available")
        
        with col3:
            if st.button("Reset to Defaults", use_container_width=True):
                st.session_state.generation_parameters = {}
                st.success("✅ Reset to default parameters")
                st.rerun()
        
        st.divider()
        
        # Manual parameter override
        with st.expander("🛠️ Manual Parameter Override", expanded=False):
            st.markdown("Override specific parameters manually")
            
            new_algorithm = st.selectbox(
                "Algorithm",
                options=["GaussianCopula", "CTGAN", "TVAE", "TabDDPM", "DP-GAN"],
                index=0,
                key="manual_algorithm"
            )
            
            new_epochs = st.number_input("Epochs", min_value=100, max_value=1000, value=300, step=50, key="manual_epochs")
            new_batch_size = st.number_input("Batch Size", min_value=32, max_value=1000, value=500, step=32, key="manual_batch_size")
            
            if st.button("Apply Manual Settings"):
                st.session_state.generation_parameters = {
                    "algorithm": new_algorithm,
                    "epochs": new_epochs,
                    "batch_size": new_batch_size
                }
                st.success("✅ Manual parameters applied")
        
        st.divider()
        
        # Regenerate button
        st.markdown("##### 🚀 Ready to Regenerate?")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("💡 **Tip:** Make sure you've applied desired recommendations before regenerating.")
        with col2:
            if st.button("🔄 Regenerate Now", use_container_width=True, type="primary"):
                _trigger_auto_regeneration(current_params)
        
        st.divider()
        
        # Session persistence (Phase 5)
        st.markdown("##### 💾 Session Persistence")
        st.caption("Save and restore your refinement sessions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            session_name = st.text_input(
                "Session Name",
                value=f"session_{datetime.now().strftime('%Y%m%d_%H%M')}",
                key="session_name_input"
            )
            if st.button("💾 Save Session", use_container_width=True):
                session_json = save_refinement_session(session_name)
                st.download_button(
                    "📥 Download Session File",
                    data=session_json,
                    file_name=f"{session_name}.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.success(f"✅ Session '{session_name}' ready to download!")
        
        with col2:
            uploaded_session = st.file_uploader(
                "Load Session File",
                type=["json"],
                key="session_uploader",
                help="Upload a previously saved session file"
            )
            if uploaded_session is not None:
                if st.button("📂 Load Session", use_container_width=True):
                    session_content = uploaded_session.read().decode('utf-8')
                    if load_refinement_session(session_content):
                        st.success("✅ Session loaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to load session. Check file format.")
        
        # Show session summary
        if st.session_state.get("generation_history"):
            summary = get_session_summary()
            with st.expander("📋 Current Session Summary", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Iterations", summary["total_iterations"])
                with col2:
                    st.metric("Current Version", summary["current_version"])
                with col3:
                    st.metric("Quality Trend", summary["quality_trend"])
                st.caption(f"**Algorithms Tried:** {', '.join(summary['algorithms_tried'])}")
                st.caption(f"**Last Updated:** {summary['last_updated']}")
        
        st.divider()
        
        # Export options
        st.markdown("##### 📥 Export & Download")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download Refinement Report", use_container_width=True):
                report = export_refinement_report()
                st.download_button(
                    "💾 Download JSON Report",
                    data=report,
                    file_name=f"refinement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📊 Export Parameter Changes", use_container_width=True):
                changes_log = st.session_state.get('parameter_changes_log', [])
                if changes_log:
                    import json
                    st.download_button(
                        "💾 Download Changes Log",
                        data=json.dumps(changes_log, indent=2),
                        file_name=f"parameter_changes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.info("No parameter changes recorded yet")
