"""
Context-Aware Feature Suggestion Engine Frontend
Displays utility suggestions, privacy suggestions, and conflict resolution UI.
"""

import streamlit as st
import requests
import pandas as pd
import os
from typing import Dict, List, Any, Optional

API_BASE = os.getenv("API_URL", "http://localhost:8000")


def _initialize_session_state():
    """Initialize session state for feature suggestions."""
    if "accepted_decisions" not in st.session_state:
        st.session_state.accepted_decisions = []
    if "resolved_conflicts" not in st.session_state:
        st.session_state.resolved_conflicts = {}
    if "suggestion_results" not in st.session_state:
        st.session_state.suggestion_results = None


def _call_context_aware_suggestions(file_id: str) -> Optional[Dict]:
    """Call backend API to get context-aware suggestions."""
    try:
        response = requests.post(
            f"{API_BASE}/eda/context-aware-suggestions",
            params={
                "file_id": file_id,
                "include_profiling": True,
                "include_pii": True,
                "include_k_anonymity": True
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to generate suggestions: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None


def _display_summary_metrics(results: Dict):
    """Display summary metrics about suggestions."""
    metadata = results.get("metadata", {})
    conflict_summary = results.get("conflict_summary", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Utility Suggestions",
            metadata.get("total_utility_suggestions", 0),
            help="Transformations to improve model performance"
        )
    
    with col2:
        st.metric(
            "Privacy Suggestions",
            metadata.get("total_privacy_suggestions", 0),
            help="Transformations to reduce privacy risk"
        )
    
    with col3:
        conflicts = metadata.get("total_conflicts", 0)
        st.metric(
            "Conflicts Detected",
            conflicts,
            delta=f"-{len(st.session_state.resolved_conflicts)} resolved" if conflicts > 0 else None,
            help="Columns with competing suggestions"
        )
    
    with col4:
        st.metric(
            "Decisions Made",
            len(st.session_state.accepted_decisions),
            help="Total transformations accepted"
        )


def _render_suggestion_table(suggestions: List[Dict], category: str, show_accept: bool = True):
    """Render a table of suggestions with accept buttons."""
    if not suggestions:
        st.info(f"No {category} suggestions for this dataset.")
        return
    
    st.markdown(f"#### {category.capitalize()} Transformations")
    
    # Create DataFrame for display
    display_data = []
    for idx, sug in enumerate(suggestions):
        display_data.append({
            "Column": sug["column"],
            "Transformation": sug["transformation"],
            "Reason": sug["reason"][:80] + "..." if len(sug.get("reason", "")) > 80 else sug.get("reason", ""),
            "Confidence/Risk": _format_confidence_or_risk(sug, category),
            "idx": idx
        })
    
    df = pd.DataFrame(display_data)
    
    # Display table
    st.dataframe(
        df[["Column", "Transformation", "Reason", "Confidence/Risk"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Accept buttons
    if show_accept:
        st.markdown("**Accept Suggestions:**")
        cols = st.columns(min(len(suggestions), 4))
        
        for idx, sug in enumerate(suggestions):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                if st.button(
                    f"✓ {sug['column']}",
                    key=f"accept_{category}_{sug['column']}_{idx}",
                    help=f"Accept {sug['transformation']} for {sug['column']}"
                ):
                    _accept_suggestion(sug, category)
                    st.rerun()


def _format_confidence_or_risk(suggestion: Dict, category: str) -> str:
    """Format confidence score or risk level for display."""
    if category == "utility":
        confidence = suggestion.get("confidence", 0.0)
        return f"{confidence:.0%}"
    else:  # privacy
        risk = suggestion.get("risk_level", "unknown")
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        return f"{risk_emoji.get(risk, '⚪')} {risk.upper()}"


def _display_conflicts(conflicts: List[Dict]):
    """Display conflict resolution UI."""
    if not conflicts:
        st.success("✅ No conflicts detected! All suggestions are compatible.")
        return
    
    st.markdown("### 🔀 Conflict Resolution Required")
    st.caption("Some columns have both utility and privacy suggestions. Choose which to apply:")
    
    for conflict in conflicts:
        column = conflict["column"]
        
        # Skip if already resolved
        if column in st.session_state.resolved_conflicts:
            continue
        
        severity = conflict["severity"]
        severity_color = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }
        
        with st.expander(
            f"{severity_color.get(severity, '⚪')} **{column}** - {severity.upper()} Priority Conflict",
            expanded=(severity == "high")
        ):
            # Display conflict details
            st.markdown(f"**📊 Utility Suggestion:**")
            st.markdown(f"- **Transform:** `{conflict['utility_transform']}`")
            st.caption(conflict['utility_reason'])
            
            st.markdown(f"**🔒 Privacy Suggestion:**")
            st.markdown(f"- **Transform:** `{conflict['privacy_transform']}`")
            st.caption(conflict['privacy_reason'])
            
            # Show recommendation
            st.info(conflict['recommendation'])
            
            # Decision buttons
            st.markdown("**Make Your Decision:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(
                    "Accept Utility",
                    key=f"conflict_utility_{column}",
                    type="secondary",
                    use_container_width=True
                ):
                    _resolve_conflict(
                        column,
                        conflict['utility_suggestion'],
                        "utility",
                        conflict
                    )
                    st.rerun()
            
            with col2:
                if st.button(
                    "Accept Privacy",
                    key=f"conflict_privacy_{column}",
                    type="primary" if severity == "high" else "secondary",
                    use_container_width=True
                ):
                    _resolve_conflict(
                        column,
                        conflict['privacy_suggestion'],
                        "privacy",
                        conflict
                    )
                    st.rerun()
            
            with col3:
                if st.button(
                    "Skip Both",
                    key=f"conflict_skip_{column}",
                    use_container_width=True
                ):
                    st.session_state.resolved_conflicts[column] = "skipped"
                    st.rerun()


def _accept_suggestion(suggestion: Dict, category: str):
    """Accept a non-conflicting suggestion."""
    decision = {
        "column": suggestion["column"],
        "transformation": suggestion["transformation"],
        "category": category,
        "params": suggestion.get("params", {}),
        "reason": suggestion.get("reason", ""),
        "metadata": suggestion.get("metrics", {})
    }
    
    # Check if already accepted
    if not any(d["column"] == decision["column"] for d in st.session_state.accepted_decisions):
        st.session_state.accepted_decisions.append(decision)
        st.success(f"✅ Accepted {category} transformation for '{suggestion['column']}'")


def _resolve_conflict(column: str, suggestion: Dict, chosen_category: str, conflict: Dict):
    """Resolve a conflict by choosing one suggestion."""
    st.session_state.resolved_conflicts[column] = chosen_category
    
    # Add to accepted decisions
    decision = {
        "column": column,
        "transformation": suggestion["transformation"],
        "category": chosen_category,
        "params": suggestion.get("params", {}),
        "reason": suggestion.get("reason", ""),
        "metadata": suggestion.get("metrics", {}),
        "conflict_resolved": True,
        "rejected_category": "privacy" if chosen_category == "utility" else "utility"
    }
    
    st.session_state.accepted_decisions.append(decision)


def _display_accepted_rules(file_id: str):
    """Display all accepted transformation rules."""
    if not st.session_state.accepted_decisions:
        st.info("ℹ️ No transformations accepted yet. Accept suggestions above to see them here.")
        return
    
    st.markdown("### ✅ Accepted Transformation Rules")
    st.caption(f"Total decisions: {len(st.session_state.accepted_decisions)}")
    
    # Group by category
    utility_decisions = [d for d in st.session_state.accepted_decisions if d["category"] == "utility"]
    privacy_decisions = [d for d in st.session_state.accepted_decisions if d["category"] == "privacy"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**📊 Utility ({len(utility_decisions)})**")
        for decision in utility_decisions:
            st.markdown(f"- `{decision['column']}` → {decision['transformation']}")
    
    with col2:
        st.markdown(f"**🔒 Privacy ({len(privacy_decisions)})**")
        for decision in privacy_decisions:
            st.markdown(f"- `{decision['column']}` → {decision['transformation']}")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", use_container_width=True):
            _save_config(file_id)
    
    with col2:
        if st.button("📥 Export as JSON", use_container_width=True):
            _export_config_json()
    
    with col3:
        if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
            st.session_state.accepted_decisions = []
            st.session_state.resolved_conflicts = {}
            st.rerun()


def _save_config(file_id: str):
    """Save accepted decisions to backend."""
    try:
        response = requests.post(
            f"{API_BASE}/eda/save-transform-config",
            params={"file_id": file_id},
            json={"decisions": st.session_state.accepted_decisions}
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Configuration saved to: {result['config_path']}")
            with st.expander("View saved configuration"):
                st.json(result["config"])
        else:
            st.error(f"Failed to save configuration: {response.text}")
    
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")


def _export_config_json():
    """Export configuration as downloadable JSON."""
    import json
    
    config = {
        "transformations": st.session_state.accepted_decisions,
        "metadata": {
            "total_decisions": len(st.session_state.accepted_decisions),
            "conflicts_resolved": len(st.session_state.resolved_conflicts)
        }
    }
    
    json_str = json.dumps(config, indent=2)
    
    st.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name="transform_config.json",
        mime="application/json"
    )


def expander_feature_suggestions():
    """Main function for context-aware feature suggestions tab."""
    _initialize_session_state()
    
    st.markdown("## 💡 Context-Aware Feature Suggestions")
    st.markdown("""
    This intelligent engine analyzes your data from two perspectives:
    - **📊 Utility:** Transformations to improve model performance and data quality
    - **🔒 Privacy:** Transformations to protect sensitive information and reduce re-identification risk
    
    When suggestions conflict, you'll be guided through resolution to make informed trade-offs.
    """)
    
    # Check for file_id
    if not st.session_state.get("file_id"):
        st.warning("⚠️ Please upload a file first (Step 1: Upload Data)")
        return
    
    file_id = st.session_state.file_id
    
    # Generate suggestions button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔍 Analyze & Generate Suggestions", type="primary", use_container_width=True):
            with st.spinner("Analyzing data for utility and privacy patterns..."):
                results = _call_context_aware_suggestions(file_id)
                if results:
                    st.session_state.suggestion_results = results
                    st.success("✅ Analysis complete!")
                    st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.suggestion_results = None
            st.session_state.accepted_decisions = []
            st.session_state.resolved_conflicts = {}
            st.rerun()
    
    # Display results if available
    if st.session_state.suggestion_results:
        results = st.session_state.suggestion_results
        
        st.markdown("---")
        
        # Summary metrics
        _display_summary_metrics(results)
        
        st.markdown("---")
        
        # Conflicts (show first - most important)
        _display_conflicts(results.get("conflicts", []))
        
        st.markdown("---")
        
        # Non-conflicting suggestions in tabs
        tab1, tab2 = st.tabs(["📊 Utility Suggestions", "🔒 Privacy Suggestions"])
        
        with tab1:
            _render_suggestion_table(
                results.get("non_conflicting_utility", []),
                "utility"
            )
        
        with tab2:
            _render_suggestion_table(
                results.get("non_conflicting_privacy", []),
                "privacy"
            )
        
        st.markdown("---")
        
        # Show accepted rules
        _display_accepted_rules(file_id)
    
    else:
        st.info("👆 Click 'Analyze & Generate Suggestions' to start the intelligent feature analysis.")
