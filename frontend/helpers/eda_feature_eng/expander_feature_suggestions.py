"""
Context-Aware Feature Suggestion Engine Frontend
Displays utility suggestions, privacy suggestions, and conflict resolution UI.
"""

import streamlit as st
import requests
import pandas as pd
import os
import base64
import gzip
import io
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
    
    # Add filters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Column filter
        all_columns = sorted(set([sug["column"] for sug in suggestions]))
        selected_column = st.selectbox(
            "Filter by Column:",
            options=["All Columns"] + all_columns,
            key=f"filter_column_{category}",
            index=0
        )
    
    with col2:
        # Transformation type filter
        all_transforms = sorted(set([sug["transformation"] for sug in suggestions]))
        selected_transform = st.selectbox(
            "Filter by Transform:",
            options=["All Transforms"] + all_transforms,
            key=f"filter_transform_{category}",
            index=0
        )
    
    # Apply filters
    filtered_suggestions = suggestions
    if selected_column != "All Columns":
        filtered_suggestions = [s for s in filtered_suggestions if s["column"] == selected_column]
    if selected_transform != "All Transforms":
        filtered_suggestions = [s for s in filtered_suggestions if s["transformation"] == selected_transform]
    
    if not filtered_suggestions:
        st.warning("No suggestions match the selected filters.")
        return
    
    st.caption(f"Showing {len(filtered_suggestions)} of {len(suggestions)} suggestions")
    
    # Create DataFrame for display
    display_data = []
    for idx, sug in enumerate(filtered_suggestions):
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
        cols = st.columns(min(len(filtered_suggestions), 4))
        
        for idx, sug in enumerate(filtered_suggestions):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                if st.button(
                    f"✓ {sug['column']}",
                    key=f"accept_{category}_{sug['column']}_{idx}_{sug['transformation']}",
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
            
            # Decision using radio button (more intuitive for single choice)
            st.markdown("**Make Your Decision:**")
            
            decision_choice = st.radio(
                "Choose one transformation to apply:",
                options=[
                    f"📊 Utility: {conflict['utility_transform']}",
                    f"🔒 Privacy: {conflict['privacy_transform']}",
                    "⏭️ Skip Both"
                ],
                key=f"radio_conflict_{column}",
                label_visibility="collapsed"
            )
            
            # Apply decision button
            if st.button(
                "✅ Confirm Decision",
                key=f"confirm_conflict_{column}",
                type="primary",
                use_container_width=True
            ):
                if "Utility" in decision_choice:
                    _resolve_conflict(
                        column,
                        conflict['utility_suggestion'],
                        "utility",
                        conflict
                    )
                elif "Privacy" in decision_choice:
                    _resolve_conflict(
                        column,
                        conflict['privacy_suggestion'],
                        "privacy",
                        conflict
                    )
                else:  # Skip
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
    
    # Log conflict resolution to backend
    _log_conflict_resolution(
        st.session_state.file_id,
        column,
        chosen_category,
        suggestion["transformation"],
        f"Resolved {conflict.get('severity', 'unknown')} priority conflict"
    )


def _log_conflict_resolution(file_id: str, column: str, chosen_category: str, chosen_transformation: str, user_note: str = None):
    """Log conflict resolution decision to backend."""
    try:
        response = requests.post(
            f"{API_BASE}/eda/resolve-conflict",
            params={
                "file_id": file_id,
                "column": column,
                "chosen_category": chosen_category,
                "chosen_transformation": chosen_transformation,
                "user_note": user_note
            }
        )
        
        if response.status_code != 200:
            # Don't block the UI, just log silently
            print(f"Warning: Failed to log conflict resolution: {response.text}")
    
    except Exception as e:
        # Don't block the UI on logging errors
        print(f"Warning: Error logging conflict resolution: {str(e)}")


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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⚡ Apply to Dataset", use_container_width=True, type="primary"):
            _apply_transformations(file_id)
    
    with col2:
        if st.button("💾 Save Configuration", use_container_width=True):
            _save_config(file_id)
    
    with col3:
        if st.button("📥 Export JSON", use_container_width=True):
            _export_config_json()
    
    with col4:
        if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
            st.session_state.accepted_decisions = []
            st.session_state.resolved_conflicts = {}
            st.rerun()
    
    # Add batch code generation button
    st.markdown("---")
    if st.button("🐍 Generate Python Code", use_container_width=True):
        _generate_batch_code(file_id)


def _decode_transformed_dataset(payload: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Decode a transformed dataset payload from the backend response."""
    if not payload:
        return None
    content = payload.get("content")
    if not content:
        return None
    raw_bytes = base64.b64decode(content)
    if payload.get("compression") == "gzip":
        raw_bytes = gzip.decompress(raw_bytes)
    text_stream = io.StringIO(raw_bytes.decode("utf-8"))
    return pd.read_csv(text_stream)


def _apply_transformations(file_id: str):
    """Apply accepted transformations to the dataset."""
    try:
        dry_run = st.session_state.get("dry_run_mode", False)
        with st.spinner("Applying transformations to dataset..."):
            response = requests.post(
                f"{API_BASE}/eda/apply-transform-config",
                params={
                    "file_id": file_id,
                    "dry_run": dry_run
                },
                json={"decisions": st.session_state.accepted_decisions}
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("summary", {})
                transformed_payload = result.get("transformed_data") or {}

                summary_message = summary.get("message") or f"✅ Applied {summary.get('applied_count', 0)} transformations successfully!"
                st.success(summary_message)

                preview_records = summary.get("preview_full") or summary.get("preview") or []
                preview_df = pd.DataFrame(preview_records) if preview_records else pd.DataFrame()
                st.session_state.feature_preview_df = preview_df
                st.session_state.feature_preview_message = summary_message
                st.session_state.last_changed_columns = summary.get("columns_modified", [])
                st.session_state.last_applied_summary = summary

                updated_df = None
                if not summary.get("dry_run", dry_run):
                    try:
                        updated_df = _decode_transformed_dataset(transformed_payload)
                    except Exception as decode_err:
                        st.warning(f"Transformations applied, but unable to load transformed dataset: {decode_err}")
                else:
                    if transformed_payload and transformed_payload.get("reason"):
                        st.info(transformed_payload["reason"])

                if updated_df is not None:
                    previous_df = st.session_state.get("df")
                    history = st.session_state.get("data_history", [])
                    if not history:
                        base_df = previous_df if isinstance(previous_df, pd.DataFrame) else st.session_state.get("uploaded_df")
                        if isinstance(base_df, pd.DataFrame):
                            history = [base_df.copy()]
                    else:
                        if isinstance(previous_df, pd.DataFrame):
                            history.append(previous_df.copy())
                    history.append(updated_df.copy())
                    st.session_state.data_history = history
                    st.session_state.df = updated_df.copy()
                    st.session_state.features_applied = True
                else:
                    if not summary.get("dry_run", dry_run) and transformed_payload and not transformed_payload.get("content"):
                        st.warning(transformed_payload.get("reason", "Transformed dataset too large to download for preview."))
                    st.session_state.features_applied = False if summary.get("dry_run", dry_run) else st.session_state.get("features_applied", False)

                # Show summary details
                with st.expander("📊 Transformation Summary", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Decisions", summary.get("total_decisions", 0))
                    with col2:
                        st.metric("Applied", summary.get("applied_count", 0))
                    with col3:
                        st.metric("Skipped", summary.get("skipped_count", 0))

                    st.markdown("**Modified Columns:**")
                    st.write(", ".join(f"`{col}`" for col in summary.get("columns_modified", [])))

                    if summary.get("skipped_actions"):
                        st.warning("⚠️ Some actions were skipped:")
                        for skip in summary["skipped_actions"]:
                            st.markdown(f"- `{skip['column']}` ({skip['transformation']}): {skip['reason']}")

                    if summary.get("preview"):
                        st.markdown("**Preview of Modified Columns:**")
                        st.dataframe(pd.DataFrame(summary["preview"]), use_container_width=True)
                    if summary.get("preview_full"):
                        st.markdown("**Preview of Updated Dataset (first 50 rows):**")
                        st.dataframe(pd.DataFrame(summary["preview_full"]), use_container_width=True)
            else:
                st.error(f"❌ Failed to apply transformations: {response.text}")
    
    except Exception as e:
        st.error(f"Error applying transformations: {str(e)}")


def _generate_batch_code(file_id: str):
    """Generate combined Python code for all accepted transformations."""
    try:
        with st.spinner("Generating Python code..."):
            response = requests.post(
                f"{API_BASE}/eda/get-transform-batch-code",
                params={"file_id": file_id},
                json={"decisions": st.session_state.accepted_decisions}
            )
            
            if response.status_code == 200:
                result = response.json()
                code = result.get("code", "")
                
                st.success(f"✅ Generated code for {result.get('total_decisions', 0)} transformations!")
                
                with st.expander("🐍 Python Code", expanded=True):
                    st.code(code, language="python")
                    st.download_button(
                        label="📥 Download Python Script",
                        data=code,
                        file_name=f"transformations_{file_id}.py",
                        mime="text/x-python"
                    )
            else:
                st.error(f"❌ Failed to generate code: {response.text}")
    
    except Exception as e:
        st.error(f"Error generating code: {str(e)}")


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


def _render_load_config_ui(file_id: str):
    """Render UI for loading saved configurations."""
    try:
        # Fetch available configs
        response = requests.get(
            f"{API_BASE}/eda/list-transform-configs",
            params={"file_id": file_id}
        )
        
        if response.status_code == 200:
            result = response.json()
            configs = result.get("configs", [])
            
            if not configs:
                st.info("No saved configurations found for this file.")
                return
            
            st.caption(f"Found {len(configs)} saved configuration(s)")
            
            # Create selectbox with config options
            config_options = {}
            for cfg in configs:
                label = f"{cfg['filename']} - {cfg['decision_count']} decisions - {cfg['modified_at'][:19]}"
                config_options[label] = cfg['filepath']
            
            selected_label = st.selectbox(
                "Select a configuration to load:",
                options=list(config_options.keys()),
                key="config_selector"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Load Selected Config", use_container_width=True):
                    _load_selected_config(config_options[selected_label])
            
            with col2:
                # Show details of selected config
                selected_cfg = next((c for c in configs if c['filepath'] == config_options[selected_label]), None)
                if selected_cfg:
                    st.metric("Decisions", selected_cfg['decision_count'])
        else:
            st.error(f"Failed to fetch configurations: {response.text}")
    
    except Exception as e:
        st.error(f"Error loading configurations: {str(e)}")


def _load_selected_config(config_path: str):
    """Load a selected configuration and populate session state."""
    try:
        response = requests.get(
            f"{API_BASE}/eda/load-transform-config",
            params={"config_path": config_path}
        )
        
        if response.status_code == 200:
            result = response.json()
            config = result.get("config", {})
            
            # Populate session state with loaded decisions
            st.session_state.accepted_decisions = config.get("transformations", [])
            
            st.success(f"✅ Loaded {len(st.session_state.accepted_decisions)} transformation decisions!")
            st.rerun()
        else:
            st.error(f"Failed to load configuration: {response.text}")
    
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")


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
    
    # Add section for loading saved configurations
    with st.expander("📂 Load Saved Configuration"):
        _render_load_config_ui(file_id)
    
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
