"""
Action Planner UI - Complete Transformation Pipeline Interface.

This module provides the UI for the full transformation pipeline:
1. Interpret Diagnostics
2. Generate Transformation Plans
3. Review & Approve Plans
4. Execute Approved Plans
5. View Validation Results

Author: DataMimicAI Team
Date: February 6, 2026
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, Optional, List
import os

# Backend API configuration
BACKEND_URL = os.getenv("API_URL", "http://localhost:8000")


def show_action_planner_tab():
    """
    Main entry point for the Action Planner tab.
    
    This tab orchestrates the complete transformation pipeline with clear
    step-by-step workflow and safety guardrails.
    """
    st.markdown("### 🤖 Action Planner - Transformation Pipeline")
    
    st.info(
        "**Safe Transformation Workflow**: This tool guides you through a controlled "
        "4-step process to transform your data:\n"
        "1. **Interpret** diagnostics to identify patterns\n"
        "2. **Generate** transformation plans (proposals only)\n"
        "3. **Approve** plans manually (human-in-the-loop)\n"
        "4. **Execute** approved plans with automatic validation"
    )
    
    # Initialize session state
    _initialize_session_state()
    
    # Display workflow steps
    st.markdown("---")
    st.markdown("#### Transformation Pipeline Steps")
    
    # Step 1: Interpret Diagnostics
    _render_step_1_interpret()
    
    st.markdown("---")
    
    # Step 2: Generate Transformation Plans
    _render_step_2_plan()
    
    st.markdown("---")
    
    # Step 3: Review & Approve Plans
    _render_step_3_approve()
    
    st.markdown("---")
    
    # Step 4: Execute Approved Plans
    _render_step_4_execute()
    
    st.markdown("---")
    
    # Navigation to Decision Report
    _render_navigation()


def _initialize_session_state():
    """Initialize session state variables for Action Planner."""
    if "ap_diagnostics" not in st.session_state:
        st.session_state.ap_diagnostics = None
    
    if "ap_interpretation" not in st.session_state:
        st.session_state.ap_interpretation = None
    
    if "ap_plans" not in st.session_state:
        st.session_state.ap_plans = None
    
    if "ap_approved_plans" not in st.session_state:
        st.session_state.ap_approved_plans = {}  # plan_id -> approval_record
    
    if "ap_execution_results" not in st.session_state:
        st.session_state.ap_execution_results = {}  # plan_id -> execution_result
    
    if "ap_file_id" not in st.session_state:
        st.session_state.ap_file_id = st.session_state.get("file_id")


# ============================================================================
# STEP 1: INTERPRET DIAGNOSTICS
# ============================================================================

def _render_step_1_interpret():
    """Render Step 1: Diagnostic Interpretation."""
    with st.expander("📊 **Step 1: Interpret Diagnostics**", expanded=True):
        st.markdown(
            "Analyze diagnostics to identify cross-cutting patterns and assess "
            "overall dataset stability. This is a **read-only analysis** with no actions."
        )
        
        # Check if diagnostics are available
        # TODO: In production, load from actual diagnostics output
        # For now, use placeholder or session state
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.caption(
                "ℹ️ Diagnostics must be run first in the EDA section. "
                "This step interprets those results."
            )
        
        with col2:
            interpret_button = st.button(
                "🔍 Interpret Diagnostics",
                key="interpret_diagnostics_btn",
                use_container_width=True,
                type="primary",
                disabled=False  # TODO: Add check for diagnostics availability
            )
        
        if interpret_button:
            _run_diagnostics_interpretation()
        
        # Display interpretation results
        if st.session_state.ap_interpretation:
            _display_interpretation_results(st.session_state.ap_interpretation)
        else:
            st.info("👆 Click 'Interpret Diagnostics' to analyze patterns in your data.")


def _run_diagnostics_interpretation():
    """Call the Diagnostics Interpreter API."""
    try:
        # TODO: Load actual diagnostics from session or EDA results
        # For now, use sample diagnostics
        sample_diagnostics = {
            "diagnostics": [
                {
                    "issue_type": "high_skew",
                    "severity": "high",
                    "column": "Volume",
                    "metrics": {"skewness": 6.9051}
                },
                {
                    "issue_type": "outliers",
                    "severity": "high",
                    "column": "Volume",
                    "metrics": {"outlier_percentage": 12.7}
                }
            ],
            "summary": {"total_issues": 2, "high_severity_count": 2},
            "metadata": {"timestamp": "2026-02-06"}
        }
        
        st.session_state.ap_diagnostics = sample_diagnostics
        
        with st.spinner("Interpreting diagnostics..."):
            response = requests.post(
                f"{BACKEND_URL}/api/diagnostics/interpret",
                json={
                    "diagnostics_input": sample_diagnostics,
                    "rag_context": None
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.ap_interpretation = result.get("interpretation")
                st.success("✅ Diagnostics interpreted successfully!")
                st.rerun()
            else:
                st.error(f"API error: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def _display_interpretation_results(interpretation: Dict[str, Any]):
    """Display interpretation results."""
    st.markdown("#### 📋 Interpretation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        assessment = interpretation.get("overall_assessment", "N/A")
        st.metric("Overall Assessment", assessment.replace("_", " ").title())
    
    with col2:
        confidence = interpretation.get("confidence", "N/A")
        st.metric("Confidence", confidence.upper())
    
    # Dominant patterns
    patterns = interpretation.get("dominant_issue_patterns", [])
    if patterns:
        st.markdown("**🔍 Dominant Issue Patterns:**")
        for pattern in patterns:
            st.markdown(f"- `{pattern.replace('_', ' ').title()}`")
    
    # Supporting evidence
    evidence = interpretation.get("supporting_evidence", [])
    if evidence:
        with st.expander("📑 Supporting Evidence", expanded=False):
            for idx, ev in enumerate(evidence, 1):
                st.markdown(f"**Evidence {idx}:**")
                st.json(ev)


# ============================================================================
# STEP 2: GENERATE TRANSFORMATION PLANS
# ============================================================================

def _render_step_2_plan():
    """Render Step 2: Generate Transformation Plans."""
    with st.expander("📝 **Step 2: Generate Transformation Plans**", expanded=False):
        st.markdown(
            "Generate **proposed** transformation plans based on interpretation. "
            "These are **proposals only** - no execution occurs at this stage."
        )
        
        # Check if prerequisites are met
        can_generate = st.session_state.ap_interpretation is not None
        
        if not can_generate:
            st.warning("⚠️ Complete Step 1 (Interpret Diagnostics) first.")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.caption("ℹ️ Plans are generated based on identified patterns. Review carefully before approval.")
        
        with col2:
            plan_button = st.button(
                "🧠 Generate Plans",
                key="generate_plans_btn",
                use_container_width=True,
                type="primary"
            )
        
        if plan_button:
            _run_transformation_planner()
        
        # Display generated plans
        if st.session_state.ap_plans:
            _display_proposed_plans(st.session_state.ap_plans)
        else:
            st.info("👆 Click 'Generate Plans' to create transformation proposals.")


def _run_transformation_planner():
    """Call the Transformation Planner API."""
    try:
        with st.spinner("Generating transformation plans..."):
            response = requests.post(
                f"{BACKEND_URL}/api/planner/create-plan",
                json={
                    "diagnostics": st.session_state.ap_diagnostics,
                    "interpretation": st.session_state.ap_interpretation,
                    "rag_context": None
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.ap_plans = result.get("proposed_plans", [])
                st.success(f"✅ Generated {len(st.session_state.ap_plans)} transformation plan(s)!")
                st.rerun()
            else:
                st.error(f"API error: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def _display_proposed_plans(plans: List[Dict[str, Any]]):
    """Display proposed transformation plans."""
    st.markdown("#### 📦 Proposed Transformation Plans")
    st.caption(f"Generated {len(plans)} plan(s). Review each plan before approval.")
    
    for idx, plan in enumerate(plans, 1):
        plan_id = plan.get("plan_id", f"Plan {idx}")
        
        with st.container():
            st.markdown(f"##### 🔖 {plan_id}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Applicable patterns
                patterns = plan.get("applicable_issue_patterns", [])
                if patterns:
                    st.markdown("**Addresses Patterns:**")
                    for pattern in patterns:
                        st.markdown(f"- {pattern.replace('_', ' ').title()}")
            
            with col2:
                # Check approval status
                approval_status = _get_plan_approval_status(plan_id)
                if approval_status == "APPROVED":
                    st.success("✅ APPROVED")
                elif approval_status == "REJECTED":
                    st.error("❌ REJECTED")
                else:
                    st.info("⏳ PENDING")
            
            # Transformations
            transformations = plan.get("proposed_transformations", [])
            if transformations:
                with st.expander("🔧 Proposed Transformations", expanded=False):
                    for t_idx, trans in enumerate(transformations, 1):
                        st.markdown(f"**Transformation {t_idx}:**")
                        st.markdown(f"- **Type**: `{trans.get('transformation', 'N/A')}`")
                        st.markdown(f"- **Target Columns**: {trans.get('target_columns', [])}")
                        st.markdown(f"- **Rationale**: {trans.get('rationale', 'N/A')}")
                        if trans.get("parameters"):
                            st.markdown(f"- **Parameters**: {trans.get('parameters')}")
                        st.markdown("---")
            
            # Rationale and risks
            with st.expander("📖 Rationale & Risks", expanded=False):
                st.markdown("**Rationale:**")
                st.write(plan.get("rationale", "N/A"))
                
                risks = plan.get("estimated_risks", [])
                if risks:
                    st.markdown("**⚠️ Estimated Risks:**")
                    for risk in risks:
                        st.markdown(f"- {risk}")
            
            st.markdown("---")


# ============================================================================
# STEP 3: REVIEW & APPROVE PLANS
# ============================================================================

def _render_step_3_approve():
    """Render Step 3: Review & Approve Plans."""
    with st.expander("✅ **Step 3: Review & Approve Plans**", expanded=False):
        st.markdown(
            "**Human-in-the-loop approval gate**. Review each plan and explicitly "
            "approve or reject. Only approved plans can be executed."
        )
        
        # Check if prerequisites are met
        can_approve = st.session_state.ap_plans is not None
        
        if not can_approve:
            st.warning("⚠️ Complete Step 2 (Generate Plans) first.")
            return
        
        if not st.session_state.ap_plans:
            st.info("No plans available for approval.")
            return
        
        st.markdown("#### 🔍 Review Plans")
        
        for idx, plan in enumerate(st.session_state.ap_plans, 1):
            plan_id = plan.get("plan_id", f"Plan {idx}")
            
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{plan_id}**")
                
                with col2:
                    approval_status = _get_plan_approval_status(plan_id)
                    if approval_status == "APPROVED":
                        st.success("✅ APPROVED")
                    elif approval_status == "REJECTED":
                        st.error("❌ REJECTED")
                    else:
                        st.info("⏳ PENDING")
                
                with col3:
                    approve_btn = st.button(
                        "✅ Approve",
                        key=f"approve_{plan_id}",
                        disabled=(approval_status != "PENDING"),
                        use_container_width=True
                    )
                    
                    if approve_btn:
                        _approve_plan(plan, "approve")
                
                with col4:
                    reject_btn = st.button(
                        "❌ Reject",
                        key=f"reject_{plan_id}",
                        disabled=(approval_status != "PENDING"),
                        use_container_width=True
                    )
                    
                    if reject_btn:
                        _approve_plan(plan, "reject")
                
                # Reviewer notes
                if approval_status == "PENDING":
                    notes = st.text_input(
                        "Reviewer Notes (optional)",
                        key=f"notes_{plan_id}",
                        placeholder="Add any notes about your decision..."
                    )
                    st.session_state[f"reviewer_notes_{plan_id}"] = notes
                
                st.markdown("---")


def _approve_plan(plan: Dict[str, Any], decision: str):
    """Call the Approval API to approve or reject a plan."""
    plan_id = plan.get("plan_id")
    reviewer_notes = st.session_state.get(f"reviewer_notes_{plan_id}", "")
    
    try:
        with st.spinner(f"{decision.title()}ing plan {plan_id}..."):
            response = requests.post(
                f"{BACKEND_URL}/api/approval/review-plan",
                json={
                    "plan": plan,
                    "decision": decision,
                    "reviewer_notes": reviewer_notes
                },
                timeout=30
            )
            
            if response.status_code == 200:
                approval_record = response.json()
                st.session_state.ap_approved_plans[plan_id] = approval_record
                st.success(f"✅ Plan {plan_id} {decision}d successfully!")
                st.rerun()
            else:
                st.error(f"API error: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


def _get_plan_approval_status(plan_id: str) -> str:
    """Get approval status for a plan."""
    if plan_id in st.session_state.ap_approved_plans:
        return st.session_state.ap_approved_plans[plan_id].get("status", "PENDING")
    return "PENDING"


# ============================================================================
# STEP 4: EXECUTE APPROVED PLANS
# ============================================================================

def _render_step_4_execute():
    """Render Step 4: Execute Approved Plans."""
    with st.expander("🚀 **Step 4: Execute Approved Plans**", expanded=False):
        st.markdown(
            "Execute **approved** transformation plans. Execution is deterministic "
            "and automatically triggers validation for before/after comparison."
        )
        
        # Check if there are any approved plans
        approved_plans = [
            plan for plan in (st.session_state.ap_plans or [])
            if _get_plan_approval_status(plan.get("plan_id")) == "APPROVED"
        ]
        
        if not approved_plans:
            st.warning("⚠️ No approved plans available. Complete Step 3 (Review & Approve) first.")
            return
        
        st.markdown("#### 🎯 Execute Approved Plans")
        st.caption(f"Found {len(approved_plans)} approved plan(s) ready for execution.")
        
        # File ID selection
        file_id = st.session_state.ap_file_id
        if not file_id:
            file_id = st.text_input(
                "📁 Dataset File ID",
                value=st.session_state.get("file_id", ""),
                help="Enter the file ID of the dataset to transform"
            )
            st.session_state.ap_file_id = file_id
        
        if not file_id:
            st.info("ℹ️ Enter a file ID to enable execution.")
            return
        
        st.markdown("---")
        
        for plan in approved_plans:
            plan_id = plan.get("plan_id")
            
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{plan_id}**")
                    if plan_id in st.session_state.ap_execution_results:
                        exec_status = st.session_state.ap_execution_results[plan_id].get("execution_status")
                        if exec_status == "SUCCESS":
                            st.success("✅ Executed Successfully")
                        else:
                            st.error("❌ Execution Failed")
                
                with col2:
                    execute_btn = st.button(
                        "🚀 Execute",
                        key=f"execute_{plan_id}",
                        disabled=(plan_id in st.session_state.ap_execution_results),
                        use_container_width=True,
                        type="primary"
                    )
                    
                    if execute_btn:
                        _execute_plan(plan_id, file_id)
                
                with col3:
                    if plan_id in st.session_state.ap_execution_results:
                        result = st.session_state.ap_execution_results[plan_id]
                        if result.get("validation_available"):
                            if st.button("📊 View Results", key=f"view_{plan_id}", use_container_width=True):
                                st.info("Navigate to Decision Report tab to view validation results.")
                
                # Show execution details
                if plan_id in st.session_state.ap_execution_results:
                    result = st.session_state.ap_execution_results[plan_id]
                    with st.expander("📋 Execution Details", expanded=False):
                        st.json(result)
                
                st.markdown("---")


def _execute_plan(plan_id: str, file_id: str):
    """Call the Execution API to execute an approved plan."""
    try:
        with st.spinner(f"Executing plan {plan_id}..."):
            response = requests.post(
                f"{BACKEND_URL}/api/execution/execute-plan",
                json={
                    "plan_id": plan_id,
                    "file_id": file_id
                },
                timeout=120  # Longer timeout for execution
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.ap_execution_results[plan_id] = result
                
                if result.get("execution_status") == "SUCCESS":
                    st.success(
                        f"✅ Plan {plan_id} executed successfully! "
                        f"Applied {len(result.get('applied_transformations', []))} transformation(s)."
                    )
                    
                    if result.get("validation_available"):
                        st.info(
                            "📊 Validation complete! Navigate to the **Decision Report** tab "
                            "to view before/after metrics."
                        )
                else:
                    st.error(f"❌ Execution failed: {result.get('error')}")
                
                st.rerun()
            
            elif response.status_code == 403:
                st.error(f"🔒 Plan {plan_id} is not approved. Cannot execute.")
            elif response.status_code == 404:
                st.error(f"❌ Plan or file not found.")
            else:
                st.error(f"API error: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


# ============================================================================
# NAVIGATION
# ============================================================================

def _render_navigation():
    """Render navigation and help section."""
    st.markdown("### 🧭 Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 View Validation Results**")
        st.markdown(
            "After executing approved plans, navigate to the **Decision Report** tab "
            "to view detailed before/after metric comparisons."
        )
    
    with col2:
        st.markdown("**🔄 Reset Workflow**")
        if st.button("🔄 Reset All", key="reset_all", use_container_width=True):
            st.session_state.ap_diagnostics = None
            st.session_state.ap_interpretation = None
            st.session_state.ap_plans = None
            st.session_state.ap_approved_plans = {}
            st.session_state.ap_execution_results = {}
            st.success("✅ Workflow reset!")
            st.rerun()
    
    # Help section
    with st.expander("❓ Help & Guidelines", expanded=False):
        st.markdown("""
        **Workflow Guidelines:**
        
        1. **Interpret Diagnostics** - Analyze patterns in your dataset
           - Read-only analysis
           - No actions taken
           - Identifies cross-cutting issues
        
        2. **Generate Plans** - Create transformation proposals
           - Agent-generated suggestions
           - Multiple plans may be proposed
           - Plans are proposals only (not executed)
        
        3. **Review & Approve** - Human approval gate
           - Review each plan carefully
           - Approve plans you trust
           - Reject plans that seem risky
           - Add notes for audit trail
        
        4. **Execute Approved** - Safe execution
           - Only approved plans can execute
           - Deterministic transformations
           - Automatic validation
           - Results viewable in Decision Report
        
        **Safety Features:**
        - ✅ No auto-execution
        - ✅ Explicit human approval required
        - ✅ Clear step-by-step workflow
        - ✅ Automatic validation after execution
        - ✅ Full audit trail
        
        **Need Help?**
        - Check the Decision Report tab for validation results
        - Review plan details before approval
        - Add reviewer notes for documentation
        """)
