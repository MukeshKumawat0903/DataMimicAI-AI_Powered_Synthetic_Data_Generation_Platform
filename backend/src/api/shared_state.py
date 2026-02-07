"""
Shared State for API Modules.

This module provides shared state objects that need to be accessed
across multiple API routers, preventing circular import issues.

LAZY LOADING: approval_gate is NOT instantiated at import time.
It is created on-demand when first accessed via get_approval_gate().
This prevents heavy computation at application startup.

Author: DataMimicAI Team
Date: February 6, 2026
"""

from typing import Dict, Any, Optional

# LAZY: Import moved inside getter function to avoid loading at startup
# from src.core.ai_assistance.approval.plan_review_gate import PlanReviewAndApprovalGate

# Global approval gate instance (lazy-loaded on first access)
# Shared between approval_api and execution_api
_approval_gate: Optional[Any] = None

# In-memory tracking of submitted plans (for pending list)
# Maps plan_id -> plan_dict
# Shared between planner_api, approval_api, and execution_api
submitted_plans: Dict[str, Dict[str, Any]] = {}


def get_approval_gate():
    """
    Get the approval gate instance (lazy singleton).
    
    LAZY LOADING: The approval gate is instantiated only when first accessed,
    not at module import time. This keeps application startup fast.
    
    Returns
    -------
    PlanReviewAndApprovalGate
        The singleton approval gate instance
    """
    global _approval_gate
    if _approval_gate is None:
        # LAZY: Import happens here, only when needed
        from src.core.ai_assistance.approval.plan_review_gate import PlanReviewAndApprovalGate
        _approval_gate = PlanReviewAndApprovalGate(storage_path="workspace/approvals.json")
    return _approval_gate
