"""
Shared State for API Modules.

This module provides shared state objects that need to be accessed
across multiple API routers, preventing circular import issues.

Author: DataMimicAI Team
Date: February 6, 2026
"""

from typing import Dict, Any
from src.core.approval.plan_review_gate import PlanReviewAndApprovalGate

# Global approval gate instance (in-memory, with optional persistence)
# Shared between approval_api and execution_api
approval_gate = PlanReviewAndApprovalGate(storage_path="workspace/approvals.json")

# In-memory tracking of submitted plans (for pending list)
# Maps plan_id -> plan_dict
# Shared between planner_api, approval_api, and execution_api
submitted_plans: Dict[str, Dict[str, Any]] = {}
