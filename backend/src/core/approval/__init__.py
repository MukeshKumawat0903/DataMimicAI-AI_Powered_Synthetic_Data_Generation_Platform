"""
Approval Module - Human-in-the-Loop Control Layer

This module provides strict, non-agentic approval gates for
transformation plans to ensure human oversight before execution.
"""

from .plan_review_gate import (
    ApprovalStatus,
    PlanReviewAndApprovalGate,
    review_transformation_plan,
)

__all__ = [
    "ApprovalStatus",
    "PlanReviewAndApprovalGate",
    "review_transformation_plan",
]
