"""
Approval API - FastAPI Endpoints for Plan Review & Approval.

This module provides REST API endpoints for the PlanReviewAndApprovalGate.
It exposes a strict human approval boundary for transformation plans.

Endpoints:
- POST /api/approval/review-plan - Review and approve/reject a plan
- GET  /api/approval/pending-plans - List plans awaiting review
- GET  /api/approval/plan-status/{plan_id} - Get approval status for a plan

Author: DataMimicAI Team
Date: February 6, 2026
"""

from fastapi import APIRouter, HTTPException, Body, Path
from typing import Dict, Any, List, Optional
import logging

from src.core.approval.plan_review_gate import (
    PlanReviewAndApprovalGate,
    ApprovalStatus
)
from src.api.shared_state import approval_gate, submitted_plans

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/approval", tags=["approval"])

# Use shared state to avoid circular imports
_approval_gate = approval_gate
_submitted_plans = submitted_plans


@router.post("/review-plan")
async def review_plan(
    plan: Dict[str, Any] = Body(
        ...,
        description="Transformation plan from TransformationPlannerAgent",
        example={
            "plan_id": "TP-001",
            "applicable_issue_patterns": ["skew_and_outliers"],
            "proposed_transformations": [
                {
                    "transformation": "log_transform",
                    "target_columns": ["Volume"],
                    "rationale": "Reduce skewness",
                    "parameters": {"base": "natural"}
                }
            ],
            "rationale": "Apply log transformation...",
            "estimated_risks": ["May introduce NaN for zero/negative values"]
        }
    ),
    decision: str = Body(
        ...,
        description="Human decision: 'approve' or 'reject'",
        example="approve"
    ),
    reviewer_notes: Optional[str] = Body(
        None,
        description="Optional notes from the reviewer",
        example="Approved after validating no negative values in Volume column"
    )
) -> Dict[str, Any]:
    """
    Review and approve/reject a transformation plan.
    
    This endpoint provides a STRICT APPROVAL BOUNDARY. It does NOT:
    - Execute transformations
    - Modify plans
    - Infer decisions
    - Call LLMs or agents
    - Re-rank or alter proposals
    
    Parameters
    ----------
    plan : Dict[str, Any]
        Transformation plan from TransformationPlannerAgent containing:
        - plan_id: Unique plan identifier
        - applicable_issue_patterns: List of patterns this plan addresses
        - proposed_transformations: List of transformations
        - rationale: Explanation for the plan
        - estimated_risks: Identified risks
    
    decision : str
        Human decision - must be "approve" or "reject"
    
    reviewer_notes : str, optional
        Optional notes from the human reviewer
    
    Returns
    -------
    Dict[str, Any]
        Approval record containing:
        - plan_id: Plan identifier
        - status: APPROVED or REJECTED
        - reviewed_by: "human"
        - reviewed_at: ISO timestamp
        - notes: Reviewer notes
    
    Raises
    ------
    HTTPException 400
        If plan structure is invalid, decision is invalid, or plan already reviewed
    HTTPException 500
        For unexpected errors
    
    Examples
    --------
    POST /api/approval/review-plan
    
    Request:
    {
        "plan": {
            "plan_id": "TP-001",
            "applicable_issue_patterns": ["skew_and_outliers"],
            "proposed_transformations": [...],
            "rationale": "...",
            "estimated_risks": [...]
        },
        "decision": "approve",
        "reviewer_notes": "Looks good after validation"
    }
    
    Response:
    {
        "plan_id": "TP-001",
        "status": "APPROVED",
        "reviewed_by": "human",
        "reviewed_at": "2026-02-06T10:30:00+00:00",
        "notes": "Looks good after validation"
    }
    """
    try:
        plan_id = plan.get("plan_id", "UNKNOWN")
        
        # Log request
        logger.info(
            f"Reviewing plan {plan_id}: decision={decision}, "
            f"has_notes={reviewer_notes is not None}"
        )
        
        # Store submitted plan for pending list tracking
        if plan_id not in _submitted_plans:
            _submitted_plans[plan_id] = plan
        
        # Call approval gate (validates and records decision)
        approval_record = _approval_gate.review_plan(
            plan=plan,
            decision=decision,
            reviewer_notes=reviewer_notes
        )
        
        logger.info(
            f"Plan {plan_id} reviewed: status={approval_record['status']}"
        )
        
        return approval_record
    
    except ValueError as e:
        # Gate validation errors (invalid plan structure, decision, or duplicate)
        logger.warning(f"Validation error in plan review: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plan review request: {str(e)}"
        )
    
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in plan review: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during plan review: {str(e)}"
        )


@router.get("/pending-plans")
async def get_pending_plans() -> Dict[str, Any]:
    """
    Get list of plans awaiting review (status = PENDING).
    
    Returns plans that have been submitted but not yet approved or rejected.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - pending_plans: List of plans with PENDING status
        - count: Number of pending plans
    
    Examples
    --------
    GET /api/approval/pending-plans
    
    Response:
    {
        "pending_plans": [
            {
                "plan_id": "TP-001",
                "applicable_issue_patterns": ["skew_and_outliers"],
                "proposed_transformations": [...],
                "status": "PENDING"
            }
        ],
        "count": 1
    }
    """
    try:
        pending_plans = []
        
        # Check each submitted plan's approval status
        for plan_id, plan in _submitted_plans.items():
            status = _approval_gate.get_approval_status(plan_id)
            
            if status == ApprovalStatus.PENDING.value:
                # Add status to plan for convenience
                plan_with_status = {**plan, "status": status}
                pending_plans.append(plan_with_status)
        
        logger.info(f"Retrieved {len(pending_plans)} pending plans")
        
        return {
            "pending_plans": pending_plans,
            "count": len(pending_plans)
        }
    
    except Exception as e:
        logger.error(f"Error retrieving pending plans: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving pending plans: {str(e)}"
        )


@router.get("/plan-status/{plan_id}")
async def get_plan_status(
    plan_id: str = Path(..., description="Plan identifier")
) -> Dict[str, Any]:
    """
    Get the approval status for a specific plan.
    
    Parameters
    ----------
    plan_id : str
        Plan identifier
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plan_id: Plan identifier
        - status: PENDING, APPROVED, or REJECTED
        - approval_record: Full approval record if approved/rejected, None if pending
    
    Raises
    ------
    HTTPException 404
        If plan_id is not found in submitted plans
    
    Examples
    --------
    GET /api/approval/plan-status/TP-001
    
    Response:
    {
        "plan_id": "TP-001",
        "status": "APPROVED",
        "approval_record": {
            "plan_id": "TP-001",
            "status": "APPROVED",
            "reviewed_by": "human",
            "reviewed_at": "2026-02-06T10:30:00+00:00",
            "notes": "Looks good"
        }
    }
    """
    try:
        # Check if plan exists in submitted plans
        if plan_id not in _submitted_plans:
            logger.warning(f"Plan status requested for unknown plan_id: {plan_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Plan not found: {plan_id}. Plan must be submitted before checking status."
            )
        
        # Get approval status and record
        status = _approval_gate.get_approval_status(plan_id)
        approval_record = _approval_gate.get_approval_record(plan_id)
        
        logger.info(f"Plan {plan_id} status: {status}")
        
        return {
            "plan_id": plan_id,
            "status": status,
            "approval_record": approval_record
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error retrieving plan status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving plan status: {str(e)}"
        )


@router.get("/approved-plans")
async def get_approved_plans() -> Dict[str, Any]:
    """
    Get list of all approved plan IDs.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - approved_plan_ids: List of approved plan IDs
        - count: Number of approved plans
    
    Examples
    --------
    GET /api/approval/approved-plans
    
    Response:
    {
        "approved_plan_ids": ["TP-001", "TP-003"],
        "count": 2
    }
    """
    try:
        approved_plan_ids = _approval_gate.list_approved_plans()
        
        logger.info(f"Retrieved {len(approved_plan_ids)} approved plans")
        
        return {
            "approved_plan_ids": approved_plan_ids,
            "count": len(approved_plan_ids)
        }
    
    except Exception as e:
        logger.error(f"Error retrieving approved plans: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving approved plans: {str(e)}"
        )


@router.get("/rejected-plans")
async def get_rejected_plans() -> Dict[str, Any]:
    """
    Get list of all rejected plan IDs.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - rejected_plan_ids: List of rejected plan IDs
        - count: Number of rejected plans
    
    Examples
    --------
    GET /api/approval/rejected-plans
    
    Response:
    {
        "rejected_plan_ids": ["TP-002"],
        "count": 1
    }
    """
    try:
        rejected_plan_ids = _approval_gate.list_rejected_plans()
        
        logger.info(f"Retrieved {len(rejected_plan_ids)} rejected plans")
        
        return {
            "rejected_plan_ids": rejected_plan_ids,
            "count": len(rejected_plan_ids)
        }
    
    except Exception as e:
        logger.error(f"Error retrieving rejected plans: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving rejected plans: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for approval API.
    
    Returns
    -------
    Dict[str, str]
        Status message indicating the API is operational
    """
    return {
        "status": "healthy",
        "service": "approval-api",
        "version": "1.0"
    }
