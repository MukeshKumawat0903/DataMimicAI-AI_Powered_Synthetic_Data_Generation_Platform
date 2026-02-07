"""
PlanReviewAndApprovalGate - Human-in-the-Loop Control Layer

This module is NOT an agent. It contains ZERO reasoning or intelligence.
It acts as a strict, auditable approval boundary for transformation plans.

PURPOSE:
    - Validate transformation plan schema
    - Record human approval/rejection decisions
    - Prevent execution without explicit human approval
    - Maintain immutable audit trail

CONSTRAINTS:
    - No LLM calls
    - No execution
    - No modification of plans
    - No decision inference
    - Deterministic behavior only
"""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ApprovalStatus(str, Enum):
    """Fixed approval states - no additions allowed."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class PlanReviewAndApprovalGate:
    """
    Non-agentic human-in-the-loop control layer for transformation plans.
    
    This is a GATEKEEPER, not a decision maker.
    
    Responsibilities:
        - Validate plan structure
        - Record human decisions
        - Prevent duplicate approvals
        - Maintain audit trail
    
    Non-Responsibilities (NEVER):
        - Execute transformations
        - Modify plans
        - Infer decisions
        - Call LLMs or agents
        - Re-rank or alter proposals
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the approval gate.
        
        Args:
            storage_path: Optional path to JSON file for persistent storage.
                         If None, uses in-memory storage only.
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._approval_records: Dict[str, Dict[str, Any]] = {}
        
        # Load existing records if storage path provided
        if self._storage_path and self._storage_path.exists():
            self._load_records()
    
    def review_plan(
        self,
        plan: Dict[str, Any],
        decision: str,
        reviewer_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a human decision for a transformation plan.
        
        This method VALIDATES and RECORDS only. It does NOT:
            - Execute transformations
            - Modify the plan
            - Make decisions
            - Call external services
        
        Args:
            plan: The transformation plan from TransformationPlannerAgent
            decision: Human decision - must be "approve" or "reject"
            reviewer_notes: Optional notes from the human reviewer
        
        Returns:
            Approval record with status, timestamp, and notes
        
        Raises:
            ValueError: If inputs are invalid or plan already approved/rejected
        """
        # Validate inputs
        self._validate_plan_schema(plan)
        self._validate_decision(decision)
        self._ensure_proposal_only(plan)
        
        plan_id = plan.get("plan_id")
        
        # Prevent re-approval
        if plan_id in self._approval_records:
            existing_status = self._approval_records[plan_id]["status"]
            if existing_status != ApprovalStatus.PENDING.value:
                raise ValueError(
                    f"Plan {plan_id} already has decision: {existing_status}. "
                    "Cannot re-approve or re-reject."
                )
        
        # Normalize decision
        normalized_decision = decision.lower()
        status = (
            ApprovalStatus.APPROVED.value
            if normalized_decision == "approve"
            else ApprovalStatus.REJECTED.value
        )
        
        # Create immutable approval record
        approval_record = {
            "plan_id": plan_id,
            "status": status,
            "reviewed_by": "human",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "notes": reviewer_notes or ""
        }
        
        # Store record
        self._approval_records[plan_id] = approval_record
        
        # Persist if storage configured
        if self._storage_path:
            self._save_records()
        
        return approval_record
    
    def get_approval_status(self, plan_id: str) -> str:
        """
        Get the current approval status for a plan.
        
        Args:
            plan_id: The plan identifier
        
        Returns:
            Approval status: PENDING, APPROVED, or REJECTED
        """
        if plan_id not in self._approval_records:
            return ApprovalStatus.PENDING.value
        
        return self._approval_records[plan_id]["status"]
    
    def get_approval_record(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the approval record for a plan.
        
        Args:
            plan_id: The plan identifier
        
        Returns:
            Full approval record if exists, None otherwise
        """
        return self._approval_records.get(plan_id)
    
    def list_approved_plans(self) -> List[str]:
        """
        Get list of all approved plan IDs.
        
        Returns:
            List of plan IDs with APPROVED status
        """
        return sorted([
            plan_id
            for plan_id, record in self._approval_records.items()
            if record["status"] == ApprovalStatus.APPROVED.value
        ])
    
    def list_rejected_plans(self) -> List[str]:
        """
        Get list of all rejected plan IDs.
        
        Returns:
            List of plan IDs with REJECTED status
        """
        return sorted([
            plan_id
            for plan_id, record in self._approval_records.items()
            if record["status"] == ApprovalStatus.REJECTED.value
        ])
    
    def clear_records(self) -> None:
        """
        Clear all approval records (for testing/reset only).
        
        WARNING: This operation is destructive.
        """
        self._approval_records.clear()
        if self._storage_path and self._storage_path.exists():
            self._storage_path.unlink()
    
    # =====================================================================
    # VALIDATION METHODS (STRICT SCHEMA ENFORCEMENT)
    # =====================================================================
    
    def _validate_plan_schema(self, plan: Dict[str, Any]) -> None:
        """
        Validate that plan has required structure.
        
        Raises:
            ValueError: If plan structure is invalid
        """
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a dictionary")
        
        # Required fields
        required_fields = ["plan_id", "applicable_issue_patterns", "proposed_transformations"]
        for field in required_fields:
            if field not in plan:
                raise ValueError(f"Plan missing required field: {field}")
        
        # Validate plan_id format
        plan_id = plan.get("plan_id")
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("plan_id must be a non-empty string")
        
        # Validate applicable_issue_patterns
        patterns = plan.get("applicable_issue_patterns")
        if not isinstance(patterns, list):
            raise ValueError("applicable_issue_patterns must be a list")
        
        # Validate proposed_transformations
        transformations = plan.get("proposed_transformations")
        if not isinstance(transformations, list):
            raise ValueError("proposed_transformations must be a list")
        
        for idx, transformation in enumerate(transformations):
            if not isinstance(transformation, dict):
                raise ValueError(
                    f"Transformation at index {idx} must be a dictionary"
                )
            
            required_transform_fields = ["transformation", "target_columns", "rationale"]
            for field in required_transform_fields:
                if field not in transformation:
                    raise ValueError(
                        f"Transformation at index {idx} missing field: {field}"
                    )
    
    def _validate_decision(self, decision: str) -> None:
        """
        Validate that decision is valid.
        
        Raises:
            ValueError: If decision is not "approve" or "reject"
        """
        if not isinstance(decision, str):
            raise ValueError("Decision must be a string")
        
        normalized = decision.lower()
        if normalized not in ["approve", "reject"]:
            raise ValueError(
                f"Decision must be 'approve' or 'reject', got: {decision}"
            )
    
    def _ensure_proposal_only(self, plan: Dict[str, Any]) -> None:
        """
        Ensure plan is proposal-only with no execution flags.
        
        Raises:
            ValueError: If plan contains execution indicators
        """
        # Check for forbidden execution fields
        forbidden_fields = ["execute", "execution_status", "executed_at", "results"]
        for field in forbidden_fields:
            if field in plan:
                raise ValueError(
                    f"Plan contains execution field '{field}'. "
                    "Only proposal-only plans can be reviewed."
                )
        
        # Check transformations don't have execution flags
        transformations = plan.get("proposed_transformations", [])
        for idx, transformation in enumerate(transformations):
            if "executed" in transformation or "execution_result" in transformation:
                raise ValueError(
                    f"Transformation at index {idx} contains execution flags. "
                    "Only proposals can be reviewed."
                )
    
    # =====================================================================
    # STORAGE METHODS (SWAPPABLE IMPLEMENTATION)
    # =====================================================================
    
    def _load_records(self) -> None:
        """Load approval records from JSON file."""
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                self._approval_records = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # If file is corrupted or unreadable, start fresh
            self._approval_records = {}
    
    def _save_records(self) -> None:
        """Persist approval records to JSON file."""
        if not self._storage_path:
            return
        
        # Ensure directory exists
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write atomically (write to temp, then rename)
        temp_path = self._storage_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self._approval_records, f, indent=2, ensure_ascii=False)
        
        temp_path.replace(self._storage_path)


# =====================================================================
# CONVENIENCE FUNCTION
# =====================================================================

def review_transformation_plan(
    plan: Dict[str, Any],
    decision: str,
    reviewer_notes: Optional[str] = None,
    storage_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to review a transformation plan.
    
    Args:
        plan: Transformation plan from TransformationPlannerAgent
        decision: Human decision - "approve" or "reject"
        reviewer_notes: Optional reviewer notes
        storage_path: Optional path for persistent storage
    
    Returns:
        Approval record
    """
    gate = PlanReviewAndApprovalGate(storage_path=storage_path)
    return gate.review_plan(plan, decision, reviewer_notes)
