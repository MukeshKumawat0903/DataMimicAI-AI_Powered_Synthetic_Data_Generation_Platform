"""
Execution API - FastAPI Endpoints for Deterministic Plan Execution.

This module provides REST API endpoints for executing approved transformation plans.
It triggers execution and validation in a safe, controlled manner.

Endpoints:
- POST /api/execution/execute-plan - Execute an approved transformation plan

Author: DataMimicAI Team
Date: February 6, 2026
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
import logging
import os
import pandas as pd

from src.core.execution.deterministic_execution_engine import (
    DeterministicExecutionEngine,
    ExecutionStatus
)
from src.core.validation.validation_feedback_loop import ValidationFeedbackLoop
from src.core.validation.validation_storage import get_validation_store
from src.core.ai_assistance.approval.plan_review_gate import ApprovalStatus
from src.api.shared_state import approval_gate, submitted_plans

logger = logging.getLogger(__name__)

# Use shared state to avoid circular imports
_approval_gate = approval_gate
_submitted_plans = submitted_plans

router = APIRouter(prefix="/api/execution", tags=["execution"])

# Upload directory for loading data files
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")


def load_df_by_file_id(file_id: str) -> pd.DataFrame:
    """
    Load DataFrame from file_id in the upload directory.
    
    Args:
        file_id: File identifier
    
    Returns:
        DataFrame
    
    Raises:
        HTTPException: If file not found
    """
    path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(404, f"File not found: {file_id}")
    return pd.read_csv(path)


@router.post("/execute-plan")
async def execute_plan(
    plan_id: str = Body(
        ...,
        description="Plan identifier to execute",
        example="TP-001"
    ),
    file_id: str = Body(
        ...,
        description="File identifier for the dataset to transform",
        example="12345"
    )
) -> Dict[str, Any]:
    """
    Execute an APPROVED transformation plan.
    
    This endpoint provides SAFE EXECUTION of approved plans. It does NOT:
    - Execute unapproved plans
    - Make decisions or inferences
    - Call LLMs or agents
    - Modify the approval status
    
    Workflow:
    1. Verify plan exists and is APPROVED
    2. Load the dataset
    3. Execute transformations deterministically
    4. Run validation feedback loop (before/after comparison)
    5. Store validation results
    6. Return execution status and validation availability
    
    Parameters
    ----------
    plan_id : str
        Plan identifier (must be APPROVED)
    
    file_id : str
        File identifier for the dataset to transform
    
    Returns
    -------
    Dict[str, Any]
        Execution result containing:
        - plan_id: Plan identifier
        - execution_status: SUCCESS or FAILED
        - applied_transformations: List of transformations applied
        - validation_available: Whether validation results are available
        - validation_plan_id: Plan ID for retrieving validation results
        - error: Error message if execution failed
    
    Raises
    ------
    HTTPException 403
        If plan is not APPROVED (PENDING or REJECTED)
    HTTPException 404
        If plan not found or file not found
    HTTPException 500
        For execution failures
    
    Examples
    --------
    POST /api/execution/execute-plan
    
    Request:
    {
        "plan_id": "TP-001",
        "file_id": "12345"
    }
    
    Response (Success):
    {
        "plan_id": "TP-001",
        "execution_status": "SUCCESS",
        "applied_transformations": ["log_transform", "winsorization"],
        "validation_available": true,
        "validation_plan_id": "TP-001",
        "error": null
    }
    
    Response (Not Approved):
    HTTP 403 Forbidden
    {
        "detail": "Plan TP-001 is not approved. Current status: PENDING. Only APPROVED plans can be executed."
    }
    """
    try:
        logger.info(f"Executing plan {plan_id} on file {file_id}")
        
        # STEP 1: Verify plan exists in submitted plans
        if plan_id not in _submitted_plans:
            logger.warning(f"Plan {plan_id} not found in submitted plans")
            raise HTTPException(
                status_code=404,
                detail=f"Plan not found: {plan_id}. Plan must be submitted through /api/planner/create-plan before execution."
            )
        
        # STEP 2: Get approval status
        approval_status = _approval_gate.get_approval_status(plan_id)
        
        # STEP 3: Verify plan is APPROVED
        if approval_status != ApprovalStatus.APPROVED.value:
            logger.warning(
                f"Plan {plan_id} is not approved. Status: {approval_status}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Plan {plan_id} is not approved. Current status: {approval_status}. "
                       f"Only APPROVED plans can be executed. Use /api/approval/review-plan to approve."
            )
        
        # STEP 4: Get approval record and plan
        approval_record = _approval_gate.get_approval_record(plan_id)
        plan = _submitted_plans[plan_id]
        
        logger.info(f"Plan {plan_id} is approved. Loading dataset...")
        
        # STEP 5: Load dataset
        try:
            original_data = load_df_by_file_id(file_id)
            logger.info(f"Loaded dataset: {original_data.shape[0]} rows, {original_data.shape[1]} columns")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error loading dataset {file_id}: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Error loading dataset {file_id}: {str(e)}"
            )
        
        # STEP 6: Execute transformations
        logger.info(f"Executing transformations for plan {plan_id}...")
        engine = DeterministicExecutionEngine()
        
        try:
            execution_result = engine.execute(
                approval_record=approval_record,
                plan=plan,
                data=original_data
            )
            
            logger.info(
                f"Execution complete: status={execution_result.execution_status}, "
                f"transformations={len(execution_result.applied_transformations)}"
            )
            
        except ValueError as e:
            # Execution precondition failures (invalid plan, data, etc.)
            logger.error(f"Execution validation error: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Execution validation error: {str(e)}"
            )
        except Exception as e:
            # Unexpected execution errors
            logger.error(f"Execution failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Execution failed: {str(e)}"
            )
        
        # STEP 7: Run validation feedback loop (if execution succeeded)
        validation_available = False
        validation_error = None
        
        if execution_result.execution_status == ExecutionStatus.SUCCESS.value:
            if execution_result.transformed_data is not None:
                logger.info(f"Running validation feedback loop for plan {plan_id}...")
                
                try:
                    validator = ValidationFeedbackLoop()
                    validation_report = validator.validate(
                        plan_id=plan_id,
                        original_data=original_data,
                        transformed_data=execution_result.transformed_data
                    )
                    
                    logger.info(
                        f"Validation complete: {validation_report.summary['metrics_compared']} metrics compared"
                    )
                    
                    # STEP 8: Store validation results
                    validation_store = get_validation_store()
                    validation_store.store_validation_result(
                        plan_id=plan_id,
                        validation_report_dict=validation_report.to_dict()
                    )
                    
                    validation_available = True
                    logger.info(f"Validation results stored for plan {plan_id}")
                    
                except Exception as e:
                    # Validation failed, but execution succeeded
                    # Don't fail the entire request
                    validation_error = str(e)
                    logger.warning(f"Validation failed (execution succeeded): {e}")
        
        # STEP 9: Build response
        response = {
            "plan_id": plan_id,
            "execution_status": execution_result.execution_status,
            "applied_transformations": execution_result.applied_transformations,
            "validation_available": validation_available,
            "validation_plan_id": plan_id if validation_available else None,
            "error": execution_result.error,
            "validation_error": validation_error
        }
        
        logger.info(f"Plan {plan_id} execution complete. Validation available: {validation_available}")
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error executing plan: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during plan execution: {str(e)}"
        )


@router.get("/execution-status/{plan_id}")
async def get_execution_status(plan_id: str) -> Dict[str, Any]:
    """
    Get execution status and validation availability for a plan.
    
    This is a simple status check that doesn't re-execute anything.
    
    Parameters
    ----------
    plan_id : str
        Plan identifier
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - plan_id: Plan identifier
        - approval_status: Current approval status
        - validation_available: Whether validation results exist
        - can_execute: Whether plan can be executed (is APPROVED)
    
    Examples
    --------
    GET /api/execution/execution-status/TP-001
    
    Response:
    {
        "plan_id": "TP-001",
        "approval_status": "APPROVED",
        "validation_available": true,
        "can_execute": true
    }
    """
    try:
        # Check if plan exists
        if plan_id not in _submitted_plans:
            raise HTTPException(
                status_code=404,
                detail=f"Plan not found: {plan_id}"
            )
        
        # Get approval status
        approval_status = _approval_gate.get_approval_status(plan_id)
        can_execute = (approval_status == ApprovalStatus.APPROVED.value)
        
        # Check if validation results exist
        validation_store = get_validation_store()
        validation_result = validation_store.get_validation_result(plan_id)
        validation_available = (validation_result is not None)
        
        return {
            "plan_id": plan_id,
            "approval_status": approval_status,
            "validation_available": validation_available,
            "can_execute": can_execute
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error checking execution status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error checking execution status: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for execution API.
    
    Returns
    -------
    Dict[str, str]
        Status message indicating the API is operational
    """
    return {
        "status": "healthy",
        "service": "execution-api",
        "version": "1.0"
    }
