"""
Planner API - FastAPI Endpoints for Transformation Planning.

This module provides REST API endpoints for the TransformationPlannerAgent.
It exposes proposal-only transformation planning without execution.

Endpoints:
- POST /api/planner/create-plan - Generate transformation plans

Author: DataMimicAI Team
Date: February 6, 2026
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
import logging
import numpy as np

# LAZY: Agent import moved inside endpoint function
# from src.core.ai_assistance.agents.transformation_planner_agent import TransformationPlannerAgent

from src.api.shared_state import submitted_plans

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/planner", tags=["planner"])


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    
    Handles numpy scalars (bool_, int64, float64, etc.), arrays, and nested structures.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.str_, str)):
        return str(obj)
    elif obj is None or isinstance(obj, (int, float, str)):
        return obj
    else:
        # For any other type, try to convert to string as fallback
        try:
            return str(obj)
        except:
            return obj


@router.post("/create-plan")
async def create_transformation_plan(
    diagnostics: Dict[str, Any] = Body(
        ...,
        description="Diagnostics output from diagnostics_builder",
        example={
            "diagnostics": [
                {
                    "issue_type": "high_skew",
                    "severity": "high",
                    "column": "<your_column_name>",
                    "metrics": {"skewness": 3.75}
                }
            ],
            "summary": {"total_issues": 1},
            "metadata": {}
        }
    ),
    interpretation: Dict[str, Any] = Body(
        ...,
        description="Interpretation output from DiagnosticsInterpreterAgent",
        example={
            "overall_assessment": "moderately_unstable",
            "dominant_issue_patterns": ["skew_and_outliers"],
            "supporting_evidence": [
                {
                    "pattern": "skew_and_outliers",
                    "columns": ["<your_column_name>"],
                    "severity": "high"
                }
            ],
            "confidence": "high"
        }
    ),
    rag_context: Optional[str] = Body(
        None,
        description="Optional RAG context for transformation naming standardization"
    ),
    signals: Optional[Dict[str, Any]] = Body(
        None,
        description="Optional signals from build_explainable_signals for column statistics (min/max)"
    )
) -> Dict[str, Any]:
    """
    Generate transformation plans based on diagnostics and interpretation.
    
    This endpoint provides PROPOSAL-ONLY transformation planning. It does NOT:
    - Execute transformations
    - Auto-approve plans
    - Persist data
    - Rank or recommend "best" plans
    
    Parameters
    ----------
    diagnostics : Dict[str, Any]
        Diagnostics output from diagnostics_builder containing:
        - diagnostics: List of detected issues
        - summary: Summary statistics
        - metadata: Tracking information
    
    interpretation : Dict[str, Any]
        Interpretation output from DiagnosticsInterpreterAgent containing:
        - overall_assessment: Stability assessment
        - dominant_issue_patterns: Identified patterns
        - supporting_evidence: Evidence for patterns
        - confidence: Confidence level
    
    rag_context : str, optional
        Pre-retrieved RAG context for naming standardization.
        Agent works without RAG; RAG does not drive logic.
    
    Returns
    -------
    Dict[str, Any]
        Transformation plans containing:
        - proposed_plans: List of transformation plans
          - plan_id: Unique plan identifier
          - transformations: List of proposed transformations
          - rationale: Explanation for the plan
          - estimated_risks: Identified risks
        - confidence: Overall confidence level
    
    Raises
    ------
    HTTPException 400
        If diagnostics or interpretation input is missing required fields
    HTTPException 500
        For unexpected errors during planning
    
    Examples
    --------
    POST /api/planner/create-plan
    
    Request:
    {
        "diagnostics": {
            "diagnostics": [
                {
                    "issue_type": "high_skew",
                    "severity": "high",
                    "column": "<your_column_name>",
                    "metrics": {"skewness": 3.75}
                }
            ],
            "summary": {"total_issues": 1},
            "metadata": {}
        },
        "interpretation": {
            "overall_assessment": "moderately_unstable",
            "dominant_issue_patterns": ["skew_and_outliers"],
            "supporting_evidence": [...],
            "confidence": "high"
        },
        "rag_context": null
    }
    
    Response:
    {
        "proposed_plans": [
            {
                "plan_id": "TP-001",
                "transformations": [
                    {
                        "transformation_type": "log_transform",
                        "target_columns": ["<your_column_name>"],
                        "parameters": {"base": "natural"},
                        "purpose": "reduce_skewness"
                    }
                ],
                "rationale": "Apply log transformation to reduce skewness...",
                "estimated_risks": ["May introduce NaN for zero/negative values"]
            }
        ],
        "confidence": "high"
    }
    """
    try:
        # Validate diagnostics structure
        if not isinstance(diagnostics, dict):
            logger.warning("Invalid diagnostics: not a dictionary")
            raise HTTPException(
                status_code=400,
                detail="diagnostics must be a dictionary"
            )
        
        if "diagnostics" not in diagnostics:
            logger.warning("Invalid diagnostics: missing 'diagnostics' field")
            raise HTTPException(
                status_code=400,
                detail="diagnostics must contain 'diagnostics' field"
            )
        
        # Validate interpretation structure
        if not isinstance(interpretation, dict):
            logger.warning("Invalid interpretation: not a dictionary")
            raise HTTPException(
                status_code=400,
                detail="interpretation must be a dictionary"
            )
        
        required_interpretation_fields = [
            "overall_assessment",
            "dominant_issue_patterns",
            "supporting_evidence",
            "confidence"
        ]
        
        for field in required_interpretation_fields:
            if field not in interpretation:
                logger.warning(f"Invalid interpretation: missing '{field}' field")
                raise HTTPException(
                    status_code=400,
                    detail=f"interpretation must contain '{field}' field"
                )
        
        # Validate dominant_issue_patterns is a list
        if not isinstance(interpretation.get("dominant_issue_patterns"), list):
            logger.warning("Invalid interpretation: 'dominant_issue_patterns' is not a list")
            raise HTTPException(
                status_code=400,
                detail="interpretation['dominant_issue_patterns'] must be a list"
            )
        
        # Log request
        num_diagnostics = len(diagnostics.get("diagnostics", []))
        num_patterns = len(interpretation.get("dominant_issue_patterns", []))
        logger.info(
            f"Creating transformation plan: "
            f"{num_diagnostics} diagnostics, {num_patterns} patterns"
        )
        
        # Build planner input
        planner_input = {
            "diagnostics": diagnostics,
            "interpretation": interpretation
        }
        
        # Include signals if provided (for column min/max filtering)
        if signals:
            planner_input["signals"] = signals
                # LAZY: Import agent only when endpoint is called (not at startup)
        from src.core.ai_assistance.agents.transformation_planner_agent import TransformationPlannerAgent
                # Instantiate agent (stateless, no side effects)
        agent = TransformationPlannerAgent(rag_context=rag_context)
        
        # Generate plans (deterministic, proposal-only)
        result = agent.plan(planner_input)
        
        # Convert to dictionary for JSON response
        response = result.to_dict()
        
        logger.info(
            f"Planning complete: "
            f"{len(result.proposed_plans)} plan(s) proposed, "
            f"confidence={result.confidence}"
        )
        
        return convert_numpy_types(response)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except ValueError as e:
        # Agent validation errors
        logger.error(f"Validation error in transformation planning: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid planner input: {str(e)}"
        )
    
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in transformation planning: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during planning: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for planner API.
    
    Returns
    -------
    Dict[str, str]
        Status message indicating the API is operational
    """
    return {
        "status": "healthy",
        "service": "planner-api",
        "version": "1.0"
    }


@router.get("/transformation-vocabulary")
async def get_transformation_vocabulary() -> Dict[str, Any]:
    """
    Get the supported transformation vocabulary.
    
    This endpoint returns the list of transformation types that the
    planner can propose. Useful for validation and UI building.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - transformation_types: List of supported transformation types
        - description: Brief description of each type
    
    Examples
    --------
    GET /api/planner/transformation-vocabulary
    
    Response:
    {
        "transformation_types": [
            {
                "type": "log_transform",
                "purpose": "reduce_skewness",
                "description": "Apply logarithmic transformation"
            },
            {
                "type": "winsorize",
                "purpose": "handle_outliers",
                "description": "Cap extreme values at percentiles"
            }
        ]
    }
    """
    # Define supported transformation vocabulary
    # This should match what the agent can propose
    vocabulary = {
        "transformation_types": [
            {
                "type": "log_transform",
                "purpose": "reduce_skewness",
                "description": "Apply logarithmic transformation to reduce skewness",
                "parameters": ["base"]
            },
            {
                "type": "sqrt_transform",
                "purpose": "reduce_skewness",
                "description": "Apply square root transformation to reduce skewness",
                "parameters": []
            },
            {
                "type": "box_cox",
                "purpose": "reduce_skewness",
                "description": "Apply Box-Cox transformation to normalize distribution",
                "parameters": ["lambda"]
            },
            {
                "type": "winsorize",
                "purpose": "handle_outliers",
                "description": "Cap extreme values at specified percentiles",
                "parameters": ["lower_percentile", "upper_percentile"]
            },
            {
                "type": "clip",
                "purpose": "handle_outliers",
                "description": "Clip values to specified range",
                "parameters": ["lower_bound", "upper_bound"]
            },
            {
                "type": "z_score_removal",
                "purpose": "handle_outliers",
                "description": "Remove rows with extreme z-scores",
                "parameters": ["threshold"]
            },
            {
                "type": "drop_column",
                "purpose": "reduce_redundancy",
                "description": "Remove redundant or low-value columns",
                "parameters": []
            },
            {
                "type": "impute_mean",
                "purpose": "handle_missing",
                "description": "Impute missing values with column mean",
                "parameters": []
            },
            {
                "type": "impute_median",
                "purpose": "handle_missing",
                "description": "Impute missing values with column median",
                "parameters": []
            },
            {
                "type": "impute_mode",
                "purpose": "handle_missing",
                "description": "Impute missing values with column mode",
                "parameters": []
            },
            {
                "type": "forward_fill",
                "purpose": "handle_missing",
                "description": "Forward fill missing values (time series)",
                "parameters": []
            }
        ],
        "total_types": 11
    }
    
    logger.info("Transformation vocabulary requested")
    return vocabulary
