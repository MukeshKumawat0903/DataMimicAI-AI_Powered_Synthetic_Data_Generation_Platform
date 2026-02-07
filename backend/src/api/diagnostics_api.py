"""
Diagnostics API - FastAPI Endpoints for Diagnostic Interpretation.

This module provides REST API endpoints for the DiagnosticsInterpreterAgent.
It exposes read-only diagnostic reasoning without actions or recommendations.

Endpoints:
- POST /api/diagnostics/interpret - Interpret diagnostics to identify patterns

Author: DataMimicAI Team
Date: February 6, 2026
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
import logging

from src.core.ai_assistance.agents.diagnostics_interpreter_agent import DiagnosticsInterpreterAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


@router.post("/interpret")
async def interpret_diagnostics(
    diagnostics_input: Dict[str, Any] = Body(
        ...,
        description="Diagnostics output from diagnostics_builder",
        example={
            "diagnostics": [
                {
                    "issue_type": "high_skew",
                    "severity": "high",
                    "column": "Volume",
                    "metrics": {"skewness": 6.9051}
                }
            ],
            "summary": {"total_issues": 1, "high_severity_count": 1},
            "metadata": {"timestamp": "2026-02-06"}
        }
    ),
    rag_context: Optional[str] = Body(
        None,
        description="Optional RAG context for pattern naming consistency"
    )
) -> Dict[str, Any]:
    """
    Interpret diagnostics to identify cross-cutting patterns and assess stability.
    
    This endpoint provides READ-ONLY interpretation of diagnostics. It does NOT:
    - Suggest actions or transformations
    - Execute any changes
    - Call LLMs (only uses RAG context if provided)
    - Persist any data
    
    Parameters
    ----------
    diagnostics_input : Dict[str, Any]
        Diagnostics output from diagnostics_builder containing:
        - diagnostics: List of detected issues
        - summary: Summary statistics
        - metadata: Tracking information
    
    rag_context : str, optional
        Pre-retrieved RAG context for pattern naming consistency.
        Agent works without RAG; RAG does not drive logic.
    
    Returns
    -------
    Dict[str, Any]
        Interpretation result containing:
        - interpretation: Structured interpretation
          - overall_assessment: Stability assessment (stable/moderately_unstable/unstable)
          - dominant_issue_patterns: List of identified patterns
          - supporting_evidence: Evidence for each pattern
          - confidence: Confidence level (high/medium/low)
    
    Raises
    ------
    HTTPException 400
        If diagnostics_input is missing required fields or has invalid structure
    HTTPException 500
        For unexpected errors during interpretation
    
    Examples
    --------
    POST /api/diagnostics/interpret
    
    Request:
    {
        "diagnostics_input": {
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
        },
        "rag_context": null
    }
    
    Response:
    {
        "interpretation": {
            "overall_assessment": "moderately_unstable",
            "dominant_issue_patterns": ["skew_and_outliers"],
            "supporting_evidence": [
                {
                    "pattern": "skew_and_outliers",
                    "columns": ["Volume"],
                    "severity": "high",
                    "description": "Columns exhibit both skewness and outlier presence"
                }
            ],
            "confidence": "high"
        }
    }
    """
    try:
        # Validate diagnostics_input structure
        if not isinstance(diagnostics_input, dict):
            logger.warning("Invalid diagnostics_input: not a dictionary")
            raise HTTPException(
                status_code=400,
                detail="diagnostics_input must be a dictionary"
            )
        
        if "diagnostics" not in diagnostics_input:
            logger.warning("Invalid diagnostics_input: missing 'diagnostics' field")
            raise HTTPException(
                status_code=400,
                detail="diagnostics_input must contain 'diagnostics' field"
            )
        
        if not isinstance(diagnostics_input.get("diagnostics"), list):
            logger.warning("Invalid diagnostics_input: 'diagnostics' is not a list")
            raise HTTPException(
                status_code=400,
                detail="diagnostics_input['diagnostics'] must be a list"
            )
        
        # Log request
        num_diagnostics = len(diagnostics_input.get("diagnostics", []))
        logger.info(f"Interpreting diagnostics with {num_diagnostics} issues")
        
        # Instantiate agent (stateless, no side effects)
        agent = DiagnosticsInterpreterAgent(rag_context=rag_context)
        
        # Perform interpretation (deterministic, read-only)
        result = agent.interpret(diagnostics_input)
        
        # Convert to dictionary for JSON response
        response = result.to_dict()
        
        logger.info(
            f"Interpretation complete: "
            f"assessment={result.overall_assessment}, "
            f"patterns={len(result.dominant_issue_patterns)}"
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except ValueError as e:
        # Agent validation errors
        logger.error(f"Validation error in diagnostics interpretation: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid diagnostics input: {str(e)}"
        )
    
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in diagnostics interpretation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during interpretation: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for diagnostics API.
    
    Returns
    -------
    Dict[str, str]
        Status message indicating the API is operational
    """
    return {
        "status": "healthy",
        "service": "diagnostics-api",
        "version": "1.0"
    }
