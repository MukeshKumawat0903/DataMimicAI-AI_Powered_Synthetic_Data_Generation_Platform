"""
Diagnostics API - FastAPI Endpoints for Diagnostic Analysis and Interpretation.

This module provides REST API endpoints for diagnostics building and interpretation.
It exposes read-only diagnostic reasoning without actions or recommendations.

Endpoints:
- POST /api/diagnostics/build - Build diagnostics from dataset
- POST /api/diagnostics/interpret - Interpret diagnostics to identify patterns

Author: DataMimicAI Team
Date: February 6, 2026
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
import logging
import numpy as np

# LAZY: Agent import moved inside endpoint function
# from src.core.ai_assistance.agents.diagnostics_interpreter_agent import DiagnosticsInterpreterAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/diagnostics", tags=["diagnostics"])


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


@router.post("/build")
async def build_diagnostics_endpoint(
    file_id: str = Body(..., description="File identifier for the dataset"),
    target_columns: Optional[list] = Body(None, description="Specific columns to analyze (None = all)")
) -> Dict[str, Any]:
    """
    Build diagnostics from a dataset file.
    
    This endpoint analyzes a dataset and produces structured diagnostics
    with detected issues and severity classifications.
    
    Parameters
    ----------
    file_id : str
        The file identifier for the uploaded dataset
    target_columns : list, optional
        Specific columns to analyze. If None, analyzes all columns.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - diagnostics: Full diagnostics structure with issues
        - summary: Summary of issues by severity
    
    Raises
    ------
    HTTPException 404
        If file not found
    HTTPException 500
        For unexpected errors during diagnostics building
    """
    try:
        import os
        import pandas as pd
        
        # LAZY: Import diagnostics builder only when needed
        from src.core.ai_assistance.explainability.llm_explainability_engine.explainable_signals import (
            build_explainable_signals
        )
        from src.core.ai_assistance.explainability.llm_explainability_engine.diagnostics_builder import (
            build_diagnostics
        )
        
        # Load dataset
        upload_dir = os.getenv("UPLOAD_DIR", "uploads")
        file_path = os.path.join(upload_dir, f"{file_id}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {file_id}")
        
        logger.info(f"Building diagnostics for file: {file_id}")
        df = pd.read_csv(file_path)
        
        # Build signals
        signals = build_explainable_signals(df)
        
        # Build diagnostics from signals
        diagnostics = build_diagnostics(signals, target_columns=target_columns)
        
        logger.info(
            f"Diagnostics built: {diagnostics['summary']['total_issues']} issues found"
        )
        
        # Convert numpy types to Python native types for JSON serialization
        result = {
            "diagnostics": diagnostics,
            "signals": signals  # Include signals for transformation filtering
        }
        
        return convert_numpy_types(result)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error building diagnostics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error building diagnostics: {str(e)}"
        )


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
                    "column": "<your_numeric_column>",
                    "metrics": {"skewness": 3.75}
                }
            ],
            "summary": {"total_issues": 1, "high_severity_count": 1},
            "metadata": {"timestamp": "2026-02-07"}
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
                    "column": "<your_column_A>",
                    "metrics": {"skewness": 3.75}
                },
                {
                    "issue_type": "outliers",
                    "severity": "high",
                    "column": "<your_column_A>",
                    "metrics": {"outlier_percentage": 13.8}
                }
            ],
            "summary": {"total_issues": 2, "high_severity_count": 2},
            "metadata": {"timestamp": "2026-02-07"}
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
                    "columns": ["<your_column_A>"],
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
        # LAZY: Import agent only when endpoint is called (not at startup)
        from src.core.ai_assistance.agents.diagnostics_interpreter_agent import DiagnosticsInterpreterAgent
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
        
        return convert_numpy_types(response)
    
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
