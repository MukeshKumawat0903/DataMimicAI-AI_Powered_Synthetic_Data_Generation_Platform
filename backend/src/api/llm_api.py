# src/api/llm_api.py

"""
LLM API - FastAPI endpoints for AI-powered data explanations.

This module provides REST API endpoints for the LLM explainability pipeline.
It follows the same pattern as other API modules in the project.
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
import pandas as pd
import logging
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from src.core.LLM import (
    build_explainable_signals,
    select_explainable_context,
    build_explanation_prompt,
    run_llama_explanation,
    validate_llm_output,
    get_validation_report
)

logger = logging.getLogger(__name__)

# Resolve UPLOAD_DIR to absolute path, defaulting to backend/uploads
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", None)
if not UPLOAD_DIR:
    current_file = os.path.abspath(__file__)
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    UPLOAD_DIR = os.path.join(backend_dir, "uploads")
else:
    UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"LLM API using UPLOAD_DIR: {UPLOAD_DIR}")

router = APIRouter(prefix="/llm")


def load_df_by_file_id(file_id: str) -> pd.DataFrame:
    """Load DataFrame from file_id in the upload directory."""
    path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return pd.read_csv(path)


# Request/Response Models
class ExplanationRequest(BaseModel):
    """Request model for explanation generation."""
    file_id: str = Field(..., description="File identifier for the uploaded dataset")
    scope: str = Field(
        default="dataset_overview",
        description="Analysis scope (dataset_overview, column_analysis, correlation_analysis, outlier_analysis, time_series_analysis)"
    )
    tone: str = Field(
        default="clear",
        description="Explanation tone (clear, concise, technical, beginner-friendly, detailed)"
    )
    columns: Optional[list] = Field(
        default=None,
        description="Specific columns to analyze (for column_analysis scope)"
    )
    max_tokens: int = Field(
        default=1500,
        description="Maximum tokens for LLM generation"
    )


class ExplanationResponse(BaseModel):
    """Response model for explanation generation."""
    explanation: str = Field(..., description="Generated explanation text")
    validated: bool = Field(..., description="Whether the explanation passed validation")
    validation_report: Dict[str, Any] = Field(..., description="Validation details")
    metadata: Dict[str, Any] = Field(..., description="Pipeline metadata")


@router.post("/explain", response_model=ExplanationResponse)
async def generate_explanation(request: ExplanationRequest = Body(...)):
    """
    Generate AI-powered data explanation using the LLM pipeline.
    
    This endpoint orchestrates the complete LLM explainability pipeline:
    1. STEP 1: Extract explainable signals from data
    2. STEP 2: Select and scope signals for focused context
    3. STEP 4: Build safe, structured prompt
    4. STEP 5: Run LLaMA inference via Groq
    5. STEP 6: Validate output against source facts
    
    Parameters
    ----------
    request : ExplanationRequest
        Request containing file_id, scope, tone, and optional parameters
    
    Returns
    -------
    ExplanationResponse
        Generated explanation with validation details
    
    Raises
    ------
    HTTPException
        - 404: File not found
        - 400: Invalid scope or parameters
        - 500: Pipeline execution error
    """
    try:
        logger.info(f"Generating explanation for file_id={request.file_id}, scope={request.scope}")
        
        # Load dataset
        try:
            df = load_df_by_file_id(request.file_id)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")
        
        # Validate scope
        valid_scopes = [
            "dataset_overview",
            "column_analysis",
            "correlation_analysis",
            "outlier_analysis",
            "time_series_analysis"
        ]
        if request.scope not in valid_scopes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scope. Must be one of: {', '.join(valid_scopes)}"
            )
        
        # Validate tone
        valid_tones = ["clear", "concise", "technical", "beginner-friendly", "detailed"]
        if request.tone not in valid_tones:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tone. Must be one of: {', '.join(valid_tones)}"
            )
        
        # STEP 1: Extract explainable signals
        logger.info("STEP 1: Extracting explainable signals")
        try:
            signals = build_explainable_signals(df)
        except Exception as e:
            logger.error(f"Error in signal extraction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting signals: {str(e)}"
            )
        
        # STEP 2: Select scoped context
        logger.info(f"STEP 2: Selecting context for scope={request.scope}")
        try:
            context_params = {"signals": signals, "scope": request.scope}
            if request.columns and request.scope == "column_analysis":
                context_params["columns"] = request.columns
            
            context = select_explainable_context(**context_params)
        except Exception as e:
            logger.error(f"Error in context selection: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error selecting context: {str(e)}"
            )
        
        # STEP 4: Build prompt
        logger.info(f"STEP 4: Building prompt with tone={request.tone}")
        try:
            prompt = build_explanation_prompt(context, tone=request.tone)
        except Exception as e:
            logger.error(f"Error in prompt building: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error building prompt: {str(e)}"
            )
        
        # STEP 5: Run LLaMA inference
        logger.info(f"STEP 5: Running LLaMA inference (max_tokens={request.max_tokens})")
        try:
            raw_explanation = run_llama_explanation(
                prompt,
                max_tokens=request.max_tokens
            )
        except Exception as e:
            logger.error(f"Error in LLM inference: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating explanation: {str(e)}"
            )
        
        # STEP 6: Validate output
        logger.info("STEP 6: Validating LLM output")
        try:
            validated_explanation = validate_llm_output(
                raw_explanation,
                context,
                max_length=3000
            )
            validation_report = get_validation_report(raw_explanation, context)
        except Exception as e:
            logger.error(f"Error in output validation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error validating output: {str(e)}"
            )
        
        # Check if validation failed (fallback message)
        fallback_msg = "The analysis highlights notable patterns in the data, but the explanation could not be confidently validated."
        is_validated = validated_explanation.strip() != fallback_msg
        
        logger.info(f"Pipeline completed. Validated: {is_validated}")
        
        return ExplanationResponse(
            explanation=validated_explanation,
            validated=is_validated,
            validation_report=validation_report,
            metadata={
                "file_id": request.file_id,
                "scope": request.scope,
                "tone": request.tone,
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "raw_explanation_length": len(raw_explanation),
                "validated_explanation_length": len(validated_explanation)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LLM pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for LLM API.
    
    Returns
    -------
    dict
        Status information
    """
    return {
        "status": "healthy",
        "service": "LLM Explainability API",
        "upload_dir": UPLOAD_DIR,
        "upload_dir_exists": os.path.exists(UPLOAD_DIR)
    }


@router.get("/scopes")
async def get_available_scopes():
    """
    Get available analysis scopes.
    
    Returns
    -------
    dict
        Available scopes and their descriptions
    """
    return {
        "scopes": {
            "dataset_overview": {
                "description": "General overview of the entire dataset",
                "requires_columns": False
            },
            "column_analysis": {
                "description": "Detailed analysis of specific columns",
                "requires_columns": True
            },
            "correlation_analysis": {
                "description": "Analysis of correlations between variables",
                "requires_columns": False
            },
            "outlier_analysis": {
                "description": "Analysis of outliers and anomalies",
                "requires_columns": False
            },
            "time_series_analysis": {
                "description": "Analysis of temporal patterns",
                "requires_columns": False
            }
        },
        "tones": {
            "clear": "Accessible, straightforward explanations",
            "concise": "Brief, focused insights",
            "technical": "Precise statistical language",
            "beginner-friendly": "Explained for non-experts",
            "detailed": "Comprehensive, thorough analysis"
        }
    }
