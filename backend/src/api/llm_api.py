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
    get_validation_report,
    build_diagnostics,
    build_diagnostics_context_for_prompt
)

# Baseline guard for protecting existing behavior
try:
    from src.core.LLM.llm_explainability_engine.baseline_guard import (
        assert_signals_structure,
        assert_context_matches_signals,
        enable_baseline_guard,
        is_baseline_guard_enabled
    )
    _baseline_guard_available = True
except ImportError:
    _baseline_guard_available = False
    logger.warning("Baseline guard not available - skipping guard checks")

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
        description="Analysis scope (diagnostics_overview, dataset_overview, column_analysis, correlation_analysis, outlier_analysis, time_series_analysis)"
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
    use_rag: bool = Field(
        default=False,
        description="Whether to augment explanation with RAG knowledge (optional)"
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
            "diagnostics_overview",
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
            
            # BASELINE GUARD: Validate signals structure at API level
            # This is an additional safety check on top of internal guards
            if _baseline_guard_available and is_baseline_guard_enabled():
                try:
                    assert_signals_structure(signals)
                    logger.debug("[BASELINE] API-level signals validation passed")
                except Exception as e:
                    logger.warning(f"[BASELINE] API-level signals validation failed: {e}")
        except Exception as e:
            logger.error(f"Error in signal extraction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting signals: {str(e)}"
            )
        
        # STEP 1.5: Build diagnostics from signals
        logger.info("STEP 1.5: Building diagnostics from signals")
        try:
            # Apply column filtering if specified by user
            target_columns = request.columns if request.columns else None
            diagnostics = build_diagnostics(signals, target_columns=target_columns)
            logger.debug(f"Built diagnostics with {diagnostics['summary']['total_issues']} issues")
        except Exception as e:
            logger.error(f"Error in diagnostics building: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error building diagnostics: {str(e)}"
            )
        
        # STEP 2: Build context from diagnostics (NEW ARCHITECTURE)
        # The Explain feature now consumes structured diagnostics instead of raw signals
        logger.info("STEP 2: Building context from diagnostics")
        try:
            # Pass scope to maintain user's analysis intent
            context = build_diagnostics_context_for_prompt(diagnostics, scope=request.scope)
            
            # BASELINE GUARD: Validate context matches signals at API level
            # This ensures end-to-end data flow consistency
            if _baseline_guard_available and is_baseline_guard_enabled():
                try:
                    assert_context_matches_signals(context, signals)
                    logger.debug("[BASELINE] API-level context validation passed")
                except Exception as e:
                    logger.warning(f"[BASELINE] API-level context validation failed: {e}")
        except Exception as e:
            logger.error(f"Error in context building: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error building context: {str(e)}"
            )
        
        # STEP 3 (Optional): Retrieve RAG knowledge if requested
        rag_context = None
        if request.use_rag:
            logger.info("STEP 3: Retrieving RAG knowledge for diagnostics")
            try:
                from src.core.LLM.rag import augment_diagnostics_with_rules
                
                # Augment diagnostics with relevant knowledge rules
                diagnostics_with_rag = augment_diagnostics_with_rules(diagnostics)
                rag_context = diagnostics_with_rag.get("rag_context")
                
                if rag_context:
                    total_rules = rag_context["retrieval_metadata"]["total_rules_retrieved"]
                    logger.info(f"Retrieved {total_rules} relevant knowledge rules")
                else:
                    logger.warning("RAG requested but no rules retrieved")
            except Exception as e:
                # RAG is optional - continue without it if it fails
                logger.warning(f"RAG retrieval failed (continuing without RAG): {str(e)}")
                rag_context = None
        
        # STEP 4: Build prompt (with optional RAG context)
        logger.info(f"STEP 4: Building prompt with tone={request.tone}, use_rag={request.use_rag}")
        try:
            prompt = build_explanation_prompt(context, rag_context=rag_context, tone=request.tone)
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
        
        response_metadata = {
            "file_id": request.file_id,
            "scope": context.get("scope", "diagnostics_overview"),
            "tone": request.tone,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "num_columns_analyzed": diagnostics["metadata"].get("num_columns_analyzed", 0),
            "total_issues_detected": diagnostics["summary"]["total_issues"],
            "high_severity_issues": diagnostics["summary"]["high_severity_count"],
            "medium_severity_issues": diagnostics["summary"]["medium_severity_count"],
            "low_severity_issues": diagnostics["summary"]["low_severity_count"],
            "raw_explanation_length": len(raw_explanation),
            "validated_explanation_length": len(validated_explanation)
        }
        
        # Include baseline guard status if available
        if _baseline_guard_available:
            response_metadata["baseline_guard_enabled"] = is_baseline_guard_enabled()
        
        return ExplanationResponse(
            explanation=validated_explanation,
            validated=is_validated,
            validation_report=validation_report,
            metadata=response_metadata
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
