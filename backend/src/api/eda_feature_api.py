# src/api/eda_feature_api.py

from fastapi import APIRouter, HTTPException, Request, Query, Body
from fastapi.responses import JSONResponse, Response
import pandas as pd
import numpy as np
import logging
import os

from src.core.eda.profiling import (
    Profiler, DataCleaner, EDAConfig, ProfilingError,
    DistributionAnalyzer, run_distribution_analysis
)
from src.core.eda.correlation import (
    CorrelationAnalyzer, CorrelationConfig, CorrelationError, make_corr_heatmap_base64,
    MutualInformationAnalyzer, compute_mi_matrix
)
from src.core.eda.outliers import (
    OutlierDetector, OutlierConfig, OutlierDetectionError, OutlierCleaner
)
from src.core.eda.drift import (
    DriftDetector, DriftConfig, DriftDetectionError
)
from src.core.feature_engineering.feature_suggester import (
    FeatureSuggester, FeatureEngConfig, FeatureEngError
)
from src.core.eda.utils import outlier_drift_report_pdf
from src.core.eda.dependencies import DependencyDetector
from src.core.eda.privacy import KAnonymityAnalyzer
from src.core.eda.pii_scan import PIIScanner, run_pii_scan_fast, run_pii_scan_deep

logger = logging.getLogger(__name__)

# Resolve UPLOAD_DIR to absolute path, defaulting to backend/uploads
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", None)
if not UPLOAD_DIR:
    # If not set, use backend/uploads relative to this file's location
    current_file = os.path.abspath(__file__)
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # Go up 3 levels: api -> src -> backend
    UPLOAD_DIR = os.path.join(backend_dir, "uploads")
else:
    UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"EDA API using UPLOAD_DIR: {UPLOAD_DIR}")

router = APIRouter(prefix="/eda")


def load_df_by_file_id(file_id: str) -> pd.DataFrame:
    """Load DataFrame from file_id in the upload directory."""
    path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return pd.read_csv(path)


def _sanitize_value_for_json(v):
    """Recursively sanitize values so they can be JSON serialized.

    Replaces non-finite floats (inf, -inf, nan) with None, converts numpy
    scalar types to native Python types, and formats datetimes as ISO strings.
    """
    import math
    import datetime

    # None
    if v is None:
        return None

    # pandas NA (pd.NA) or numpy NaN
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # numpy scalar types
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)

    # native python float
    if isinstance(v, float):
        return v if math.isfinite(v) else None

    # datetimes
    if isinstance(v, (datetime.datetime, datetime.date, pd.Timestamp)):
        try:
            return v.isoformat()
        except Exception:
            return str(v)

    # Lists / tuples
    if isinstance(v, (list, tuple)):
        return [_sanitize_value_for_json(x) for x in v]

    # Dicts
    if isinstance(v, dict):
        return {str(k): _sanitize_value_for_json(val) for k, val in v.items()}

    # Fallback for numpy generic types
    try:
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass

    return v
# ============================================================================
# 0. Data Access Endpoint (for frontend visualizations)
# ============================================================================

@router.get("/get-data/{file_id}")
async def get_data(
    file_id: str,
    sample_size: int = Query(None, description="Optional: return only N random rows for performance"),
    columns: str = Query(None, description="Optional: comma-separated list of columns to return")
):
    """
    Retrieve dataset for frontend visualization.
    Supports sampling and column filtering for performance.
    
    Args:
        file_id: File identifier
        sample_size: If provided, returns random sample of N rows
        columns: If provided, returns only specified columns (comma-separated)
    
    Returns:
        JSON with dataset records and metadata
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Filter columns if specified
        if columns:
            requested_cols = [c.strip() for c in columns.split(",")]
            available_cols = [c for c in requested_cols if c in df.columns]
            if available_cols:
                df = df[available_cols]
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled data to {len(df)} rows")
        
        # Clean data for JSON serialization
        # 1. Replace infinity values with None
        df = df.replace([np.inf, -np.inf], None)
        
        # 2. Replace NaN values with None
        df = df.where(pd.notna(df), None)
        
        # 3. Convert to dict and sanitize values for JSON serialization
        data_dict = df.to_dict(orient='records')

        # Recursively sanitize every value in each record
        sanitized_records = []
        for rec in data_dict:
            sanitized_rec = {str(k): _sanitize_value_for_json(v) for k, v in rec.items()}
            sanitized_records.append(sanitized_rec)

        logger.info(f"Successfully prepared {len(sanitized_records)} sanitized records for JSON response")

        # Return data with metadata
        return JSONResponse(content={
            "data": sanitized_records,
            "columns": list(df.columns),
            "shape": list(df.shape),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving data for file_id={file_id}: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Error retrieving data: {str(e)}")

# ============================================================================
# 1. Data Profiling & Cleaning
# ============================================================================

@router.post("/profile")
async def profile_data(file_id: str = Query(...)):
    """Run automated profiling on uploaded dataset."""
    df = load_df_by_file_id(file_id)
    try:
        config = EDAConfig()
        profiler = Profiler(df, config=config)
        results = profiler.dataset_profile()
        return JSONResponse(results)
    except ProfilingError as e:
        raise HTTPException(422, str(e))

@router.post("/fix-missing")
async def fix_missing(
    file_id: str = Query(...),
    columns: str = Query(...),
    method: str = Query("mean")
):
    """Impute missing values in selected columns."""
    df = load_df_by_file_id(file_id)
    cols = columns.split(",")
    try:
        df_fixed = DataCleaner.impute_missing(df, cols, method=method)
        df_fixed.to_csv(os.path.join(UPLOAD_DIR, f"{file_id}.csv"), index=False)
        return {"status": "fixed", "columns": cols, "method": method}
    except ProfilingError as e:
        raise HTTPException(422, str(e))

@router.post("/drop-columns")
async def drop_columns_endpoint(
    file_id: str = Query(...),
    columns: str = Query(...)
):
    """Drop columns from dataset."""
    df = load_df_by_file_id(file_id)
    cols = columns.split(",")
    df_new = DataCleaner.drop_columns(df, cols)
    df_new.to_csv(os.path.join(UPLOAD_DIR, f"{file_id}.csv"), index=False)
    return {"status": "dropped", "columns": cols}

@router.post("/encode-high-cardinality")
async def encode_high_cardinality_endpoint(
    file_id: str = Query(...),
    columns: str = Query(...),
    max_values: int = Query(10)
):
    """Encode high-cardinality categorical columns."""
    df = load_df_by_file_id(file_id)
    cols = columns.split(",")
    df_new = DataCleaner.encode_high_cardinality(df, cols, max_values)
    df_new.to_csv(os.path.join(UPLOAD_DIR, f"{file_id}.csv"), index=False)
    return {"status": "encoded", "columns": cols}

# ============================================================================
# 2. Correlation & Pattern Discovery
# ============================================================================

@router.post("/correlation")
async def correlation_endpoint(
    file_id: str = Query(...),
    top_k: int = Query(10)
):
    """Run correlation and pattern analysis."""
    df = load_df_by_file_id(file_id)
    try:
        config = CorrelationConfig(top_k=top_k)
        analyzer = CorrelationAnalyzer(df, config=config)
        result = analyzer.analyze()
        # Add heatmap base64 image
        if result.get("corr_columns"):
            corr_heatmap_base64 = make_corr_heatmap_base64(df[result["corr_columns"]])
        else:
            corr_heatmap_base64 = None
        result["corr_heatmap_base64"] = corr_heatmap_base64
        return JSONResponse(result)
    except CorrelationError as e:
        raise HTTPException(422, str(e))

# ============================================================================
# 3. Outlier Detection & Cleaning
# ============================================================================

@router.post("/detect-outliers")
async def api_detect_outliers(file_id: str = Query(...)):
    """Detect outliers in dataset."""
    df = load_df_by_file_id(file_id)
    try:
        detector = OutlierDetector(df)
        results = detector.detect()
        return JSONResponse(results)
    except OutlierDetectionError as e:
        raise HTTPException(422, str(e))

@router.post("/remove-outliers")
async def api_remove_outliers(
    file_id: str = Query(...),
    columns: str = Query(...)
):
    """Remove outliers from columns using IQR/Z-score rules."""
    df = load_df_by_file_id(file_id)
    cols = columns.split(",")
    try:
        detector = OutlierDetector(df)
        df_new = detector.remove(cols)
        df_new.to_csv(os.path.join(UPLOAD_DIR, f"{file_id}.csv"), index=False)
        return {"status": "outliers_removed", "columns": cols}
    except OutlierDetectionError as e:
        raise HTTPException(422, str(e))

@router.post("/drop-outliers")
async def drop_outliers_endpoint(
    file_id: str = Query(...),
    columns: str = Query(...),
    z_thresh: float = Query(3.0)
):
    """Drop all rows where any column is > z_thresh standard deviations from mean."""
    df = load_df_by_file_id(file_id)
    cols = columns.split(",")
    df_new = OutlierCleaner.drop_outliers(df, cols, z_thresh)
    df_new.to_csv(os.path.join(UPLOAD_DIR, f"{file_id}.csv"), index=False)
    return {
        "status": "dropped_outliers",
        "columns": cols,
        "rows_remaining": len(df_new)
    }

# ============================================================================
# 4. Drift Detection
# ============================================================================

@router.post("/detect-drift")
async def api_detect_drift(
    real_file_id: str = Query(...),
    synth_file_id: str = Query(...)
):
    """Detect distributional drift between real and synthetic datasets."""
    real_df = load_df_by_file_id(real_file_id)
    synth_df = load_df_by_file_id(synth_file_id)
    try:
        detector = DriftDetector(real_df, synth_df)
        results = detector.detect()
        return JSONResponse(results)
    except DriftDetectionError as e:
        raise HTTPException(422, str(e))

# ============================================================================
# 5. Feature Engineering & Suggestions
# ============================================================================

@router.post("/feature-suggestions")
async def feature_suggestions(
    file_id: str = Query(...),
    target_col: str = Query(None),
    max_pairs: int = Query(3),
    max_suggestions: int = Query(5)
):
    """Get AI-powered feature suggestions."""
    df = load_df_by_file_id(file_id)
    try:
        config = FeatureEngConfig(max_pairs=max_pairs)
        suggester = FeatureSuggester(df, config=config)
        results = suggester.suggest(target_col=target_col, max_suggestions=max_suggestions)
        return JSONResponse(results)
    except FeatureEngError as e:
        raise HTTPException(422, str(e))

@router.post("/apply-features")
async def apply_features(
    file_id: str = Query(...),
    body: dict = Body(...)
):
    """Apply selected engineered features to data."""
    code_blocks = body.get("code_blocks", [])
    df = load_df_by_file_id(file_id)
    try:
        suggester = FeatureSuggester(df)
        df_new = suggester.apply(code_blocks)
        df_new.to_csv(os.path.join(UPLOAD_DIR, f"{file_id}.csv"), index=False)
        return {"status": "applied", "applied_features": code_blocks}
    except FeatureEngError as e:
        raise HTTPException(422, str(e))

@router.post("/export-pipeline")
async def export_pipeline(
    file_id: str = Query(...),
    target_col: str = Query(None),
    max_pairs: int = Query(3),
    body: dict = Body(None)
):
    """
    Export Python code for the feature engineering pipeline.
    Uses selected code_blocks if given, otherwise recomputes from data.
    """
    df = load_df_by_file_id(file_id)
    try:
        code_blocks = body.get("code_blocks", []) if body else None
        if not code_blocks:
            config = FeatureEngConfig(max_pairs=max_pairs)
            suggester = FeatureSuggester(df, config=config)
            result = suggester.suggest(target_col=target_col)
            code_blocks = result["code_blocks"]
        code = suggester.export_pipeline_code(code_blocks)
        return {"pipeline_code": code}
    except FeatureEngError as e:
        raise HTTPException(422, str(e))

@router.post("/explain-feature")
async def explain_feature(
    feature_code: str = Query(...),
    openai_key: str = Query(None),
    file_id: str = Query(None)
):
    """Get a natural language explanation for a feature transformation."""
    try:
        suggester = FeatureSuggester(pd.DataFrame())
        explanation = suggester.explain_feature(feature_code, openai_key)
        return {"explanation": explanation}
    except FeatureEngError as e:
        raise HTTPException(422, str(e))

# ============================================================================
# 6. Reporting & Download Utilities
# ============================================================================

@router.post("/report-outlier-drift")
async def report_outlier_drift(
    file_id: str = Query(...),
    synth_file_id: str = Query(None)
):
    """
    Generate PDF report of outlier and drift statistics.
    """
    df = load_df_by_file_id(file_id)
    out_stats = OutlierDetector(df).detect()["stats"]
    drift_stats = []
    if synth_file_id:
        df_synth = load_df_by_file_id(synth_file_id)
        drift_stats = DriftDetector(df, df_synth).detect()["drift_stats"]
    pdf_path = outlier_drift_report_pdf(out_stats, drift_stats)
    with open(pdf_path, "rb") as f:
        content = f.read()
    return Response(content, media_type="application/pdf")

@router.post("/explain-outlier-drift")
async def explain_outlier_drift(
    file_id: str = Query(...),
    synth_file_id: str = Query(None)
):
    """
    Provide natural language explanation for outlier and drift results.
    """
    df = load_df_by_file_id(file_id)
    out_stats = OutlierDetector(df).detect()["stats"]
    drift_stats = []
    if synth_file_id:
        df_synth = load_df_by_file_id(synth_file_id)
        drift_stats = DriftDetector(df, df_synth).detect()["drift_stats"]

    message = "### Automated Explanation\n"
    message += f"- Outlier detection flagged {sum(row['outlier_count'] for row in out_stats)} outlier values across {len(out_stats)} columns.\n"
    if drift_stats:
        drifted = [row['column'] for row in drift_stats if row.get('drifted')]
        if drifted:
            message += f"- Significant data drift detected in: {', '.join(drifted)}.\n"
        else:
            message += "- No major drift detected between real and synthetic datasets.\n"
    message += "\n**Tip:** Outliers can distort models and drift means synthetic data is outdated or mismatched."
    return {"explanation": message}

@router.get("/download")
async def download_csv(file_id: str = Query(...)):
    """Download the current CSV file for the given file_id."""
    path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    with open(path, "rb") as f:
        content = f.read()
    return Response(content, media_type="text/csv")


# ============================================================================
# DEPENDENCIES & PRIVACY ENDPOINTS
# ============================================================================

@router.get("/detect-dependencies/{file_id}")
async def detect_dependencies(
    file_id: str,
    max_determinant_cardinality: int = Query(1000, description="Max unique values for determinant column"),
    min_confidence: float = Query(0.95, description="Minimum confidence threshold (0-1)")
):
    """
    Detect functional dependencies (X → Y mappings) between columns.
    
    Args:
        file_id: CSV file identifier
        max_determinant_cardinality: Max unique values for X column
        min_confidence: Minimum confidence to report dependency
    
    Returns:
        Detected dependencies with confidence scores and graph data
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Detecting dependencies for file {file_id}, shape: {df.shape}")
        
        detector = DependencyDetector(df)
        dependencies = detector.find_dependencies(
            max_determinant_cardinality=max_determinant_cardinality
        )
        
        # Filter by confidence threshold
        filtered_deps = [
            dep for dep in dependencies 
            if dep["confidence"] >= min_confidence
        ]
        
        # Get graph visualization data
        graph_data = detector.get_dependency_graph_data(filtered_deps)
        
        # Persist dependencies to metadata storage for generation constraints
        try:
            metadata_file = detector.save_to_metadata(file_id)
            logger.info(f"Saved dependency metadata to: {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to save dependency metadata: {str(e)}")
        
        return {
            "dependencies": filtered_deps,
            "total_found": len(dependencies),
            "filtered_count": len(filtered_deps),
            "graph_data": graph_data,
            "parameters": {
                "max_determinant_cardinality": max_determinant_cardinality,
                "min_confidence": min_confidence
            }
        }
    
    except Exception as e:
        logger.error(f"Error detecting dependencies: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Dependency detection failed: {str(e)}")


@router.post("/validate-dependencies")
async def validate_dependencies(
    request: Request,
    real_file_id: str = Body(..., embed=True),
    synthetic_file_id: str = Body(..., embed=True),
    min_confidence: float = Body(0.95, embed=True)
):
    """
    Validate if synthetic data preserves functional dependencies from real data.
    
    Args:
        real_file_id: Real dataset file ID
        synthetic_file_id: Synthetic dataset file ID
        min_confidence: Minimum confidence threshold
    
    Returns:
        Validation results showing preserved vs violated dependencies
    """
    try:
        real_df = load_df_by_file_id(real_file_id)
        synthetic_df = load_df_by_file_id(synthetic_file_id)
        
        logger.info(f"Validating dependencies - Real: {real_df.shape}, Synthetic: {synthetic_df.shape}")
        
        # Detect dependencies in real data
        detector = DependencyDetector(real_df)
        real_deps = detector.find_dependencies()
        
        # Filter by confidence
        filtered_deps = [dep for dep in real_deps if dep["confidence"] >= min_confidence]
        
        # Validate against synthetic
        validation = detector.validate_against_synthetic(synthetic_df)
        
        # Categorize results based on status field
        preserved = [v for v in validation if v["status"] == "PRESERVED"]
        violated = [v for v in validation if v["status"] in ["VIOLATED", "MISSING_COLUMNS"]]
        
        return {
            "total_dependencies": len(filtered_deps),
            "preserved_count": len(preserved),
            "violated_count": len(violated),
            "preserved": preserved,
            "violated": violated,
            "preservation_rate": len(preserved) / len(validation) if validation else 0
        }
    
    except Exception as e:
        logger.error(f"Error validating dependencies: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Dependency validation failed: {str(e)}")


@router.get("/suggest-quasi-identifiers/{file_id}")
async def suggest_quasi_identifiers(
    file_id: str,
    max_cardinality: int = Query(100, description="Max unique values for QI candidate"),
    min_cardinality: int = Query(2, description="Min unique values for QI candidate")
):
    """
    Identify potential quasi-identifiers for privacy risk assessment.
    
    Args:
        file_id: CSV file identifier
        max_cardinality: Maximum unique values for QI candidate
        min_cardinality: Minimum unique values for QI candidate
    
    Returns:
        List of potential QI columns with risk metrics
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Identifying QIs for file {file_id}, shape: {df.shape}")
        
        analyzer = KAnonymityAnalyzer(df)
        potential_qis = analyzer.identify_potential_qis(
            max_cardinality=max_cardinality,
            min_cardinality=min_cardinality
        )
        
        return {
            "potential_qis": potential_qis,
            "total_candidates": len(potential_qis),
            "high_risk_count": len([qi for qi in potential_qis if qi["risk_level"] == "High"]),
            "parameters": {
                "max_cardinality": max_cardinality,
                "min_cardinality": min_cardinality
            }
        }
    
    except Exception as e:
        logger.error(f"Error suggesting QIs: {str(e)}", exc_info=True)
        raise HTTPException(500, f"QI suggestion failed: {str(e)}")


@router.post("/compute-k-anonymity")
async def compute_k_anonymity_endpoint(
    request: Request,
    file_id: str = Body(..., embed=True),
    quasi_identifiers: list = Body(..., embed=True),
    k_threshold: int = Body(5, embed=True)
):
    """
    Compute k-anonymity risk assessment for selected quasi-identifiers.
    
    Args:
        file_id: CSV file identifier
        quasi_identifiers: List of QI column names
        k_threshold: K-value threshold for vulnerable records
    
    Returns:
        K-anonymity analysis with risk statistics and recommendations
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Computing k-anonymity for file {file_id} with QIs: {quasi_identifiers}")
        
        analyzer = KAnonymityAnalyzer(df)
        
        # Compute k-anonymity
        k_result = analyzer.compute_k_anonymity(quasi_identifiers)
        
        # Generate comprehensive report
        risk_report = analyzer.generate_risk_report()
        
        # Get vulnerable records (limited to avoid large responses)
        vulnerable_df = analyzer.get_vulnerable_records(k_threshold=k_threshold)
        vulnerable_count = len(vulnerable_df)
        
        # Persist k-anonymity results to metadata storage
        try:
            metadata_file = analyzer.save_to_metadata(file_id)
            logger.info(f"Saved k-anonymity metadata to: {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to save k-anonymity metadata: {str(e)}")
        
        return {
            "risk_report": risk_report,
            "vulnerable_count": vulnerable_count,
            "k_threshold": k_threshold,
            "total_records": len(df)
        }
    
    except ValueError as e:
        logger.error(f"Invalid QI columns: {str(e)}")
        raise HTTPException(400, f"Invalid quasi-identifiers: {str(e)}")
    except Exception as e:
        logger.error(f"Error computing k-anonymity: {str(e)}", exc_info=True)
        raise HTTPException(500, f"K-anonymity computation failed: {str(e)}")


# ============================================================================
# Advanced Diagnostics: Statistical Fingerprinting, PII Detection, MI Analysis
# ============================================================================

@router.post("/compute-distributions/{file_id}")
async def compute_distributions_endpoint(file_id: str):
    """
    Compute statistical distribution fingerprinting for all numeric columns.
    Uses distfit library to identify best-fit probability distributions.
    
    Args:
        file_id: CSV file identifier
    
    Returns:
        Distribution analysis results with best-fit distributions, parameters,
        goodness-of-fit scores, and recommendations
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Computing distribution fingerprinting for file {file_id}")
        
        # Run distribution analysis
        analyzer = DistributionAnalyzer(df)
        distributions = analyzer.fit_distributions()
        summary = analyzer.get_summary()
        
        # Persist results to metadata
        try:
            metadata_file = analyzer.save_results(file_id)
            logger.info(f"Saved distribution metadata to: {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to save distribution metadata: {str(e)}")
        
        return {
            "distributions": _sanitize_value_for_json(distributions),
            "summary": _sanitize_value_for_json(summary),
            "file_id": file_id
        }
    
    except ProfilingError as e:
        logger.error(f"Distribution analysis error: {str(e)}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Error computing distributions: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Distribution analysis failed: {str(e)}")


@router.post("/pii-scan-fast/{file_id}")
async def pii_scan_fast_endpoint(file_id: str, sample_size: int = 1000):
    """
    Perform fast PII detection using regex patterns and column name heuristics.
    
    Args:
        file_id: CSV file identifier
        sample_size: Number of rows to sample for scanning (default 1000)
    
    Returns:
        PII detection results with detected entities, confidence scores,
        and privacy recommendations
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Running fast PII scan for file {file_id} (sample_size={sample_size})")
        
        # Run fast PII scan
        scanner = PIIScanner(df)
        results = scanner.run_fast_scan(sample_size=sample_size)
        summary = scanner.get_pii_summary()
        
        # Persist results to metadata
        try:
            metadata_file = scanner.save_results(file_id)
            logger.info(f"Saved PII scan metadata to: {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to save PII metadata: {str(e)}")
        
        return {
            "scan_type": "fast",
            "results": _sanitize_value_for_json(results),
            "summary": _sanitize_value_for_json(summary),
            "sample_size": sample_size,
            "file_id": file_id
        }
    
    except Exception as e:
        logger.error(f"Error in fast PII scan: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Fast PII scan failed: {str(e)}")


@router.post("/pii-scan-deep")
async def pii_scan_deep_endpoint(
    file_id: str = Body(..., embed=True),
    columns: list = Body(None, embed=True),
    sample_size: int = Body(100, embed=True)
):
    """
    Perform deep PII detection using Microsoft Presidio AI analyzer.
    More accurate but slower than fast scan.
    
    Args:
        file_id: CSV file identifier
        columns: Optional list of columns to scan (defaults to all)
        sample_size: Number of rows to sample (default 100)
    
    Returns:
        Deep PII detection results with entity types, confidence scores,
        and detailed findings
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Running deep PII scan for file {file_id} (columns={columns}, sample={sample_size})")
        
        # Run deep PII scan (will fallback to fast scan if Presidio not available)
        scanner = PIIScanner(df)
        
        # Check if Presidio is available
        if not scanner.presidio_available:
            logger.warning("Presidio not installed - falling back to fast PII scan")
            results = scanner.run_fast_scan(sample_size=sample_size)
            summary = scanner.get_pii_summary()
            
            # Return with informative message about fallback
            return {
                "scan_type": "fast",
                "results": _sanitize_value_for_json(results),
                "summary": _sanitize_value_for_json(summary),
                "columns_scanned": columns or "all",
                "sample_size": sample_size,
                "file_id": file_id,
                "message": "⚠️ Deep scan requires presidio-analyzer library. Performed fast scan instead. Install with: pip install presidio-analyzer",
                "fallback": True
            }
        
        # Run deep scan if Presidio available
        results = scanner.run_deep_scan(columns=columns, sample_size=sample_size)
        summary = scanner.get_pii_summary()
        
        # Persist results to metadata
        try:
            metadata_file = scanner.save_results(file_id)
            logger.info(f"Saved deep PII scan metadata to: {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to save PII metadata: {str(e)}")
        
        return {
            "scan_type": "deep",
            "results": _sanitize_value_for_json(results),
            "summary": _sanitize_value_for_json(summary),
            "columns_scanned": columns or "all",
            "sample_size": sample_size,
            "file_id": file_id,
            "fallback": False
        }
    
    except Exception as e:
        logger.error(f"Error in deep PII scan: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Deep PII scan failed: {str(e)}")


@router.post("/compute-mi-matrix/{file_id}")
async def compute_mi_matrix_endpoint(
    file_id: str,
    columns: list = Body(None, embed=True),
    normalize: bool = Body(True, embed=True)
):
    """
    Compute Mutual Information (MI) matrix for non-linear correlation analysis.
    
    Args:
        file_id: CSV file identifier
        columns: Optional list of columns to analyze (defaults to all numeric)
        normalize: Whether to normalize MI scores to [0, 1] range
    
    Returns:
        MI matrix, top MI pairs, and comparison with Pearson correlation
        to identify non-linear relationships
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Computing MI matrix for file {file_id} (normalize={normalize})")
        
        # Run MI analysis
        analyzer = MutualInformationAnalyzer(df)
        mi_matrix = analyzer.compute_mutual_information_matrix(
            columns=columns,
            normalize=normalize
        )
        
        # Get analysis results
        top_pairs = analyzer.get_top_mi_pairs(top_k=20)
        comparison = analyzer.compare_with_pearson()
        
        # Persist results to metadata
        try:
            metadata_file = analyzer.save_results(file_id)
            logger.info(f"Saved MI analysis metadata to: {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to save MI metadata: {str(e)}")
        
        return {
            "mi_matrix": _sanitize_value_for_json(mi_matrix.round(4).to_dict()),
            "mi_columns": mi_matrix.columns.tolist(),
            "mi_heatmap": _sanitize_value_for_json(mi_matrix.values.tolist()),
            "top_mi_pairs": [
                {
                    "column1": col1,
                    "column2": col2,
                    "mi_score": float(score)
                }
                for col1, col2, score in top_pairs
            ],
            "nonlinear_analysis": _sanitize_value_for_json(comparison),
            "normalized": normalize,
            "file_id": file_id
        }
    
    except CorrelationError as e:
        logger.error(f"MI computation error: {str(e)}")
        raise HTTPException(400, str(e))
    except ImportError:
        raise HTTPException(
            400,
            "MI analysis requires ennemi library. "
            "Install with: pip install ennemi"
        )
    except Exception as e:
        logger.error(f"Error computing MI matrix: {str(e)}", exc_info=True)
        raise HTTPException(500, f"MI matrix computation failed: {str(e)}")


# ============================================================================
# END OF FILE
# ============================================================================

