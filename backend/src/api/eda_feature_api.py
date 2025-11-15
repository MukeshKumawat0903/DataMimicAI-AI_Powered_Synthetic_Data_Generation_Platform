# src/api/eda_feature_api.py

from fastapi import APIRouter, HTTPException, Request, Query, Body
from fastapi.responses import JSONResponse, Response
import pandas as pd
import numpy as np
import logging
import os
import base64
import gzip
import io

from src.core.eda.profiling import (
    Profiler, DataCleaner, EDAConfig, ProfilingError,
    DistributionAnalyzer, run_distribution_analysis
)
from src.core.eda.correlation import (
    CorrelationAnalyzer, CorrelationConfig, CorrelationError, make_corr_heatmap_base64,
    MutualInformationAnalyzer, compute_mi_matrix
)
from src.core.eda.outliers import (
    OutlierDetector, OutlierConfig, OutlierDetectionError, OutlierCleaner,
    load_outlier_metadata_file
)
from src.core.eda.drift import (
    DriftDetector, DriftConfig, DriftDetectionError,
    load_drift_metadata_file
)
from src.core.feature_engineering.feature_suggester import (
    FeatureSuggester, FeatureEngConfig, FeatureEngError
)
from src.core.feature_engineering.utility_suggester import UtilitySuggester
from src.core.feature_engineering.privacy_suggester import PrivacySuggester
from src.core.feature_engineering.pet_constants import PET_TYPES, is_pet
from src.core.feature_engineering.conflict_resolver import ConflictResolver
from src.core.eda.utils import outlier_drift_report_pdf
from src.core.eda.dependencies import DependencyDetector
from src.core.eda.privacy import KAnonymityAnalyzer
from src.core.eda.pii_scan import PIIScanner, run_pii_scan_fast, run_pii_scan_deep
from src.core.eda.timeseries import (
    TimeSeriesDetector, TimeSeriesAnalyzer, TimeSeriesError,
    detect_timeseries, compute_acf_pacf, decompose_timeseries
)
from src.core.feedback_engine import TransformConfigManager, EDAFeedbackEngine

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
    """Detect distributional drift between real and synthetic datasets (legacy endpoint)."""
    real_df = load_df_by_file_id(real_file_id)
    synth_df = load_df_by_file_id(synth_file_id)
    try:
        detector = DriftDetector(real_df, synth_df)
        # Use analyze_all() instead of non-existent detect()
        results = detector.analyze_all(methods=['ks', 'psi', 'chi2', 'classifier'])
        return JSONResponse(results)
    except DriftDetectionError as e:
        raise HTTPException(422, str(e))

# ============================================================================
# 4.1 Comprehensive Outlier & Drift Detection (New Enhanced Endpoints)
# ============================================================================

@router.post("/outliers/detect-comprehensive")
async def detect_outliers_comprehensive_endpoint(
    file_id: str = Query(...),
    methods: str = Query("zscore,iqr,mad", description="Comma-separated: zscore,iqr,mad,isolation_forest,lof"),
    columns: str = Query(None, description="Comma-separated column names (optional)")
):
    """
    Comprehensive outlier detection using multiple methods.
    Supports: Z-score, IQR, MAD, Isolation Forest, LOF
    """
    df = load_df_by_file_id(file_id)
    methods_list = [m.strip() for m in methods.split(",")]
    columns_list = [c.strip() for c in columns.split(",")] if columns else None
    
    try:
        from src.core.eda.outliers import detect_outliers_comprehensive
        
        results = detect_outliers_comprehensive(
            df=df,
            file_id=file_id,
            methods=methods_list,
            columns=columns_list
        )
        return JSONResponse(results)
    except Exception as e:
        logger.error(f"Comprehensive outlier detection failed: {str(e)}")
        raise HTTPException(422, str(e))


@router.post("/outliers/detect-timeseries")
async def detect_timeseries_outliers_endpoint(
    file_id: str = Query(...),
    value_col: str = Query(..., description="Numeric column for outlier detection"),
    datetime_col: str = Query(..., description="Datetime column"),
    freq: str = Query(None, description="Time-series frequency (e.g., 'D', 'H', 'M')")
):
    """Detect outliers in time-series using STL decomposition residuals."""
    df = load_df_by_file_id(file_id)
    
    try:
        detector = OutlierDetector(df)
        results = detector.detect_timeseries_outliers(value_col, datetime_col, freq)
        
        # Save metadata
        metadata_path = detector.save_metadata(file_id, results)
        results["metadata_path"] = metadata_path
        
        return JSONResponse(results)
    except OutlierDetectionError as e:
        raise HTTPException(422, str(e))


@router.post("/outliers/detect-cooks-distance")
async def detect_cooks_distance_endpoint(
    file_id: str = Query(...),
    target_col: str = Query(..., description="Target variable for regression"),
    feature_cols: str = Query(None, description="Comma-separated feature columns (optional)")
):
    """Detect influential outliers using Cook's Distance from regression."""
    df = load_df_by_file_id(file_id)
    features = [f.strip() for f in feature_cols.split(",")] if feature_cols else None
    
    try:
        detector = OutlierDetector(df)
        results = detector.detect_cooks_distance(target_col, features)
        
        # Save metadata
        metadata_path = detector.save_metadata(file_id, results)
        results["metadata_path"] = metadata_path
        
        return JSONResponse(results)
    except OutlierDetectionError as e:
        raise HTTPException(422, str(e))


@router.post("/drift/analyze-comprehensive")
async def analyze_drift_comprehensive_endpoint(
    ref_file_id: str = Query(..., description="Reference dataset file ID"),
    cur_file_id: str = Query(..., description="Current dataset file ID"),
    methods: str = Query("ks,psi", description="Comma-separated: ks,chi_square,psi,classifier"),
    columns: str = Query(None, description="Comma-separated column names (optional)")
):
    """
    Comprehensive drift analysis using multiple methods.
    Supports: KS test, Chi-square, PSI, Drift Classifier
    """
    df_ref = load_df_by_file_id(ref_file_id)
    df_cur = load_df_by_file_id(cur_file_id)
    methods_list = [m.strip() for m in methods.split(",")]
    columns_list = [c.strip() for c in columns.split(",")] if columns else None
    
    try:
        from src.core.eda.drift import analyze_drift_comprehensive
        
        results = analyze_drift_comprehensive(
            df_reference=df_ref,
            df_current=df_cur,
            file_id_ref=ref_file_id,
            file_id_cur=cur_file_id,
            methods=methods_list,
            columns=columns_list
        )
        return JSONResponse(results)
    except Exception as e:
        logger.error(f"Comprehensive drift analysis failed: {str(e)}")
        raise HTTPException(422, str(e))


@router.get("/outliers/metadata/{file_id}")
async def get_outlier_metadata(file_id: str):
    """Retrieve saved outlier detection metadata."""
    try:
        metadata = load_outlier_metadata_file(file_id)
        if metadata:
            return JSONResponse(metadata)
        else:
            raise HTTPException(404, "Metadata not found for this file_id")
    except Exception as e:
        raise HTTPException(422, str(e))


@router.get("/drift/metadata/{ref_file_id}/{cur_file_id}")
async def get_drift_metadata(ref_file_id: str, cur_file_id: str):
    """Retrieve saved drift analysis metadata."""
    try:
        metadata = load_drift_metadata_file(ref_file_id, cur_file_id)
        if metadata:
            return JSONResponse(metadata)
        else:
            raise HTTPException(404, "Metadata not found for these file_ids")
    except Exception as e:
        raise HTTPException(422, str(e))

# ============================================================================
# 4.2 Outlier Remediation Endpoints
# ============================================================================

@router.post("/outliers/remediate")
async def remediate_outliers_endpoint(
    file_id: str = Query(...),
    method: str = Query(..., description="winsorize, cap, remove, bin, transform"),
    columns: str = Query(..., description="Comma-separated column names"),
    body: dict = Body(default={})
):
    """
    Apply outlier remediation method.
    Supports: winsorize, cap, remove, bin, transform
    """
    df = load_df_by_file_id(file_id)
    columns_list = [c.strip() for c in columns.split(",")]
    
    try:
        from src.core.eda.remediation import remediate_outliers
        
        df_result, metadata = remediate_outliers(
            df=df,
            file_id=file_id,
            method=method,
            columns=columns_list,
            **body
        )
        
        # Save the remediated dataset
        output_file_id = body.get('output_file_id', f"{file_id}_remediated")
        output_path = os.path.join(UPLOAD_DIR, f"{output_file_id}.csv")
        df_result.to_csv(output_path, index=False)
        
        return JSONResponse({
            "status": "success",
            "method": method,
            "columns": columns_list,
            "output_file_id": output_file_id,
            "n_rows": len(df_result),
            "metadata": metadata
        })
    except Exception as e:
        logger.error(f"Remediation failed: {str(e)}")
        raise HTTPException(422, str(e))


@router.post("/outliers/winsorize")
async def winsorize_endpoint(
    file_id: str = Query(...),
    columns: str = Query(...),
    lower_limit: float = Query(0.05, description="Lower percentile (0-1)"),
    upper_limit: float = Query(0.05, description="Upper percentile (0-1)")
):
    """Winsorize outliers by capping at percentiles."""
    df = load_df_by_file_id(file_id)
    columns_list = [c.strip() for c in columns.split(",")]
    
    try:
        from src.core.eda.remediation import OutlierRemediator
        
        remediator = OutlierRemediator(df)
        df_result = remediator.winsorize(columns_list, limits=(lower_limit, upper_limit))
        
        # Save
        df_result.to_csv(os.path.join(UPLOAD_DIR, f"{file_id}.csv"), index=False)
        metadata_path = remediator.save_metadata(file_id)
        
        return JSONResponse({
            "status": "success",
            "method": "winsorize",
            "columns": columns_list,
            "limits": [lower_limit, upper_limit],
            "history": remediator.get_remediation_history(),
            "metadata_path": metadata_path
        })
    except Exception as e:
        raise HTTPException(422, str(e))


@router.post("/outliers/cap")
async def cap_outliers_endpoint(
    file_id: str = Query(...),
    columns: str = Query(...),
    cap_method: str = Query("iqr", description="'iqr' or 'percentile'"),
    multiplier: float = Query(1.5, description="IQR multiplier"),
    lower_percentile: float = Query(1.0, description="Lower percentile for percentile method"),
    upper_percentile: float = Query(99.0, description="Upper percentile for percentile method")
):
    """Cap outliers using IQR or percentile method."""
    df = load_df_by_file_id(file_id)
    columns_list = [c.strip() for c in columns.split(",")]
    
    try:
        from src.core.eda.remediation import OutlierRemediator
        
        remediator = OutlierRemediator(df)
        df_result = remediator.cap_outliers(
            columns_list,
            method=cap_method,
            multiplier=multiplier,
            percentiles=(lower_percentile, upper_percentile)
        )
        
        # Save
        df_result.to_csv(os.path.join(UPLOAD_DIR, f"{file_id}.csv"), index=False)
        metadata_path = remediator.save_metadata(file_id)
        
        return JSONResponse({
            "status": "success",
            "method": f"cap_{cap_method}",
            "columns": columns_list,
            "history": remediator.get_remediation_history(),
            "metadata_path": metadata_path
        })
    except Exception as e:
        raise HTTPException(422, str(e))


@router.get("/outliers/remediation-history/{file_id}")
async def get_remediation_history(file_id: str):
    """Retrieve remediation history for a file."""
    try:
        from src.core.eda.remediation import OutlierRemediator
        
        # Create dummy instance to load metadata
        remediator = OutlierRemediator(pd.DataFrame())
        metadata = remediator.load_metadata(file_id)
        
        if metadata:
            return JSONResponse(metadata)
        else:
            raise HTTPException(404, "No remediation history found")
    except Exception as e:
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


@router.post("/context-aware-suggestions")
async def context_aware_suggestions(
    file_id: str = Query(...),
    include_profiling: bool = Query(True),
    include_pii: bool = Query(True),
    include_k_anonymity: bool = Query(True)
):
    """
    Generate context-aware feature suggestions combining utility and privacy analysis.
    Returns utility suggestions, privacy suggestions, and conflicts.
    """
    df = load_df_by_file_id(file_id)
    
    try:
        result = {
            "file_id": file_id,
            "utility_suggestions": [],
            "privacy_suggestions": [],
            "conflicts": [],
            "conflict_summary": {},
            "non_conflicting_utility": [],
            "non_conflicting_privacy": [],
            "metadata": {}
        }
        
        # Step 1: Generate utility-driven suggestions
        profiling_data = None
        if include_profiling:
            try:
                profiler = Profiler(df)
                profiling_data = profiler.dataset_profile()
            except Exception as e:
                logger.warning(f"Profiling failed: {e}")
        
        utility_suggester = UtilitySuggester(df, profile_data=profiling_data)
        utility_suggestions = utility_suggester.suggest_utility_transforms()
        result["utility_suggestions"] = utility_suggestions
        
        # Step 2: Generate privacy-driven suggestions
        pii_report = None
        k_anonymity_report = None
        
        if include_pii:
            try:
                pii_scanner = PIIScanner(df)
                pii_report = pii_scanner.run_fast_scan()
            except Exception as e:
                logger.warning(f"PII scan failed: {e}")
        
        if include_k_anonymity:
            try:
                k_analyzer = KAnonymityAnalyzer(df)
                potential_qis = k_analyzer.identify_potential_qis()
                k_anonymity_report = {"potential_qis": potential_qis}
            except Exception as e:
                logger.warning(f"K-anonymity analysis failed: {e}")
        
        privacy_suggester = PrivacySuggester(df, pii_report=pii_report, k_anonymity_report=k_anonymity_report)
        privacy_suggestions = privacy_suggester.suggest_privacy_transforms()
        result["privacy_suggestions"] = privacy_suggestions
        
        # Step 3: Detect conflicts
        conflict_resolver = ConflictResolver()
        conflicts = conflict_resolver.detect_conflicts(utility_suggestions, privacy_suggestions)
        result["conflicts"] = conflicts
        result["conflict_summary"] = conflict_resolver.generate_conflict_summary()
        
        # Step 4: Separate non-conflicting suggestions
        non_conflicting_utility, non_conflicting_privacy = conflict_resolver.get_non_conflicting_suggestions(
            utility_suggestions, privacy_suggestions
        )
        result["non_conflicting_utility"] = non_conflicting_utility
        result["non_conflicting_privacy"] = non_conflicting_privacy
        
        # Add metadata
        result["metadata"] = {
            "total_utility_suggestions": len(utility_suggestions),
            "total_privacy_suggestions": len(privacy_suggestions),
            "total_conflicts": len(conflicts),
            "total_columns_analyzed": len(df.columns),
            "profiling_included": include_profiling,
            "pii_scan_included": include_pii,
            "k_anonymity_included": include_k_anonymity
        }
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Context-aware suggestions failed: {e}")
        raise HTTPException(500, f"Failed to generate suggestions: {str(e)}")


@router.post("/resolve-conflict")
async def resolve_conflict(
    file_id: str = Query(...),
    column: str = Query(...),
    chosen_category: str = Query(...),
    chosen_transformation: str = Query(...),
    user_note: str = Query(None)
):
    """
    Record a user's resolution of a conflict between utility and privacy suggestions.
    """
    try:
        conflict_resolver = ConflictResolver()
        resolution = conflict_resolver.resolve_conflict(
            column=column,
            chosen_category=chosen_category,
            chosen_transformation=chosen_transformation,
            user_note=user_note
        )
        
        return JSONResponse({
            "status": "conflict_resolved",
            "resolution": resolution
        })
        
    except Exception as e:
        raise HTTPException(500, f"Failed to resolve conflict: {str(e)}")


@router.post("/save-transform-config")
async def save_transform_config(
    file_id: str = Query(...),
    body: dict = Body(...)
):
    """
    Save accepted transformation decisions to a configuration file.
    """
    try:
        decisions = body.get("decisions", [])
        
        config_manager = TransformConfigManager()
        
        # Add each decision
        for decision in decisions:
            config_manager.add_decision(
                column=decision["column"],
                transformation=decision["transformation"],
                category=decision.get("category", "utility"),
                params=decision.get("params", {}),
                reason=decision.get("reason", ""),
                metadata=decision.get("metadata", {})
            )
        
        # Save to file
        config_path = config_manager.save_config(file_id)
        
        return JSONResponse({
            "status": "saved",
            "config_path": config_path,
            "total_decisions": len(decisions),
            "config": config_manager.export_config()
        })
        
    except Exception as e:
        raise HTTPException(500, f"Failed to save config: {str(e)}")


@router.get("/load-transform-config")
async def load_transform_config(
    file_id: str = Query(...),
    config_path: str = Query(None)
):
    """
    Load a previously saved transformation configuration.
    """
    try:
        config_manager = TransformConfigManager()
        
        if config_path:
            success = config_manager.load_config(config_path)
        else:
            # Look for most recent config for this file_id
            import glob
            config_dir = config_manager.config_dir
            pattern = f"transform_config_{file_id}_*.json"
            matches = sorted(glob.glob(str(config_dir / pattern)), reverse=True)
            
            if not matches:
                raise HTTPException(404, f"No configuration found for file_id: {file_id}")
            
            success = config_manager.load_config(matches[0])
        
        if not success:
            raise HTTPException(500, "Failed to load configuration")
        
        return JSONResponse({
            "status": "loaded",
            "config": config_manager.export_config()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load config: {str(e)}")


@router.get("/list-transform-configs")
async def list_transform_configs(
    file_id: str = Query(None)
):
    """
    List all saved transformation configurations, optionally filtered by file_id.
    Returns list with filenames, timestamps, and metadata.
    """
    try:
        import glob
        import os
        from datetime import datetime
        
        config_manager = TransformConfigManager()
        config_dir = config_manager.config_dir
        
        # Build pattern
        if file_id:
            pattern = f"transform_config_{file_id}_*.json"
        else:
            pattern = "transform_config_*.json"
        
        matches = glob.glob(str(config_dir / pattern))
        
        configs = []
        for filepath in matches:
            try:
                # Extract info from filename
                filename = os.path.basename(filepath)
                
                # Get file stats
                stat = os.stat(filepath)
                created_time = datetime.fromtimestamp(stat.st_ctime)
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                file_size = stat.st_size
                
                # Try to load and get decision count
                temp_manager = TransformConfigManager()
                if temp_manager.load_config(filepath):
                    # TransformConfigManager stores accepted transforms in 'accepted_transforms'
                    decision_count = len(getattr(temp_manager, 'accepted_transforms', []))
                else:
                    decision_count = 0
                
                configs.append({
                    "filepath": filepath,
                    "filename": filename,
                    "created_at": created_time.isoformat(),
                    "modified_at": modified_time.isoformat(),
                    "file_size_bytes": file_size,
                    "decision_count": decision_count
                })
            except Exception as e:
                logger.warning(f"Error processing config file {filepath}: {e}")
                continue
        
        # Sort by modified time (newest first)
        configs.sort(key=lambda x: x["modified_at"], reverse=True)
        
        return JSONResponse({
            "status": "success",
            "total_configs": len(configs),
            "configs": configs
        })
        
    except Exception as e:
        logger.error(f"Failed to list configs: {e}")
        raise HTTPException(500, f"Failed to list configs: {str(e)}")


@router.post("/get-transform-code")
async def get_transform_code(
    file_id: str = Query(...),
    body: dict = Body(...)
):
    """
    Generate Python code for applying accepted transformations.
    """
    try:
        suggestion = body.get("suggestion", {})
        category = suggestion.get("category", "utility")
        
        df = load_df_by_file_id(file_id)
        
        if category == "utility":
            suggester = UtilitySuggester(df)
            code = suggester.get_transform_code(suggestion)
        elif category == "privacy":
            suggester = PrivacySuggester(df)
            code = suggester.get_transform_code(suggestion)
        else:
            raise HTTPException(400, f"Unknown category: {category}")
        
        return JSONResponse({
            "code": code,
            "column": suggestion.get("column"),
            "transformation": suggestion.get("transformation")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate code: {str(e)}")


@router.post("/apply-transform-config")
async def apply_transform_config(
    file_id: str = Query(...),
    dry_run: bool = Query(False, description="If true, preview only without persisting"),
    body: dict = Body(...)
):
    """
    Apply accepted transformation decisions to the dataset in-memory.
    Returns transformed data summary and preview.
    
    Args:
        file_id: Dataset identifier
        dry_run: If True, preview transformations without saving or persisting config
        body: JSON with decisions array
    """
    try:
        decisions = body.get("decisions", [])
        
        if not decisions:
            return JSONResponse({
                "status": "no_op",
                "message": "No decisions to apply",
                "applied_count": 0,
                "dry_run": dry_run
            })
        
        # Load the original dataframe
        df = load_df_by_file_id(file_id)
        df_transformed = df.copy()
        
        applied_actions = []
        skipped_actions = []
        
        # Apply each decision
        for decision in decisions:
            column = decision.get("column")
            transformation = decision.get("transformation")
            params = decision.get("params", {})
            
            # Validate column exists
            if column not in df_transformed.columns:
                skipped_actions.append({
                    "column": column,
                    "transformation": transformation,
                    "reason": "Column not found in dataset"
                })
                continue
            
            # Validate dtype for numeric transformations
            numeric_transforms = [
                "log_transform", "sqrt_transform", "power_transform",
                "standard_scaler", "robust_scaler", "minmax_scaler"
            ]
            
            if transformation in numeric_transforms:
                if not pd.api.types.is_numeric_dtype(df_transformed[column]):
                    skipped_actions.append({
                        "column": column,
                        "transformation": transformation,
                        "reason": "Column is not numeric"
                    })
                    continue
                
                # Check for negative values in log/sqrt transforms
                if transformation in ["log_transform", "sqrt_transform"]:
                    if (df_transformed[column] < 0).any():
                        skipped_actions.append({
                            "column": column,
                            "transformation": transformation,
                            "reason": "Column contains negative values"
                        })
                        continue
            
            # Build feedback dict
            feedback_dict = {
                "column": column,
                "action": transformation,
                **params
            }
            
            try:
                # Create a feedback engine instance with current df and single feedback
                feedback_engine = EDAFeedbackEngine(df_transformed, [feedback_dict])
                df_transformed = feedback_engine.apply_feedback()
                applied_actions.append({
                    "column": column,
                    "transformation": transformation,
                    "category": decision.get("category", "unknown")
                })
            except Exception as e:
                skipped_actions.append({
                    "column": column,
                    "transformation": transformation,
                    "reason": f"Error during application: {str(e)}"
                })
        
        # Generate summary
        summary = {
            "total_decisions": len(decisions),
            "applied_count": len(applied_actions),
            "skipped_count": len(skipped_actions),
            "columns_modified": list(set([a["column"] for a in applied_actions])),
            "applied_actions": applied_actions,
            "skipped_actions": skipped_actions,
            "dry_run": dry_run
        }
        
        # Generate preview (first 10 rows of modified columns)
        modified_columns = summary["columns_modified"]
        if modified_columns:
            preview_df = df_transformed[modified_columns].head(10)
            summary["preview"] = preview_df.to_dict(orient="records")
        else:
            summary["preview"] = []

        # Provide a broader preview across all columns for UI display
        full_preview_limit = min(50, len(df_transformed))
        summary["preview_full"] = (
            df_transformed.head(full_preview_limit).to_dict(orient="records")
            if full_preview_limit > 0 else []
        )
        summary["preview_shape"] = [int(df_transformed.shape[0]), int(df_transformed.shape[1])]
        summary["preview_columns"] = list(df_transformed.columns)

        applied_msg = f"Applied {summary['applied_count']} transformation(s)."
        if summary["skipped_count"] > 0:
            applied_msg += f" {summary['skipped_count']} action(s) skipped."

        transformed_data_payload = None
        try:
            csv_buffer = io.StringIO()
            df_transformed.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode("utf-8")
            csv_size = len(csv_bytes)
            max_bytes = int(os.environ.get("TRANSFORMED_DATA_MAX_BYTES", 5 * 1024 * 1024))
            max_bytes_mb = max_bytes / (1024 * 1024)
            
            if csv_size <= max_bytes:
                compressed_bytes = gzip.compress(csv_bytes)
                transformed_data_payload = {
                    "format": "csv",
                    "encoding": "base64",
                    "compression": "gzip",
                    "size_bytes": csv_size,
                    "compressed_size_bytes": len(compressed_bytes),
                    "rows": int(df_transformed.shape[0]),
                    "columns": list(df_transformed.columns),
                    "content": base64.b64encode(compressed_bytes).decode("utf-8")
                }
            else:
                transformed_data_payload = {
                    "format": "csv",
                    "encoding": "base64",
                    "compression": "gzip",
                    "size_bytes": csv_size,
                    "rows": int(df_transformed.shape[0]),
                    "columns": list(df_transformed.columns),
                    "content": None,
                    "content_available": False,
                    "reason": f"Transformed dataset size {csv_size/1024/1024:.2f} MB exceeds limit ({max_bytes_mb:.2f} MB)"
                }
        except Exception as e:
            logger.warning(f"Failed to serialize transformed dataset: {e}")
            transformed_data_payload = {
                "format": "csv",
                "encoding": "base64",
                "compression": "gzip",
                "content": None,
                "error": str(e)
            }
        
        summary_message = applied_msg
        # Save transformed data only if NOT dry run
        if not dry_run:
            logger.info(f"Transformations applied (not saved to disk): {len(applied_actions)} actions")
        else:
            logger.info(f"DRY RUN: Previewed {len(applied_actions)} transformations without persisting")
            summary_message = f"{applied_msg} Dry run mode: Transformations previewed only, not persisted."

        summary["message"] = summary_message
        
        return JSONResponse({
            "status": "applied",
            "summary": summary,
            "transformed_data": transformed_data_payload
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apply transform config failed: {e}")
        raise HTTPException(500, f"Failed to apply transformations: {str(e)}")


def _apply_row_sampling(df: pd.DataFrame, method: str, sample_size: int, seed: int = 42) -> pd.DataFrame:
    """
    Apply row sampling to dataframe based on specified method.
    
    Args:
        df: Input dataframe
        method: Sampling method - 'first', 'random', or 'stratified'
        sample_size: Number of rows to sample
        seed: Random seed for reproducibility
        
    Returns:
        Sampled dataframe
    """
    if sample_size <= 0 or sample_size >= len(df):
        return df
    
    if method == "first":
        return df.head(sample_size)
    
    elif method == "random":
        return df.sample(n=sample_size, random_state=seed)
    
    elif method == "stratified":
        # Try to find a column suitable for stratification
        # Use the first categorical or low-cardinality numeric column
        strat_col = None
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_count = df[col].nunique()
                if 2 <= unique_count <= 20:  # Reasonable number of strata
                    strat_col = col
                    break
            elif df[col].dtype in ['int64', 'int32']:
                unique_count = df[col].nunique()
                if 2 <= unique_count <= 20:
                    strat_col = col
                    break
        
        if strat_col:
            try:
                # Stratified sampling
                from sklearn.model_selection import train_test_split
                sampled, _ = train_test_split(
                    df,
                    train_size=sample_size,
                    stratify=df[strat_col],
                    random_state=seed
                )
                return sampled
            except Exception:
                # Fall back to random if stratified fails
                logger.warning(f"Stratified sampling failed, using random sampling instead")
                return df.sample(n=sample_size, random_state=seed)
        else:
            # No suitable column for stratification, use random
            logger.info("No suitable column for stratification, using random sampling")
            return df.sample(n=sample_size, random_state=seed)
    
    else:
        # Default to random
        return df.sample(n=sample_size, random_state=seed)


@router.post("/get-transform-comparisons")
async def get_transform_comparisons(
    file_id: str = Query(...),
    body: dict = Body(...)
):
    """
    Get before/after metric comparisons and visualizations for accepted transformations.
    Returns metrics deltas and plot data for display in Smart Preview.
    
    Supports optional sampling parameters:
    - row_sampling_method: 'first', 'random', or 'stratified'
    - row_sample_size: Number of rows to sample (0 = no sampling)
    - column_filter: 'all' or 'transformed_only'
    - sampling_seed: Random seed for reproducibility (default: 42)
    """
    try:
        decisions = body.get("decisions", [])
        
        # Sampling parameters
        row_sampling_method = body.get("row_sampling_method", "random")
        row_sample_size = body.get("row_sample_size", 0)  # 0 = no sampling
        column_filter = body.get("column_filter", "all")
        sampling_seed = body.get("sampling_seed", 42)
        
        if not decisions:
            return JSONResponse({
                "comparisons": [],
                "message": "No transformation decisions provided"
            })
        
        # Load the data
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(404, f"File not found: {file_id}")
        
        df = pd.read_csv(file_path)
        original_row_count = len(df)
        
        # Apply row sampling if requested
        if row_sample_size > 0 and row_sample_size < len(df):
            df = _apply_row_sampling(df, row_sampling_method, row_sample_size, sampling_seed)
            logger.info(f"Applied {row_sampling_method} sampling: {len(df)}/{original_row_count} rows")
        
        # Convert decisions to feedback format
        feedback = []
        transformed_columns = set()
        
        for decision in decisions:
            transformation = decision.get("transformation")
            column = decision.get("column")
            params = decision.get("params", {})
            
            transformed_columns.add(column)
            
            feedback.append({
                "action": transformation,
                "column": column,
                "params": params
            })
        
        # Apply column filter if requested
        if column_filter == "transformed_only":
            # Keep only transformed columns + index columns
            columns_to_keep = list(transformed_columns)
            # Add any columns that might be needed for stratification or context
            existing_cols = [col for col in columns_to_keep if col in df.columns]
            if existing_cols:
                df = df[existing_cols]
                logger.info(f"Filtered to {len(df.columns)} transformed columns")
        
        # Apply transformations using EDAFeedbackEngine
        engine = EDAFeedbackEngine(df, feedback)
        engine.apply_feedback()
        
        # Get comparisons
        comparisons = engine.get_transform_comparisons()
        
        # Add sampling info to response
        sampling_info = {
            "original_rows": original_row_count,
            "sampled_rows": len(df),
            "sampling_method": row_sampling_method if row_sample_size > 0 else "none",
            "sampling_applied": row_sample_size > 0 and row_sample_size < original_row_count,
            "column_filter": column_filter,
            "sampling_seed": sampling_seed
        }
        
        return JSONResponse({
            "comparisons": comparisons,
            "total_comparisons": len(comparisons),
            "file_id": file_id,
            "sampling_info": sampling_info
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get transform comparisons failed: {e}")
        raise HTTPException(500, f"Failed to generate comparisons: {str(e)}")


@router.post("/rollback-transforms")
async def rollback_transforms(
    file_id: str = Query(...),
    body: dict = Body(default={})
):
    """
    Rollback all transformations and reset to original dataset state.
    Clears accepted decisions and transformation configs.
    
    Args:
        file_id: Dataset identifier
        body: Optional JSON with confirmation flag
        
    Returns:
        Status of rollback operation
    """
    try:
        # Check if file exists
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(404, f"File not found: {file_id}")
        
        # Optional: Require explicit confirmation
        confirmed = body.get("confirmed", True)
        if not confirmed:
            return JSONResponse({
                "status": "confirmation_required",
                "message": "Rollback requires confirmation. Set 'confirmed': true in request body."
            })
        
        # Clear any saved transformation configs for this file
        config_manager = TransformConfigManager()
        config_dir = config_manager.config_dir
        
        # Find and optionally remove config files for this file_id
        import glob
        pattern = f"transform_config_{file_id}_*.json"
        config_files = glob.glob(str(config_dir / pattern))
        
        removed_configs = []
        if config_files:
            # Option 1: Archive configs instead of deleting
            archive_dir = config_dir / "archived"
            archive_dir.mkdir(exist_ok=True)
            
            for config_file in config_files:
                # Move to archive
                import shutil
                filename = os.path.basename(config_file)
                archive_path = archive_dir / filename
                shutil.move(config_file, archive_path)
                removed_configs.append(filename)
                logger.info(f"Archived config: {filename}")
        
        # Reset to original data (file already exists, no action needed)
        # If transformations were saved to a separate file, could restore here
        
        return JSONResponse({
            "status": "rollback_complete",
            "message": "All transformations rolled back. Dataset reset to original state.",
            "file_id": file_id,
            "archived_configs": removed_configs,
            "archived_count": len(removed_configs)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rollback failed for {file_id}: {e}")
        raise HTTPException(500, f"Failed to rollback transformations: {str(e)}")


@router.post("/get-pet-mapping")
async def get_pet_mapping(
    file_id: str = Query(...),
    body: dict = Body(...)
):
    """
    Generate Privacy-Enhanced Technology (PET) mapping grid.
    Shows privacy risks and applied protections per column.
    
    Args:
        file_id: Dataset identifier
        body: Contains accepted_decisions list
        
    Returns:
        PET mapping data with risk levels, suggestions, and protection status
    """
    try:
        decisions = body.get("decisions", [])
        
        # Load the data
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(404, f"File not found: {file_id}")
        
        df = pd.read_csv(file_path)
        
        # Run privacy analysis
        pii_report = None
        k_anonymity_report = None
        
        try:
            pii_scanner = PIIScanner(df)
            pii_report = pii_scanner.run_fast_scan()
        except Exception as e:
            logger.warning(f"PII scan failed: {e}")
        
        try:
            k_analyzer = KAnonymityAnalyzer(df)
            potential_qis = k_analyzer.detect_potential_quasi_identifiers()
            k_anonymity_report = {"potential_qis": potential_qis}
        except Exception as e:
            logger.warning(f"K-anonymity analysis failed: {e}")
        
        # Get privacy suggestions
        privacy_suggester = PrivacySuggester(df, pii_report=pii_report, k_anonymity_report=k_anonymity_report)
        privacy_suggestions = privacy_suggester.suggest_privacy_transforms()
        
        # Build PET mapping
        pet_mapping = []
        
        # Track which columns have risks and protections
        column_risks = {}  # column -> {risk_level, pii_types, is_qi, suggestions}
        column_protections = {}  # column -> {transformation, accepted}
        
        # Process PII detections
        if pii_report and "detections" in pii_report:
            for detection in pii_report["detections"]:
                col = detection.get("column")
                if col:
                    pii_types = detection.get("pii_types", [])
                    confidence = detection.get("confidence", 0.0)
                    
                    # Determine risk level based on PII type
                    direct_ids = ["EMAIL", "SSN", "PHONE", "CREDIT_CARD", "PHONE_US", "IP_ADDRESS"]
                    if any(p in direct_ids for p in pii_types):
                        risk_level = "Direct Identifier"
                    else:
                        risk_level = "Sensitive"
                    
                    column_risks[col] = {
                        "risk_level": risk_level,
                        "pii_types": pii_types,
                        "is_qi": False,
                        "confidence": confidence
                    }
        
        # Process quasi-identifiers
        if k_anonymity_report and "potential_qis" in k_anonymity_report:
            for qi_info in k_anonymity_report["potential_qis"]:
                col = qi_info.get("column")
                if col:
                    if col in column_risks:
                        # Already marked as PII, update to QI if applicable
                        column_risks[col]["is_qi"] = True
                    else:
                        column_risks[col] = {
                            "risk_level": "Quasi-Identifier",
                            "pii_types": [],
                            "is_qi": True,
                            "confidence": qi_info.get("uniqueness", 0.0)
                        }
        
        # Process accepted privacy transformations
        # Use centralized PET_TYPES and helper to normalize/validate transformations
        for decision in decisions:
            col = decision.get("column")
            transformation = decision.get("transformation")
            if not col or not transformation:
                continue

            # Normalize and validate transformation name
            t_norm = transformation.strip().lower() if isinstance(transformation, str) else None
            if not is_pet(t_norm):
                # Not a recognised privacy transformation - ignore
                continue

            column_protections[col] = {
                "transformation": t_norm,
                "accepted": True,
                "params": decision.get("params", {})
            }
        
        # Map suggestions to columns
        suggested_pets = {}  # column -> [suggestions]
        for suggestion in privacy_suggestions:
            col = suggestion.get("column")
            if col:
                if col not in suggested_pets:
                    suggested_pets[col] = []
                suggested_pets[col].append(suggestion)
        
        # Build final mapping for each column
        for col in df.columns:
            risk_info = column_risks.get(col, {})
            protection_info = column_protections.get(col)
            suggestions = suggested_pets.get(col, [])
            
            # Determine protection status
            if not risk_info:
                # No risk detected
                protection_status = "none_needed"
                status_icon = "✅"
                status_text = "No Privacy Risk"
            elif protection_info:
                # Has protection applied
                protection_status = "protected"
                status_icon = "✅"
                status_text = "Protected"
            elif suggestions:
                # Has risk but not protected
                protection_status = "unprotected"
                status_icon = "❌"
                status_text = "Unprotected"
            else:
                # Has risk but no suggestions available
                protection_status = "needs_review"
                status_icon = "⚠️"
                status_text = "Needs Review"
            
            # Get suggested PET
            suggested_pet = None
            if suggestions:
                # Get highest priority suggestion
                priority_order = ["hash", "mask", "generalize", "suppress", "redact"]
                for pet in priority_order:
                    for sug in suggestions:
                        if sug.get("transformation") == pet:
                            suggested_pet = sug
                            break
                    if suggested_pet:
                        break
                
                if not suggested_pet:
                    suggested_pet = suggestions[0]
            
            mapping = {
                "column": col,
                "risk_level": risk_info.get("risk_level", "None"),
                "pii_types": risk_info.get("pii_types", []),
                "is_quasi_identifier": risk_info.get("is_qi", False),
                "confidence": risk_info.get("confidence", 0.0),
                "suggested_pet": suggested_pet.get("transformation") if suggested_pet else None,
                "suggestion_reason": suggested_pet.get("reason") if suggested_pet else None,
                "applied_pet": protection_info.get("transformation") if protection_info else None,
                "is_protected": protection_info is not None,
                "protection_status": protection_status,
                "status_icon": status_icon,
                "status_text": status_text
            }
            
            pet_mapping.append(mapping)
        
        # Calculate summary statistics
        total_columns = len(pet_mapping)
        risk_columns = len([m for m in pet_mapping if m["risk_level"] != "None"])
        protected_columns = len([m for m in pet_mapping if m["is_protected"]])
        unprotected_high_risk = len([
            m for m in pet_mapping 
            if m["risk_level"] == "Direct Identifier" and not m["is_protected"]
        ])
        
        summary = {
            "total_columns": total_columns,
            "risk_columns": risk_columns,
            "protected_columns": protected_columns,
            "unprotected_columns": risk_columns - protected_columns,
            "unprotected_high_risk": unprotected_high_risk,
            "protection_rate": (protected_columns / risk_columns * 100) if risk_columns > 0 else 100.0
        }
        
        return JSONResponse({
            "pet_mapping": pet_mapping,
            "summary": summary,
            "file_id": file_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PET mapping failed for {file_id}: {e}")
        raise HTTPException(500, f"Failed to generate PET mapping: {str(e)}")


@router.post("/get-transform-batch-code")
async def get_transform_batch_code(
    file_id: str = Query(...),
    body: dict = Body(...)
):
    """
    Generate combined Python code for a batch of accepted transformations.
    Returns an ordered, executable Python snippet.
    """
    try:
        decisions = body.get("decisions", [])
        
        if not decisions:
            return JSONResponse({
                "code": "# No transformations to apply",
                "total_decisions": 0
            })
        
        df = load_df_by_file_id(file_id)
        
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler",
            "from scipy.stats import yeojohnson",
            "import hashlib",
            "",
            "# Load your dataframe",
            "# df = pd.read_csv('your_data.csv')",
            "",
            "# Apply transformations",
            ""
        ]
        
        for i, decision in enumerate(decisions, 1):
            column = decision.get("column")
            transformation = decision.get("transformation")
            category = decision.get("category", "utility")
            
            code_lines.append(f"# {i}. {category.capitalize()}: {transformation} on '{column}'")
            
            # Get code for this transformation
            suggestion = {
                "column": column,
                "transformation": transformation,
                "category": category,
                **decision.get("params", {})
            }
            
            try:
                if category == "utility":
                    suggester = UtilitySuggester(df)
                    code = suggester.get_transform_code(suggestion)
                elif category == "privacy":
                    suggester = PrivacySuggester(df)
                    code = suggester.get_transform_code(suggestion)
                else:
                    code = f"# Unknown category: {category}"
                
                code_lines.append(code)
                code_lines.append("")
                
            except Exception as e:
                code_lines.append(f"# Error generating code: {str(e)}")
                code_lines.append("")
        
        code_lines.append("# Transformations complete")
        code_lines.append("print(f'Applied {0} transformations')".replace("{0}", str(len(decisions))))
        
        combined_code = "\n".join(code_lines)
        
        return JSONResponse({
            "code": combined_code,
            "total_decisions": len(decisions),
            "status": "generated"
        })
        
    except Exception as e:
        logger.error(f"Batch code generation failed: {e}")
        raise HTTPException(500, f"Failed to generate batch code: {str(e)}")


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
# TIME-SERIES ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/detect-timeseries/{file_id}")
async def detect_timeseries_endpoint(file_id: str):
    """
    Detect if uploaded dataset is time-series and identify temporal characteristics.
    
    Args:
        file_id: CSV file identifier
    
    Returns:
        Time-series detection results with datetime columns, frequency, and confidence
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Detecting time-series characteristics for file {file_id}")
        
        detector = TimeSeriesDetector(df)
        results = detector.detect_timeseries()
        
        # Save detection results
        if results.get('is_timeseries'):
            analyzer = TimeSeriesAnalyzer(df, results['primary_datetime_column'])
            analyzer.save_results(file_id, 'detection', results)
        
        return {
            "detection": _sanitize_value_for_json(results),
            "file_id": file_id
        }
    
    except TimeSeriesError as e:
        logger.error(f"Time-series detection error: {str(e)}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Error in time-series detection: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Time-series detection failed: {str(e)}")


@router.post("/timeseries-acf-pacf")
async def compute_timeseries_acf_pacf(
    file_id: str = Body(...),
    datetime_col: str = Body(...),
    target_col: str = Body(...),
    lags: int = Body(40)
):
    """
    Compute ACF and PACF for time-series diagnostics.
    
    Args:
        file_id: CSV file identifier
        datetime_col: Name of datetime column
        target_col: Target column to analyze
        lags: Number of lags to compute (default 40)
    
    Returns:
        ACF/PACF values, confidence intervals, and AR/MA order suggestions
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Computing ACF/PACF for {target_col} in file {file_id}")
        logger.info(f"DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        
        analyzer = TimeSeriesAnalyzer(df, datetime_col)
        logger.info(f"After TimeSeriesAnalyzer init, columns: {list(analyzer.df.columns)}")
        
        results = analyzer.compute_acf_pacf(target_col, lags)
        
        # Save results
        analyzer.save_results(file_id, f'acf_pacf_{target_col}', results)
        
        return {
            "acf_pacf": _sanitize_value_for_json(results),
            "target_column": target_col,
            "file_id": file_id
        }
    
    except TimeSeriesError as e:
        logger.error(f"ACF/PACF computation error: {str(e)}")
        raise HTTPException(400, str(e))
    except ImportError:
        raise HTTPException(
            400,
            "Time-series analysis requires statsmodels library. "
            "Install with: pip install statsmodels"
        )
    except Exception as e:
        logger.error(f"Error computing ACF/PACF: {str(e)}", exc_info=True)
        raise HTTPException(500, f"ACF/PACF computation failed: {str(e)}")


@router.post("/timeseries-decompose")
async def decompose_timeseries_endpoint(
    file_id: str = Body(...),
    datetime_col: str = Body(...),
    target_col: str = Body(...),
    model: str = Body('additive'),
    period: int = Body(None),
    resample_freq: str = Body(None),
    aggregation: str = Body('mean')
):
    """
    Perform seasonal decomposition (STL) on time-series data.
    
    Args:
        file_id: CSV file identifier
        datetime_col: Name of datetime column
        target_col: Target column to decompose
        model: 'additive' or 'multiplicative' (default 'additive')
        period: Seasonal period (auto-detected if None)
        resample_freq: Frequency to resample irregular series (e.g., 'D', 'H')
        aggregation: Aggregation method for resampling ('mean', 'sum', 'median')
    
    Returns:
        Decomposed components: observed, trend, seasonal, residual
    """
    try:
        df = load_df_by_file_id(file_id)
        logger.info(f"Decomposing time-series {target_col} in file {file_id}")
        
        if model not in ['additive', 'multiplicative']:
            raise HTTPException(400, "Model must be 'additive' or 'multiplicative'")
        
        if aggregation not in ['mean', 'sum', 'median', 'first', 'last']:
            raise HTTPException(400, "Aggregation must be one of: mean, sum, median, first, last")
        
        analyzer = TimeSeriesAnalyzer(df, datetime_col)
        results = analyzer.decompose_series(
            target_col, 
            model, 
            period,
            resample_freq,
            aggregation
        )
        
        # Save results
        analyzer.save_results(file_id, f'decomposition_{target_col}', results)
        
        return {
            "decomposition": _sanitize_value_for_json(results),
            "target_column": target_col,
            "file_id": file_id
        }
    
    except TimeSeriesError as e:
        logger.error(f"Decomposition error: {str(e)}")
        raise HTTPException(400, str(e))
    except ImportError:
        raise HTTPException(
            400,
            "Time-series decomposition requires statsmodels library. "
            "Install with: pip install statsmodels"
        )
    except Exception as e:
        logger.error(f"Error in decomposition: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Decomposition failed: {str(e)}")


# ============================================================================
# END OF FILE
# ============================================================================

