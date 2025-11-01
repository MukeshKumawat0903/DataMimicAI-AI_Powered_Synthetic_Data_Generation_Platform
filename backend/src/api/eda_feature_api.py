# src/api/eda_feature_api.py

from fastapi import APIRouter, HTTPException, Request, Query, Body
from fastapi.responses import JSONResponse, Response
import pandas as pd
import logging
import os

from src.core.eda.profiling import (
    Profiler, DataCleaner, EDAConfig, ProfilingError
)
from src.core.eda.correlation import (
    CorrelationAnalyzer, CorrelationConfig, CorrelationError, make_corr_heatmap_base64
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
    max_pairs: int = Query(3)
):
    """Get AI-powered feature suggestions."""
    df = load_df_by_file_id(file_id)
    try:
        config = FeatureEngConfig(max_pairs=max_pairs)
        suggester = FeatureSuggester(df, config=config)
        results = suggester.suggest(target_col=target_col)
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
# END OF FILE
# ============================================================================
