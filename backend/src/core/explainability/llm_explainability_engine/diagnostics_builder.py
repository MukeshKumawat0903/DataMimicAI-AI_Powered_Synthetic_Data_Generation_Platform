"""
Diagnostics Builder - Deterministic EDA Issue Classification.

This module takes raw EDA outputs and produces structured diagnostic entries
with severity levels. It uses fixed thresholds and contains NO LLM calls,
NO recommendations, and NO external service dependencies.

The output is deterministic and repeatable for the same input.

Author: DataMimicAI Team
Date: February 2026
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# THRESHOLD CONFIGURATION (Fixed, Deterministic)
# ============================================================================

@dataclass
class DiagnosticsThresholds:
    """
    Fixed thresholds for diagnostic classification.
    
    All thresholds are deterministic and well-defined.
    Modify these values to adjust sensitivity of issue detection.
    """
    
    # Skewness thresholds (absolute value)
    skewness_low: float = 0.5      # Low concern: 0.5 to 1.0
    skewness_medium: float = 1.0   # Medium concern: 1.0 to 2.0
    skewness_high: float = 2.0     # High concern: > 2.0
    
    # Missing value percentage thresholds
    missing_low: float = 5.0       # Low concern: 5% to 20%
    missing_medium: float = 20.0   # Medium concern: 20% to 50%
    missing_high: float = 50.0     # High concern: > 50%
    
    # Outlier percentage thresholds
    outlier_low: float = 2.0       # Low concern: 2% to 5%
    outlier_medium: float = 5.0    # Medium concern: 5% to 10%
    outlier_high: float = 10.0     # High concern: > 10%
    
    # Drift thresholds (KS statistic)
    drift_ks_low: float = 0.1      # Low drift: 0.1 to 0.2
    drift_ks_medium: float = 0.2   # Medium drift: 0.2 to 0.3
    drift_ks_high: float = 0.3     # High drift: > 0.3
    
    # Drift thresholds (PSI - Population Stability Index)
    drift_psi_low: float = 0.1     # Low drift: 0.1 to 0.2
    drift_psi_medium: float = 0.2  # Medium drift: 0.2 to 0.3
    drift_psi_high: float = 0.3    # High drift: > 0.3
    
    # Imbalance thresholds (categorical)
    imbalance_low: float = 0.7     # Low concern: 70% to 85%
    imbalance_medium: float = 0.85 # Medium concern: 85% to 95%
    imbalance_high: float = 0.95   # High concern: > 95%
    
    # High cardinality thresholds (categorical)
    cardinality_low: int = 50      # Low concern: 50 to 100
    cardinality_medium: int = 100  # Medium concern: 100 to 500
    cardinality_high: int = 500    # High concern: > 500
    
    # Correlation thresholds (absolute value)
    correlation_low: float = 0.6   # Low concern: 0.6 to 0.8
    correlation_medium: float = 0.8  # Medium concern: 0.8 to 0.95
    correlation_high: float = 0.95   # High concern: > 0.95


# ============================================================================
# INPUT VALIDATION (Graceful, Non-Breaking)
# ============================================================================

def validate_signals_input(signals: Any) -> tuple[bool, Optional[str]]:
    """
    Validate signals input structure.
    
    This function checks that signals is a valid dictionary with at least
    one of the required keys. It does NOT raise exceptions.
    
    Parameters
    ----------
    signals : Any
        Input to validate (expected to be a dict)
    
    Returns
    -------
    tuple[bool, Optional[str]]
        (is_valid, error_message)
        - If valid: (True, None)
        - If invalid: (False, "error description")
    """
    # Check if signals is a dictionary
    if not isinstance(signals, dict):
        return False, f"signals must be a dictionary, got {type(signals).__name__}"
    
    # Check if at least one required key exists
    required_keys = ["columns", "dataset_summary", "correlations"]
    has_required_key = any(key in signals for key in required_keys)
    
    if not has_required_key:
        return False, f"signals must contain at least one of: {required_keys}"
    
    return True, None


def get_current_utc_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.
    
    Returns
    -------
    str
        ISO 8601 formatted UTC timestamp
    """
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# SEVERITY CLASSIFICATION FUNCTIONS (Pure, Deterministic)
# ============================================================================

def classify_skewness_severity(skewness_value: float, thresholds: DiagnosticsThresholds) -> str:
    """
    Classify skewness severity based on absolute value.
    
    Parameters
    ----------
    skewness_value : float
        Absolute skewness value
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    str
        Severity level: 'low', 'medium', or 'high'
    """
    abs_skew = abs(skewness_value)
    
    if abs_skew >= thresholds.skewness_high:
        return "high"
    elif abs_skew >= thresholds.skewness_medium:
        return "medium"
    elif abs_skew >= thresholds.skewness_low:
        return "low"
    else:
        return None  # Below threshold, not an issue


def classify_missing_severity(missing_pct: float, thresholds: DiagnosticsThresholds) -> str:
    """
    Classify missing value severity.
    
    Parameters
    ----------
    missing_pct : float
        Missing value percentage (0-100)
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    str
        Severity level: 'low', 'medium', or 'high'
    """
    if missing_pct >= thresholds.missing_high:
        return "high"
    elif missing_pct >= thresholds.missing_medium:
        return "medium"
    elif missing_pct >= thresholds.missing_low:
        return "low"
    else:
        return None  # Below threshold


def classify_outlier_severity(outlier_pct: float, thresholds: DiagnosticsThresholds) -> str:
    """
    Classify outlier severity.
    
    Parameters
    ----------
    outlier_pct : float
        Outlier percentage (0-100)
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    str
        Severity level: 'low', 'medium', or 'high'
    """
    if outlier_pct >= thresholds.outlier_high:
        return "high"
    elif outlier_pct >= thresholds.outlier_medium:
        return "medium"
    elif outlier_pct >= thresholds.outlier_low:
        return "low"
    else:
        return None


def classify_drift_ks_severity(ks_stat: float, thresholds: DiagnosticsThresholds) -> str:
    """
    Classify drift severity based on KS statistic.
    
    Parameters
    ----------
    ks_stat : float
        Kolmogorov-Smirnov statistic (0-1)
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    str
        Severity level: 'low', 'medium', or 'high'
    """
    if ks_stat >= thresholds.drift_ks_high:
        return "high"
    elif ks_stat >= thresholds.drift_ks_medium:
        return "medium"
    elif ks_stat >= thresholds.drift_ks_low:
        return "low"
    else:
        return None


def classify_drift_psi_severity(psi_value: float, thresholds: DiagnosticsThresholds) -> str:
    """
    Classify drift severity based on PSI (Population Stability Index).
    
    Parameters
    ----------
    psi_value : float
        PSI value
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    str
        Severity level: 'low', 'medium', or 'high'
    """
    if psi_value >= thresholds.drift_psi_high:
        return "high"
    elif psi_value >= thresholds.drift_psi_medium:
        return "medium"
    elif psi_value >= thresholds.drift_psi_low:
        return "low"
    else:
        return None


def classify_imbalance_severity(imbalance_ratio: float, thresholds: DiagnosticsThresholds) -> str:
    """
    Classify imbalance severity for categorical columns.
    
    Parameters
    ----------
    imbalance_ratio : float
        Ratio of most frequent category (0-1)
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    str
        Severity level: 'low', 'medium', or 'high'
    """
    if imbalance_ratio >= thresholds.imbalance_high:
        return "high"
    elif imbalance_ratio >= thresholds.imbalance_medium:
        return "medium"
    elif imbalance_ratio >= thresholds.imbalance_low:
        return "low"
    else:
        return None


def classify_cardinality_severity(cardinality: int, thresholds: DiagnosticsThresholds) -> str:
    """
    Classify high cardinality severity for categorical columns.
    
    Parameters
    ----------
    cardinality : int
        Number of unique values
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    str
        Severity level: 'low', 'medium', or 'high'
    """
    if cardinality >= thresholds.cardinality_high:
        return "high"
    elif cardinality >= thresholds.cardinality_medium:
        return "medium"
    elif cardinality >= thresholds.cardinality_low:
        return "low"
    else:
        return None


def classify_correlation_severity(correlation: float, thresholds: DiagnosticsThresholds) -> str:
    """
    Classify correlation severity.
    
    Parameters
    ----------
    correlation : float
        Absolute correlation value (0-1)
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    str
        Severity level: 'low', 'medium', or 'high'
    """
    abs_corr = abs(correlation)
    
    if abs_corr >= thresholds.correlation_high:
        return "high"
    elif abs_corr >= thresholds.correlation_medium:
        return "medium"
    elif abs_corr >= thresholds.correlation_low:
        return "low"
    else:
        return None


# ============================================================================
# DIAGNOSTIC DETECTION FUNCTIONS (Pure, Deterministic)
# ============================================================================

def detect_skewness_issues(
    signals: Dict[str, Any],
    thresholds: DiagnosticsThresholds
) -> List[Dict[str, Any]]:
    """
    Detect high skewness issues in numeric columns.
    
    Parameters
    ----------
    signals : dict
        Signals from build_explainable_signals()
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    list of dict
        Diagnostic entries for skewness issues
    """
    diagnostics = []
    columns = signals.get("columns", {})
    
    if not isinstance(columns, dict):
        return diagnostics
    
    for col_name, col_info in columns.items():
        if not isinstance(col_info, dict):
            continue
        
        if col_info.get("type") != "numeric":
            continue
        
        skewness = col_info.get("skewness")
        if skewness is None:
            continue
        
        try:
            skewness_float = float(skewness)
            severity = classify_skewness_severity(skewness_float, thresholds)
            if severity:
                diagnostics.append({
                    "issue_type": "high_skew",
                    "affected_columns": [col_name],
                    "metric_name": "skewness",
                    "metric_value": round(skewness_float, 4),
                    "severity": severity
                })
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping column {col_name} due to invalid skewness value: {e}")
            continue
    
    return diagnostics


def detect_missing_value_issues(
    signals: Dict[str, Any],
    thresholds: DiagnosticsThresholds
) -> List[Dict[str, Any]]:
    """
    Detect missing value issues.
    
    Parameters
    ----------
    signals : dict
        Signals from build_explainable_signals()
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    list of dict
        Diagnostic entries for missing value issues
    """
    diagnostics = []
    columns = signals.get("columns", {})
    
    if not isinstance(columns, dict):
        return diagnostics
    
    for col_name, col_info in columns.items():
        if not isinstance(col_info, dict):
            continue
        
        missing_pct = col_info.get("missing_pct", 0.0)
        
        try:
            missing_pct_float = float(missing_pct)
            severity = classify_missing_severity(missing_pct_float, thresholds)
            if severity:
                diagnostics.append({
                    "issue_type": "missing_values",
                    "affected_columns": [col_name],
                    "metric_name": "missing_percentage",
                    "metric_value": round(missing_pct_float, 2),
                    "severity": severity
                })
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping column {col_name} due to invalid missing_pct value: {e}")
            continue
    
    return diagnostics


def detect_outlier_issues(
    signals: Dict[str, Any],
    thresholds: DiagnosticsThresholds
) -> List[Dict[str, Any]]:
    """
    Detect outlier issues in numeric columns.
    
    Parameters
    ----------
    signals : dict
        Signals from build_explainable_signals()
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    list of dict
        Diagnostic entries for outlier issues
    """
    diagnostics = []
    columns = signals.get("columns", {})
    
    if not isinstance(columns, dict):
        return diagnostics
    
    for col_name, col_info in columns.items():
        if not isinstance(col_info, dict):
            continue
        
        if col_info.get("type") != "numeric":
            continue
        
        outlier_pct = col_info.get("outlier_pct")
        if outlier_pct is None:
            continue
        
        try:
            outlier_pct_float = float(outlier_pct)
            severity = classify_outlier_severity(outlier_pct_float, thresholds)
            if severity:
                diagnostics.append({
                    "issue_type": "outliers",
                    "affected_columns": [col_name],
                    "metric_name": "outlier_percentage",
                    "metric_value": round(outlier_pct_float, 2),
                    "severity": severity
                })
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping column {col_name} due to invalid outlier_pct value: {e}")
            continue
    
    return diagnostics


def detect_drift_issues(
    signals: Dict[str, Any],
    thresholds: DiagnosticsThresholds
) -> List[Dict[str, Any]]:
    """
    Detect drift issues (if drift metrics are present).
    
    Parameters
    ----------
    signals : dict
        Signals from build_explainable_signals()
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    list of dict
        Diagnostic entries for drift issues
    """
    diagnostics = []
    
    # Check if drift information exists in signals
    # This is optional and may not always be present
    drift_info = signals.get("drift", {})
    if not drift_info or not isinstance(drift_info, dict):
        return diagnostics
    
    columns = drift_info.get("columns", {})
    if not isinstance(columns, dict):
        return diagnostics
    
    for col_name, drift_data in columns.items():
        if not isinstance(drift_data, dict):
            continue
        
        # KS statistic drift
        ks_stat = drift_data.get("ks_statistic")
        if ks_stat is not None:
            try:
                ks_stat_float = float(ks_stat)
                severity = classify_drift_ks_severity(ks_stat_float, thresholds)
                if severity:
                    diagnostics.append({
                        "issue_type": "drift",
                        "affected_columns": [col_name],
                        "metric_name": "ks_statistic",
                        "metric_value": round(ks_stat_float, 4),
                        "severity": severity
                    })
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping ks_statistic for {col_name}: {e}")
        
        # PSI drift
        psi_value = drift_data.get("psi")
        if psi_value is not None:
            try:
                psi_value_float = float(psi_value)
                severity = classify_drift_psi_severity(psi_value_float, thresholds)
                if severity:
                    diagnostics.append({
                        "issue_type": "drift",
                        "affected_columns": [col_name],
                        "metric_name": "psi",
                        "metric_value": round(psi_value_float, 4),
                        "severity": severity
                    })
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping psi for {col_name}: {e}")
    
    return diagnostics


def detect_imbalance_issues(
    signals: Dict[str, Any],
    thresholds: DiagnosticsThresholds
) -> List[Dict[str, Any]]:
    """
    Detect class imbalance issues in categorical columns.
    
    Parameters
    ----------
    signals : dict
        Signals from build_explainable_signals()
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    list of dict
        Diagnostic entries for imbalance issues
    """
    diagnostics = []
    columns = signals.get("columns", {})
    
    if not isinstance(columns, dict):
        return diagnostics
    
    for col_name, col_info in columns.items():
        if not isinstance(col_info, dict):
            continue
        
        if col_info.get("type") != "categorical":
            continue
        
        # Check for imbalance ratio
        # explainable_signals.py provides 'dominant_category_pct'
        imbalance_ratio = col_info.get("imbalance_ratio")
        if imbalance_ratio is None:
            # Primary key used by explainable_signals.py
            dominant_pct = col_info.get("dominant_category_pct")
            if dominant_pct is not None:
                try:
                    imbalance_ratio = float(dominant_pct) / 100.0
                except (ValueError, TypeError):
                    pass
            else:
                # Fallback for other formats
                top_cat_pct = col_info.get("top_category_pct")
                if top_cat_pct is not None:
                    try:
                        imbalance_ratio = float(top_cat_pct) / 100.0
                    except (ValueError, TypeError):
                        pass
        
        if imbalance_ratio is None:
            continue
        
        try:
            imbalance_ratio_float = float(imbalance_ratio)
            severity = classify_imbalance_severity(imbalance_ratio_float, thresholds)
            if severity:
                diagnostics.append({
                    "issue_type": "imbalance",
                    "affected_columns": [col_name],
                    "metric_name": "imbalance_ratio",
                    "metric_value": round(imbalance_ratio_float, 4),
                    "severity": severity
                })
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping column {col_name} due to invalid imbalance_ratio value: {e}")
            continue
    
    return diagnostics


def detect_high_cardinality_issues(
    signals: Dict[str, Any],
    thresholds: DiagnosticsThresholds
) -> List[Dict[str, Any]]:
    """
    Detect high cardinality issues in categorical columns.
    
    Parameters
    ----------
    signals : dict
        Signals from build_explainable_signals()
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    list of dict
        Diagnostic entries for high cardinality issues
    """
    diagnostics = []
    columns = signals.get("columns", {})
    
    if not isinstance(columns, dict):
        return diagnostics
    
    for col_name, col_info in columns.items():
        if not isinstance(col_info, dict):
            continue
        
        if col_info.get("type") != "categorical":
            continue
        
        cardinality = col_info.get("cardinality")
        if cardinality is None:
            continue
        
        try:
            cardinality_int = int(cardinality)
            severity = classify_cardinality_severity(cardinality_int, thresholds)
            if severity:
                diagnostics.append({
                    "issue_type": "high_cardinality",
                    "affected_columns": [col_name],
                    "metric_name": "cardinality",
                    "metric_value": cardinality_int,
                    "severity": severity
                })
        except (ValueError, TypeError) as e:
            logger.debug(f"Skipping column {col_name} due to invalid cardinality value: {e}")
            continue
    
    return diagnostics


def detect_correlation_issues(
    signals: Dict[str, Any],
    thresholds: DiagnosticsThresholds
) -> List[Dict[str, Any]]:
    """
    Detect high correlation issues between numeric columns.
    
    Parameters
    ----------
    signals : dict
        Signals from build_explainable_signals()
    thresholds : DiagnosticsThresholds
        Threshold configuration
    
    Returns
    -------
    list of dict
        Diagnostic entries for correlation issues
    """
    diagnostics = []
    correlations = signals.get("correlations", {})
    
    # Handle both dict format {"pearson": [...], "spearman": [...]} and list format
    if isinstance(correlations, dict):
        # New format from explainable_signals.py
        pearson_corrs = correlations.get("pearson", [])
        spearman_corrs = correlations.get("spearman", [])
        
        # Process Pearson correlations
        if isinstance(pearson_corrs, list):
            for corr_info in pearson_corrs:
                if not isinstance(corr_info, dict):
                    continue
                # Support both 'column_1'/'column_2' and 'column1'/'column2' keys
                col1 = corr_info.get("column_1") or corr_info.get("column1")
                col2 = corr_info.get("column_2") or corr_info.get("column2")
                correlation = corr_info.get("correlation")
                
                if not col1 or not col2 or correlation is None:
                    continue
                
                try:
                    corr_float = float(correlation)
                    severity = classify_correlation_severity(abs(corr_float), thresholds)
                    if severity:
                        diagnostics.append({
                            "issue_type": "high_correlation",
                            "affected_columns": [col1, col2],
                            "metric_name": "pearson_correlation",
                            "metric_value": round(corr_float, 4),
                            "severity": severity
                        })
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping pearson correlation for {col1}-{col2}: {e}")
        
        # Process Spearman correlations
        if isinstance(spearman_corrs, list):
            for corr_info in spearman_corrs:
                if not isinstance(corr_info, dict):
                    continue
                # Support both 'column_1'/'column_2' and 'column1'/'column2' keys
                col1 = corr_info.get("column_1") or corr_info.get("column1")
                col2 = corr_info.get("column_2") or corr_info.get("column2")
                correlation = corr_info.get("correlation")
                
                if not col1 or not col2 or correlation is None:
                    continue
                
                try:
                    corr_float = float(correlation)
                    severity = classify_correlation_severity(abs(corr_float), thresholds)
                    if severity:
                        diagnostics.append({
                            "issue_type": "high_correlation",
                            "affected_columns": [col1, col2],
                            "metric_name": "spearman_correlation",
                            "metric_value": round(corr_float, 4),
                            "severity": severity
                        })
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping spearman correlation for {col1}-{col2}: {e}")
    
    elif isinstance(correlations, list):
        # Legacy format support (if needed)
        for corr_info in correlations:
            if not isinstance(corr_info, dict):
                continue
            col1 = corr_info.get("column1")
            col2 = corr_info.get("column2")
            
            if not col1 or not col2:
                continue
            
            # Check Pearson correlation
            pearson = corr_info.get("pearson")
            if pearson is not None:
                try:
                    pearson_float = float(pearson)
                    severity = classify_correlation_severity(abs(pearson_float), thresholds)
                    if severity:
                        diagnostics.append({
                            "issue_type": "high_correlation",
                            "affected_columns": [col1, col2],
                            "metric_name": "pearson_correlation",
                            "metric_value": round(pearson_float, 4),
                            "severity": severity
                        })
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping pearson correlation for {col1}-{col2}: {e}")
            
            # Check Spearman correlation
            spearman = corr_info.get("spearman")
            if spearman is not None:
                try:
                    spearman_float = float(spearman)
                    severity = classify_correlation_severity(abs(spearman_float), thresholds)
                    if severity:
                        diagnostics.append({
                            "issue_type": "high_correlation",
                            "affected_columns": [col1, col2],
                            "metric_name": "spearman_correlation",
                            "metric_value": round(spearman_float, 4),
                            "severity": severity
                        })
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skipping spearman correlation for {col1}-{col2}: {e}")
    
    return diagnostics


# ============================================================================
# MAIN DIAGNOSTICS BUILDER (Pure, Deterministic)
# ============================================================================

def build_diagnostics(
    signals: Dict[str, Any],
    thresholds: Optional[DiagnosticsThresholds] = None,
    target_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build structured diagnostics from raw EDA signals.
    
    This is the main entry point for diagnostic classification.
    It takes signals from build_explainable_signals() and produces
    a structured diagnostics dictionary with severity-classified issues.
    
    This function is:
    - Deterministic: Same input always produces same output
    - Pure: No side effects, no external calls
    - Repeatable: No randomness or non-deterministic operations
    
    Parameters
    ----------
    signals : dict
        Signals from build_explainable_signals() in explainable_signals.py
    thresholds : DiagnosticsThresholds, optional
        Custom threshold configuration. If None, uses default thresholds.
    target_columns : list of str, optional
        Specific columns to analyze. If None, analyzes all columns.
    
    Returns
    -------
    dict
        Structured diagnostics with format:
        {
            "diagnostics": [
                {
                    "issue_type": str,
                    "affected_columns": list,
                    "metric_name": str,
                    "metric_value": float,
                    "severity": str
                },
                ...
            ],
            "summary": {
                "total_issues": int,
                "high_severity_count": int,
                "medium_severity_count": int,
                "low_severity_count": int,
                "issue_types": {
                    "high_skew": int,
                    "missing_values": int,
                    "outliers": int,
                    "drift": int,
                    "imbalance": int,
                    "high_cardinality": int,
                    "high_correlation": int
                }
            }
        }
    
    Notes
    -----
    This function does NOT:
    - Call any LLM or external service
    - Generate recommendations or actions
    - Produce any non-deterministic output
    - Have any side effects
    
    Examples
    --------
    >>> from src.core.explainability.llm_explainability_engine import build_explainable_signals
    >>> signals = build_explainable_signals(df)
    >>> diagnostics = build_diagnostics(signals)
    >>> print(diagnostics["summary"]["total_issues"])
    5
    >>> print(diagnostics["diagnostics"][0]["issue_type"])
    'high_skew'
    """
    # Input validation (graceful, non-breaking)
    is_valid, error_message = validate_signals_input(signals)
    if not is_valid:
        logger.warning(f"Invalid signals input: {error_message}")
        return {
            "diagnostics": [],
            "summary": {
                "total_issues": 0,
                "high_severity_count": 0,
                "medium_severity_count": 0,
                "low_severity_count": 0,
                "issue_types": {
                    "high_skew": 0,
                    "missing_values": 0,
                    "outliers": 0,
                    "drift": 0,
                    "imbalance": 0,
                    "high_cardinality": 0,
                    "high_correlation": 0
                }
            },
            "metadata": {
                "signals_version": "1.0",
                "generated_at": get_current_utc_timestamp(),
                "num_columns_analyzed": 0,
                "error": error_message
            }
        }
    
    if thresholds is None:
        thresholds = DiagnosticsThresholds()
    
    # Filter signals if target_columns specified
    filtered_signals = signals
    if target_columns is not None and len(target_columns) > 0:
        logger.debug(f"Filtering diagnostics to columns: {target_columns}")
        filtered_signals = dict(signals)  # Shallow copy
        
        # Filter column-level data
        if "columns" in filtered_signals and isinstance(filtered_signals["columns"], dict):
            filtered_columns = {}
            for col in target_columns:
                if col in filtered_signals["columns"]:
                    filtered_columns[col] = filtered_signals["columns"][col]
            filtered_signals["columns"] = filtered_columns
        
        # Filter correlations to only include pairs with target columns
        if "correlations" in filtered_signals:
            corrs = filtered_signals["correlations"]
            if isinstance(corrs, dict):
                filtered_corrs = {"pearson": [], "spearman": []}
                for corr_type in ["pearson", "spearman"]:
                    if corr_type in corrs and isinstance(corrs[corr_type], list):
                        filtered_corrs[corr_type] = [
                            c for c in corrs[corr_type]
                            if isinstance(c, dict) and (
                                c.get("column_1") in target_columns or
                                c.get("column_2") in target_columns or
                                c.get("column1") in target_columns or
                                c.get("column2") in target_columns
                            )
                        ]
                filtered_signals["correlations"] = filtered_corrs
    else:
        logger.debug("Building diagnostics from all signals")
    
    # Collect all diagnostics
    all_diagnostics = []
    
    # Detect each type of issue (use filtered_signals, not original signals)
    all_diagnostics.extend(detect_skewness_issues(filtered_signals, thresholds))
    all_diagnostics.extend(detect_missing_value_issues(filtered_signals, thresholds))
    all_diagnostics.extend(detect_outlier_issues(filtered_signals, thresholds))
    # Note: Drift detection disabled - explainable_signals.py doesn't provide drift metrics yet
    # all_diagnostics.extend(detect_drift_issues(filtered_signals, thresholds))
    all_diagnostics.extend(detect_imbalance_issues(filtered_signals, thresholds))
    all_diagnostics.extend(detect_high_cardinality_issues(filtered_signals, thresholds))
    all_diagnostics.extend(detect_correlation_issues(filtered_signals, thresholds))
    
    # Build summary statistics
    severity_counts = {
        "high": 0,
        "medium": 0,
        "low": 0
    }
    
    issue_type_counts = {
        "high_skew": 0,
        "missing_values": 0,
        "outliers": 0,
        "drift": 0,
        "imbalance": 0,
        "high_cardinality": 0,
        "high_correlation": 0
    }
    
    for diagnostic in all_diagnostics:
        severity = diagnostic.get("severity")
        if severity in severity_counts:
            severity_counts[severity] += 1
        
        issue_type = diagnostic.get("issue_type")
        if issue_type in issue_type_counts:
            issue_type_counts[issue_type] += 1
    
    summary = {
        "total_issues": len(all_diagnostics),
        "high_severity_count": severity_counts["high"],
        "medium_severity_count": severity_counts["medium"],
        "low_severity_count": severity_counts["low"],
        "issue_types": issue_type_counts
    }
    
    logger.debug(f"Built {len(all_diagnostics)} diagnostic entries")
    
    # Count columns analyzed (use filtered_signals if filtering was applied)
    columns = filtered_signals.get("columns", {})
    num_columns = len(columns) if isinstance(columns, dict) else 0
    
    return {
        "diagnostics": all_diagnostics,
        "summary": summary,
        "metadata": {
            "signals_version": "1.0",
            "generated_at": get_current_utc_timestamp(),
            "num_columns_analyzed": num_columns
        }
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_diagnostics_by_severity(
    diagnostics_dict: Dict[str, Any],
    severity: str
) -> List[Dict[str, Any]]:
    """
    Filter diagnostics by severity level.
    
    Parameters
    ----------
    diagnostics_dict : dict
        Output from build_diagnostics()
    severity : str
        Severity level: 'low', 'medium', or 'high'
    
    Returns
    -------
    list of dict
        Filtered diagnostic entries
    """
    all_diagnostics = diagnostics_dict.get("diagnostics", [])
    return [d for d in all_diagnostics if d.get("severity") == severity]


def get_diagnostics_by_type(
    diagnostics_dict: Dict[str, Any],
    issue_type: str
) -> List[Dict[str, Any]]:
    """
    Filter diagnostics by issue type.
    
    Parameters
    ----------
    diagnostics_dict : dict
        Output from build_diagnostics()
    issue_type : str
        Issue type: 'high_skew', 'missing_values', 'outliers', etc.
    
    Returns
    -------
    list of dict
        Filtered diagnostic entries
    """
    all_diagnostics = diagnostics_dict.get("diagnostics", [])
    return [d for d in all_diagnostics if d.get("issue_type") == issue_type]


def get_affected_columns(diagnostics_dict: Dict[str, Any]) -> List[str]:
    """
    Get all columns affected by any diagnostic issue.
    
    Parameters
    ----------
    diagnostics_dict : dict
        Output from build_diagnostics()
    
    Returns
    -------
    list of str
        Unique column names with issues
    """
    all_diagnostics = diagnostics_dict.get("diagnostics", [])
    affected = set()
    
    for diagnostic in all_diagnostics:
        columns = diagnostic.get("affected_columns", [])
        affected.update(columns)
    
    return sorted(list(affected))


# Example usage (for testing only)
if __name__ == "__main__":
    # Sample signals for testing
    sample_signals = {
        "dataset_summary": {
            "num_rows": 1000,
            "num_columns": 5
        },
        "columns": {
            "age": {
                "type": "numeric",
                "missing_pct": 5.5,
                "skewness": 1.8,
                "outlier_pct": 3.2
            },
            "income": {
                "type": "numeric",
                "missing_pct": 25.0,
                "skewness": 2.5,
                "outlier_pct": 12.0
            },
            "department": {
                "type": "categorical",
                "cardinality": 150,
                "imbalance_ratio": 0.88
            }
        },
        "correlations": [
            {
                "column1": "age",
                "column2": "income",
                "pearson": 0.92,
                "spearman": 0.87
            }
        ]
    }
    
    diagnostics = build_diagnostics(sample_signals)
    
    print("Diagnostics Summary:")
    print(f"Total Issues: {diagnostics['summary']['total_issues']}")
    print(f"High Severity: {diagnostics['summary']['high_severity_count']}")
    print(f"Medium Severity: {diagnostics['summary']['medium_severity_count']}")
    print(f"Low Severity: {diagnostics['summary']['low_severity_count']}")
    
    print("\nDiagnostic Entries:")
    for diag in diagnostics["diagnostics"]:
        print(f"- {diag['issue_type']}: {diag['affected_columns']} "
              f"({diag['metric_name']}={diag['metric_value']}, severity={diag['severity']})")
