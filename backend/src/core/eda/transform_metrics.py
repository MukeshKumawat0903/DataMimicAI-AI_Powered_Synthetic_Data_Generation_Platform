"""
Transformation Metrics Module
Calculates before/after metrics for data transformations.
Part of the EDA transformation analysis system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy import stats


def generate_metric_deltas(
    column: str,
    original_data: pd.Series,
    transformed_data: pd.Series
) -> Dict[str, Any]:
    """
    Generate before/after metric comparison for a transformed column.
    
    This function calculates comprehensive statistics for both original
    and transformed data, including:
    - Central tendency (mean, median)
    - Spread (std dev, min, max)
    - Distribution shape (skewness, kurtosis)
    - Outlier detection (IQR method)
    
    Args:
        column: Column name
        original_data: Original column data (before transformation)
        transformed_data: Transformed column data (after transformation)
        
    Returns:
        Dictionary with the following structure:
        {
            "column": str,
            "is_numeric": bool,
            "original": {
                "mean": float,
                "median": float,
                "std": float,
                "min": float,
                "max": float,
                "skewness": float,
                "kurtosis": float,
                "outlier_count": int,
                "outlier_percentage": float
            },
            "transformed": { ... },
            "deltas": {
                "skewness": {
                    "absolute": float,
                    "percentage": float,
                    "formatted": str
                },
                ...
            }
        }
        
    Examples:
        >>> original = pd.Series([1, 2, 3, 100])
        >>> transformed = pd.Series(np.log1p([1, 2, 3, 100]))
        >>> metrics = generate_metric_deltas("value", original, transformed)
        >>> metrics["deltas"]["skewness"]["formatted"]
        '2.89 → 0.12'
    """
    metrics = {
        "column": column,
        "original": {},
        "transformed": {},
        "deltas": {}
    }
    
    # Check if data is numeric
    if not pd.api.types.is_numeric_dtype(original_data):
        metrics["is_numeric"] = False
        return metrics
    
    metrics["is_numeric"] = True
    
    # Calculate original metrics
    orig_clean = original_data.dropna()
    if len(orig_clean) > 0:
        metrics["original"] = _calculate_statistics(orig_clean)
    
    # Calculate transformed metrics
    trans_clean = transformed_data.dropna()
    if len(trans_clean) > 0 and pd.api.types.is_numeric_dtype(transformed_data):
        metrics["transformed"] = _calculate_statistics(trans_clean)
        
        # Calculate deltas
        metrics["deltas"] = _calculate_deltas(
            metrics["original"],
            metrics["transformed"]
        )
    
    return metrics


def _calculate_statistics(data: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a numeric series.
    
    Args:
        data: Cleaned numeric series (no NaN values)
        
    Returns:
        Dictionary with calculated statistics
    """
    import math
    
    # Clean data: replace inf/-inf with NaN and drop all NaN
    data_clean = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Check minimum count for meaningful statistics
    if len(data_clean) < 3:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "skewness": None,
            "kurtosis": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "error": "Insufficient data (need at least 3 values)"
        }
    
    # Calculate basic statistics with safety checks
    try:
        mean_val = float(data_clean.mean())
        median_val = float(data_clean.median())
        std_val = float(data_clean.std())
        min_val = float(data_clean.min())
        max_val = float(data_clean.max())
        
        # Ensure finite values
        stats_dict = {
            "mean": mean_val if math.isfinite(mean_val) else None,
            "median": median_val if math.isfinite(median_val) else None,
            "std": std_val if math.isfinite(std_val) else None,
            "min": min_val if math.isfinite(min_val) else None,
            "max": max_val if math.isfinite(max_val) else None,
        }
        
        # Calculate skewness and kurtosis (may fail for small/uniform data)
        try:
            skew_val = float(stats.skew(data_clean))
            stats_dict["skewness"] = skew_val if math.isfinite(skew_val) else None
        except Exception:
            stats_dict["skewness"] = None
        
        try:
            kurt_val = float(stats.kurtosis(data_clean))
            stats_dict["kurtosis"] = kurt_val if math.isfinite(kurt_val) else None
        except Exception:
            stats_dict["kurtosis"] = None
        
        # Outlier detection using IQR method
        try:
            Q1 = data_clean.quantile(0.25)
            Q3 = data_clean.quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((data_clean < (Q1 - 1.5 * IQR)) | (data_clean > (Q3 + 1.5 * IQR))).sum()
            stats_dict["outlier_count"] = int(outliers)
            stats_dict["outlier_percentage"] = float(outliers / len(data_clean) * 100)
        except Exception:
            stats_dict["outlier_count"] = 0
            stats_dict["outlier_percentage"] = 0.0
        
        return stats_dict
        
    except Exception as e:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "skewness": None,
            "kurtosis": None,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "error": f"Calculation error: {str(e)}"
        }


def _calculate_deltas(
    original: Dict[str, float],
    transformed: Dict[str, float]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate delta (change) metrics between original and transformed.
    
    Args:
        original: Original statistics dictionary
        transformed: Transformed statistics dictionary
        
    Returns:
        Dictionary of delta metrics with absolute, percentage, and formatted values
    """
    deltas = {}
    
    # Metrics to track deltas for
    metric_keys = ["mean", "median", "std", "skewness", "kurtosis", "outlier_percentage"]
    
    for key in metric_keys:
        if key in original and key in transformed:
            orig_val = original[key]
            trans_val = transformed[key]
            
            # Skip if either value is None (calculation failed)
            if orig_val is None or trans_val is None:
                continue
            
            # Calculate absolute and percentage change
            absolute_change = trans_val - orig_val
            percentage_change = (absolute_change / orig_val * 100) if orig_val != 0 else 0
            
            deltas[key] = {
                "absolute": float(absolute_change),
                "percentage": float(percentage_change),
                "formatted": f"{orig_val:.2f} → {trans_val:.2f}"
            }
    
    return deltas


def calculate_improvement_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate an overall improvement score based on transformation metrics.
    
    Considers reduction in skewness and outliers as positive improvements.
    
    Args:
        metrics: Metrics dictionary from generate_metric_deltas()
        
    Returns:
        Dictionary with improvement scores and interpretation
    """
    if not metrics.get("is_numeric") or not metrics.get("deltas"):
        return {
            "overall_score": 0,
            "interpretation": "Not applicable for non-numeric data"
        }
    
    score = 0
    factors = []
    
    deltas = metrics["deltas"]
    
    # Skewness improvement (closer to 0 is better)
    if "skewness" in deltas:
        orig_skew_abs = abs(metrics["original"]["skewness"])
        trans_skew_abs = abs(metrics["transformed"]["skewness"])
        
        if trans_skew_abs < orig_skew_abs:
            reduction = ((orig_skew_abs - trans_skew_abs) / orig_skew_abs * 100) if orig_skew_abs != 0 else 0
            score += min(reduction, 50)  # Cap at 50 points
            factors.append(f"Skewness reduced by {reduction:.1f}%")
    
    # Outlier improvement (fewer outliers is better)
    if "outlier_percentage" in deltas:
        orig_outliers = metrics["original"]["outlier_percentage"]
        trans_outliers = metrics["transformed"]["outlier_percentage"]
        
        if trans_outliers < orig_outliers:
            reduction = ((orig_outliers - trans_outliers) / orig_outliers * 100) if orig_outliers != 0 else 0
            score += min(reduction / 2, 50)  # Cap at 50 points
            factors.append(f"Outliers reduced by {reduction:.1f}%")
    
    # Interpret the score
    if score >= 75:
        interpretation = "Excellent improvement"
    elif score >= 50:
        interpretation = "Good improvement"
    elif score >= 25:
        interpretation = "Moderate improvement"
    elif score > 0:
        interpretation = "Minor improvement"
    else:
        interpretation = "No significant improvement"
    
    return {
        "overall_score": round(score, 2),
        "interpretation": interpretation,
        "factors": factors
    }
