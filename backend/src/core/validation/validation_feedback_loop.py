"""
ValidationFeedbackLoop - Deterministic Transformation Impact Evaluator.

This module evaluates the impact of executed transformation plans by comparing
diagnostic metrics before and after transformation. It contains ZERO reasoning,
ZERO decision-making, and ZERO LLM usage.

This is NOT an agent. It is a pure comparison and reporting module.

Purpose:
--------
After a transformation plan is executed, this module computes the same metrics
on both the original and transformed datasets, then reports the deltas clearly.

Design Principles:
------------------
1. No reasoning or judgment - just metric computation and comparison
2. No success/failure classification - only delta reporting
3. Deterministic - same inputs produce same outputs
4. Reuses existing metric functions from EDA/diagnostics
5. Graceful error handling - unavailable metrics are recorded as such

Integration:
------------
- Does NOT modify execution engine
- Does NOT modify approval logic
- Does NOT feed results back into agents automatically
- Read-only and evaluative

Author: DataMimicAI Team
Date: February 6, 2026
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MetricStatus(str, Enum):
    """Status of metric computation."""
    SUCCESS = "SUCCESS"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


@dataclass
class MetricComparison:
    """
    Comparison result for a single metric on a single column.
    
    Attributes
    ----------
    metric : str
        Name of the metric (e.g., "skewness", "missing_pct")
    column : str
        Column name (or "dataset" for dataset-level metrics)
    before : float | None
        Metric value before transformation
    after : float | None
        Metric value after transformation
    delta : float | None
        Change in metric (after - before)
    status : MetricStatus
        Whether metric was successfully computed
    error : str | None
        Error message if status is ERROR
    """
    metric: str
    column: str
    before: Optional[float]
    after: Optional[float]
    delta: Optional[float]
    status: MetricStatus
    error: Optional[str] = None


@dataclass
class ValidationReport:
    """
    Complete validation report comparing original and transformed datasets.
    
    Attributes
    ----------
    plan_id : str
        Reference to the executed transformation plan
    validation_results : List[MetricComparison]
        List of all metric comparisons
    summary : Dict[str, Any]
        Summary statistics about the validation
    """
    plan_id: str
    validation_results: List[MetricComparison]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        def convert_value(val):
            """Convert NaN to None for JSON serialization."""
            if val is None:
                return None
            if isinstance(val, float) and np.isnan(val):
                return None
            return val
        
        return {
            "plan_id": self.plan_id,
            "validation_results": [
                {
                    "metric": comp.metric,
                    "column": comp.column,
                    "before": convert_value(comp.before),
                    "after": convert_value(comp.after),
                    "delta": convert_value(comp.delta),
                    "status": comp.status.value,
                    "error": comp.error
                }
                for comp in self.validation_results
            ],
            "summary": self.summary
        }


# ============================================================================
# VALIDATION FEEDBACK LOOP (Non-Agentic Evaluator)
# ============================================================================

class ValidationFeedbackLoop:
    """
    Deterministic transformation impact evaluator.
    
    This class compares metrics before and after transformation execution.
    It contains NO reasoning, NO decision-making, and NO LLM usage.
    
    Methods
    -------
    validate(plan_id, original_data, transformed_data) -> ValidationReport
        Compute and compare metrics between original and transformed datasets
    """
    
    def __init__(self):
        """Initialize the validation feedback loop."""
        logger.info("ValidationFeedbackLoop initialized (non-agentic evaluator)")
    
    def validate(
        self,
        plan_id: str,
        original_data: pd.DataFrame,
        transformed_data: pd.DataFrame
    ) -> ValidationReport:
        """
        Evaluate transformation impact by comparing metrics.
        
        This method computes the same metrics on both datasets and reports
        the deltas. It does NOT judge success or failure.
        
        Parameters
        ----------
        plan_id : str
            Reference to the executed transformation plan
        original_data : pd.DataFrame
            Original dataset (pre-transformation)
        transformed_data : pd.DataFrame
            Transformed dataset (post-transformation)
        
        Returns
        -------
        ValidationReport
            Structured comparison of metrics before and after
        
        Raises
        ------
        ValueError
            If inputs are invalid (not DataFrames, empty, etc.)
        """
        # Precondition 1: Validate plan_id
        if not plan_id or not isinstance(plan_id, str):
            raise ValueError(
                f"plan_id must be a non-empty string, got: {type(plan_id).__name__}"
            )
        
        # Precondition 2: Validate original_data
        if not isinstance(original_data, pd.DataFrame):
            raise ValueError(
                f"original_data must be a pandas DataFrame, "
                f"got: {type(original_data).__name__}"
            )
        
        if original_data.empty:
            raise ValueError("original_data cannot be empty")
        
        # Precondition 3: Validate transformed_data
        if not isinstance(transformed_data, pd.DataFrame):
            raise ValueError(
                f"transformed_data must be a pandas DataFrame, "
                f"got: {type(transformed_data).__name__}"
            )
        
        if transformed_data.empty:
            raise ValueError("transformed_data cannot be empty")
        
        logger.info(
            f"Starting validation for plan_id={plan_id} "
            f"(original: {original_data.shape}, transformed: {transformed_data.shape})"
        )
        
        # Collect all metric comparisons
        validation_results: List[MetricComparison] = []
        
        # Compute column-level metrics
        validation_results.extend(
            self._compare_column_metrics(original_data, transformed_data)
        )
        
        # Compute dataset-level metrics
        validation_results.extend(
            self._compare_dataset_metrics(original_data, transformed_data)
        )
        
        # Compute correlation metrics (if applicable)
        validation_results.extend(
            self._compare_correlation_metrics(original_data, transformed_data)
        )
        
        # Build summary
        summary = self._build_summary(validation_results, original_data, transformed_data)
        
        logger.info(
            f"Validation complete for plan_id={plan_id}: "
            f"{summary['metrics_compared']} metrics compared, "
            f"{summary['metrics_unavailable']} unavailable"
        )
        
        return ValidationReport(
            plan_id=plan_id,
            validation_results=validation_results,
            summary=summary
        )
    
    # ========================================================================
    # COLUMN-LEVEL METRICS
    # ========================================================================
    
    def _compare_column_metrics(
        self,
        original: pd.DataFrame,
        transformed: pd.DataFrame
    ) -> List[MetricComparison]:
        """
        Compare column-level metrics (skewness, missing %, outlier %).
        
        Parameters
        ----------
        original : pd.DataFrame
            Original dataset
        transformed : pd.DataFrame
            Transformed dataset
        
        Returns
        -------
        List[MetricComparison]
            List of metric comparisons for each column
        """
        comparisons: List[MetricComparison] = []
        
        # Get common columns (intersection)
        original_cols = set(original.columns)
        transformed_cols = set(transformed.columns)
        common_cols = original_cols & transformed_cols
        
        # Note: Columns may differ if transformations added/removed columns
        # We only compare metrics for common columns
        
        for col in sorted(common_cols):  # Sorted for determinism
            # Skewness (numeric only)
            if pd.api.types.is_numeric_dtype(original[col]) and \
               pd.api.types.is_numeric_dtype(transformed[col]):
                comparisons.append(
                    self._compute_metric(
                        metric_name="skewness",
                        column=col,
                        original_series=original[col],
                        transformed_series=transformed[col],
                        metric_func=self._compute_skewness
                    )
                )
                
                # Outlier percentage (numeric only)
                comparisons.append(
                    self._compute_metric(
                        metric_name="outlier_pct",
                        column=col,
                        original_series=original[col],
                        transformed_series=transformed[col],
                        metric_func=self._compute_outlier_percentage
                    )
                )
            
            # Missing value percentage (all types)
            comparisons.append(
                self._compute_metric(
                    metric_name="missing_pct",
                    column=col,
                    original_series=original[col],
                    transformed_series=transformed[col],
                    metric_func=self._compute_missing_percentage
                )
            )
        
        return comparisons
    
    def _compute_metric(
        self,
        metric_name: str,
        column: str,
        original_series: pd.Series,
        transformed_series: pd.Series,
        metric_func: Any  # Callable
    ) -> MetricComparison:
        """
        Compute a metric on both series and return comparison.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric
        column : str
            Column name
        original_series : pd.Series
            Original column data
        transformed_series : pd.Series
            Transformed column data
        metric_func : Callable
            Function to compute the metric
        
        Returns
        -------
        MetricComparison
            Comparison result
        """
        try:
            before = metric_func(original_series)
            after = metric_func(transformed_series)
            
            # Handle None values (metric unavailable)
            if before is None or after is None:
                return MetricComparison(
                    metric=metric_name,
                    column=column,
                    before=before,
                    after=after,
                    delta=None,
                    status=MetricStatus.UNAVAILABLE,
                    error="Metric computation returned None"
                )
            
            # Compute delta
            delta = after - before
            
            return MetricComparison(
                metric=metric_name,
                column=column,
                before=before,
                after=after,
                delta=delta,
                status=MetricStatus.SUCCESS
            )
        
        except Exception as e:
            logger.warning(
                f"Failed to compute {metric_name} for column {column}: {e}"
            )
            return MetricComparison(
                metric=metric_name,
                column=column,
                before=None,
                after=None,
                delta=None,
                status=MetricStatus.ERROR,
                error=str(e)
            )
    
    # ========================================================================
    # DATASET-LEVEL METRICS
    # ========================================================================
    
    def _compare_dataset_metrics(
        self,
        original: pd.DataFrame,
        transformed: pd.DataFrame
    ) -> List[MetricComparison]:
        """
        Compare dataset-level metrics (row count, column count, etc.).
        
        Parameters
        ----------
        original : pd.DataFrame
            Original dataset
        transformed : pd.DataFrame
            Transformed dataset
        
        Returns
        -------
        List[MetricComparison]
            List of dataset-level metric comparisons
        """
        comparisons: List[MetricComparison] = []
        
        # Row count
        comparisons.append(
            MetricComparison(
                metric="row_count",
                column="dataset",
                before=float(len(original)),
                after=float(len(transformed)),
                delta=float(len(transformed) - len(original)),
                status=MetricStatus.SUCCESS
            )
        )
        
        # Column count
        comparisons.append(
            MetricComparison(
                metric="column_count",
                column="dataset",
                before=float(len(original.columns)),
                after=float(len(transformed.columns)),
                delta=float(len(transformed.columns) - len(original.columns)),
                status=MetricStatus.SUCCESS
            )
        )
        
        # Overall missing percentage
        orig_missing_pct = (original.isnull().sum().sum() / original.size) * 100
        trans_missing_pct = (transformed.isnull().sum().sum() / transformed.size) * 100
        
        comparisons.append(
            MetricComparison(
                metric="overall_missing_pct",
                column="dataset",
                before=orig_missing_pct,
                after=trans_missing_pct,
                delta=trans_missing_pct - orig_missing_pct,
                status=MetricStatus.SUCCESS
            )
        )
        
        return comparisons
    
    # ========================================================================
    # CORRELATION METRICS
    # ========================================================================
    
    def _compare_correlation_metrics(
        self,
        original: pd.DataFrame,
        transformed: pd.DataFrame
    ) -> List[MetricComparison]:
        """
        Compare correlation metrics between datasets.
        
        Computes average absolute correlation for numeric columns.
        
        Parameters
        ----------
        original : pd.DataFrame
            Original dataset
        transformed : pd.DataFrame
            Transformed dataset
        
        Returns
        -------
        List[MetricComparison]
            List of correlation metric comparisons
        """
        comparisons: List[MetricComparison] = []
        
        try:
            # Get common numeric columns
            orig_numeric = original.select_dtypes(include=[np.number])
            trans_numeric = transformed.select_dtypes(include=[np.number])
            
            common_numeric = set(orig_numeric.columns) & set(trans_numeric.columns)
            
            # Need at least 2 numeric columns for correlation
            if len(common_numeric) < 2:
                return [
                    MetricComparison(
                        metric="avg_correlation",
                        column="dataset",
                        before=None,
                        after=None,
                        delta=None,
                        status=MetricStatus.UNAVAILABLE,
                        error="Need at least 2 common numeric columns"
                    )
                ]
            
            # Compute correlations
            orig_corr = orig_numeric[list(common_numeric)].corr()
            trans_corr = trans_numeric[list(common_numeric)].corr()
            
            # Extract upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(orig_corr, dtype=bool), k=1)
            orig_values = orig_corr.where(mask).stack().values
            trans_values = trans_corr.where(mask).stack().values
            
            # Compute average absolute correlation
            orig_avg = float(np.abs(orig_values).mean())
            trans_avg = float(np.abs(trans_values).mean())
            
            # Check for NaN (happens with constant columns)
            if np.isnan(orig_avg) or np.isnan(trans_avg):
                comparisons.append(
                    MetricComparison(
                        metric="avg_correlation",
                        column="dataset",
                        before=None,
                        after=None,
                        delta=None,
                        status=MetricStatus.UNAVAILABLE,
                        error="Correlation matrix contains NaN (likely constant columns)"
                    )
                )
            else:
                comparisons.append(
                    MetricComparison(
                        metric="avg_correlation",
                        column="dataset",
                        before=orig_avg,
                        after=trans_avg,
                        delta=trans_avg - orig_avg,
                        status=MetricStatus.SUCCESS
                    )
                )
        
        except Exception as e:
            logger.warning(f"Failed to compute correlation metrics: {e}")
            comparisons.append(
                MetricComparison(
                    metric="avg_correlation",
                    column="dataset",
                    before=None,
                    after=None,
                    delta=None,
                    status=MetricStatus.ERROR,
                    error=str(e)
                )
            )
        
        return comparisons
    
    # ========================================================================
    # METRIC COMPUTATION FUNCTIONS (Reuse Existing Logic)
    # ========================================================================
    
    def _compute_skewness(self, series: pd.Series) -> Optional[float]:
        """
        Compute skewness for a numeric series.
        
        Parameters
        ----------
        series : pd.Series
            Numeric series
        
        Returns
        -------
        float | None
            Skewness value, or None if cannot compute
        """
        try:
            # Remove NaN values
            clean = series.dropna()
            
            # Need at least 3 values
            if len(clean) < 3:
                return None
            
            # Compute skewness (pandas uses Fisher's definition)
            skew_value = clean.skew()
            
            # Check if result is finite
            if not np.isfinite(skew_value):
                return None
            
            return float(skew_value)
        
        except Exception:
            return None
    
    def _compute_missing_percentage(self, series: pd.Series) -> Optional[float]:
        """
        Compute missing value percentage.
        
        Parameters
        ----------
        series : pd.Series
            Any series
        
        Returns
        -------
        float | None
            Missing percentage (0-100), or None if cannot compute
        """
        try:
            if len(series) == 0:
                return None
            
            missing_count = series.isnull().sum()
            missing_pct = (missing_count / len(series)) * 100
            
            return float(missing_pct)
        
        except Exception:
            return None
    
    def _compute_outlier_percentage(self, series: pd.Series) -> Optional[float]:
        """
        Compute outlier percentage using IQR method.
        
        Parameters
        ----------
        series : pd.Series
            Numeric series
        
        Returns
        -------
        float | None
            Outlier percentage (0-100), or None if cannot compute
        """
        try:
            # Remove NaN values
            clean = series.dropna()
            
            # Need at least 4 values for IQR
            if len(clean) < 4:
                return None
            
            # Compute IQR
            Q1 = clean.quantile(0.25)
            Q3 = clean.quantile(0.75)
            IQR = Q3 - Q1
            
            # Check for zero IQR (constant values)
            if IQR == 0:
                return 0.0
            
            # Count outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((clean < lower_bound) | (clean > upper_bound)).sum()
            outlier_pct = (outlier_count / len(clean)) * 100
            
            return float(outlier_pct)
        
        except Exception:
            return None
    
    # ========================================================================
    # SUMMARY BUILDING
    # ========================================================================
    
    def _build_summary(
        self,
        validation_results: List[MetricComparison],
        original: pd.DataFrame,
        transformed: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Build summary statistics for the validation report.
        
        Parameters
        ----------
        validation_results : List[MetricComparison]
            All metric comparisons
        original : pd.DataFrame
            Original dataset
        transformed : pd.DataFrame
            Transformed dataset
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        # Count metrics by status
        total_metrics = len(validation_results)
        success_count = sum(
            1 for comp in validation_results
            if comp.status == MetricStatus.SUCCESS
        )
        unavailable_count = sum(
            1 for comp in validation_results
            if comp.status == MetricStatus.UNAVAILABLE
        )
        error_count = sum(
            1 for comp in validation_results
            if comp.status == MetricStatus.ERROR
        )
        
        # Count affected columns
        columns_with_comparisons = {
            comp.column for comp in validation_results
            if comp.column != "dataset" and comp.status == MetricStatus.SUCCESS
        }
        
        return {
            "metrics_compared": success_count,
            "metrics_unavailable": unavailable_count,
            "metrics_errored": error_count,
            "total_metrics": total_metrics,
            "columns_affected": len(columns_with_comparisons),
            "original_shape": {
                "rows": len(original),
                "columns": len(original.columns)
            },
            "transformed_shape": {
                "rows": len(transformed),
                "columns": len(transformed.columns)
            }
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def validate_transformation_impact(
    plan_id: str,
    original_data: pd.DataFrame,
    transformed_data: pd.DataFrame
) -> ValidationReport:
    """
    Convenience function to validate transformation impact.
    
    Creates a ValidationFeedbackLoop instance and calls validate().
    
    Parameters
    ----------
    plan_id : str
        Reference to the executed transformation plan
    original_data : pd.DataFrame
        Original dataset (pre-transformation)
    transformed_data : pd.DataFrame
        Transformed dataset (post-transformation)
    
    Returns
    -------
    ValidationReport
        Structured comparison of metrics before and after
    
    Raises
    ------
    ValueError
        If inputs are invalid
    
    Examples
    --------
    >>> report = validate_transformation_impact(
    ...     plan_id="TP-001",
    ...     original_data=df_original,
    ...     transformed_data=df_transformed
    ... )
    >>> print(f"Metrics compared: {report.summary['metrics_compared']}")
    >>> for result in report.validation_results:
    ...     if result.status == "SUCCESS":
    ...         print(f"{result.column}.{result.metric}: {result.delta:.4f}")
    """
    validator = ValidationFeedbackLoop()
    return validator.validate(plan_id, original_data, transformed_data)
