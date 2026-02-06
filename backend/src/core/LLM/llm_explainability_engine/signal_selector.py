"""
Signal Selector for LLM Context Preparation (STEP 2).

This module takes the structured signals from STEP 1 (explainable_signals.py)
and filters/selects/normalizes them into focused contexts based on different
analysis scopes. The output is optimized for LLM consumption.

Key Principle: This module ONLY selects and filters existing signals.
It does NOT compute new statistics or modify STEP 1 logic.

Author: DataMimicAI Team
Date: February 2026
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import warnings
import logging

# Baseline guard for protecting existing behavior
try:
    from .baseline_guard import (
        capture_signals_snapshot,
        assert_context_matches_signals,
        is_baseline_guard_enabled
    )
    _baseline_guard_available = True
except ImportError:
    _baseline_guard_available = False
    
logger = logging.getLogger(__name__)


@dataclass
class SelectorConfig:
    """Configuration for signal selection behavior."""
    
    # Default limits
    default_max_items: int = 5
    max_correlations: int = 10
    max_outlier_columns: int = 5
    max_top_categories: int = 5
    
    # Column importance thresholds
    min_correlation_threshold: float = 0.3
    high_outlier_threshold: float = 5.0  # percentage
    
    # Scope validation
    valid_scopes: List[str] = None
    
    def __post_init__(self):
        """Initialize valid scopes after dataclass creation."""
        if self.valid_scopes is None:
            self.valid_scopes = [
                "dataset_overview",
                "column_analysis",
                "correlation_analysis",
                "outlier_analysis",
                "time_series_analysis"
            ]


class DatasetOverviewSelector:
    """Selects high-level dataset overview information."""
    
    @staticmethod
    def select(signals: Dict[str, Any], config: SelectorConfig) -> Dict[str, Any]:
        """
        Extract dataset overview facts.
        
        Parameters
        ----------
        signals : dict
            Full signals dictionary from STEP 1
        config : SelectorConfig
            Configuration object
        
        Returns
        -------
        dict
            Dataset overview facts
        """
        dataset_summary = signals.get("dataset_summary", {})
        numeric_summary = signals.get("numeric_summary", {})
        categorical_summary = signals.get("categorical_summary", {})
        time_series = signals.get("time_series", {})
        
        # Calculate overall missing data percentage
        total_rows = dataset_summary.get("num_rows", 0)
        total_values = total_rows * dataset_summary.get("num_columns", 0)
        total_missing = dataset_summary.get("total_missing_values", 0)
        missing_pct = round((total_missing / total_values * 100), 2) if total_values > 0 else 0.0
        
        facts = {
            "basic_info": {
                "num_rows": dataset_summary.get("num_rows", 0),
                "num_columns": dataset_summary.get("num_columns", 0),
                "memory_usage_mb": dataset_summary.get("memory_usage_mb", 0.0),
                "duplicate_rows": dataset_summary.get("duplicate_rows", 0)
            },
            "data_quality": {
                "total_missing_values": total_missing,
                "overall_missing_pct": missing_pct,
                "is_empty": dataset_summary.get("is_empty", False)
            },
            "column_types": {
                "num_numeric": numeric_summary.get("num_numeric_columns", 0),
                "num_categorical": categorical_summary.get("num_categorical_columns", 0),
                "num_datetime": time_series.get("num_datetime_columns", 0)
            },
            "key_issues": {
                "columns_with_outliers": numeric_summary.get("num_columns_with_outliers", 0),
                "imbalanced_columns": categorical_summary.get("num_imbalanced_columns", 0),
                "high_cardinality_columns": categorical_summary.get("num_high_cardinality_columns", 0)
            }
        }
        
        # Add distribution summary if available
        if numeric_summary.get("distribution_shapes"):
            facts["distribution_summary"] = numeric_summary["distribution_shapes"]
        
        return facts


class ColumnAnalysisSelector:
    """Selects detailed information for specific columns."""
    
    def __init__(self, config: SelectorConfig):
        """Initialize selector with configuration."""
        self.config = config
    
    def select(self, signals: Dict[str, Any], columns: Optional[List[str]] = None, 
               max_items: int = 5) -> Dict[str, Any]:
        """
        Extract column-specific analysis facts.
        
        Parameters
        ----------
        signals : dict
            Full signals dictionary from STEP 1
        columns : list, optional
            Specific columns to analyze. If None, selects top-N important columns.
        max_items : int
            Maximum number of columns to include if not specified
        
        Returns
        -------
        dict
            Column analysis facts
        """
        all_columns = signals.get("columns", {})
        
        if not all_columns:
            return {"columns": {}, "message": "No column information available"}
        
        # If columns not specified, select the most "interesting" ones
        if columns is None:
            columns = self._select_important_columns(all_columns, max_items)
        else:
            # Filter to only valid columns
            columns = [col for col in columns if col in all_columns]
        
        # Extract facts for selected columns
        column_facts = {}
        for col in columns:
            col_info = all_columns[col]
            column_facts[col] = self._extract_column_facts(col_info)
        
        return {
            "columns": column_facts,
            "num_columns_analyzed": len(column_facts)
        }
    
    def _select_important_columns(self, all_columns: Dict[str, Any], max_items: int) -> List[str]:
        """
        Select most important/interesting columns based on various criteria.
        
        Prioritizes columns with:
        - High missing values
        - Outliers (for numeric)
        - Imbalance (for categorical)
        - High variance
        """
        scored_columns = []
        
        for col_name, col_info in all_columns.items():
            score = 0
            
            # Missing data contributes to importance
            missing_pct = col_info.get("missing_pct", 0)
            score += missing_pct * 0.5
            
            # Type-specific scoring
            col_type = col_info.get("type", "unknown")
            
            if col_type == "numeric":
                # Outliers make it interesting
                outlier_pct = col_info.get("outlier_pct", 0)
                score += outlier_pct
                
                # High skewness is interesting
                skewness = abs(col_info.get("skewness", 0))
                score += skewness * 10
            
            elif col_type == "categorical":
                # Imbalanced columns are interesting
                if col_info.get("is_imbalanced", False):
                    score += 20
                
                # High cardinality is interesting
                if col_info.get("is_high_cardinality", False):
                    score += 15
            
            scored_columns.append((col_name, score))
        
        # Sort by score descending and take top N
        scored_columns.sort(key=lambda x: x[1], reverse=True)
        return [col_name for col_name, _ in scored_columns[:max_items]]
    
    def _extract_column_facts(self, col_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant facts for a single column."""
        col_type = col_info.get("type", "unknown")
        
        # Base facts for all columns
        facts = {
            "type": col_type,
            "missing_pct": col_info.get("missing_pct", 0),
            "cardinality": col_info.get("cardinality", 0)
        }
        
        # Type-specific facts
        if col_type == "numeric":
            facts["statistics"] = {
                "mean": col_info.get("mean"),
                "std": col_info.get("std"),
                "min": col_info.get("min"),
                "max": col_info.get("max"),
                "median": col_info.get("median")
            }
            facts["distribution"] = {
                "shape": col_info.get("distribution_shape", "unknown"),
                "skewness": col_info.get("skewness")
            }
            facts["outliers"] = {
                "count": col_info.get("outlier_count", 0),
                "percentage": col_info.get("outlier_pct", 0)
            }
        
        elif col_type == "categorical":
            facts["top_categories"] = col_info.get("top_categories", [])[:self.config.max_top_categories]
            facts["is_imbalanced"] = col_info.get("is_imbalanced", False)
            facts["is_high_cardinality"] = col_info.get("is_high_cardinality", False)
            facts["entropy"] = col_info.get("entropy")
        
        elif col_type == "datetime":
            facts["date_range"] = {
                "min_date": col_info.get("min_date"),
                "max_date": col_info.get("max_date"),
                "range_days": col_info.get("range_days", 0)
            }
            facts["trend"] = col_info.get("trend", "unknown")
        
        elif col_type == "text":
            facts["text_stats"] = {
                "avg_length": col_info.get("avg_length", 0),
                "max_length": col_info.get("max_length", 0),
                "min_length": col_info.get("min_length", 0)
            }
        
        return facts


class CorrelationAnalysisSelector:
    """Selects strongest correlations for analysis."""
    
    def __init__(self, config: SelectorConfig):
        """Initialize selector with configuration."""
        self.config = config
    
    def select(self, signals: Dict[str, Any], max_items: int = 5) -> Dict[str, Any]:
        """
        Extract strongest correlation facts.
        
        Parameters
        ----------
        signals : dict
            Full signals dictionary from STEP 1
        max_items : int
            Maximum number of correlation pairs to include
        
        Returns
        -------
        dict
            Correlation analysis facts
        """
        correlations = signals.get("correlations", {})
        pearson = correlations.get("pearson", [])
        spearman = correlations.get("spearman", [])
        
        # Limit to top N correlations
        top_pearson = pearson[:max_items]
        top_spearman = spearman[:max_items]
        
        facts = {
            "pearson_correlations": top_pearson,
            "spearman_correlations": top_spearman,
            "num_pearson_found": len(pearson),
            "num_spearman_found": len(spearman),
            "num_pearson_shown": len(top_pearson),
            "num_spearman_shown": len(top_spearman)
        }
        
        # Add interpretation hints
        if not top_pearson and not top_spearman:
            facts["message"] = "No strong correlations found in the dataset"
        
        return facts


class OutlierAnalysisSelector:
    """Selects columns with significant outliers."""
    
    def __init__(self, config: SelectorConfig):
        """Initialize selector with configuration."""
        self.config = config
    
    def select(self, signals: Dict[str, Any], max_items: int = 5) -> Dict[str, Any]:
        """
        Extract outlier analysis facts.
        
        Parameters
        ----------
        signals : dict
            Full signals dictionary from STEP 1
        max_items : int
            Maximum number of columns to analyze
        
        Returns
        -------
        dict
            Outlier analysis facts
        """
        all_columns = signals.get("columns", {})
        numeric_summary = signals.get("numeric_summary", {})
        
        # Find numeric columns with outliers
        outlier_columns = []
        for col_name, col_info in all_columns.items():
            if col_info.get("type") == "numeric":
                outlier_pct = col_info.get("outlier_pct", 0)
                outlier_count = col_info.get("outlier_count", 0)
                
                if outlier_count > 0:
                    outlier_columns.append({
                        "column": col_name,
                        "outlier_count": outlier_count,
                        "outlier_pct": outlier_pct,
                        "distribution_shape": col_info.get("distribution_shape", "unknown"),
                        "skewness": col_info.get("skewness"),
                        "mean": col_info.get("mean"),
                        "std": col_info.get("std")
                    })
        
        # Sort by outlier percentage descending
        outlier_columns.sort(key=lambda x: x["outlier_pct"], reverse=True)
        
        # Limit to top N
        top_outlier_columns = outlier_columns[:max_items]
        
        facts = {
            "columns_with_outliers": top_outlier_columns,
            "total_outlier_columns": len(outlier_columns),
            "total_outliers": numeric_summary.get("total_outliers", 0),
            "num_columns_shown": len(top_outlier_columns)
        }
        
        if not outlier_columns:
            facts["message"] = "No outliers detected in numeric columns"
        
        return facts


class TimeSeriesAnalysisSelector:
    """Selects time-series related information."""
    
    @staticmethod
    def select(signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract time series analysis facts.
        
        Parameters
        ----------
        signals : dict
            Full signals dictionary from STEP 1
        
        Returns
        -------
        dict
            Time series analysis facts
        """
        time_series = signals.get("time_series", {})
        
        if not time_series.get("has_datetime_columns", False):
            return {
                "has_datetime_columns": False,
                "message": "No datetime columns found in the dataset"
            }
        
        datetime_columns = time_series.get("datetime_columns", {})
        
        # Extract facts for each datetime column
        column_facts = {}
        for col_name, col_info in datetime_columns.items():
            column_facts[col_name] = {
                "min_date": col_info.get("min_date"),
                "max_date": col_info.get("max_date"),
                "range_days": col_info.get("range_days", 0),
                "trend": col_info.get("trend", "unknown"),
                "is_monotonic": col_info.get("is_monotonic", False)
            }
        
        facts = {
            "has_datetime_columns": True,
            "num_datetime_columns": time_series.get("num_datetime_columns", 0),
            "datetime_columns": column_facts
        }
        
        return facts


class SignalContextSelector:
    """
    Main orchestrator for selecting and filtering signals into focused contexts.
    
    This class coordinates different scope selectors to produce optimized
    contexts for LLM consumption.
    """
    
    def __init__(self, config: Optional[SelectorConfig] = None):
        """
        Initialize the selector with optional custom configuration.
        
        Parameters
        ----------
        config : SelectorConfig, optional
            Custom configuration. If None, uses default configuration.
        """
        self.config = config or SelectorConfig()
        
        # Initialize scope selectors
        self.dataset_overview_selector = DatasetOverviewSelector()
        self.column_analysis_selector = ColumnAnalysisSelector(self.config)
        self.correlation_analysis_selector = CorrelationAnalysisSelector(self.config)
        self.outlier_analysis_selector = OutlierAnalysisSelector(self.config)
        self.time_series_analysis_selector = TimeSeriesAnalysisSelector()
    
    def select_context(
        self,
        signals: Dict[str, Any],
        scope: str,
        columns: Optional[List[str]] = None,
        max_items: int = 5
    ) -> Dict[str, Any]:
        """
        Select and filter signals into a focused context based on scope.
        
        Parameters
        ----------
        signals : dict
            Full signals dictionary from STEP 1 (explainable_signals.py)
        scope : str
            Analysis scope. Must be one of:
            - "dataset_overview"
            - "column_analysis"
            - "correlation_analysis"
            - "outlier_analysis"
            - "time_series_analysis"
        columns : list of str, optional
            Specific columns to analyze (only used for "column_analysis" scope)
        max_items : int, default=5
            Maximum number of items to include in the context
        
        Returns
        -------
        dict
            Filtered context with structure:
            {
                "scope": "<scope_name>",
                "facts": { ... },
                "metadata": {
                    "columns_used": [...],
                    "generated_at": "<iso timestamp>",
                    "max_items": <int>
                }
            }
        
        Raises
        ------
        No exceptions are raised. Invalid scopes return an error message in facts.
        
        Examples
        --------
        >>> selector = SignalContextSelector()
        >>> context = selector.select_context(signals, "dataset_overview")
        >>> print(context["scope"])
        'dataset_overview'
        """
        # Validate input
        if not signals:
            return self._error_response("Empty signals provided", scope)
        
        if scope not in self.config.valid_scopes:
            return self._error_response(
                f"Invalid scope '{scope}'. Valid scopes: {self.config.valid_scopes}",
                scope
            )
        
        # Route to appropriate selector
        try:
            if scope == "dataset_overview":
                facts = self.dataset_overview_selector.select(signals, self.config)
                columns_used = []
            
            elif scope == "column_analysis":
                facts = self.column_analysis_selector.select(signals, columns, max_items)
                columns_used = list(facts.get("columns", {}).keys())
            
            elif scope == "correlation_analysis":
                facts = self.correlation_analysis_selector.select(signals, max_items)
                # Extract column names from correlations
                columns_used = self._extract_corr_columns(facts)
            
            elif scope == "outlier_analysis":
                facts = self.outlier_analysis_selector.select(signals, max_items)
                columns_used = [col["column"] for col in facts.get("columns_with_outliers", [])]
            
            elif scope == "time_series_analysis":
                facts = self.time_series_analysis_selector.select(signals)
                columns_used = list(facts.get("datetime_columns", {}).keys())
            
            else:
                # Should not reach here due to validation, but safe fallback
                return self._error_response(f"Unhandled scope: {scope}", scope)
        
        except Exception as e:
            # Graceful error handling - never raise uncaught exceptions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._error_response(f"Error processing scope '{scope}': {str(e)}", scope)
        
        # Build final context
        context = {
            "scope": scope,
            "facts": facts,
            "metadata": {
                "columns_used": columns_used,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "max_items": max_items
            }
        }
        
        return context
    
    def _error_response(self, message: str, scope: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            "scope": scope,
            "facts": {
                "error": True,
                "message": message
            },
            "metadata": {
                "columns_used": [],
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "max_items": 0
            }
        }
    
    def _extract_corr_columns(self, facts: Dict[str, Any]) -> List[str]:
        """Extract unique column names from correlation facts."""
        columns = set()
        
        for corr in facts.get("pearson_correlations", []):
            columns.add(corr.get("column_1", ""))
            columns.add(corr.get("column_2", ""))
        
        for corr in facts.get("spearman_correlations", []):
            columns.add(corr.get("column_1", ""))
            columns.add(corr.get("column_2", ""))
        
        # Remove empty strings
        columns.discard("")
        
        return sorted(list(columns))


def select_explainable_context(
    signals: Dict[str, Any],
    scope: str,
    columns: Optional[List[str]] = None,
    max_items: int = 5,
    config: Optional[SelectorConfig] = None
) -> Dict[str, Any]:
    """
    Select and filter explainable signals into a focused context for LLM consumption.
    
    This is the main convenience function that wraps the class-based implementation.
    It takes the structured signals from STEP 1 and produces a filtered, normalized
    context optimized for a specific analysis scope.
    
    Parameters
    ----------
    signals : dict
        Full signals dictionary from build_explainable_signals() in STEP 1
    scope : str
        Analysis scope determining what information to include. Valid values:
        - "dataset_overview": High-level dataset summary
        - "column_analysis": Detailed column-specific information
        - "correlation_analysis": Strongest correlations between features
        - "outlier_analysis": Columns with significant outliers
        - "time_series_analysis": Temporal patterns and trends
    columns : list of str, optional
        Specific columns to analyze (only used for "column_analysis" scope).
        If None, automatically selects most interesting columns.
    max_items : int, default=5
        Maximum number of items to include (columns, correlations, etc.)
    config : SelectorConfig, optional
        Custom configuration for selection behavior
    
    Returns
    -------
    dict
        Filtered context with structure:
        {
            "scope": str,              # The analysis scope
            "facts": dict,             # Filtered facts for this scope
            "metadata": {
                "columns_used": list,  # Columns included in analysis
                "generated_at": str,   # ISO timestamp
                "max_items": int       # Items limit applied
            }
        }
    
    Notes
    -----
    This function performs NO computation. It only selects, filters, and
    reorganizes existing signals from STEP 1. All statistical computations
    should be done in the explainable_signals module.
    
    The output is designed to be:
    - JSON-serializable
    - Focused and concise
    - Ready for LLM consumption
    - Free of redundant information
    
    Examples
    --------
    >>> from core.LLM.llm_explainability_engine import build_explainable_signals, select_explainable_context
    >>> 
    >>> # Step 1: Extract all signals
    >>> signals = build_explainable_signals(df)
    >>> 
    >>> # Step 2: Select context for dataset overview
    >>> context = select_explainable_context(signals, scope="dataset_overview")
    >>> print(context["facts"]["basic_info"])
    {'num_rows': 1000, 'num_columns': 15, ...}
    >>> 
    >>> # Step 2: Analyze specific columns
    >>> context = select_explainable_context(
    ...     signals,
    ...     scope="column_analysis",
    ...     columns=["age", "salary"],
    ...     max_items=2
    ... )
    >>> 
    >>> # Step 2: Find strongest correlations
    >>> context = select_explainable_context(
    ...     signals,
    ...     scope="correlation_analysis",
    ...     max_items=10
    ... )
    """
    # BASELINE GUARD: Capture input signals snapshot (STEP 2 input)
    # This validates that STEP 2 receives the same EDA inputs as before
    if _baseline_guard_available and is_baseline_guard_enabled():
        try:
            capture_signals_snapshot(
                signals,
                label=f"STEP2_select_context_input_scope={scope}",
                log_level="debug"
            )
        except Exception as e:
            logger.warning(f"[BASELINE] Input snapshot failed: {e}")
    
    selector = SignalContextSelector(config)
    context = selector.select_context(signals, scope, columns, max_items)
    
    # BASELINE GUARD: Validate context matches signals
    # This ensures STEP 2 correctly derives context from STEP 1 signals
    if _baseline_guard_available and is_baseline_guard_enabled():
        try:
            assert_context_matches_signals(context, signals)
        except Exception as e:
            logger.warning(f"[BASELINE] Context validation failed: {e}")
    
    return context


# Example usage and testing (for development only - remove in production)
if __name__ == "__main__":
    # Sample signals for testing
    sample_signals = {
        "dataset_summary": {
            "num_rows": 1000,
            "num_columns": 5,
            "memory_usage_mb": 0.5,
            "duplicate_rows": 10,
            "total_missing_values": 50
        },
        "columns": {
            "age": {
                "type": "numeric",
                "missing_pct": 5.0,
                "cardinality": 50,
                "mean": 35.5,
                "outlier_pct": 2.5,
                "distribution_shape": "normal"
            },
            "salary": {
                "type": "numeric",
                "missing_pct": 0.0,
                "cardinality": 900,
                "mean": 75000,
                "outlier_pct": 8.5,
                "distribution_shape": "skewed_right"
            }
        },
        "numeric_summary": {
            "num_numeric_columns": 2,
            "total_outliers": 110
        },
        "categorical_summary": {
            "num_categorical_columns": 2
        },
        "correlations": {
            "pearson": [
                {"column_1": "age", "column_2": "salary", "correlation": 0.75}
            ],
            "spearman": []
        },
        "time_series": {
            "has_datetime_columns": False
        }
    }
    
    # Test different scopes
    print("Testing dataset_overview:")
    context = select_explainable_context(sample_signals, "dataset_overview")
    print(f"Scope: {context['scope']}")
    print(f"Facts keys: {list(context['facts'].keys())}")
    
    print("\nTesting column_analysis:")
    context = select_explainable_context(sample_signals, "column_analysis", columns=["age"])
    print(f"Columns analyzed: {context['metadata']['columns_used']}")
    
    print("\nTesting invalid scope:")
    context = select_explainable_context(sample_signals, "invalid_scope")
    print(f"Error: {context['facts'].get('message', 'No error')}")
