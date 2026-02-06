"""
Explainable Signals Extractor for LLM-Based Data Insights.

This module extracts structured, deterministic signals from tabular data
to serve as ground-truth facts for LLM-based explanations. It computes
dataset-level and column-level statistics without performing any ML modeling
or LLM inference.

Author: DataMimicAI Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import warnings
from dataclasses import dataclass
import logging

# Baseline guard for protecting existing behavior
try:
    from .baseline_guard import (
        capture_signals_snapshot,
        assert_signals_structure,
        is_baseline_guard_enabled
    )
    _baseline_guard_available = True
except ImportError:
    _baseline_guard_available = False
    
logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration class for signal extraction thresholds and parameters."""
    
    # Correlation thresholds
    strong_correlation_threshold: float = 0.6
    
    # Outlier detection
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0
    
    # Categorical analysis
    imbalance_threshold: float = 0.7
    top_categories_limit: int = 10
    high_cardinality_threshold: int = 50
    
    # Distribution classification
    skewness_threshold: float = 0.5
    normality_alpha: float = 0.05


class ColumnTypeInferrer:
    """Handles column type inference for different data types."""
    
    @staticmethod
    def infer_column_type(series: pd.Series) -> str:
        """
        Infer the semantic type of a column.
        
        Parameters
        ----------
        series : pd.Series
            Column to analyze
        
        Returns
        -------
        str
            One of: 'numeric', 'categorical', 'datetime', 'text', 'unknown'
        """
        # Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        
        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        
        # Object/string types
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Try to parse as datetime
            try:
                pd.to_datetime(series.dropna().head(100), errors='raise')
                return "datetime"
            except:
                pass
            
            # Categorical or text based on cardinality
            cardinality = series.nunique()
            if cardinality < len(series) * 0.5:  # If unique values < 50% of total
                return "categorical"
            else:
                return "text"
        
        # Boolean
        if pd.api.types.is_bool_dtype(series):
            return "categorical"
        
        return "unknown"


class NumericColumnAnalyzer:
    """Analyzes numeric columns for statistical insights."""
    
    def __init__(self, config: SignalConfig):
        """
        Initialize the analyzer with configuration.
        
        Parameters
        ----------
        config : SignalConfig
            Configuration object with analysis thresholds
        """
        self.config = config
    
    def analyze(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze a numeric column and extract statistical insights.
        
        Parameters
        ----------
        series : pd.Series
            Numeric column
        
        Returns
        -------
        dict
            Numeric statistics including mean, std, skewness, outliers, etc.
        """
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return self._empty_result()
        
        # Basic statistics
        mean_val = float(clean_series.mean())
        std_val = float(clean_series.std())
        
        # Skewness and kurtosis
        skewness = float(clean_series.skew())
        kurtosis = float(clean_series.kurtosis())
        
        # Distribution shape classification
        distribution_shape = self._classify_distribution_shape(clean_series, skewness)
        
        # Outlier detection (IQR method)
        outlier_info = self._detect_outliers_iqr(clean_series)
        
        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": float(clean_series.min()),
            "max": float(clean_series.max()),
            "median": float(clean_series.median()),
            "q1": float(clean_series.quantile(0.25)),
            "q3": float(clean_series.quantile(0.75)),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "distribution_shape": distribution_shape,
            "outlier_count": outlier_info["count"],
            "outlier_pct": outlier_info["percentage"],
            "zeros_count": int((clean_series == 0).sum()),
            "negative_count": int((clean_series < 0).sum())
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure for columns with no data."""
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "skewness": None,
            "kurtosis": None,
            "distribution_shape": "unknown",
            "outlier_count": 0,
            "outlier_pct": 0.0
        }
    
    def _classify_distribution_shape(self, series: pd.Series, skewness: float) -> str:
        """
        Classify the distribution shape of a numeric series.
        
        Parameters
        ----------
        series : pd.Series
            Numeric series (no NaN values)
        skewness : float
            Pre-computed skewness value
        
        Returns
        -------
        str
            One of: "normal", "skewed_right", "skewed_left", "multimodal", "uniform", "unknown"
        """
        if len(series) < 10:
            return "unknown"
        
        # Test for normality
        try:
            if len(series) <= 5000:
                # Shapiro-Wilk test
                stat, p_value = stats.shapiro(series)
                is_normal = p_value > self.config.normality_alpha
            else:
                # Anderson-Darling test (for larger samples)
                result = stats.anderson(series, dist='norm')
                is_normal = result.statistic < result.critical_values[2]  # 5% significance
        except:
            is_normal = False
        
        # Check for multimodality (using kurtosis as proxy)
        kurtosis = series.kurtosis()
        is_multimodal = kurtosis < -1  # Negative kurtosis suggests multimodality
        
        # Check for uniformity
        try:
            # Kolmogorov-Smirnov test against uniform distribution
            min_val = series.min()
            max_val = series.max()
            # Handle constant values (all same)
            if max_val - min_val > 0:
                normalized = (series - min_val) / (max_val - min_val)
                ks_stat, ks_p = stats.kstest(normalized, 'uniform')
                is_uniform = ks_p > self.config.normality_alpha
            else:
                is_uniform = False  # Constant values cannot be uniform
        except:
            is_uniform = False
        
        # Classification logic
        if is_normal:
            return "normal"
        elif is_multimodal:
            return "multimodal"
        elif is_uniform:
            return "uniform"
        elif abs(skewness) < self.config.skewness_threshold:
            return "normal"  # Approximately symmetric
        elif skewness > self.config.skewness_threshold:
            return "skewed_right"
        elif skewness < -self.config.skewness_threshold:
            return "skewed_left"
        else:
            return "unknown"
    
    def _detect_outliers_iqr(self, series: pd.Series) -> Dict[str, Any]:
        """
        Detect outliers using the IQR (Interquartile Range) method.
        
        Parameters
        ----------
        series : pd.Series
            Numeric series (no NaN values)
        
        Returns
        -------
        dict
            Outlier count and percentage
        """
        if len(series) == 0:
            return {"count": 0, "percentage": 0.0}
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.config.outlier_iqr_multiplier * iqr
        upper_bound = q3 + self.config.outlier_iqr_multiplier * iqr
        
        outliers = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outliers.sum())
        outlier_pct = round(outlier_count / len(series) * 100, 2)
        
        return {
            "count": outlier_count,
            "percentage": outlier_pct,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }


class CategoricalColumnAnalyzer:
    """Analyzes categorical columns for distribution and imbalance."""
    
    def __init__(self, config: SignalConfig):
        """
        Initialize the analyzer with configuration.
        
        Parameters
        ----------
        config : SignalConfig
            Configuration object with analysis thresholds
        """
        self.config = config
    
    def analyze(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze a categorical column and extract insights.
        
        Parameters
        ----------
        series : pd.Series
            Categorical column
        
        Returns
        -------
        dict
            Categorical statistics including top categories, imbalance flag, etc.
        """
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                "top_categories": [],
                "is_imbalanced": False,
                "entropy": 0.0
            }
        
        # Value counts
        value_counts = clean_series.value_counts()
        total_count = len(clean_series)
        
        # Top categories with frequencies
        top_categories = []
        for category, count in value_counts.head(self.config.top_categories_limit).items():
            top_categories.append({
                "category": str(category),
                "count": int(count),
                "percentage": round(count / total_count * 100, 2)
            })
        
        # Imbalance detection
        if len(value_counts) > 0:
            top_category_ratio = value_counts.iloc[0] / total_count
            is_imbalanced = top_category_ratio > self.config.imbalance_threshold
        else:
            is_imbalanced = False
        
        # Entropy (measure of diversity)
        probabilities = value_counts / total_count
        # Filter out zero probabilities to avoid log(0) error
        probabilities_nonzero = probabilities[probabilities > 0]
        entropy = float(-np.sum(probabilities_nonzero * np.log2(probabilities_nonzero)))
        
        return {
            "top_categories": top_categories,
            "is_imbalanced": is_imbalanced,
            "dominant_category": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "dominant_category_pct": round(value_counts.iloc[0] / total_count * 100, 2) if len(value_counts) > 0 else 0.0,
            "entropy": round(entropy, 4),
            "is_high_cardinality": series.nunique() > self.config.high_cardinality_threshold
        }


class DatetimeColumnAnalyzer:
    """Analyzes datetime columns for temporal patterns."""
    
    @staticmethod
    def analyze(series: pd.Series) -> Dict[str, Any]:
        """
        Analyze a datetime column and extract temporal insights.
        
        Parameters
        ----------
        series : pd.Series
            Datetime column
        
        Returns
        -------
        dict
            Datetime statistics including range, monotonicity, etc.
        """
        # Convert to datetime if not already
        try:
            dt_series = pd.to_datetime(series, errors='coerce')
        except:
            return {"error": "Failed to parse datetime"}
        
        clean_series = dt_series.dropna()
        
        if len(clean_series) == 0:
            return {
                "min_date": None,
                "max_date": None,
                "range_days": 0
            }
        
        min_date = clean_series.min()
        max_date = clean_series.max()
        range_days = (max_date - min_date).days
        
        # Check monotonicity
        is_monotonic_increasing = clean_series.is_monotonic_increasing
        is_monotonic_decreasing = clean_series.is_monotonic_decreasing
        
        # Determine trend
        if is_monotonic_increasing:
            trend = "monotonic_increasing"
        elif is_monotonic_decreasing:
            trend = "monotonic_decreasing"
        else:
            trend = "irregular"
        
        return {
            "min_date": str(min_date),
            "max_date": str(max_date),
            "range_days": range_days,
            "trend": trend,
            "is_monotonic": is_monotonic_increasing or is_monotonic_decreasing
        }


class TextColumnAnalyzer:
    """Analyzes text columns for length and patterns."""
    
    @staticmethod
    def analyze(series: pd.Series) -> Dict[str, Any]:
        """
        Analyze a text column (high-cardinality string data).
        
        Parameters
        ----------
        series : pd.Series
            Text column
        
        Returns
        -------
        dict
            Text statistics including average length, etc.
        """
        clean_series = series.dropna().astype(str)
        
        if len(clean_series) == 0:
            return {
                "avg_length": 0,
                "max_length": 0
            }
        
        lengths = clean_series.str.len()
        
        return {
            "avg_length": round(lengths.mean(), 2),
            "max_length": int(lengths.max()),
            "min_length": int(lengths.min()),
            "empty_strings": int((clean_series == "").sum())
        }


class CorrelationAnalyzer:
    """Analyzes correlations between numeric features."""
    
    def __init__(self, config: SignalConfig):
        """
        Initialize the analyzer with configuration.
        
        Parameters
        ----------
        config : SignalConfig
            Configuration object with correlation thresholds
        """
        self.config = config
    
    def analyze(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract strong correlations (Pearson & Spearman) among numeric columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        numeric_cols : list
            List of numeric column names
        
        Returns
        -------
        dict
            Dictionary with 'pearson' and 'spearman' correlation lists
        """
        correlations = {
            "pearson": [],
            "spearman": []
        }
        
        if len(numeric_cols) < 2:
            return correlations
        
        # Pearson correlation
        try:
            pearson_corr = df[numeric_cols].corr(method='pearson')
            correlations["pearson"] = self._extract_strong_correlations(pearson_corr, numeric_cols)
        except Exception as e:
            pass  # Skip if computation fails
        
        # Spearman correlation
        try:
            spearman_corr = df[numeric_cols].corr(method='spearman')
            correlations["spearman"] = self._extract_strong_correlations(spearman_corr, numeric_cols)
        except Exception as e:
            pass  # Skip if computation fails
        
        return correlations
    
    def _extract_strong_correlations(self, corr_matrix: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
        """
        Extract strong correlation pairs from a correlation matrix.
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            Correlation matrix
        columns : list
            List of column names
        
        Returns
        -------
        list
            List of strong correlation pairs
        """
        strong_correlations = []
        
        # Iterate over upper triangle (avoid duplicates and self-correlation)
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if j <= i:
                    continue
                
                corr_value = corr_matrix.loc[col1, col2]
                
                # Skip NaN correlations
                if pd.isna(corr_value):
                    continue
                
                # Check if correlation is strong
                if abs(corr_value) >= self.config.strong_correlation_threshold:
                    strong_correlations.append({
                        "column_1": col1,
                        "column_2": col2,
                        "correlation": round(float(corr_value), 4),
                        "strength": "strong" if abs(corr_value) >= 0.8 else "moderate"
                    })
        
        # Sort by absolute correlation value (descending)
        strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return strong_correlations


class ExplainableSignalsExtractor:
    """
    Main class for extracting explainable signals from tabular data.
    
    This orchestrator coordinates all analyzers to produce a comprehensive
    dictionary of dataset and column-level insights for LLM consumption.
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        """
        Initialize the extractor with optional custom configuration.
        
        Parameters
        ----------
        config : SignalConfig, optional
            Custom configuration. If None, uses default configuration.
        """
        self.config = config or SignalConfig()
        
        # Initialize analyzers
        self.type_inferrer = ColumnTypeInferrer()
        self.numeric_analyzer = NumericColumnAnalyzer(self.config)
        self.categorical_analyzer = CategoricalColumnAnalyzer(self.config)
        self.datetime_analyzer = DatetimeColumnAnalyzer()
        self.text_analyzer = TextColumnAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer(self.config)
    
    def extract(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract all explainable signals from a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (assumed to be cleaned/loaded)
        
        Returns
        -------
        dict
            Structured dictionary containing all extracted signals
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Initialize output structure
            signals = {
                "dataset_summary": {},
                "columns": {},
                "numeric_summary": {},
                "categorical_summary": {},
                "correlations": {"pearson": [], "spearman": []},
                "time_series": {}
            }
            
            # Validate input
            if df is None or df.empty:
                signals["dataset_summary"] = {
                    "num_rows": 0,
                    "num_columns": 0,
                    "is_empty": True
                }
                return signals
            
            # 1. Dataset-level summary
            signals["dataset_summary"] = self._extract_dataset_summary(df)
            
            # 2. Column-level analysis
            signals["columns"] = self._extract_column_details(df)
            
            # 3. Numeric summary (aggregated)
            signals["numeric_summary"] = self._extract_numeric_summary(df, signals["columns"])
            
            # 4. Categorical summary (aggregated)
            signals["categorical_summary"] = self._extract_categorical_summary(df, signals["columns"])
            
            # 5. Correlations (Pearson & Spearman)
            numeric_cols = [col for col, info in signals["columns"].items() if info["type"] == "numeric"]
            signals["correlations"] = self.correlation_analyzer.analyze(df, numeric_cols)
            
            # 6. Time series insights (if applicable)
            signals["time_series"] = self._extract_time_series_info(df, signals["columns"])
            
            return signals
    
    def _extract_dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract high-level dataset statistics."""
        summary = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_values": int(df.isnull().sum().sum()),
            "is_empty": False
        }
        return summary
    
    def _extract_column_details(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Extract detailed information for each column."""
        columns_info = {}
        
        for col in df.columns:
            series = df[col]
            col_type = self.type_inferrer.infer_column_type(series)
            
            # Base information for all columns
            col_info = {
                "type": col_type,
                "missing_count": int(series.isnull().sum()),
                "missing_pct": round(series.isnull().sum() / len(series) * 100, 2),
                "cardinality": int(series.nunique()),
                "dtype": str(series.dtype)
            }
            
            # Type-specific analysis
            if col_type == "numeric":
                col_info.update(self.numeric_analyzer.analyze(series))
            elif col_type == "categorical":
                col_info.update(self.categorical_analyzer.analyze(series))
            elif col_type == "datetime":
                col_info.update(self.datetime_analyzer.analyze(series))
            elif col_type == "text":
                col_info.update(self.text_analyzer.analyze(series))
            
            columns_info[col] = col_info
        
        return columns_info
    
    def _extract_numeric_summary(self, df: pd.DataFrame, columns_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract aggregated insights across all numeric columns."""
        numeric_cols = [col for col, info in columns_info.items() if info["type"] == "numeric"]
        
        if not numeric_cols:
            return {
                "num_numeric_columns": 0,
                "total_outliers": 0,
                "columns_with_outliers": []
            }
        
        # Count columns with outliers
        cols_with_outliers = [
            col for col in numeric_cols
            if columns_info[col].get("outlier_count", 0) > 0
        ]
        
        # Total outliers across all numeric columns
        total_outliers = sum(columns_info[col].get("outlier_count", 0) for col in numeric_cols)
        
        # Distribution shapes summary
        distribution_shapes = {}
        for col in numeric_cols:
            shape = columns_info[col].get("distribution_shape", "unknown")
            distribution_shapes[shape] = distribution_shapes.get(shape, 0) + 1
        
        return {
            "num_numeric_columns": len(numeric_cols),
            "total_outliers": total_outliers,
            "columns_with_outliers": cols_with_outliers,
            "num_columns_with_outliers": len(cols_with_outliers),
            "distribution_shapes": distribution_shapes
        }
    
    def _extract_categorical_summary(self, df: pd.DataFrame, columns_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract aggregated insights across all categorical columns."""
        categorical_cols = [col for col, info in columns_info.items() if info["type"] == "categorical"]
        
        if not categorical_cols:
            return {
                "num_categorical_columns": 0,
                "imbalanced_columns": []
            }
        
        # Count imbalanced columns
        imbalanced_cols = [
            col for col in categorical_cols
            if columns_info[col].get("is_imbalanced", False)
        ]
        
        # High cardinality columns
        high_cardinality_cols = [
            col for col in categorical_cols
            if columns_info[col].get("is_high_cardinality", False)
        ]
        
        return {
            "num_categorical_columns": len(categorical_cols),
            "imbalanced_columns": imbalanced_cols,
            "num_imbalanced_columns": len(imbalanced_cols),
            "high_cardinality_columns": high_cardinality_cols,
            "num_high_cardinality_columns": len(high_cardinality_cols)
        }
    
    def _extract_time_series_info(self, df: pd.DataFrame, columns_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract time series insights if datetime columns exist."""
        datetime_cols = [col for col, info in columns_info.items() if info["type"] == "datetime"]
        
        if not datetime_cols:
            return {
                "has_datetime_columns": False,
                "num_datetime_columns": 0
            }
        
        # Aggregate datetime information
        time_info = {
            "has_datetime_columns": True,
            "num_datetime_columns": len(datetime_cols),
            "datetime_columns": {}
        }
        
        for col in datetime_cols:
            col_info = columns_info[col]
            time_info["datetime_columns"][col] = {
                "min_date": col_info.get("min_date"),
                "max_date": col_info.get("max_date"),
                "range_days": col_info.get("range_days", 0),
                "trend": col_info.get("trend", "unknown"),
                "is_monotonic": col_info.get("is_monotonic", False)
            }
        
        return time_info


def build_explainable_signals(df: pd.DataFrame, config: Optional[SignalConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to build explainable signals from a DataFrame.
    
    This function maintains backward compatibility with the original API
    while using the new class-based implementation internally.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (assumed to be cleaned/loaded)
    config : SignalConfig, optional
        Custom configuration for signal extraction
    
    Returns
    -------
    dict
        Structured dictionary containing:
        - dataset_summary: Overall dataset statistics
        - columns: Per-column detailed information
        - numeric_summary: Aggregated numeric insights
        - categorical_summary: Aggregated categorical insights
        - correlations: Strong correlation pairs (Pearson & Spearman)
        - time_series: Temporal insights if datetime columns exist
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'age': [25, 30, 35], 'salary': [50000, 60000, 70000]})
    >>> signals = build_explainable_signals(df)
    >>> signals['dataset_summary']['num_rows']
    3
    
    >>> # Using custom configuration
    >>> custom_config = SignalConfig(strong_correlation_threshold=0.7)
    >>> signals = build_explainable_signals(df, config=custom_config)
    """
    extractor = ExplainableSignalsExtractor(config)
    signals = extractor.extract(df)
    
    # BASELINE GUARD: Capture snapshot of signals (STEP 1)
    # This protects existing behavior by logging the raw EDA statistics
    if _baseline_guard_available and is_baseline_guard_enabled():
        try:
            capture_signals_snapshot(
                signals,
                label="STEP1_build_explainable_signals",
                log_level="debug"
            )
            assert_signals_structure(signals)
        except Exception as e:
            logger.warning(f"[BASELINE] Guard check failed: {e}")
    
    return signals


# Example usage (for testing purposes only - remove or comment out in production)
if __name__ == "__main__":
    # Create sample data for testing
    sample_df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 100, 28, 32, 38, 42],
        'salary': [50000, 60000, 70000, 80000, 90000, 200000, 55000, 65000, 75000, 85000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'IT', 'Finance', 'HR', 'IT'],
        'join_date': pd.date_range('2020-01-01', periods=10, freq='M')
    })
    
    signals = build_explainable_signals(sample_df)
    
    # Print sample output
    print("Dataset Summary:")
    print(signals['dataset_summary'])
    print("\nNumeric Summary:")
    print(signals['numeric_summary'])
    print("\nCorrelations:")
    print(signals['correlations'])
