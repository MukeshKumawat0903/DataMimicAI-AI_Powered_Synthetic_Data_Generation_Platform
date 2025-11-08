"""
src/core/eda/profiling.py

Class-based modular EDA Profiler with configuration, strict error handling,
and extensibility for automated data profiling in DataMimicAI.

Usage:
    from src.core.eda.profiling import Profiler, EDAConfig, ProfilingError

    profiler = Profiler(df, config=EDAConfig(high_missing_thresh=0.6))
    profile_result = profiler.dataset_profile()
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# ===========================
# 1. Custom Error Type
# ===========================

class ProfilingError(Exception):
    """Raised when an error occurs during data profiling."""

# ===========================
# 2. Config Class
# ===========================

@dataclass(frozen=True)
class EDAConfig:
    high_missing_thresh: float = 0.5         # Drop suggestion if > 50% missing
    outlier_z_thresh: float = 3.0            # Z-score threshold for outlier flag
    min_high_cardinality: int = 50           # For object columns: suggest encoding if unique > this

# ===========================
# 3. Main Profiler Class
# ===========================

class Profiler:
    """
    Modular EDA Profiler for pandas DataFrames.
    - Performs column-wise and dataset-wise profiling.
    - Generates fix/action suggestions for EDA/FE.
    """
    def __init__(self, df: pd.DataFrame, config: Optional[EDAConfig] = None):
        if df is None or not isinstance(df, pd.DataFrame):
            raise ProfilingError("Input must be a pandas DataFrame.")
        self.df = df
        self.config = config or EDAConfig()

    def column_profile(self, s: pd.Series) -> Dict[str, Any]:
        """
        Profile a single Series/column and extract statistics & quickfix cues.
        """
        if s is None or not isinstance(s, pd.Series):
            raise ProfilingError(f"Invalid Series passed to column_profile: {type(s)}")
        out = {
            "column": s.name,
            "dtype": str(s.dtype),
            "n_missing": int(s.isnull().sum()),
            "missing_%": float(100 * s.isnull().mean()),
            "n_unique": int(s.nunique(dropna=True)),
            "sample_unique": s.dropna().unique()[:5].tolist(),
        }
        # Numeric profiling
        if np.issubdtype(s.dtype, np.number):
            min_v = max_v = mean_v = std_v = skew_v = kurt_v = None
            if not s.isnull().all():
                min_v = np.nanmin(s)
                max_v = np.nanmax(s)
                mean_v = np.nanmean(s)
                std_v = np.nanstd(s)
                skew_v = s.skew()
                kurt_v = s.kurtosis()
            out.update({
                "min": float(min_v) if min_v is not None else None,
                "max": float(max_v) if max_v is not None else None,
                "mean": float(mean_v) if mean_v is not None else None,
                "std": float(std_v) if std_v is not None else None,
                "skew": float(skew_v) if skew_v is not None else None,
                "kurtosis": float(kurt_v) if kurt_v is not None else None,
                "n_outliers": int(
                    ((s - mean_v).abs() > self.config.outlier_z_thresh * std_v).sum()
                ) if std_v and not np.isnan(std_v) else 0,
                "outlier_%": float(
                    100 * ((s - mean_v).abs() > self.config.outlier_z_thresh * std_v).mean()
                ) if std_v and not np.isnan(std_v) else 0,
            })
        # Categorical/object profiling
        else:
            vc = s.value_counts(dropna=True)
            out.update({
                "most_common": vc.index[0] if not vc.empty else None,
                "most_common_count": int(vc.iloc[0]) if not vc.empty else 0,
                "least_common": vc.index[-1] if len(vc) > 1 else None,
                "rare_values": vc[vc < 5].index.tolist() if len(vc) > 0 else [],
            })
        return out

    def dataset_profile(self) -> Dict[str, Any]:
        """
        Full dataset profiling: per-column stats, summary, actionable suggestions.
        Returns structure compatible with API/frontend usage.
        """
        profile: List[Dict[str, Any]] = [self.column_profile(self.df[col]) for col in self.df.columns]

        summary: Dict[str, Any] = {
            "n_rows": int(len(self.df)),
            "n_columns": int(len(self.df.columns)),
            "total_missing": int(self.df.isnull().sum().sum()),
            "columns_with_missing": int((self.df.isnull().sum() > 0).sum()),
            "constant_columns": [col for col in self.df.columns if self.df[col].nunique(dropna=True) <= 1],
            "duplicated_columns": self.df.columns[self.df.T.duplicated()].tolist(),
            "columns_high_missing": [
                col for col in self.df.columns if self.df[col].isnull().mean() > self.config.high_missing_thresh
            ],
        }

        # ---- Suggestions (used by your frontend!) ----
        suggestions: List[Dict[str, Any]] = []
        for stat in profile:
            col = stat["column"]
            # Missing value suggestion
            if stat["n_missing"] > 0:
                if stat["dtype"].startswith("float") or stat["dtype"].startswith("int"):
                    method = "median" if abs(stat.get("skew", 0)) > 1 else "mean"
                    suggestions.append({
                        "column": col,
                        "action": f"Impute missing ({stat['n_missing']} values) in '{col}' with {method}",
                        "fix_type": "impute",
                        "method": method
                    })
                else:
                    suggestions.append({
                        "column": col,
                        "action": f"Impute missing ({stat['n_missing']}) in '{col}' with mode",
                        "fix_type": "impute",
                        "method": "mode"
                    })
            # High cardinality categorical suggestion
            if stat["dtype"] == "object" and stat["n_unique"] > self.config.min_high_cardinality:
                suggestions.append({
                    "column": col,
                    "action": f"High cardinality in '{col}' ({stat['n_unique']} unique); consider encoding or grouping.",
                    "fix_type": "encode"
                })
            # Low cardinality check
            if stat["dtype"] in ["object", "category"] and stat["n_unique"] < 3:
                suggestions.append({
                    "column": col,
                    "action": f"Low cardinality in '{col}' ({stat['n_unique']} unique); check for errors or merge with other columns.",
                    "fix_type": "review"
                })
            # Outlier flag
            if stat.get("n_outliers", 0) > 0:
                suggestions.append({
                    "column": col,
                    "action": f"Outliers detected in '{col}' ({stat['n_outliers']} values > {self.config.outlier_z_thresh} std dev)",
                    "fix_type": "outlier"
                })

        # Drop constant/duplicated/high-missing
        for col in summary["constant_columns"]:
            suggestions.append({
                "column": col,
                "action": f"Column '{col}' is constant; recommend dropping.",
                "fix_type": "drop_constant"
            })
        for col in summary["duplicated_columns"]:
            suggestions.append({
                "column": col,
                "action": f"Column '{col}' is a duplicate; recommend dropping.",
                "fix_type": "drop_duplicate"
            })
        for col in summary["columns_high_missing"]:
            suggestions.append({
                "column": col,
                "action": f"Column '{col}' has >{int(100*self.config.high_missing_thresh)}% missing; consider dropping or advanced imputation.",
                "fix_type": "drop_high_missing"
            })

        return {
            "profile": profile,
            "summary": summary,
            "suggestions": suggestions
        }

class DataCleaner:
    """Utility for imputation, dropping, encoding."""
    @staticmethod
    def impute_missing(df: pd.DataFrame, columns: List[str], method: str = "mean", k: int = 3) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if method == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif method == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif method == "knn":
                    try:
                        from sklearn.impute import KNNImputer
                        imputer = KNNImputer(n_neighbors=k)
                        df[[col]] = imputer.fit_transform(df[[col]])
                    except ImportError:
                        raise ProfilingError("scikit-learn is required for KNN imputation.")
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        return df

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return df.drop(columns=columns)

    @staticmethod
    def encode_high_cardinality(df: pd.DataFrame, columns: List[str], max_values: int = 10) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            freq = df[col].value_counts()
            common = freq[:max_values].index
            df[col] = df[col].apply(lambda x: x if x in common else "Other")
        return df


# ===========================
# 4. Statistical Fingerprinting
# ===========================

class DistributionAnalyzer:
    """
    Statistical fingerprinting using distribution fitting to identify
    the best-fit probability distribution for numeric columns.
    Uses distfit library for automated distribution detection.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DistributionAnalyzer with a DataFrame.
        
        Args:
            df: pandas DataFrame to analyze
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise ProfilingError("Input must be a pandas DataFrame.")
        self.df = df
        self._distributions = None
    
    def fit_distributions(self, columns: Optional[List[str]] = None, 
                          dist_list: Optional[List[str]] = None,
                          max_samples: int = 10000) -> Dict[str, Dict[str, Any]]:
        """
        Fit statistical distributions to numeric columns using distfit.
        
        Args:
            columns: List of columns to analyze. If None, analyzes all numeric columns.
            dist_list: List of distribution names to test. If None, uses common distributions.
            max_samples: Maximum number of samples to use per column (default 10000 for performance).
                        Set to None to use all samples.
        
        Returns:
            Dictionary mapping column names to distribution metadata:
            {
                'column_name': {
                    'best_distribution': str,  # e.g., 'norm', 'lognorm', 'gamma'
                    'params': dict,            # {'loc': float, 'scale': float, ...}
                    'score': float,            # RSS or goodness-of-fit metric
                    'method': str,             # fitting method used
                    'n_samples': int,          # number of samples used
                    'has_negatives': bool,     # whether column contains negative values
                    'skewness': float,         # statistical skewness
                    'kurtosis': float          # statistical kurtosis
                }
            }
        """
        try:
            from distfit import distfit
        except ImportError:
            raise ProfilingError(
                "distfit library is required for distribution analysis. "
                "Install with: pip install distfit"
            )
        
        # Default distribution list (fast and commonly used)
        if dist_list is None:
            dist_list = ['norm', 'lognorm', 'expon', 'gamma', 'beta', 
                        'uniform', 'chi2', 'weibull_min', 't']
        
        # Select numeric columns
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [c for c in columns if c in self.df.columns 
                          and pd.api.types.is_numeric_dtype(self.df[c])]
        
        results = {}
        
        for col in numeric_cols:
            try:
                # Extract clean data (drop NaNs)
                data = self.df[col].dropna()
                
                # Apply sampling if needed for performance
                original_count = len(data)
                if max_samples is not None and len(data) > max_samples:
                    data = data.sample(n=max_samples, random_state=42)
                
                if len(data) < 10:  # Need minimum samples
                    results[col] = {
                        'best_distribution': 'insufficient_data',
                        'params': {},
                        'score': None,
                        'method': None,
                        'n_samples': len(data),
                        'n_total': original_count,
                        'has_negatives': bool((data < 0).any()),
                        'skewness': None,
                        'kurtosis': None,
                        'error': 'Insufficient samples (minimum 10 required)'
                    }
                    continue
                
                # Statistical metadata
                has_negatives = bool((data < 0).any())
                skewness = float(data.skew())
                kurtosis = float(data.kurtosis())
                
                # Initialize distfit
                dfit = distfit(distr=dist_list, method='rss', verbose=0)
                
                # Fit distributions
                dfit.fit_transform(data.values)
                
                # Extract best fit results
                if hasattr(dfit, 'model') and dfit.model is not None:
                    best_dist = dfit.model['name']
                    params = dfit.model.get('params', {})
                    score = dfit.model.get('score', None)  # RSS or similar
                    
                    # Convert params to serializable format
                    params_dict = {}
                    if isinstance(params, dict):
                        params_dict = {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                     for k, v in params.items()}
                    elif isinstance(params, (tuple, list)):
                        # Many distributions return params as tuple (shape, loc, scale)
                        if len(params) >= 2:
                            params_dict = {
                                'loc': float(params[-2]),
                                'scale': float(params[-1])
                            }
                            if len(params) > 2:
                                # Shape parameters
                                for i, p in enumerate(params[:-2]):
                                    params_dict[f'shape_{i}'] = float(p)
                    
                    results[col] = {
                        'best_distribution': best_dist,
                        'params': params_dict,
                        'score': float(score) if score is not None else None,
                        'method': 'rss',
                        'n_samples': int(len(data)),
                        'n_total': original_count,
                        'has_negatives': has_negatives,
                        'skewness': skewness,
                        'kurtosis': kurtosis
                    }
                else:
                    # Fallback if fitting failed
                    results[col] = {
                        'best_distribution': 'unknown',
                        'params': {},
                        'score': None,
                        'method': 'rss',
                        'n_samples': int(len(data)),
                        'n_total': original_count,
                        'has_negatives': has_negatives,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'error': 'Distribution fitting failed'
                    }
                    
            except Exception as e:
                results[col] = {
                    'best_distribution': 'error',
                    'params': {},
                    'score': None,
                    'method': None,
                    'n_samples': int(len(self.df[col].dropna())),
                    'n_total': int(len(self.df[col].dropna())),
                    'has_negatives': None,
                    'skewness': None,
                    'kurtosis': None,
                    'error': str(e)
                }
        
        self._distributions = results
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of distribution analysis.
        
        Returns:
            Dictionary with summary statistics and recommendations
        """
        if self._distributions is None:
            raise ProfilingError("Must call fit_distributions() first")
        
        # Count distribution types
        dist_counts = {}
        for col, meta in self._distributions.items():
            dist_name = meta['best_distribution']
            dist_counts[dist_name] = dist_counts.get(dist_name, 0) + 1
        
        # Identify problematic columns
        skewed_columns = [col for col, meta in self._distributions.items()
                         if meta.get('skewness') is not None and abs(meta['skewness']) > 1]
        
        heavy_tailed = [col for col, meta in self._distributions.items()
                       if meta.get('kurtosis') is not None and abs(meta['kurtosis']) > 3]
        
        return {
            'total_columns_analyzed': len(self._distributions),
            'distribution_counts': dist_counts,
            'highly_skewed_columns': skewed_columns,
            'heavy_tailed_columns': heavy_tailed,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on distribution analysis."""
        recommendations = []
        
        for col, meta in self._distributions.items():
            dist = meta['best_distribution']
            
            # Log-normal recommendation
            if dist == 'lognorm':
                recommendations.append(
                    f"Column '{col}' follows log-normal distribution. "
                    f"Consider log-transformation before modeling."
                )
            
            # Exponential recommendation
            elif dist == 'expon':
                recommendations.append(
                    f"Column '{col}' follows exponential distribution. "
                    f"Suitable for time-between-events modeling."
                )
            
            # Skewness recommendations
            skew = meta.get('skewness')
            if skew is not None and abs(skew) > 2:
                recommendations.append(
                    f"Column '{col}' is highly skewed (skewness={skew:.2f}). "
                    f"Apply Box-Cox or Yeo-Johnson transformation."
                )
            
            # Beta distribution (bounded)
            if dist == 'beta':
                recommendations.append(
                    f"Column '{col}' follows beta distribution (bounded [0,1]). "
                    f"Check if data represents probabilities or proportions."
                )
        
        return recommendations
    
    def save_results(self, file_id: str, metadata_path: str = "workspace/metadata") -> str:
        """
        Save distribution analysis results to JSON metadata file.
        
        Args:
            file_id: Unique identifier for the dataset
            metadata_path: Directory to store metadata files
        
        Returns:
            Path to saved metadata file
        """
        if self._distributions is None:
            raise ProfilingError("No distribution results to save. Call fit_distributions() first.")
        
        Path(metadata_path).mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'file_id': file_id,
            'analysis_type': 'statistical_fingerprinting',
            'distributions': self._distributions,
            'summary': self.get_summary()
        }
        
        filepath = os.path.join(metadata_path, f"{file_id}_distributions.json")
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath


# Helper functions for easy access

def run_distribution_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None,
                             max_samples: int = 10000) -> Dict[str, Any]:
    """
    Convenience function to run distribution analysis on a DataFrame.
    
    Args:
        df: pandas DataFrame to analyze
        columns: Optional list of columns to analyze (defaults to all numeric)
        max_samples: Maximum samples to use per column for performance (default 10000)
    
    Returns:
        Dictionary containing distribution analysis results
    """
    analyzer = DistributionAnalyzer(df)
    distributions = analyzer.fit_distributions(columns=columns, max_samples=max_samples)
    summary = analyzer.get_summary()
    
    return {
        'distributions': distributions,
        'summary': summary
    }


def load_distribution_results(file_id: str, metadata_path: str = "workspace/metadata") -> Optional[Dict[str, Any]]:
    """
    Load saved distribution analysis results from metadata file.
    
    Args:
        file_id: Unique identifier for the dataset
        metadata_path: Directory containing metadata files
    
    Returns:
        Dictionary with distribution analysis results, or None if not found
    """
    filepath = os.path.join(metadata_path, f"{file_id}_distributions.json")
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ProfilingError(f"Failed to load distribution results: {str(e)}")


# Alias for requirements spec compatibility
def compute_distribution_fits(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                              max_samples: int = 10000) -> Dict[str, Any]:
    """
    Alias for run_distribution_analysis() to match requirements specification.
    
    Compute statistical distribution fingerprinting for numeric columns.
    
    Args:
        df: pandas DataFrame to analyze
        columns: Optional list of columns to analyze (defaults to all numeric)
        max_samples: Maximum samples to use per column for performance
    
    Returns:
        Dictionary containing distribution analysis results
    """
    return run_distribution_analysis(df, columns, max_samples=max_samples)