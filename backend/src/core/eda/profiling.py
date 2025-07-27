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