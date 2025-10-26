"""
src/core/eda/outliers.py

Class-based outlier detection module for DataMimicAI.
Supports IQR and Z-score based outlier detection for numeric columns.

Usage:
    from backend.src.core.eda.outliers import OutlierDetector, OutlierConfig, OutlierDetectionError

    detector = OutlierDetector(df, config=OutlierConfig(z_thresh=3.0))
    outlier_result = detector.detect()
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

# ===========================
# 1. Custom Error Type
# ===========================

class OutlierDetectionError(Exception):
    """Raised when an error occurs during outlier detection."""

# ===========================
# 2. Config Class
# ===========================

@dataclass(frozen=True)
class OutlierConfig:
    z_thresh: float = 3.0         # Z-score threshold
    iqr_multiplier: float = 1.5   # IQR multiplier for outlier detection

# ===========================
# 3. Main OutlierDetector Class
# ===========================

class OutlierDetector:
    """
    OutlierDetector: Detects and removes outliers in numeric columns using
    both Z-score and IQR methods.
    """
    def __init__(self, df: pd.DataFrame, config: Optional[OutlierConfig] = None):
        if df is None or not isinstance(df, pd.DataFrame):
            raise OutlierDetectionError("Input must be a pandas DataFrame.")
        self.df = df
        self.config = config or OutlierConfig()

    def detect(self) -> Dict[str, Any]:
        """
        Detect outliers in all numeric columns. 
        Returns outlier indices and summary stats per column.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_indices = {}
        stats = []

        for col in numeric_cols:
            s = self.df[col]
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            # Outlier masks
            iqr_mask = (s < Q1 - self.config.iqr_multiplier * IQR) | (s > Q3 + self.config.iqr_multiplier * IQR)
            # For std, protect against zero/NaN
            std_val = s.std()
            z_mask = (np.abs((s - s.mean()) / std_val) > self.config.z_thresh) if std_val else pd.Series(False, index=s.index)
            outliers = iqr_mask | z_mask
            outlier_idx = s[outliers].index.tolist()
            outlier_indices[col] = outlier_idx
            stats.append({
                "column": col,
                "outlier_count": int(outliers.sum()),
                "outlier_percent": 100 * float(outliers.mean()),
                "method": "IQR/Z-score"
            })
        return {"outlier_indices": outlier_indices, "stats": stats}

    def remove(self, columns: List[str]) -> pd.DataFrame:
        """
        Remove rows with outliers in any of the specified columns (IQR and Z-score).
        Returns a new DataFrame.
        """
        if not columns:
            raise OutlierDetectionError("No columns specified for outlier removal.")
        mask = np.ones(len(self.df), dtype=bool)
        for col in columns:
            if col not in self.df.columns:
                raise OutlierDetectionError(f"Column '{col}' not found in DataFrame.")
            s = self.df[col]
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            iqr_mask = (s >= Q1 - self.config.iqr_multiplier * IQR) & (s <= Q3 + self.config.iqr_multiplier * IQR)
            std_val = s.std()
            z_mask = (np.abs((s - s.mean()) / std_val) <= self.config.z_thresh) if std_val else pd.Series(True, index=s.index)
            mask = mask & iqr_mask & z_mask
        return self.df[mask].reset_index(drop=True)


class OutlierCleaner:
    @staticmethod
    def drop_outliers(df: pd.DataFrame, columns: list, z_thresh: float = 3.0) -> pd.DataFrame:
        mask = np.ones(len(df), dtype=bool)
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_mean = df[col].mean()
                col_std = df[col].std()
                z = (df[col] - col_mean) / col_std
                mask &= np.abs(z) < z_thresh
        return df[mask].reset_index(drop=True)
