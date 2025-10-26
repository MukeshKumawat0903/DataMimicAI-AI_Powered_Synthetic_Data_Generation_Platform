"""
src/core/eda/drift.py

Class-based drift detection module for DataMimicAI.
Uses Wasserstein distance and KS-test to detect drift in numeric features.

Usage:
    from backend.src.core.eda.drift import DriftDetector, DriftConfig, DriftDetectionError

    detector = DriftDetector(real_df, synth_df, config=DriftConfig())
    drift_result = detector.detect()
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp

# ===========================
# 1. Custom Error Type
# ===========================

class DriftDetectionError(Exception):
    """Raised when an error occurs during drift detection."""

# ===========================
# 2. Config Class
# ===========================

class DriftConfig:
    def __init__(self, ks_pvalue_thresh: float = 0.05, wasserstein_multiplier: float = 0.5):
        self.ks_pvalue_thresh = ks_pvalue_thresh
        self.wasserstein_multiplier = wasserstein_multiplier
# ===========================
# 3. Main DriftDetector Class
# ===========================

class DriftDetector:
    """
    DriftDetector: Compares numeric features in two DataFrames to detect drift
    using Wasserstein distance and KS test.
    """
    def __init__(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        config: Optional[DriftConfig] = None,
    ):
        if real_df is None or not isinstance(real_df, pd.DataFrame):
            raise DriftDetectionError("real_df must be a pandas DataFrame.")
        if synth_df is None or not isinstance(synth_df, pd.DataFrame):
            raise DriftDetectionError("synth_df must be a pandas DataFrame.")

        # Never mutate input dataframes in-place!
        self.real_df = real_df.copy()
        self.synth_df = synth_df.copy()
        self.config = config or DriftConfig()

        # Try to convert object columns to numeric (safe, in-place on the copy)
        for df in [self.real_df, self.synth_df]:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='ignore')

    def detect(self) -> Dict[str, Any]:
        """Detects drift between matching numeric columns in real and synthetic DataFrames."""
        def normalize_colname(name):
            return name.strip().lower()

        real_cols_norm = {normalize_colname(c): c for c in self.real_df.columns}
        synth_cols_norm = {normalize_colname(c): c for c in self.synth_df.columns}
        common_norm = set(real_cols_norm) & set(synth_cols_norm)

        numeric_cols = [
            real_cols_norm[c] for c in common_norm
            if pd.api.types.is_numeric_dtype(self.real_df[real_cols_norm[c]])
            and pd.api.types.is_numeric_dtype(self.synth_df[synth_cols_norm[c]])
        ]

        drift_stats: List[Dict[str, Any]] = []
        if not numeric_cols:
            return {
                "error": "No common numeric columns found between real and synthetic data.",
                "drift_stats": []
            }
        for col in numeric_cols:
            real, synth = self.real_df[col].dropna(), self.synth_df[col].dropna()
            if len(real) == 0 or len(synth) == 0:
                continue
            try:
                w_dist = wasserstein_distance(real, synth)
                ks_stat, ks_p = ks_2samp(real, synth)
                std_real = np.std(real)
                drifted = (
                    (ks_p < self.config.ks_pvalue_thresh)
                    or (w_dist > std_real * self.config.wasserstein_multiplier)
                )
                drift_stats.append({
                    "column": col,
                    "wasserstein_distance": float(w_dist),
                    "ks_stat": float(ks_stat),
                    "ks_pvalue": float(ks_p),
                    "drifted": bool(drifted)
                })
            except Exception as e:
                drift_stats.append({
                    "column": col,
                    "error": str(e),
                    "wasserstein_distance": None,
                    "ks_stat": None,
                    "ks_pvalue": None,
                    "drifted": None
                })
        return {"drift_stats": drift_stats}

