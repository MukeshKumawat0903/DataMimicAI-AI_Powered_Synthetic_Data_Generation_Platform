"""
src/core/eda/correlation.py

Modular class-based Smart Correlation & Pattern Discovery for DataMimicAI.

Usage:
    from src.core.eda.correlation import CorrelationAnalyzer, CorrelationConfig, CorrelationError

    analyzer = CorrelationAnalyzer(df, config=CorrelationConfig(top_k=15))
    corr_result = analyzer.analyze()
"""
import base64
import io

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# ===========================
# 1. Custom Error Type
# ===========================

class CorrelationError(Exception):
    """Raised when an error occurs during correlation analysis."""

# ===========================
# 2. Config Class
# ===========================

@dataclass(frozen=True)
class CorrelationConfig:
    top_k: int = 10           # Number of top correlations to return
    high_corr_thresh: float = 0.99  # Threshold for data leakage/perfect corr detection
    min_categorical: int = 2        # Minimum unique categories for Cramér's V
    max_categorical: int = 80       # Ignore categorical features with too many uniques

# ===========================
# 3. Main CorrelationAnalyzer Class
# ===========================

class CorrelationAnalyzer:
    """
    Class for computing linear/nonlinear correlation matrices,
    top correlated pairs, categorical associations (Cramér’s V),
    and data leakage detection.
    """
    def __init__(self, df: pd.DataFrame, config: Optional[CorrelationConfig] = None):
        if df is None or not isinstance(df, pd.DataFrame):
            raise CorrelationError("Input must be a pandas DataFrame.")
        self.df = df
        self.config = config or CorrelationConfig()

    def analyze(self) -> Dict[str, Any]:
        """
        Perform correlation and association analysis.
        Returns a dict for use by API/frontend.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # ---- Numeric Correlations ----
        corr_matrix = self.df[numeric_cols].corr().fillna(0) if len(numeric_cols) > 1 else pd.DataFrame()
        spearman = self.df[numeric_cols].corr(method='spearman').fillna(0) if len(numeric_cols) > 1 else pd.DataFrame()
        kendall = self.df[numeric_cols].corr(method='kendall').fillna(0) if len(numeric_cols) > 1 else pd.DataFrame()

        # Top absolute Pearson correlations (excluding diagonal)
        abs_corr = corr_matrix.abs().where(~np.eye(corr_matrix.shape[0], dtype=bool))
        pairs = abs_corr.stack().sort_values(ascending=False)
        top_pairs = pairs.head(self.config.top_k).index.tolist() if not pairs.empty else []
        top_corrs = [
            (i, j, float(corr_matrix.loc[i, j]))
            for i, j in top_pairs
        ] if top_pairs else []

        # ---- Categorical Associations: Cramér's V ----
        cat_results = []
        # Limit categorical cols for performance
        cat_cols_valid = [
            col for col in cat_cols
            if self.df[col].nunique() <= self.config.max_categorical and self.df[col].nunique() >= self.config.min_categorical
        ]
        if len(cat_cols_valid) >= 2:
            for i in range(len(cat_cols_valid)):
                for j in range(i+1, len(cat_cols_valid)):
                    table = pd.crosstab(self.df[cat_cols_valid[i]], self.df[cat_cols_valid[j]])
                    if table.shape[0] > 1 and table.shape[1] > 1:
                        try:
                            chi2 = chi2_contingency(table)[0]
                            n = table.sum().sum()
                            phi2 = chi2 / n
                            r, k = table.shape
                            # Bias correction
                            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                            rcorr = r - ((r-1)**2)/(n-1)
                            kcorr = k - ((k-1)**2)/(n-1)
                            denom = min((kcorr-1), (rcorr-1))
                            cramer_v = np.sqrt(phi2corr / denom) if denom > 0 else np.nan
                            cat_results.append({
                                "Feature 1": cat_cols_valid[i],
                                "Feature 2": cat_cols_valid[j],
                                "CramersV": float(cramer_v)
                            })
                        except Exception:
                            continue

        # ---- Data Leakage: High Linear Correlations ----
        leakage_pairs = []
        if not corr_matrix.empty:
            for i in numeric_cols:
                for j in numeric_cols:
                    if i != j and abs(corr_matrix.loc[i, j]) > self.config.high_corr_thresh:
                        leakage_pairs.append((i, j, float(corr_matrix.loc[i, j])))

        # For heatmap: return Pearson matrix as 2D array (for plotting)
        corr_heatmap = corr_matrix.values.tolist() if not corr_matrix.empty else []
        corr_columns = corr_matrix.columns.tolist() if not corr_matrix.empty else []

        return {
            "pearson_corr_matrix": corr_matrix.round(3).to_dict(),
            "spearman_corr_matrix": spearman.round(3).to_dict(),
            "kendall_corr_matrix": kendall.round(3).to_dict(),
            "corr_heatmap": corr_heatmap,
            "corr_columns": corr_columns,
            "top_corrs": top_corrs,
            "categorical_assoc": cat_results,
            "leakage_pairs": leakage_pairs,
        }

def make_corr_heatmap_base64(df, columns=None, title="Feature Correlation Heatmap"):
    if columns is not None:
        df = df[columns]
    corr = df.corr()
    plt.figure(figsize=(max(10, len(corr.columns)), 8))
    ax = sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        linewidths=0.5, linecolor="gray", cbar=True,
        square=False, annot_kws={"fontsize":11}
    )
    plt.title(title, fontsize=16, fontweight="bold", loc="left", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
