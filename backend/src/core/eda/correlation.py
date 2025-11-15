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
import json
import os
import logging
from pathlib import Path

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)

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


# ===========================
# 4. Mutual Information Analysis
# ===========================

class MutualInformationAnalyzer:
    """
    Compute non-linear correlation matrix using Mutual Information (MI).
    MI captures dependencies that linear correlation (Pearson) may miss.
    Uses ennemi library for efficient MI estimation.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize MutualInformationAnalyzer with a DataFrame.
        
        Args:
            df: pandas DataFrame to analyze
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise CorrelationError("Input must be a pandas DataFrame.")
        self.df = df
        self._mi_matrix = None
    
    def compute_mutual_information_matrix(self, columns: Optional[List[str]] = None,
                                         normalize: bool = True,
                                         k: int = 3,
                                         max_samples: int = 5000) -> pd.DataFrame:
        """
        Compute pairwise Mutual Information matrix for numeric columns.
        
        Args:
            columns: List of columns to analyze. If None, uses all numeric columns.
            normalize: If True, normalize MI to [0, 1] range using min-max scaling
            k: Number of nearest neighbors for MI estimation (default=3)
            max_samples: Maximum number of samples to use (default 5000 for performance).
                        Set to None to use all samples.
        
        Returns:
            DataFrame containing MI matrix (symmetric, diagonal = max MI)
        """
        try:
            from ennemi import estimate_mi
        except ImportError:
            raise CorrelationError(
                "ennemi library is required for Mutual Information analysis. "
                "Install with: pip install ennemi"
            )
        
        # Select numeric columns
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [c for c in columns if c in self.df.columns 
                          and pd.api.types.is_numeric_dtype(self.df[c])]
        
        if len(numeric_cols) < 2:
            raise CorrelationError("Need at least 2 numeric columns for MI matrix")
        
        # Drop NaN values for MI computation
        df_clean = self.df[numeric_cols].dropna()
        
        # Apply sampling if needed for performance
        original_count = len(df_clean)
        if max_samples is not None and len(df_clean) > max_samples:
            df_clean = df_clean.sample(n=max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} of {original_count} rows for MI computation")
        
        if len(df_clean) < 10:
            raise CorrelationError("Insufficient samples after dropping NaNs (minimum 10 required)")
        
        n = len(numeric_cols)
        mi_matrix = np.zeros((n, n))
        
        # Compute pairwise MI
        for i in range(n):
            for j in range(i, n):  # Symmetric matrix, only compute upper triangle
                if i == j:
                    # Self-MI (entropy) - set to NaN for now, will handle later
                    mi_matrix[i, j] = np.nan
                else:
                    try:
                        # Extract column data as 1D arrays
                        x = df_clean[numeric_cols[i]].values
                        y = df_clean[numeric_cols[j]].values
                        
                        # Estimate MI using ennemi
                        mi = estimate_mi(y, x, k=k, mask=None)
                        mi_value = mi[0] if isinstance(mi, np.ndarray) else mi
                        
                        # Store in both positions (symmetric)
                        mi_matrix[i, j] = mi_value
                        mi_matrix[j, i] = mi_value
                    except Exception as e:
                        # If MI estimation fails, set to 0
                        mi_matrix[i, j] = 0.0
                        mi_matrix[j, i] = 0.0
        
        # Handle diagonal: set to max MI value (represents self-information)
        max_mi = np.nanmax(mi_matrix) if not np.all(np.isnan(mi_matrix)) else 1.0
        np.fill_diagonal(mi_matrix, max_mi)
        
        # Normalize to [0, 1] if requested
        if normalize and max_mi > 0:
            mi_matrix = mi_matrix / max_mi
        
        # Convert to DataFrame
        mi_df = pd.DataFrame(mi_matrix, index=numeric_cols, columns=numeric_cols)
        self._mi_matrix = mi_df
        
        return mi_df
    
    def get_top_mi_pairs(self, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Get top-K column pairs by MI score (excluding diagonal).
        
        Args:
            top_k: Number of top pairs to return
        
        Returns:
            List of tuples: [(col1, col2, mi_score), ...]
        """
        if self._mi_matrix is None:
            raise CorrelationError("Must call compute_mutual_information_matrix() first")
        
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones(self._mi_matrix.shape), k=1).astype(bool)
        upper_triangle = self._mi_matrix.where(mask)
        
        # Stack and sort
        pairs = upper_triangle.stack().sort_values(ascending=False)
        top_pairs = pairs.head(top_k)
        
        return [(i, j, float(val)) for (i, j), val in top_pairs.items()]
    
    def compare_with_pearson(self) -> Dict[str, Any]:
        """
        Compare MI matrix with Pearson correlation to identify non-linear relationships.
        
        Returns:
            Dictionary with comparison metrics and columns showing non-linearity
        """
        if self._mi_matrix is None:
            raise CorrelationError("Must call compute_mutual_information_matrix() first")
        
        # Compute Pearson correlation for same columns
        pearson = self.df[self._mi_matrix.columns].corr().abs()
        
        # Find pairs where MI is high but Pearson is low (non-linear relationships)
        mi_threshold = 0.3
        pearson_threshold = 0.3
        
        nonlinear_pairs = []
        for col1 in self._mi_matrix.columns:
            for col2 in self._mi_matrix.columns:
                if col1 < col2:  # Avoid duplicates
                    mi_val = self._mi_matrix.loc[col1, col2]
                    pearson_val = pearson.loc[col1, col2]
                    
                    # High MI but low Pearson suggests non-linearity
                    if mi_val > mi_threshold and pearson_val < pearson_threshold:
                        nonlinear_pairs.append({
                            'column1': col1,
                            'column2': col2,
                            'mi_score': float(mi_val),
                            'pearson_score': float(pearson_val),
                            'difference': float(mi_val - pearson_val)
                        })
        
        # Sort by difference
        nonlinear_pairs.sort(key=lambda x: x['difference'], reverse=True)
        
        return {
            'nonlinear_relationships': nonlinear_pairs,
            'recommendations': self._generate_mi_recommendations(nonlinear_pairs)
        }
    
    def _generate_mi_recommendations(self, nonlinear_pairs: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on MI analysis."""
        recommendations = []
        
        if len(nonlinear_pairs) > 0:
            recommendations.append(
                f"Found {len(nonlinear_pairs)} column pairs with non-linear relationships. "
                f"Consider polynomial features or interaction terms for these pairs."
            )
            
            for pair in nonlinear_pairs[:5]:  # Top 5
                recommendations.append(
                    f"Non-linear relationship detected between '{pair['column1']}' and '{pair['column2']}' "
                    f"(MI={pair['mi_score']:.3f}, Pearson={pair['pearson_score']:.3f}). "
                    f"Try polynomial or spline transformations."
                )
        else:
            recommendations.append(
                "All detected relationships appear to be linear. "
                "Pearson correlation is sufficient for this dataset."
            )
        
        return recommendations
    
    def save_results(self, file_id: str, metadata_path: str = "workspace/metadata") -> str:
        """
        Save MI analysis results to JSON metadata file.
        
        Args:
            file_id: Unique identifier for the dataset
            metadata_path: Directory to store metadata files
        
        Returns:
            Path to saved metadata file
        """
        if self._mi_matrix is None:
            raise CorrelationError("No MI results to save. Call compute_mutual_information_matrix() first.")
        
        Path(metadata_path).mkdir(parents=True, exist_ok=True)
        
        # Get analysis results
        top_pairs = self.get_top_mi_pairs(top_k=20)
        comparison = self.compare_with_pearson()
        
        metadata = {
            'file_id': file_id,
            'analysis_type': 'mutual_information',
            'mi_matrix': self._mi_matrix.round(4).to_dict(),
            'mi_columns': self._mi_matrix.columns.tolist(),
            'mi_heatmap': self._mi_matrix.values.tolist(),
            'top_mi_pairs': [
                {'column1': col1, 'column2': col2, 'mi_score': score}
                for col1, col2, score in top_pairs
            ],
            'nonlinear_analysis': comparison
        }
        
        filepath = os.path.join(metadata_path, f"{file_id}_mutual_information.json")
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath


# Helper functions for easy access

def compute_mi_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                     normalize: bool = True, max_samples: int = 5000) -> pd.DataFrame:
    """
    Convenience function to compute MI matrix.
    
    Args:
        df: pandas DataFrame to analyze
        columns: Optional list of columns (defaults to all numeric)
        normalize: Whether to normalize MI to [0, 1]
        max_samples: Maximum number of samples to use (default 5000)
    
    Returns:
        DataFrame containing MI matrix
    """
    analyzer = MutualInformationAnalyzer(df)
    return analyzer.compute_mutual_information_matrix(
        columns=columns, 
        normalize=normalize, 
        max_samples=max_samples
    )


def load_mi_results(file_id: str, metadata_path: str = "workspace/metadata") -> Optional[Dict[str, Any]]:
    """
    Load saved MI analysis results from metadata file.
    
    Args:
        file_id: Unique identifier for the dataset
        metadata_path: Directory containing metadata files
    
    Returns:
        Dictionary with MI analysis results, or None if not found
    """
    filepath = os.path.join(metadata_path, f"{file_id}_mutual_information.json")
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise CorrelationError(f"Failed to load MI results: {str(e)}")
