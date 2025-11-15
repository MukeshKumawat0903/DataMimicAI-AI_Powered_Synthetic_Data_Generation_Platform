"""
Outlier remediation module for data cleaning and transformation.
Provides methods: winsorize, remove, bin, cap outliers.
Integrates with feedback_engine for tracking remediation actions.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy.stats import mstats


class RemediationError(Exception):
    """Raised when remediation fails."""
    pass


@dataclass
class RemediationConfig:
    """Configuration for outlier remediation."""
    winsorize_limits: Tuple[float, float] = (0.05, 0.05)  # Lower and upper percentiles
    cap_method: str = "iqr"  # 'iqr' or 'percentile'
    cap_multiplier: float = 1.5  # For IQR method
    cap_percentiles: Tuple[float, float] = (1, 99)  # For percentile method
    bin_strategy: str = "quantile"  # 'quantile', 'uniform', 'kmeans'
    n_bins: int = 5


class OutlierRemediator:
    """
    Comprehensive outlier remediation with multiple strategies.
    """
    def __init__(self, df: pd.DataFrame, config: RemediationConfig = None):
        if df.empty:
            raise RemediationError("DataFrame is empty.")
        self.df = df.copy()
        self.config = config or RemediationConfig()
        self.metadata_dir = Path("workspace/metadata")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.remediation_history = []

    # ============= WINSORIZATION =============
    
    def winsorize(self, columns: List[str], limits: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Winsorize outliers by capping extreme values at percentiles.
        limits: (lower_percentile, upper_percentile) e.g., (0.05, 0.05) caps at 5th and 95th percentiles
        """
        if limits is None:
            limits = self.config.winsorize_limits
        
        df_result = self.df.copy()
        remediation_stats = []
        
        for col in columns:
            if col not in df_result.columns:
                raise RemediationError(f"Column '{col}' not found.")
            if not pd.api.types.is_numeric_dtype(df_result[col]):
                raise RemediationError(f"Column '{col}' is not numeric.")
            
            original_data = df_result[col].dropna()
            if len(original_data) == 0:
                continue
            
            # Winsorize
            winsorized = mstats.winsorize(original_data, limits=limits)
            df_result.loc[original_data.index, col] = winsorized
            
            # Track changes
            n_capped = int((original_data != winsorized).sum())
            lower_bound = np.percentile(original_data, limits[0] * 100)
            upper_bound = np.percentile(original_data, 100 - limits[1] * 100)
            
            remediation_stats.append({
                "column": col,
                "method": "winsorize",
                "n_values_capped": n_capped,
                "percent_capped": 100 * n_capped / len(original_data),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "limits": limits
            })
        
        self.remediation_history.append({
            "action": "winsorize",
            "columns": columns,
            "stats": remediation_stats,
            "timestamp": datetime.now().isoformat()
        })
        
        return df_result

    # ============= CAPPING =============
    
    def cap_outliers(self, columns: List[str], method: Optional[str] = None, 
                     multiplier: Optional[float] = None, percentiles: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Cap outliers using IQR or percentile method.
        method: 'iqr' or 'percentile'
        multiplier: For IQR method (default 1.5)
        percentiles: For percentile method e.g., (1, 99)
        """
        method = method or self.config.cap_method
        multiplier = multiplier or self.config.cap_multiplier
        percentiles = percentiles or self.config.cap_percentiles
        
        df_result = self.df.copy()
        remediation_stats = []
        
        for col in columns:
            if col not in df_result.columns:
                raise RemediationError(f"Column '{col}' not found.")
            if not pd.api.types.is_numeric_dtype(df_result[col]):
                raise RemediationError(f"Column '{col}' is not numeric.")
            
            data = df_result[col].dropna()
            if len(data) == 0:
                continue
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
            elif method == 'percentile':
                lower_bound = np.percentile(data, percentiles[0])
                upper_bound = np.percentile(data, percentiles[1])
            else:
                raise RemediationError(f"Invalid method '{method}'. Use 'iqr' or 'percentile'.")
            
            # Cap values
            original = df_result[col].copy()
            df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Track changes
            n_capped = int((original != df_result[col]).sum())
            
            remediation_stats.append({
                "column": col,
                "method": f"cap_{method}",
                "n_values_capped": n_capped,
                "percent_capped": 100 * n_capped / len(data),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "multiplier": multiplier if method == 'iqr' else None,
                "percentiles": percentiles if method == 'percentile' else None
            })
        
        self.remediation_history.append({
            "action": "cap_outliers",
            "columns": columns,
            "method": method,
            "stats": remediation_stats,
            "timestamp": datetime.now().isoformat()
        })
        
        return df_result

    # ============= REMOVAL =============
    
    def remove_outliers(self, outlier_indices: Dict[str, List[int]], 
                       method: str = "union") -> pd.DataFrame:
        """
        Remove rows containing outliers.
        outlier_indices: Dict mapping column names to lists of outlier indices
        method: 'union' (remove if outlier in ANY column) or 'intersection' (remove if outlier in ALL columns)
        """
        if not outlier_indices:
            raise RemediationError("No outlier indices provided.")
        
        df_result = self.df.copy()
        
        if method == "union":
            # Remove rows that are outliers in ANY column
            all_outlier_idx = set()
            for idx_list in outlier_indices.values():
                all_outlier_idx.update(idx_list)
            mask = ~df_result.index.isin(all_outlier_idx)
        elif method == "intersection":
            # Remove rows that are outliers in ALL columns
            outlier_sets = [set(idx_list) for idx_list in outlier_indices.values()]
            all_outlier_idx = set.intersection(*outlier_sets) if outlier_sets else set()
            mask = ~df_result.index.isin(all_outlier_idx)
        else:
            raise RemediationError(f"Invalid method '{method}'. Use 'union' or 'intersection'.")
        
        df_result = df_result[mask].reset_index(drop=True)
        
        n_removed = len(self.df) - len(df_result)
        
        self.remediation_history.append({
            "action": "remove_outliers",
            "columns": list(outlier_indices.keys()),
            "method": method,
            "n_rows_removed": n_removed,
            "percent_removed": 100 * n_removed / len(self.df),
            "timestamp": datetime.now().isoformat()
        })
        
        return df_result

    # ============= BINNING =============
    
    def bin_outliers(self, columns: List[str], strategy: Optional[str] = None, 
                     n_bins: Optional[int] = None, labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Bin continuous values to reduce outlier impact.
        strategy: 'quantile' (equal frequency), 'uniform' (equal width), 'kmeans' (clustering)
        n_bins: Number of bins
        labels: Custom labels for bins (optional)
        """
        strategy = strategy or self.config.bin_strategy
        n_bins = n_bins or self.config.n_bins
        
        df_result = self.df.copy()
        remediation_stats = []
        
        for col in columns:
            if col not in df_result.columns:
                raise RemediationError(f"Column '{col}' not found.")
            if not pd.api.types.is_numeric_dtype(df_result[col]):
                raise RemediationError(f"Column '{col}' is not numeric.")
            
            data = df_result[col].dropna()
            if len(data) == 0:
                continue
            
            binned_col_name = f"{col}_binned"
            
            if strategy == 'quantile':
                df_result[binned_col_name] = pd.qcut(df_result[col], q=n_bins, labels=labels, duplicates='drop')
            elif strategy == 'uniform':
                df_result[binned_col_name] = pd.cut(df_result[col], bins=n_bins, labels=labels)
            elif strategy == 'kmeans':
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
                valid_mask = df_result[col].notna()
                clusters = np.full(len(df_result), np.nan)
                clusters[valid_mask] = kmeans.fit_predict(df_result.loc[valid_mask, col].values.reshape(-1, 1))
                df_result[binned_col_name] = clusters
            else:
                raise RemediationError(f"Invalid strategy '{strategy}'. Use 'quantile', 'uniform', or 'kmeans'.")
            
            remediation_stats.append({
                "column": col,
                "binned_column": binned_col_name,
                "method": f"bin_{strategy}",
                "n_bins": n_bins,
                "strategy": strategy
            })
        
        self.remediation_history.append({
            "action": "bin_outliers",
            "columns": columns,
            "strategy": strategy,
            "n_bins": n_bins,
            "stats": remediation_stats,
            "timestamp": datetime.now().isoformat()
        })
        
        return df_result

    # ============= TRANSFORMATION =============
    
    def transform_outliers(self, columns: List[str], method: str = "log") -> pd.DataFrame:
        """
        Transform data to reduce outlier impact.
        method: 'log', 'sqrt', 'boxcox', 'zscore'
        """
        from scipy.stats import boxcox
        
        df_result = self.df.copy()
        remediation_stats = []
        
        for col in columns:
            if col not in df_result.columns:
                raise RemediationError(f"Column '{col}' not found.")
            if not pd.api.types.is_numeric_dtype(df_result[col]):
                raise RemediationError(f"Column '{col}' is not numeric.")
            
            data = df_result[col].dropna()
            if len(data) == 0:
                continue
            
            transformed_col_name = f"{col}_{method}"
            
            try:
                if method == 'log':
                    # Add small constant to handle zeros
                    min_val = data.min()
                    offset = 1 - min_val if min_val <= 0 else 0
                    df_result[transformed_col_name] = np.log(df_result[col] + offset + 1)
                
                elif method == 'sqrt':
                    min_val = data.min()
                    offset = -min_val if min_val < 0 else 0
                    df_result[transformed_col_name] = np.sqrt(df_result[col] + offset)
                
                elif method == 'boxcox':
                    # Box-Cox requires positive values
                    if data.min() <= 0:
                        offset = 1 - data.min()
                        df_result[transformed_col_name], lambda_param = boxcox(df_result[col] + offset)
                    else:
                        df_result[transformed_col_name], lambda_param = boxcox(df_result[col])
                    remediation_stats.append({"lambda": float(lambda_param)})
                
                elif method == 'zscore':
                    mean = data.mean()
                    std = data.std()
                    df_result[transformed_col_name] = (df_result[col] - mean) / std if std > 0 else 0
                
                else:
                    raise RemediationError(f"Invalid method '{method}'. Use 'log', 'sqrt', 'boxcox', or 'zscore'.")
                
                remediation_stats.append({
                    "column": col,
                    "transformed_column": transformed_col_name,
                    "method": method
                })
            
            except Exception as e:
                raise RemediationError(f"Transformation failed for '{col}': {str(e)}")
        
        self.remediation_history.append({
            "action": "transform_outliers",
            "columns": columns,
            "method": method,
            "stats": remediation_stats,
            "timestamp": datetime.now().isoformat()
        })
        
        return df_result

    # ============= METADATA & FEEDBACK =============
    
    def get_remediation_history(self) -> List[Dict[str, Any]]:
        """Get history of all remediation actions performed."""
        return self.remediation_history

    def save_metadata(self, file_id: str) -> str:
        """Save remediation history to JSON metadata."""
        metadata = {
            "file_id": file_id,
            "timestamp": datetime.now().isoformat(),
            "remediation_history": self.remediation_history,
            "config": {
                "winsorize_limits": self.config.winsorize_limits,
                "cap_method": self.config.cap_method,
                "cap_multiplier": self.config.cap_multiplier,
                "bin_strategy": self.config.bin_strategy,
                "n_bins": self.config.n_bins
            }
        }
        
        metadata_path = self.metadata_dir / f"{file_id}_remediation.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(metadata_path)

    def load_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Load remediation metadata from JSON."""
        metadata_path = self.metadata_dir / f"{file_id}_remediation.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

    def send_feedback(self, file_id: str, action: str, feedback: Dict[str, Any]) -> None:
        """
        Send remediation action to feedback_engine for tracking.
        Integrates with the platform's feedback loop.
        """
        try:
            from src.core.feedback_engine import log_eda_feedback
            
            log_eda_feedback(
                file_id=file_id,
                feature_type="outlier_remediation",
                action=action,
                feedback=feedback
            )
        except ImportError:
            # Fallback: save to local feedback file
            feedback_path = self.metadata_dir / f"{file_id}_feedback.json"
            feedback_data = {
                "file_id": file_id,
                "action": action,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            }
            
            existing_feedback = []
            if feedback_path.exists():
                with open(feedback_path, 'r') as f:
                    existing_feedback = json.load(f)
            
            existing_feedback.append(feedback_data)
            
            with open(feedback_path, 'w') as f:
                json.dump(existing_feedback, f, indent=2)


# ============= CONVENIENCE FUNCTIONS FOR API =============

def remediate_outliers(
    df: pd.DataFrame,
    file_id: str,
    method: str,
    columns: List[str],
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply remediation method and return result with metadata.
    method: 'winsorize', 'cap', 'remove', 'bin', 'transform'
    """
    remediator = OutlierRemediator(df)
    
    if method == 'winsorize':
        df_result = remediator.winsorize(columns, kwargs.get('limits'))
    elif method == 'cap':
        df_result = remediator.cap_outliers(
            columns,
            kwargs.get('cap_method'),
            kwargs.get('multiplier'),
            kwargs.get('percentiles')
        )
    elif method == 'remove':
        df_result = remediator.remove_outliers(
            kwargs.get('outlier_indices', {}),
            kwargs.get('removal_method', 'union')
        )
    elif method == 'bin':
        df_result = remediator.bin_outliers(
            columns,
            kwargs.get('strategy'),
            kwargs.get('n_bins'),
            kwargs.get('labels')
        )
    elif method == 'transform':
        df_result = remediator.transform_outliers(columns, kwargs.get('transform_method', 'log'))
    else:
        raise RemediationError(f"Invalid method '{method}'.")
    
    # Save metadata
    metadata_path = remediator.save_metadata(file_id)
    
    # Send feedback
    feedback = {
        "method": method,
        "columns": columns,
        "history": remediator.get_remediation_history()
    }
    remediator.send_feedback(file_id, method, feedback)
    
    return df_result, {
        "metadata_path": metadata_path,
        "history": remediator.get_remediation_history()
    }
