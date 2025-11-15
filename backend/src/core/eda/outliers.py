"""
Comprehensive outlier detection module for EDA.
Supports univariate (Z-score, IQR, MAD), multivariate (Isolation Forest, LOF),
time-series (STL residuals), and regression-based (Cook's Distance) methods.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime


class OutlierDetectionError(Exception):
    """Raised when outlier detection fails."""
    pass


@dataclass
class OutlierConfig:
    """Configuration for outlier detection."""
    z_thresh: float = 3.0
    iqr_multiplier: float = 1.5
    mad_threshold: float = 3.5
    contamination: float = 0.1  # For Isolation Forest / LOF
    n_neighbors: int = 20  # For LOF
    stl_threshold: float = 3.0  # For time-series residuals
    cooks_threshold: float = 4.0  # Cook's distance (4/n rule)


class OutlierDetector:
    """
    Comprehensive outlier detection using multiple methods.
    """
    def __init__(self, df: pd.DataFrame, config: OutlierConfig = None):
        if df.empty:
            raise OutlierDetectionError("DataFrame is empty.")
        self.df = df.copy()
        self.config = config or OutlierConfig()
        self.metadata_dir = Path("workspace/metadata")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    # ============= UNIVARIATE METHODS =============
    
    def detect_zscore(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers using Z-score method.
        Returns outlier indices, counts, and summary stats per column.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_indices = {}
        stats = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
                
            s = self.df[col].dropna()
            if len(s) < 3:
                continue
                
            mean = s.mean()
            std = s.std()
            
            if std == 0 or np.isnan(std):
                outlier_indices[col] = []
                stats.append({
                    "column": col,
                    "method": "Z-score",
                    "outlier_count": 0,
                    "outlier_percent": 0.0,
                    "threshold": self.config.z_thresh
                })
                continue
            
            z_scores = np.abs((s - mean) / std)
            outliers = z_scores > self.config.z_thresh
            outlier_idx = s[outliers].index.tolist()
            
            outlier_indices[col] = outlier_idx
            stats.append({
                "column": col,
                "method": "Z-score",
                "outlier_count": int(outliers.sum()),
                "outlier_percent": 100 * float(outliers.sum() / len(s)),
                "threshold": self.config.z_thresh,
                "mean": float(mean),
                "std": float(std)
            })
        
        return {"outlier_indices": outlier_indices, "stats": stats}

    def detect_iqr(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers using IQR (Interquartile Range) method.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_indices = {}
        stats = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
                
            s = self.df[col].dropna()
            if len(s) < 4:
                continue
            
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                outlier_indices[col] = []
                stats.append({
                    "column": col,
                    "method": "IQR",
                    "outlier_count": 0,
                    "outlier_percent": 0.0,
                    "Q1": float(Q1),
                    "Q3": float(Q3),
                    "IQR": 0.0
                })
                continue
            
            lower = Q1 - self.config.iqr_multiplier * IQR
            upper = Q3 + self.config.iqr_multiplier * IQR
            outliers = (s < lower) | (s > upper)
            outlier_idx = s[outliers].index.tolist()
            
            outlier_indices[col] = outlier_idx
            stats.append({
                "column": col,
                "method": "IQR",
                "outlier_count": int(outliers.sum()),
                "outlier_percent": 100 * float(outliers.sum() / len(s)),
                "Q1": float(Q1),
                "Q3": float(Q3),
                "IQR": float(IQR),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "multiplier": self.config.iqr_multiplier
            })
        
        return {"outlier_indices": outlier_indices, "stats": stats}

    def detect_mad(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers using MAD (Median Absolute Deviation) - more robust than Z-score.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_indices = {}
        stats = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
                
            s = self.df[col].dropna()
            if len(s) < 3:
                continue
            
            median = s.median()
            mad = np.median(np.abs(s - median))
            
            if mad == 0:
                outlier_indices[col] = []
                stats.append({
                    "column": col,
                    "method": "MAD",
                    "outlier_count": 0,
                    "outlier_percent": 0.0,
                    "median": float(median),
                    "mad": 0.0
                })
                continue
            
            # Modified Z-score using MAD: 0.6745 is the constant for normal distribution
            modified_z = 0.6745 * (s - median) / mad
            outliers = np.abs(modified_z) > self.config.mad_threshold
            outlier_idx = s[outliers].index.tolist()
            
            outlier_indices[col] = outlier_idx
            stats.append({
                "column": col,
                "method": "MAD",
                "outlier_count": int(outliers.sum()),
                "outlier_percent": 100 * float(outliers.sum() / len(s)),
                "median": float(median),
                "mad": float(mad),
                "threshold": self.config.mad_threshold
            })
        
        return {"outlier_indices": outlier_indices, "stats": stats}

    # ============= MULTIVARIATE METHODS =============
    
    def detect_isolation_forest(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect multivariate outliers using Isolation Forest.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter valid numeric columns
        valid_cols = [c for c in columns if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c])]
        
        if len(valid_cols) == 0:
            raise OutlierDetectionError("No valid numeric columns for Isolation Forest.")
        
        # Prepare data
        df_subset = self.df[valid_cols].dropna()
        if len(df_subset) < 10:
            raise OutlierDetectionError("Insufficient data for Isolation Forest (need at least 10 samples).")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_subset)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.config.contamination,
            random_state=42,
            n_estimators=100
        )
        predictions = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)
        
        # -1 indicates outlier
        outliers = predictions == -1
        outlier_idx = df_subset[outliers].index.tolist()
        
        result = {
            "outlier_indices": {"multivariate": outlier_idx},
            "stats": [{
                "method": "Isolation Forest",
                "columns": valid_cols,
                "outlier_count": int(outliers.sum()),
                "outlier_percent": 100 * float(outliers.sum() / len(df_subset)),
                "contamination": self.config.contamination,
                "mean_anomaly_score": float(np.mean(anomaly_scores)),
                "min_anomaly_score": float(np.min(anomaly_scores)),
                "max_anomaly_score": float(np.max(anomaly_scores))
            }]
        }
        
        return result

    def detect_lof(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect multivariate outliers using Local Outlier Factor.
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter valid numeric columns
        valid_cols = [c for c in columns if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c])]
        
        if len(valid_cols) == 0:
            raise OutlierDetectionError("No valid numeric columns for LOF.")
        
        # Prepare data
        df_subset = self.df[valid_cols].dropna()
        min_samples = max(self.config.n_neighbors + 1, 10)
        if len(df_subset) < min_samples:
            raise OutlierDetectionError(f"Insufficient data for LOF (need at least {min_samples} samples).")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_subset)
        
        # Fit LOF
        lof = LocalOutlierFactor(
            n_neighbors=self.config.n_neighbors,
            contamination=self.config.contamination
        )
        predictions = lof.fit_predict(X_scaled)
        lof_scores = lof.negative_outlier_factor_
        
        # -1 indicates outlier
        outliers = predictions == -1
        outlier_idx = df_subset[outliers].index.tolist()
        
        result = {
            "outlier_indices": {"multivariate": outlier_idx},
            "stats": [{
                "method": "Local Outlier Factor",
                "columns": valid_cols,
                "outlier_count": int(outliers.sum()),
                "outlier_percent": 100 * float(outliers.sum() / len(df_subset)),
                "contamination": self.config.contamination,
                "n_neighbors": self.config.n_neighbors,
                "mean_lof_score": float(np.mean(lof_scores)),
                "min_lof_score": float(np.min(lof_scores)),
                "max_lof_score": float(np.max(lof_scores))
            }]
        }
        
        return result

    # ============= TIME-SERIES METHODS =============
    
    def detect_timeseries_outliers(self, value_col: str, datetime_col: str, freq: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect outliers in time-series using STL decomposition residuals.
        """
        from statsmodels.tsa.seasonal import STL
        
        if value_col not in self.df.columns:
            raise OutlierDetectionError(f"Value column '{value_col}' not found.")
        if datetime_col not in self.df.columns:
            raise OutlierDetectionError(f"Datetime column '{datetime_col}' not found.")
        
        # Prepare time-series
        df_ts = self.df[[datetime_col, value_col]].copy()
        df_ts[datetime_col] = pd.to_datetime(df_ts[datetime_col])
        df_ts = df_ts.dropna().sort_values(datetime_col)
        df_ts.set_index(datetime_col, inplace=True)
        
        if len(df_ts) < 20:
            raise OutlierDetectionError("Need at least 20 observations for STL decomposition.")
        
        # Infer frequency if not provided
        if freq is None:
            freq = pd.infer_freq(df_ts.index)
            if freq is None:
                # Estimate frequency from median time difference
                time_diffs = df_ts.index.to_series().diff().dropna()
                if len(time_diffs) > 0:
                    median_diff = time_diffs.median()
                    # Convert Timedelta to frequency string
                    if median_diff <= pd.Timedelta(minutes=1):
                        freq = 'T'  # Minute
                    elif median_diff <= pd.Timedelta(hours=1):
                        freq = 'H'  # Hour
                    elif median_diff <= pd.Timedelta(days=1):
                        freq = 'D'  # Day
                    elif median_diff <= pd.Timedelta(weeks=1):
                        freq = 'W'  # Week
                    else:
                        freq = 'M'  # Month (approximate)
                    
                    # Resample to inferred frequency
                    df_ts = df_ts.resample(freq).mean().dropna()
                else:
                    raise OutlierDetectionError("Could not infer time-series frequency from data.")
        
        # Determine seasonal period based on frequency
        freq_to_period = {
            'T': 60, 'H': 24, 'D': 7, 'W': 52, 'M': 12,
            'min': 60, 'h': 24, 'D': 7, 'W': 52, 'MS': 12, 'M': 12
        }
        seasonal_period = freq_to_period.get(freq, 13)  # Default to 13 if unknown
        
        # Ensure we have enough data for the seasonal period
        if len(df_ts) < 2 * seasonal_period:
            # Fall back to simpler period or non-seasonal decomposition
            seasonal_period = max(7, len(df_ts) // 3)
        
        # STL decomposition
        try:
            stl = STL(df_ts[value_col], seasonal=seasonal_period, robust=True)
            result_stl = stl.fit()
            residuals = result_stl.resid
        except Exception as e:
            raise OutlierDetectionError(f"STL decomposition failed: {str(e)}")
        
        # Detect outliers from residuals using Z-score
        residual_mean = residuals.mean()
        residual_std = residuals.std()
        
        if residual_std == 0 or np.isnan(residual_std):
            outlier_idx = []
        else:
            z_residuals = np.abs((residuals - residual_mean) / residual_std)
            outliers = z_residuals > self.config.stl_threshold
            outlier_idx = residuals[outliers].index.tolist()
        
        result = {
            "outlier_indices": {value_col: [str(idx) for idx in outlier_idx]},
            "stats": [{
                "method": "STL Residuals",
                "column": value_col,
                "datetime_column": datetime_col,
                "outlier_count": len(outlier_idx),
                "outlier_percent": 100 * len(outlier_idx) / len(residuals),
                "threshold": self.config.stl_threshold,
                "residual_mean": float(residual_mean),
                "residual_std": float(residual_std),
                "frequency": str(freq)
            }]
        }
        
        return result

    # ============= REGRESSION-BASED METHODS =============
    
    def detect_cooks_distance(self, target_col: str, feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect influential outliers using Cook's Distance from linear regression.
        """
        from sklearn.linear_model import LinearRegression
        
        if target_col not in self.df.columns:
            raise OutlierDetectionError(f"Target column '{target_col}' not found.")
        
        if feature_cols is None:
            feature_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols:
                feature_cols.remove(target_col)
        
        if len(feature_cols) == 0:
            raise OutlierDetectionError("No feature columns available for regression.")
        
        # Prepare data
        df_clean = self.df[[target_col] + feature_cols].dropna()
        if len(df_clean) < len(feature_cols) + 5:
            raise OutlierDetectionError("Insufficient data for Cook's Distance calculation.")
        
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        n, p = X.shape
        
        # Fit regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Calculate Cook's Distance
        # D_i = (r_i^2 / p) * (h_ii / (1 - h_ii))
        # Where r_i is standardized residual and h_ii is leverage
        
        # Leverage (hat values)
        X_with_intercept = np.column_stack([np.ones(n), X])
        hat_matrix = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
        leverage = np.diag(hat_matrix)
        
        # Guard against leverage too close to 1 (numerical instability)
        leverage = np.clip(leverage, 0, 0.9999)
        
        # Standardized residuals
        mse = np.sum(residuals**2) / (n - p - 1)
        std_residuals = residuals / np.sqrt(np.maximum(mse * (1 - leverage), 1e-10))
        
        # Cook's Distance with safe division
        cooks_d = (std_residuals**2 / p) * (leverage / (1 - leverage))
        
        # Threshold: typically 4/n
        threshold = self.config.cooks_threshold / n
        outliers = cooks_d > threshold
        outlier_idx = df_clean[outliers].index.tolist()
        
        result = {
            "outlier_indices": {"regression_influential": outlier_idx},
            "stats": [{
                "method": "Cook's Distance",
                "target": target_col,
                "features": feature_cols,
                "outlier_count": int(outliers.sum()),
                "outlier_percent": 100 * float(outliers.sum() / n),
                "threshold": float(threshold),
                "max_cooks_d": float(np.max(cooks_d)),
                "mean_cooks_d": float(np.mean(cooks_d))
            }]
        }
        
        return result

    # ============= UNIFIED DETECTION =============
    
    def detect_all(self, columns: Optional[List[str]] = None, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run multiple outlier detection methods and combine results.
        methods: list of ['zscore', 'iqr', 'mad', 'isolation_forest', 'lof']
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'mad']
        
        all_results = {}
        all_stats = []
        
        for method in methods:
            try:
                if method == 'zscore':
                    result = self.detect_zscore(columns)
                elif method == 'iqr':
                    result = self.detect_iqr(columns)
                elif method == 'mad':
                    result = self.detect_mad(columns)
                elif method == 'isolation_forest':
                    result = self.detect_isolation_forest(columns)
                elif method == 'lof':
                    result = self.detect_lof(columns)
                else:
                    continue
                
                all_results[method] = result['outlier_indices']
                all_stats.extend(result['stats'])
            except Exception as e:
                all_stats.append({
                    "method": method,
                    "error": str(e)
                })
        
        return {"results": all_results, "stats": all_stats}

    # ============= METADATA PERSISTENCE =============
    
    def save_metadata(self, file_id: str, detection_results: Dict[str, Any]) -> str:
        """Save outlier detection metadata to JSON."""
        metadata = {
            "file_id": file_id,
            "timestamp": datetime.now().isoformat(),
            "detection_results": detection_results,
            "config": {
                "z_thresh": self.config.z_thresh,
                "iqr_multiplier": self.config.iqr_multiplier,
                "mad_threshold": self.config.mad_threshold,
                "contamination": self.config.contamination,
                "n_neighbors": self.config.n_neighbors
            }
        }
        
        metadata_path = self.metadata_dir / f"{file_id}_outliers.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(metadata_path)

    def load_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Load outlier detection metadata from JSON."""
        metadata_path = self.metadata_dir / f"{file_id}_outliers.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None


# ============= CONVENIENCE FUNCTIONS FOR API =============

def detect_outliers_comprehensive(
    df: pd.DataFrame,
    file_id: str,
    methods: List[str] = None,
    columns: List[str] = None,
    config: OutlierConfig = None
) -> Dict[str, Any]:
    """
    Comprehensive outlier detection with multiple methods.
    Returns combined results and saves metadata.
    """
    detector = OutlierDetector(df, config)
    results = detector.detect_all(columns=columns, methods=methods)
    
    # Save metadata
    metadata_path = detector.save_metadata(file_id, results)
    results["metadata_path"] = metadata_path
    
    return results


def get_outlier_summary(df: pd.DataFrame, methods: List[str] = None) -> pd.DataFrame:
    """Get a summary DataFrame of outlier detection results."""
    detector = OutlierDetector(df)
    results = detector.detect_all(methods=methods or ['zscore', 'iqr', 'mad'])
    
    # Convert stats to DataFrame
    if 'stats' in results and len(results['stats']) > 0:
        return pd.DataFrame(results['stats'])
    return pd.DataFrame()


class OutlierCleaner:
    """Legacy compatibility class."""
    @staticmethod
    def drop_outliers(df: pd.DataFrame, columns: list, z_thresh: float = 3.0) -> pd.DataFrame:
        """Drop outliers using Z-score method (legacy)."""
        detector = OutlierDetector(df, OutlierConfig(z_thresh=z_thresh))
        result = detector.detect_zscore(columns)
        
        # Collect all outlier indices
        all_outlier_idx = set()
        for idx_list in result['outlier_indices'].values():
            all_outlier_idx.update(idx_list)
        
        # Remove outliers
        mask = ~df.index.isin(all_outlier_idx)
        return df[mask].reset_index(drop=True)


# ============= STANDALONE METADATA LOADERS (for API use) =============

def load_outlier_metadata_file(file_id: str) -> Optional[Dict[str, Any]]:
    """
    Load outlier detection metadata from JSON without instantiating detector.
    Use this in API endpoints to avoid empty DataFrame errors.
    """
    metadata_dir = Path("workspace/metadata")
    metadata_path = metadata_dir / f"{file_id}_outliers.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return None
    return None

