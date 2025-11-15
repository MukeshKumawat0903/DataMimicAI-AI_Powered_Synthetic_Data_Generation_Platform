"""
Time-Series Analysis Module for DataMimicAI

Provides detection, diagnostics, and visualization utilities for temporal datasets:
- Time-series detection and frequency inference
- ACF/PACF analysis for AR/MA order suggestions
- STL seasonal decomposition
- Generator configuration hints

Author: DataMimicAI Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class TimeSeriesError(Exception):
    """Custom exception for time-series analysis errors"""
    pass


class TimeSeriesDetector:
    """
    Detects time-series characteristics in uploaded datasets.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize detector with DataFrame.
        
        Args:
            df: pandas DataFrame to analyze
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise TimeSeriesError("Input must be a pandas DataFrame")
        self.df = df
    
    def detect_timeseries(self) -> Dict[str, Any]:
        """
        Detect if dataset is time-series and identify temporal characteristics.
        
        Returns:
            Dictionary with detection results:
            {
                'is_timeseries': bool,
                'datetime_columns': List[str],
                'primary_datetime_column': str or None,
                'frequency': str or None,
                'is_regular': bool,
                'start_date': str,
                'end_date': str,
                'periods': int,
                'potential_targets': List[str],
                'detection_confidence': str  # 'high', 'medium', 'low'
            }
        """
        result = {
            'is_timeseries': False,
            'datetime_columns': [],
            'primary_datetime_column': None,
            'frequency': None,
            'is_regular': False,
            'start_date': None,
            'end_date': None,
            'periods': 0,
            'potential_targets': [],
            'detection_confidence': 'low'
        }
        
        # Step 1: Find datetime columns with strict validation
        datetime_cols = []
        for col in self.df.columns:
            # Check if already datetime type
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                datetime_cols.append(col)
            # Skip numeric columns - they should not be treated as datetime
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            else:
                # Try to parse string-like columns as datetime
                try:
                    # Sample first few values to check if they look like dates
                    sample = self.df[col].dropna().head(10)
                    if len(sample) == 0:
                        continue
                    
                    # Check if values are string-like
                    if not all(isinstance(v, (str, pd.Timestamp)) for v in sample):
                        continue
                    
                    # Try parsing
                    parsed = pd.to_datetime(self.df[col], errors='coerce')
                    valid_ratio = parsed.notna().sum() / len(self.df)
                    
                    # Require high success rate and check if parsed values look reasonable
                    if valid_ratio > 0.8:
                        # Additional validation: check if dates are in reasonable range
                        valid_dates = parsed.dropna()
                        if len(valid_dates) > 0:
                            min_date = valid_dates.min()
                            max_date = valid_dates.max()
                            # Dates should be between 1900 and 2100
                            if pd.Timestamp('1900-01-01') <= min_date and max_date <= pd.Timestamp('2100-12-31'):
                                datetime_cols.append(col)
                except Exception:
                    # Non-parsable column
                    pass
        
        result['datetime_columns'] = datetime_cols
        
        if not datetime_cols:
            return result
        
        # Step 2: Select primary datetime column
        # Priority: 'date', 'datetime', 'timestamp', first datetime column
        primary_col = None
        for preferred in ['date', 'datetime', 'timestamp', 'time']:
            for col in datetime_cols:
                if preferred in col.lower():
                    primary_col = col
                    break
            if primary_col:
                break
        
        if not primary_col:
            primary_col = datetime_cols[0]
        
        result['primary_datetime_column'] = primary_col
        
        # Step 3: Analyze temporal characteristics
        try:
            # Ensure datetime type
            if not pd.api.types.is_datetime64_any_dtype(self.df[primary_col]):
                dt_series = pd.to_datetime(self.df[primary_col], errors='coerce')
            else:
                dt_series = self.df[primary_col]
            
            # Remove NaT values
            dt_series = dt_series.dropna()
            
            if len(dt_series) < 2:
                return result
            
            # Sort to get proper start/end
            dt_sorted = dt_series.sort_values()
            
            result['start_date'] = dt_sorted.iloc[0].isoformat()
            result['end_date'] = dt_sorted.iloc[-1].isoformat()
            result['periods'] = len(dt_sorted)
            
            # Step 4: Infer frequency
            try:
                # infer_freq expects a DatetimeIndex; ensure we pass a DatetimeIndex
                freq = pd.infer_freq(pd.DatetimeIndex(dt_sorted))
                if freq:
                    result['frequency'] = freq
                    result['is_regular'] = True
                    result['detection_confidence'] = 'high'
                else:
                    # Try to detect common patterns manually
                    diffs = dt_sorted.diff().dropna()
                    mode_diff = diffs.mode()
                    if len(mode_diff) > 0:
                        common_diff = mode_diff.iloc[0]
                        # Check if most intervals match the mode
                        matching = (diffs == common_diff).sum()
                        if matching / len(diffs) > 0.8:
                            result['is_regular'] = True
                            result['frequency'] = self._describe_timedelta(common_diff)
                            result['detection_confidence'] = 'medium'
            except Exception as e:
                logger.warning(f"Frequency inference failed: {e}")
            
            # Step 5: Identify potential target columns (numeric)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude index-like columns
            potential_targets = [col for col in numeric_cols 
                               if col.lower() not in ['id', 'index', 'row']]
            result['potential_targets'] = potential_targets
            
            # Mark as time-series if we have datetime + numeric targets
            if result['datetime_columns'] and result['potential_targets']:
                result['is_timeseries'] = True
                if result['detection_confidence'] == 'low':
                    result['detection_confidence'] = 'medium'
            
        except Exception as e:
            logger.error(f"Time-series detection error: {e}")
            raise TimeSeriesError(f"Detection failed: {str(e)}")
        
        return result
    
    def _describe_timedelta(self, td: pd.Timedelta) -> str:
        """Convert timedelta to human-readable frequency string."""
        days = td.days
        seconds = td.seconds
        
        if days == 1 and seconds == 0:
            return 'D'  # Daily
        elif days == 7:
            return 'W'  # Weekly
        elif days >= 28 and days <= 31:
            return 'M'  # Monthly
        elif days >= 365:
            return 'Y'  # Yearly
        elif seconds == 3600:
            return 'H'  # Hourly
        elif seconds == 60:
            return 'T'  # Minutely
        else:
            return f'{days}D {seconds}S'  # Custom


class TimeSeriesAnalyzer:
    """
    Performs ACF/PACF analysis and seasonal decomposition for time-series data.
    """
    
    def __init__(self, df: pd.DataFrame, datetime_col: str):
        """
        Initialize analyzer.
        
        Args:
            df: pandas DataFrame
            datetime_col: Name of datetime column
        """
        self.df = df.copy()
        self.datetime_col = datetime_col
        self.original_columns = list(df.columns)  # Store original columns
        
        # Ensure datetime index
        if datetime_col in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[datetime_col]):
                self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])
            
            # Set index but keep the column
            if self.df.index.name != datetime_col:
                self.df = self.df.set_index(datetime_col, drop=False)
                self.df = self.df.sort_index()
        else:
            # datetime_col might already be the index
            if not pd.api.types.is_datetime64_any_dtype(self.df.index):
                self.df.index = pd.to_datetime(self.df.index)
            self.df = self.df.sort_index()
    
    def compute_acf_pacf(self, target_col: str, lags: int = 40) -> Dict[str, Any]:
        """
        Compute Autocorrelation and Partial Autocorrelation.
        
        Args:
            target_col: Column to analyze
            lags: Number of lags to compute
        
        Returns:
            Dictionary with ACF/PACF values and confidence intervals
        """
        try:
            from statsmodels.tsa.stattools import acf, pacf
        except ImportError:
            raise TimeSeriesError(
                "statsmodels library required for ACF/PACF. "
                "Install with: pip install statsmodels"
            )
        
        if target_col not in self.df.columns:
            available_cols = list(self.df.columns)
            raise TimeSeriesError(
                f"Column '{target_col}' not found. "
                f"Available columns: {available_cols}"
            )
        
        series = self.df[target_col].dropna()
        n_obs = len(series)

        # Validate minimum observations
        if n_obs < 10:
            raise TimeSeriesError(
                f"Series too short for reliable ACF/PACF analysis. "
                f"Need at least 10 observations, got {n_obs}. "
                f"Consider collecting more data or using simpler analysis methods."
            )
        
        if n_obs < 20:
            logger.warning(f"Short series ({n_obs} obs) - ACF/PACF results may be unreliable")

        # cap lags to a sensible value relative to available observations
        lags = min(lags, max(1, n_obs - 1))
        
        try:
            # Compute ACF with confidence intervals
            acf_values, acf_confint = acf(
                series,
                nlags=lags,
                alpha=0.05,
                fft=True
            )
            
            # Compute PACF with confidence intervals
            pacf_values, pacf_confint = pacf(
                series,
                nlags=lags,
                alpha=0.05,
                method='ywm'
            )
            
            # Extract absolute confidence bounds (not offsets)
            acf_lower = acf_confint[:, 0]
            acf_upper = acf_confint[:, 1]
            
            pacf_lower = pacf_confint[:, 0]
            pacf_upper = pacf_confint[:, 1]
            
            # Suggest AR/MA orders based on cutoffs (use actual n_obs for thresholds)
            suggestions = self._suggest_arma_orders(acf_values, pacf_values, n_obs)
            
            return {
                'acf': {
                    'values': acf_values.tolist(),
                    'lower': acf_lower.tolist(),
                    'upper': acf_upper.tolist(),
                    'lags': list(range(len(acf_values)))
                },
                'pacf': {
                    'values': pacf_values.tolist(),
                    'lower': pacf_lower.tolist(),
                    'upper': pacf_upper.tolist(),
                    'lags': list(range(len(pacf_values)))
                },
                'suggestions': suggestions,
                'n_obs': n_obs
            }
            
        except Exception as e:
            logger.error(f"ACF/PACF computation error: {e}")
            raise TimeSeriesError(f"ACF/PACF failed: {str(e)}")
    
    def _suggest_arma_orders(self, acf_vals: np.ndarray, pacf_vals: np.ndarray, n_obs: int) -> Dict[str, Any]:
        """Suggest AR/MA orders based on ACF/PACF patterns."""
        suggestions = {
            'ar_order': None,
            'ma_order': None,
            'model_type': 'Unknown',
            'explanation': ''
        }
        # Find cutoff points using actual number of observations for confidence threshold
        try:
            conf_threshold = 1.96 / np.sqrt(max(1, n_obs))
        except Exception:
            conf_threshold = 1.96 / np.sqrt(max(1, len(acf_vals)))

        # PACF cutoff suggests AR order: find last lag where PACF is significant
        pacf_significant = np.abs(pacf_vals[1:]) > conf_threshold
        if pacf_significant.any():
            sig_idx = np.where(pacf_significant)[0]
            # last significant lag index -> add 1 to convert from zero-based (lag1->index0)
            suggestions['ar_order'] = int(sig_idx[-1] + 1)

        # ACF cutoff suggests MA order: find last lag where ACF is significant
        acf_significant = np.abs(acf_vals[1:]) > conf_threshold
        if acf_significant.any():
            sig_idx = np.where(acf_significant)[0]
            suggestions['ma_order'] = int(sig_idx[-1] + 1)
        
        # Determine model type
        if suggestions['ar_order'] and not suggestions['ma_order']:
            suggestions['model_type'] = f"AR({suggestions['ar_order']})"
            suggestions['explanation'] = f"PACF cuts off at lag {suggestions['ar_order']}, ACF tails off → AR model suggested"
        elif suggestions['ma_order'] and not suggestions['ar_order']:
            suggestions['model_type'] = f"MA({suggestions['ma_order']})"
            suggestions['explanation'] = f"ACF cuts off at lag {suggestions['ma_order']}, PACF tails off → MA model suggested"
        elif suggestions['ar_order'] and suggestions['ma_order']:
            suggestions['model_type'] = f"ARMA({suggestions['ar_order']},{suggestions['ma_order']})"
            suggestions['explanation'] = "Both ACF and PACF tail off → ARMA model suggested"
        else:
            suggestions['model_type'] = "White Noise or Complex Pattern"
            suggestions['explanation'] = "No clear cutoff detected; consider advanced diagnostics"
        
        return suggestions
    
    def decompose_series(self, target_col: str, model: str = 'additive', 
                        period: Optional[int] = None,
                        resample_freq: Optional[str] = None,
                        aggregation: str = 'mean') -> Dict[str, Any]:
        """
        Perform STL (Seasonal-Trend decomposition using Loess) or classical decomposition.
        
        Args:
            target_col: Column to decompose
            model: 'additive' or 'multiplicative'
            period: Seasonal period (auto-detected if None)
            resample_freq: Frequency to resample to (e.g., 'D', 'H') for irregular series
            aggregation: Aggregation method for resampling ('mean', 'sum', 'median')
        
        Returns:
            Dictionary with trend, seasonal, residual components
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            raise TimeSeriesError(
                "statsmodels required. Install with: pip install statsmodels"
            )
        
        if target_col not in self.df.columns:
            available_cols = list(self.df.columns)
            raise TimeSeriesError(
                f"Column '{target_col}' not found. "
                f"Available columns: {available_cols}"
            )
        
        series = self.df[target_col].dropna()
        
        # Check if series is irregular and handle resampling
        is_irregular = False
        try:
            freq = pd.infer_freq(series.index)
            if freq is None:
                is_irregular = True
                logger.warning(f"Irregular time-series detected for '{target_col}'")
        except Exception:
            is_irregular = True
        
        # Resample if requested or if irregular
        if resample_freq or is_irregular:
            if not resample_freq:
                # Auto-select reasonable frequency based on data range
                time_range = series.index[-1] - series.index[0]
                if time_range.days > 365:
                    resample_freq = 'D'  # Daily for yearly+ data
                elif time_range.days > 7:
                    resample_freq = 'H'  # Hourly for weekly+ data
                else:
                    resample_freq = 'T'  # Minutely for short periods
                logger.info(f"Auto-selected resample frequency: {resample_freq}")
            
            # Apply resampling
            agg_methods = {
                'mean': lambda x: x.mean(),
                'sum': lambda x: x.sum(),
                'median': lambda x: x.median(),
                'first': lambda x: x.first(),
                'last': lambda x: x.last()
            }
            agg_func = agg_methods.get(aggregation, lambda x: x.mean())
            series = series.resample(resample_freq).apply(agg_func).dropna()
            logger.info(f"Resampled series from {len(self.df)} to {len(series)} points")
        
        # Validate minimum length for decomposition
        if len(series) < 2 * (period or 2):
            raise TimeSeriesError(
                f"Series too short for decomposition. Need at least {2 * (period or 2)} points, got {len(series)}"
            )
        
        # Auto-detect period if not provided
        if period is None:
            freq = pd.infer_freq(series.index)
            if freq:
                period = self._freq_to_period(freq)
            else:
                period = min(len(series) // 2, 12)  # Default to 12 or half series
        
        # Ensure period is valid
        if period >= len(series) // 2:
            period = max(2, len(series) // 3)
            logger.warning(f"Adjusted period to {period} based on series length")
        
        try:
            result = seasonal_decompose(
                series,
                model=model,
                period=period,
                extrapolate_trend='freq'
            )
            
            return {
                'observed': {
                    'values': series.values.tolist(),
                    'index': series.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                },
                'trend': {
                    'values': result.trend.tolist(),
                    'index': result.trend.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                },
                'seasonal': {
                    'values': result.seasonal.tolist(),
                    'index': result.seasonal.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                },
                'residual': {
                    'values': result.resid.tolist(),
                    'index': result.resid.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                },
                'model': model,
                'period': period,
                'strength_of_trend': self._compute_strength(series, result.trend, result.resid),
                'strength_of_seasonality': self._compute_strength(series, result.seasonal, result.resid)
            }
            
        except Exception as e:
            logger.error(f"Decomposition error: {e}")
            raise TimeSeriesError(f"Decomposition failed: {str(e)}")
    
    def _freq_to_period(self, freq: str) -> int:
        """Convert pandas frequency string to period for decomposition."""
        # Normalize frequency string (handle complex cases like 'MS', '15T', etc.)
        try:
            # Extract base frequency (first letter or pattern)
            if not freq:
                return 12
            
            # Handle offset aliases
            freq_upper = freq.upper()
            
            # Map common patterns
            freq_map = {
                'D': 7,      # Daily → weekly seasonality
                'B': 5,      # Business day → weekly
                'W': 52,     # Weekly → yearly seasonality
                'M': 12,     # Monthly → yearly seasonality
                'MS': 12,    # Month start
                'ME': 12,    # Month end  
                'Q': 4,      # Quarterly → yearly seasonality
                'QS': 4,     # Quarter start
                'Y': 1,      # Yearly
                'A': 1,      # Annual (alias for Y)
                'H': 24,     # Hourly → daily seasonality
                'T': 60,     # Minutely → hourly seasonality
                'MIN': 60,   # Minute (alias)
                'S': 60,     # Secondly → minutely seasonality
                'L': 1000,   # Millisecond
                'U': 1000,   # Microsecond
            }
            
            # Try direct match first
            if freq_upper in freq_map:
                return freq_map[freq_upper]
            
            # Try first character
            first_char = freq_upper[0] if freq_upper else 'M'
            return freq_map.get(first_char, 12)
            
        except Exception as e:
            logger.warning(f"Could not parse frequency '{freq}': {e}. Using default period=12")
            return 12
    
    def _compute_strength(self, original: pd.Series, component: pd.Series, 
                         residual: pd.Series) -> float:
        """Compute strength of trend or seasonality (0-1 scale)."""
        try:
            var_resid = np.var(residual.dropna())
            var_detrended = np.var((original - component).dropna())
            if var_detrended == 0:
                return 0.0
            strength = max(0, 1 - (var_resid / var_detrended))
            return float(strength)
        except:
            return 0.0
    
    def save_results(self, file_id: str, analysis_type: str, results: Dict) -> str:
        """
        Save time-series analysis results to metadata directory.
        
        Args:
            file_id: File identifier
            analysis_type: 'detection', 'acf_pacf', 'decomposition'
            results: Analysis results dictionary
        
        Returns:
            Path to saved metadata file
        """
        # Resolve repository root (go up from `backend/src/core/eda`) and use top-level `workspace/metadata`
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
        workspace_dir = os.path.join(repo_root, 'workspace', 'metadata')
        os.makedirs(workspace_dir, exist_ok=True)
        
        filename = f"timeseries_{analysis_type}_{file_id}.json"
        filepath = os.path.join(workspace_dir, filename)
        
        metadata = {
            "file_id": file_id,
            "analysis_type": analysis_type,
            "timestamp": datetime.utcnow().isoformat(),
            "results": results
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved {analysis_type} results to {filepath}")
        return filepath


# Helper functions for API endpoints

def detect_timeseries(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to detect time-series characteristics.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Detection results dictionary
    """
    detector = TimeSeriesDetector(df)
    return detector.detect_timeseries()


def compute_acf_pacf(df: pd.DataFrame, datetime_col: str, target_col: str, 
                     lags: int = 40) -> Dict[str, Any]:
    """
    Convenience function for ACF/PACF computation.
    
    Args:
        df: Input DataFrame
        datetime_col: Name of datetime column
        target_col: Target column to analyze
        lags: Number of lags
    
    Returns:
        ACF/PACF results dictionary
    """
    analyzer = TimeSeriesAnalyzer(df, datetime_col)
    return analyzer.compute_acf_pacf(target_col, lags)


def decompose_timeseries(df: pd.DataFrame, datetime_col: str, target_col: str,
                        model: str = 'additive', period: Optional[int] = None,
                        resample_freq: Optional[str] = None,
                        aggregation: str = 'mean') -> Dict[str, Any]:
    """
    Convenience function for seasonal decomposition.
    
    Args:
        df: Input DataFrame
        datetime_col: Name of datetime column
        target_col: Target column to decompose
        model: 'additive' or 'multiplicative'
        period: Seasonal period
        resample_freq: Frequency to resample irregular series (e.g., 'D', 'H')
        aggregation: Aggregation method for resampling
    
    Returns:
        Decomposition results dictionary
    """
    analyzer = TimeSeriesAnalyzer(df, datetime_col)
    return analyzer.decompose_series(target_col, model, period, resample_freq, aggregation)
