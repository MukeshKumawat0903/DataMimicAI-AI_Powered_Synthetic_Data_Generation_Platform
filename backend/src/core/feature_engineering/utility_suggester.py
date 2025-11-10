"""
Utility-Driven Feature Transformation Suggester

Analyzes statistical profiles to recommend transformations that improve:
- Distribution normality
- Variance stability
- Outlier handling
- Model performance
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class UtilitySuggestion:
    """Represents a utility-driven transformation suggestion."""
    column: str
    transformation: str
    reason: str
    confidence: float  # 0-1 scale
    params: Dict[str, Any]
    metrics: Dict[str, Any]  # Supporting statistics


class UtilitySuggester:
    """
    Generates utility-driven transformation suggestions based on statistical profiling.
    """
    
    def __init__(self, df: pd.DataFrame, profile_data: Optional[Dict] = None):
        """
        Initialize utility suggester.
        
        Args:
            df: Input DataFrame
            profile_data: Pre-computed profiling results (from profiling.py)
        """
        self.df = df
        self.profile_data = profile_data or {}
        self.suggestions = []
    
    def suggest_utility_transforms(self) -> List[Dict[str, Any]]:
        """
        Generate all utility-driven transformation suggestions.
        
        Returns:
            List of suggestion dictionaries
        """
        self.suggestions = []
        
        # Analyze each numeric column
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_suggestions = self._analyze_column(col)
            self.suggestions.extend(col_suggestions)
        
        return [self._suggestion_to_dict(s) for s in self.suggestions]
    
    def _analyze_column(self, column: str) -> List[UtilitySuggestion]:
        """
        Analyze a single column and generate suggestions.
        
        Args:
            column: Column name to analyze
            
        Returns:
            List of UtilitySuggestion objects
        """
        suggestions = []
        col_data = self.df[column].dropna()
        
        if len(col_data) == 0:
            return suggestions
        
        # Calculate statistics
        skewness = col_data.skew()
        kurtosis = col_data.kurtosis()
        std = col_data.std()
        mean = col_data.mean()
        cv = std / mean if mean != 0 else 0  # Coefficient of variation
        
        # Calculate outlier percentage (IQR method)
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        outlier_mask = (col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))
        outlier_pct = outlier_mask.sum() / len(col_data) * 100
        
        # Get variance compared to other numeric columns
        all_variances = self.df.select_dtypes(include=[np.number]).var()
        variance_rank = (all_variances > col_data.var()).sum() / len(all_variances)
        
        metrics = {
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "cv": float(cv),
            "outlier_pct": float(outlier_pct),
            "variance_rank": float(variance_rank)
        }
        
        # Optional: Test for log-normal distribution using distfit
        is_lognormal = self._test_lognormal_fit(col_data)
        if is_lognormal is not None:
            metrics["lognormal_fit"] = is_lognormal
        
        # Rule 1: Log/Power Transform for high skewness OR log-normal distribution
        if (abs(skewness) > 2.0 or is_lognormal) and col_data.min() >= 0:
            transform_type = "log" if skewness > 0 or is_lognormal else "power"
            
            # Higher confidence if both skewed AND log-normal
            confidence = min(0.9, abs(skewness) / 5.0)
            if is_lognormal:
                confidence = min(0.95, confidence + 0.15)
            
            reason = f"Highly skewed distribution (skewness: {skewness:.2f})"
            if is_lognormal:
                reason += " with confirmed log-normal fit"
            reason += f". {transform_type.capitalize()} transform reduces skewness and improves linearity."
            
            suggestions.append(UtilitySuggestion(
                column=column,
                transformation=f"{transform_type}_transform",
                reason=reason,
                confidence=confidence,
                params={"base": "e" if transform_type == "log" else 2},
                metrics=metrics
            ))
        
        # Rule 2: RobustScaler for high outlier percentage
        elif outlier_pct > 5.0:
            confidence = min(0.85, outlier_pct / 20.0)
            
            suggestions.append(UtilitySuggestion(
                column=column,
                transformation="robust_scaler",
                reason=f"High outlier percentage ({outlier_pct:.1f}%). RobustScaler uses median and IQR, making it resilient to outliers.",
                confidence=confidence,
                params={"quantile_range": (25.0, 75.0)},
                metrics=metrics
            ))
        
        # Rule 3: StandardScaler for high variance compared to others
        elif variance_rank < 0.3 and cv > 1.0:
            confidence = 0.75
            
            suggestions.append(UtilitySuggestion(
                column=column,
                transformation="standard_scaler",
                reason=f"High variance relative to other features (top {variance_rank*100:.0f}% by variance). StandardScaler normalizes for better model convergence.",
                confidence=confidence,
                params={"with_mean": True, "with_std": True},
                metrics=metrics
            ))
        
        # Rule 4: Square root transform for variance stabilization (moderate skew)
        elif 0.75 < abs(skewness) <= 2.0 and col_data.min() >= 0:
            confidence = 0.70
            
            suggestions.append(UtilitySuggestion(
                column=column,
                transformation="sqrt_transform",
                reason=f"Moderate skewness ({skewness:.2f}) and non-negative values. Square root transform stabilizes variance.",
                confidence=confidence,
                params={},
                metrics=metrics
            ))
        
        # Rule 5: MinMaxScaler for bounded features
        elif col_data.min() >= 0 and outlier_pct < 2.0:
            # Only suggest if not already normalized
            if not (col_data.min() == 0 and col_data.max() == 1):
                confidence = 0.65
                
                suggestions.append(UtilitySuggestion(
                    column=column,
                    transformation="minmax_scaler",
                    reason=f"Non-negative values with low outliers ({outlier_pct:.1f}%). MinMaxScaler bounds to [0,1] range.",
                    confidence=confidence,
                    params={"feature_range": (0, 1)},
                    metrics=metrics
                ))
        
        return suggestions
    
    def _test_lognormal_fit(self, col_data: pd.Series) -> Optional[bool]:
        """
        Test if data fits a log-normal distribution using distfit (optional).
        
        Args:
            col_data: Column data to test
            
        Returns:
            True if log-normal fit is good, False otherwise, None if distfit unavailable
        """
        try:
            # Try to import distfit (optional dependency)
            from distfit import distfit
            
            # Only test if data is positive
            if col_data.min() <= 0:
                return False
            
            # Fit distributions
            dfit = distfit(distr=['lognorm', 'norm', 'expon'], verbose=0)
            dfit.fit_transform(col_data.values)
            
            # Get best fit
            if hasattr(dfit, 'model') and 'name' in dfit.model:
                best_fit = dfit.model['name']
                
                # Check if log-normal is the best fit
                if best_fit == 'lognorm':
                    # Verify the fit quality using p-value
                    if 'score' in dfit.model and dfit.model.get('score', 0) > 0.05:
                        return True
            
            return False
            
        except ImportError:
            # distfit not installed, skip this check
            return None
        except Exception:
            # Any error during fitting, skip
            return None
    
    def _suggestion_to_dict(self, suggestion: UtilitySuggestion) -> Dict[str, Any]:
        """Convert UtilitySuggestion to dictionary."""
        return {
            "column": suggestion.column,
            "transformation": suggestion.transformation,
            "reason": suggestion.reason,
            "confidence": suggestion.confidence,
            "params": suggestion.params,
            "metrics": suggestion.metrics,
            "category": "utility"
        }
    
    def get_transform_code(self, suggestion: Dict[str, Any]) -> str:
        """
        Generate Python code snippet for applying a transformation.
        
        Args:
            suggestion: Suggestion dictionary
            
        Returns:
            Python code string
        """
        col = suggestion["column"]
        transform = suggestion["transformation"]
        
        code_templates = {
            "log_transform": f"df['{col}_log'] = np.log1p(df['{col}'])",
            "power_transform": f"from sklearn.preprocessing import PowerTransformer\ntransformer = PowerTransformer()\ndf['{col}_power'] = transformer.fit_transform(df[['{col}']])",
            "robust_scaler": f"from sklearn.preprocessing import RobustScaler\nscaler = RobustScaler()\ndf['{col}_robust'] = scaler.fit_transform(df[['{col}']])",
            "standard_scaler": f"from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ndf['{col}_scaled'] = scaler.fit_transform(df[['{col}']])",
            "sqrt_transform": f"df['{col}_sqrt'] = np.sqrt(df['{col}'])",
            "minmax_scaler": f"from sklearn.preprocessing import MinMaxScaler\nscaler = MinMaxScaler()\ndf['{col}_minmax'] = scaler.fit_transform(df[['{col}']])"
        }
        
        return code_templates.get(transform, f"# Transform: {transform}")
