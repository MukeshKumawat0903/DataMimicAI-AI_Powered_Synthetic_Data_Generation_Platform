# src/core/feedback_engine.py

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer

# Import transformation analysis modules
from src.core.eda.transform_metrics import generate_metric_deltas
from src.core.eda.transform_viz import plot_overlay_histograms


class TransformConfigManager:
    """
    Manages storage and retrieval of accepted transformation decisions.
    Stores decisions as JSON for audit, versioning, and reproducibility.
    """
    
    def __init__(self, config_dir: str = "workspace/transform_configs"):
        """
        Initialize config manager.
        
        Args:
            config_dir: Directory to store transformation configs
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_transforms = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "total_decisions": 0
        }
    
    def add_decision(
        self,
        column: str,
        transformation: str,
        category: str,
        params: Dict[str, Any],
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add an accepted transformation decision.
        
        Args:
            column: Column name
            transformation: Transformation type
            category: "utility" or "privacy"
            params: Transformation parameters
            reason: Reason for the transformation
            metadata: Additional metadata
            
        Returns:
            Decision record
        """
        decision = {
            "column": column,
            "transformation": transformation,
            "category": category,
            "params": params,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "decision_id": f"{column}_{transformation}_{datetime.now().timestamp()}",
            "metadata": metadata or {}
        }
        
        self.accepted_transforms.append(decision)
        self.metadata["total_decisions"] = len(self.accepted_transforms)
        
        return decision
    
    def remove_decision(self, column: str) -> bool:
        """
        Remove all decisions for a specific column.
        
        Args:
            column: Column name
            
        Returns:
            True if decisions were removed, False otherwise
        """
        initial_count = len(self.accepted_transforms)
        self.accepted_transforms = [
            d for d in self.accepted_transforms if d["column"] != column
        ]
        self.metadata["total_decisions"] = len(self.accepted_transforms)
        
        return len(self.accepted_transforms) < initial_count
    
    def get_decisions_by_column(self, column: str) -> List[Dict[str, Any]]:
        """Get all decisions for a specific column."""
        return [d for d in self.accepted_transforms if d["column"] == column]
    
    def get_decisions_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all decisions for a specific category (utility/privacy)."""
        return [d for d in self.accepted_transforms if d["category"] == category]
    
    def save_config(self, file_id: str, include_metadata: bool = True) -> str:
        """
        Save transformation config to JSON file.
        
        Args:
            file_id: Unique identifier for the dataset
            include_metadata: Whether to include metadata
            
        Returns:
            Path to saved config file
        """
        config = {
            "file_id": file_id,
            "transformations": self.accepted_transforms
        }
        
        if include_metadata:
            config["metadata"] = self.metadata
        
        filename = f"transform_config_{file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.config_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(filepath)
    
    def load_config(self, filepath: str) -> bool:
        """
        Load transformation config from JSON file.
        
        Args:
            filepath: Path to config file
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.accepted_transforms = config.get("transformations", [])
            self.metadata = config.get("metadata", {})
            
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def export_config(self) -> Dict[str, Any]:
        """
        Export current config as dictionary.
        
        Returns:
            Config dictionary
        """
        return {
            "transformations": self.accepted_transforms,
            "metadata": self.metadata
        }
    
    def clear_all(self):
        """Clear all accepted transformations."""
        self.accepted_transforms = []
        self.metadata["total_decisions"] = 0


class EDAFeedbackEngine:
    """
    Applies EDA/feature feedback to a DataFrame before synthetic generation.
    Enhanced to support utility and privacy transformations.
    """

    def __init__(self, df: pd.DataFrame, feedback: list):
        self.df = df.copy()
        self.original_df = df.copy()  # Store original for comparison
        self.feedback = feedback   # List of feedback dicts

    def apply_feedback(self, return_log=False):
        log = []
        for fb in self.feedback:
            action = fb.get("action")
            col = fb.get("column")
            params = fb.get("params", {})
            
            # Original actions
            if action == "impute":
                method = fb.get("method", "mean")
                val = None
                if method == "mean":
                    val = self.df[col].mean()
                elif method == "median":
                    val = self.df[col].median()
                elif method == "mode":
                    val = self.df[col].mode()[0]
                self.df[col].fillna(val, inplace=True)
                log.append(f"Imputed {col} with {method} ({val})")
            
            elif action == "drop":
                self.df.drop(columns=[col], inplace=True)
                log.append(f"Dropped column {col}")
            
            elif action == "encode":
                self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
                log.append(f"One-hot encoded {col}")
            
            elif action == "bin":
                bins = fb.get("bins", 4)
                self.df[f"{col}_bin"] = pd.cut(self.df[col], bins)
                log.append(f"Binned {col} into {bins} bins")
            
            # Utility transformations
            elif action == "log_transform":
                self.df[f"{col}_log"] = np.log1p(self.df[col])
                log.append(f"Applied log transform to {col}")
            
            elif action == "sqrt_transform":
                self.df[f"{col}_sqrt"] = np.sqrt(self.df[col])
                log.append(f"Applied square root transform to {col}")
            
            elif action == "power_transform":
                transformer = PowerTransformer()
                self.df[f"{col}_power"] = transformer.fit_transform(self.df[[col]])
                log.append(f"Applied power transform to {col}")
            
            elif action == "standard_scaler":
                scaler = StandardScaler()
                self.df[f"{col}_scaled"] = scaler.fit_transform(self.df[[col]])
                log.append(f"Applied standard scaling to {col}")
            
            elif action == "robust_scaler":
                scaler = RobustScaler()
                self.df[f"{col}_robust"] = scaler.fit_transform(self.df[[col]])
                log.append(f"Applied robust scaling to {col}")
            
            elif action == "minmax_scaler":
                scaler = MinMaxScaler()
                self.df[f"{col}_minmax"] = scaler.fit_transform(self.df[[col]])
                log.append(f"Applied min-max scaling to {col}")
            
            # Privacy transformations
            elif action == "redact":
                replacement = params.get("replacement", "[REDACTED]")
                self.df[col] = replacement
                log.append(f"Redacted {col}")
            
            elif action == "hash":
                self.df[f"{col}_hashed"] = self.df[col].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()
                )
                log.append(f"Hashed {col}")
            
            elif action == "mask":
                mask_pct = params.get("mask_percentage", 0.5)
                self.df[f"{col}_masked"] = self.df[col].apply(
                    lambda x: self._mask_value(x, mask_pct) if isinstance(x, str) else x
                )
                log.append(f"Masked {col}")
            
            elif action == "generalize":
                min_freq = params.get("min_frequency", 5)
                replacement = params.get("replacement", "Other")
                value_counts = self.df[col].value_counts()
                rare_values = value_counts[value_counts < min_freq].index
                self.df[f"{col}_generalized"] = self.df[col].apply(
                    lambda x: replacement if x in rare_values else x
                )
                log.append(f"Generalized {col}")
            
            elif action == "suppress":
                self.df.drop(columns=[col], inplace=True)
                log.append(f"Suppressed {col}")
        
        return (self.df, log) if return_log else self.df
    
    def get_transform_comparisons(self) -> List[Dict[str, Any]]:
        """
        Generate before/after comparisons for all transformed columns.
        
        Returns:
            List of comparison dictionaries with metrics and plots
        """
        comparisons = []
        
        for fb in self.feedback:
            action = fb.get("action")
            col = fb.get("column")
            
            # Determine the transformed column name
            transformed_col = None
            
            if action in ["log_transform"]:
                transformed_col = f"{col}_log"
            elif action == "sqrt_transform":
                transformed_col = f"{col}_sqrt"
            elif action == "power_transform":
                transformed_col = f"{col}_power"
            elif action == "standard_scaler":
                transformed_col = f"{col}_scaled"
            elif action == "robust_scaler":
                transformed_col = f"{col}_robust"
            elif action == "minmax_scaler":
                transformed_col = f"{col}_minmax"
            elif action == "hash":
                transformed_col = f"{col}_hashed"
            elif action == "mask":
                transformed_col = f"{col}_masked"
            elif action == "generalize":
                transformed_col = f"{col}_generalized"
            elif action in ["impute", "encode", "bin"]:
                # These modify the column in place or create different structures
                transformed_col = col
            
            # Only generate comparison if we have both original and transformed
            if transformed_col and col in self.original_df.columns and transformed_col in self.df.columns:
                original_data = self.original_df[col]
                transformed_data = self.df[transformed_col]
                
                # Generate metrics
                metrics = generate_metric_deltas(col, original_data, transformed_data)
                
                # Generate plots (only for numeric data)
                plot_data = None
                if metrics.get("is_numeric", False):
                    plot_data = plot_overlay_histograms(col, original_data, transformed_data)
                
                comparisons.append({
                    "column": col,
                    "action": action,
                    "transformed_column": transformed_col,
                    "metrics": metrics,
                    "plot": plot_data,
                    "params": fb.get("params", {})
                })
        
        return comparisons
    
    @staticmethod
    def _mask_value(value: str, mask_percentage: float) -> str:
        """Mask a portion of a string value."""
        if not isinstance(value, str) or len(value) == 0:
            return value
        
        mask_len = int(len(value) * mask_percentage)
        keep_len = len(value) - mask_len
        
        return value[:keep_len] + '*' * mask_len
