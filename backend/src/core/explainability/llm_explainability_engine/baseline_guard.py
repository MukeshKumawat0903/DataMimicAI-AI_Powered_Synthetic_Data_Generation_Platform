"""
Baseline Guard for LLM Explainability Engine.

This module provides lightweight snapshot and validation mechanisms to ensure
that the Explain feature continues to receive the same EDA inputs as before
any refactoring or changes.

DO NOT MODIFY:
- EDA computation logic
- Outlier detection
- Drift detection
- Synthetic data generation
- Validation metrics

This module only captures, logs, and validates the data flow.

Author: DataMimicAI Team
Date: February 2026
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime
from copy import deepcopy

logger = logging.getLogger(__name__)


class BaselineSnapshot:
    """Captures a lightweight snapshot of EDA statistics for baseline validation."""
    
    def __init__(self, signals: Dict[str, Any], label: str = ""):
        """
        Create a snapshot of explainable signals.
        
        Parameters
        ----------
        signals : dict
            The signals dictionary from build_explainable_signals()
        label : str, optional
            Human-readable label for this snapshot (e.g., "dataset_overview")
        """
        self.timestamp = datetime.utcnow().isoformat()
        self.label = label
        self.signals = deepcopy(signals)  # Deep copy to prevent mutations
        self.signature = self._compute_signature(signals)
        self.summary = self._build_summary(signals)
    
    def _compute_signature(self, signals: Dict[str, Any]) -> str:
        """
        Compute a stable hash signature of the signals structure.
        
        This allows quick comparison to detect if the signal structure
        or values have changed.
        """
        try:
            # Serialize to JSON with sorted keys for stable hashing
            json_str = json.dumps(signals, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Could not compute signature: {e}")
            return "N/A"
    
    def _build_summary(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a lightweight summary of the signals for quick inspection.
        """
        summary = {
            "top_level_keys": list(signals.keys()),
            "dataset_summary": signals.get("dataset_summary", {}),
            "num_columns_analyzed": len(signals.get("columns", {})),
            "num_correlations": len(signals.get("correlations", [])),
            "has_time_series": bool(signals.get("time_series", {}).get("num_datetime_columns", 0) > 0)
        }
        
        # Extract key metrics
        if "numeric_summary" in signals:
            summary["numeric_summary"] = {
                "num_numeric_columns": signals["numeric_summary"].get("num_numeric_columns", 0),
                "num_columns_with_outliers": signals["numeric_summary"].get("num_columns_with_outliers", 0)
            }
        
        if "categorical_summary" in signals:
            summary["categorical_summary"] = {
                "num_categorical_columns": signals["categorical_summary"].get("num_categorical_columns", 0),
                "num_imbalanced_columns": signals["categorical_summary"].get("num_imbalanced_columns", 0)
            }
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Export snapshot as dictionary."""
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "signature": self.signature,
            "summary": self.summary
        }
    
    def log_summary(self, level: str = "info"):
        """Log a summary of this snapshot."""
        log_func = getattr(logger, level, logger.info)
        log_func(
            f"[BASELINE] Snapshot '{self.label}' | "
            f"Signature: {self.signature} | "
            f"Columns: {self.summary.get('num_columns_analyzed', 0)} | "
            f"Correlations: {self.summary.get('num_correlations', 0)}"
        )


class BaselineGuard:
    """
    Validates that EDA signals remain consistent across pipeline steps.
    
    This guard does NOT modify any logic - it only validates data flow.
    """
    
    def __init__(self):
        """Initialize the baseline guard."""
        self.snapshots = []
        self.enabled = True  # Can be disabled in production
    
    def capture_signals_snapshot(
        self,
        signals: Dict[str, Any],
        label: str = "",
        log_level: str = "debug"
    ) -> BaselineSnapshot:
        """
        Capture a snapshot of explainable signals.
        
        Parameters
        ----------
        signals : dict
            Signals from build_explainable_signals()
        label : str
            Description of this capture point
        log_level : str
            Logging level ('debug', 'info', 'warning')
        
        Returns
        -------
        BaselineSnapshot
            The captured snapshot
        """
        if not self.enabled:
            return None
        
        snapshot = BaselineSnapshot(signals, label)
        self.snapshots.append(snapshot)
        snapshot.log_summary(log_level)
        
        return snapshot
    
    def assert_signals_structure(self, signals: Dict[str, Any]) -> None:
        """
        Assert that signals have the expected structure.
        
        This is a lightweight structural check to catch breaking changes early.
        
        Parameters
        ----------
        signals : dict
            Signals to validate
        
        Raises
        ------
        AssertionError
            If required structure is missing
        """
        if not self.enabled:
            return
        
        # Required top-level keys
        required_keys = ["dataset_summary", "columns", "numeric_summary", 
                        "categorical_summary", "correlations"]
        
        for key in required_keys:
            assert key in signals, f"Missing required key in signals: {key}"
        
        # dataset_summary checks
        ds = signals["dataset_summary"]
        assert "num_rows" in ds, "Missing 'num_rows' in dataset_summary"
        assert "num_columns" in ds, "Missing 'num_columns' in dataset_summary"
        assert isinstance(ds["num_rows"], (int, type(None))), "num_rows must be int or None"
        assert isinstance(ds["num_columns"], (int, type(None))), "num_columns must be int or None"
        
        # columns checks
        columns = signals["columns"]
        assert isinstance(columns, dict), "signals['columns'] must be a dict"
        
        logger.debug(f"[BASELINE] Structure assertion passed for {len(columns)} columns")
    
    def assert_context_matches_signals(
        self,
        context: Dict[str, Any],
        signals: Dict[str, Any]
    ) -> None:
        """
        Assert that context was properly derived from signals.
        
        This ensures STEP 2 (select_explainable_context) correctly uses
        signals from STEP 1 (build_explainable_signals).
        
        Parameters
        ----------
        context : dict
            Context from select_explainable_context()
        signals : dict
            Original signals from build_explainable_signals()
        
        Raises
        ------
        AssertionError
            If context doesn't match expected derivation from signals
        """
        if not self.enabled:
            return
        
        # Context must have these keys
        assert "scope" in context, "Context missing 'scope'"
        assert "facts" in context, "Context missing 'facts'"
        assert "metadata" in context, "Context missing 'metadata'"
        
        # If context has dataset info, it should match signals
        if "facts" in context and "basic_info" in context["facts"]:
            basic_info = context["facts"]["basic_info"]
            ds_summary = signals["dataset_summary"]
            
            if "num_rows" in basic_info and "num_rows" in ds_summary:
                assert basic_info["num_rows"] == ds_summary["num_rows"], \
                    f"Row count mismatch: context={basic_info['num_rows']} vs signals={ds_summary['num_rows']}"
            
            if "num_columns" in basic_info and "num_columns" in ds_summary:
                assert basic_info["num_columns"] == ds_summary["num_columns"], \
                    f"Column count mismatch: context={basic_info['num_columns']} vs signals={ds_summary['num_columns']}"
        
        logger.debug(f"[BASELINE] Context matches signals for scope='{context['scope']}'")
    
    def get_snapshot_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all captured snapshots.
        
        Returns
        -------
        dict
            Summary with count and details of snapshots
        """
        return {
            "num_snapshots": len(self.snapshots),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "enabled": self.enabled
        }
    
    def clear_snapshots(self):
        """Clear all captured snapshots."""
        self.snapshots.clear()
        logger.debug("[BASELINE] Snapshots cleared")


# Global instance for convenience
_global_guard = BaselineGuard()


def capture_signals_snapshot(
    signals: Dict[str, Any],
    label: str = "",
    log_level: str = "debug"
) -> BaselineSnapshot:
    """
    Convenience function to capture a signals snapshot.
    
    Parameters
    ----------
    signals : dict
        Signals from build_explainable_signals()
    label : str
        Description of this capture point
    log_level : str
        Logging level
    
    Returns
    -------
    BaselineSnapshot
        The captured snapshot
    """
    return _global_guard.capture_signals_snapshot(signals, label, log_level)


def assert_signals_structure(signals: Dict[str, Any]) -> None:
    """
    Convenience function to assert signals structure.
    
    Parameters
    ----------
    signals : dict
        Signals to validate
    """
    _global_guard.assert_signals_structure(signals)


def assert_context_matches_signals(
    context: Dict[str, Any],
    signals: Dict[str, Any]
) -> None:
    """
    Convenience function to assert context matches signals.
    
    Parameters
    ----------
    context : dict
        Context from select_explainable_context()
    signals : dict
        Original signals from build_explainable_signals()
    """
    _global_guard.assert_context_matches_signals(context, signals)


def get_snapshot_summary() -> Dict[str, Any]:
    """Get summary of all captured snapshots."""
    return _global_guard.get_snapshot_summary()


def enable_baseline_guard():
    """Enable the baseline guard."""
    _global_guard.enabled = True
    logger.info("[BASELINE] Baseline guard enabled")


def disable_baseline_guard():
    """Disable the baseline guard (for production)."""
    _global_guard.enabled = False
    logger.info("[BASELINE] Baseline guard disabled")


def is_baseline_guard_enabled() -> bool:
    """Check if baseline guard is enabled."""
    return _global_guard.enabled
