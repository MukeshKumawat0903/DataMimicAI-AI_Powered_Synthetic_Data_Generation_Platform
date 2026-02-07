"""
Diagnostics Explainer - STEP 3 (Diagnostics Interpretation Layer).

This module transforms structured diagnostics from diagnostics_builder.py
into presentation-ready context for the Explain feature. It acts as a pure
presentation layer with zero interpretation logic.

STRICT RESPONSIBILITIES:
- Consume structured diagnostics output
- Format diagnostics for presentation
- Group by severity and type
- NO recommendations, NO fixes, NO prioritization

Author: DataMimicAI Team
Date: February 2026
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def format_diagnostics_for_explanation(
    diagnostics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format structured diagnostics into explanation-ready context.
    
    This function is a pure presentation formatter. It does NOT:
    - Add recommendations
    - Suggest fixes
    - Rank or prioritize issues
    - Add interpretations beyond severity
    
    Parameters
    ----------
    diagnostics : dict
        Output from build_diagnostics() with structure:
        {
            "diagnostics": [...],
            "summary": {...},
            "metadata": {...}
        }
    
    Returns
    -------
    dict
        Formatted context for prompt building:
        {
            "has_issues": bool,
            "error_message": str or None,
            "dataset_overview": {...},
            "detected_issues": {
                "high": [...],
                "medium": [...],
                "low": [...]
            },
            "severity_summary": {...},
            "metadata": {...}
        }
    
    Notes
    -----
    This function maintains strict boundaries:
    - Does NOT interpret what issues mean
    - Does NOT suggest actions
    - Only organizes and formats data
    """
    logger.debug("Formatting diagnostics for explanation")
    
    # Check for errors in diagnostics
    metadata = diagnostics.get("metadata", {})
    error_message = metadata.get("error")
    
    if error_message:
        logger.warning(f"Diagnostics contains error: {error_message}")
        return {
            "has_issues": False,
            "error_message": error_message,
            "dataset_overview": {},
            "detected_issues": {"high": [], "medium": [], "low": []},
            "severity_summary": {
                "total_issues": 0,
                "high_severity_count": 0,
                "medium_severity_count": 0,
                "low_severity_count": 0
            },
            "metadata": metadata
        }
    
    summary = diagnostics.get("summary", {})
    all_diagnostics = diagnostics.get("diagnostics", [])
    
    # Check if there are any issues
    has_issues = summary.get("total_issues", 0) > 0
    
    # Group issues by severity
    issues_by_severity = {
        "high": [],
        "medium": [],
        "low": []
    }
    
    for diagnostic in all_diagnostics:
        severity = diagnostic.get("severity")
        if severity in issues_by_severity:
            issues_by_severity[severity].append({
                "issue_type": diagnostic.get("issue_type"),
                "affected_columns": diagnostic.get("affected_columns", []),
                "metric_name": diagnostic.get("metric_name"),
                "metric_value": diagnostic.get("metric_value"),
                "severity": severity
            })
    
    # Build dataset overview from metadata
    dataset_overview = {
        "num_columns_analyzed": metadata.get("num_columns_analyzed", 0),
        "signals_version": metadata.get("signals_version", "unknown"),
        "generated_at": metadata.get("generated_at", "unknown")
    }
    
    # Build severity summary
    severity_summary = {
        "total_issues": summary.get("total_issues", 0),
        "high_severity_count": summary.get("high_severity_count", 0),
        "medium_severity_count": summary.get("medium_severity_count", 0),
        "low_severity_count": summary.get("low_severity_count", 0),
        "issue_types": summary.get("issue_types", {})
    }
    
    logger.debug(
        f"Formatted {severity_summary['total_issues']} issues: "
        f"{severity_summary['high_severity_count']} high, "
        f"{severity_summary['medium_severity_count']} medium, "
        f"{severity_summary['low_severity_count']} low"
    )
    
    return {
        "has_issues": has_issues,
        "error_message": None,
        "dataset_overview": dataset_overview,
        "detected_issues": issues_by_severity,
        "severity_summary": severity_summary,
        "metadata": metadata
    }


def build_diagnostics_context_for_prompt(
    diagnostics: Dict[str, Any],
    scope: str = "diagnostics_overview"
) -> Dict[str, Any]:
    """
    Build context dictionary for prompt building from diagnostics.
    
    This is the main entry point for converting diagnostics into
    a format suitable for LLM prompt generation.
    
    Parameters
    ----------
    diagnostics : dict
        Output from build_diagnostics()
    scope : str, optional
        Analysis scope (e.g., 'dataset_overview', 'column_analysis')
        Default is 'diagnostics_overview'
    
    Returns
    -------
    dict
        Context dictionary with structure:
        {
            "scope": str,
            "facts": {
                "has_issues": bool,
                "error_message": str or None,
                "dataset_overview": {...},
                "detected_issues": {...},
                "severity_summary": {...}
            },
            "metadata": {...}
        }
    
    Notes
    -----
    This function enforces the new architecture:
    - Diagnostics are the ONLY source of truth
    - No raw EDA metrics are passed to Explain
    - All interpretation logic stays in diagnostics_builder
    """
    formatted = format_diagnostics_for_explanation(diagnostics)
    
    return {
        "scope": scope,
        "facts": formatted,
        "metadata": {
            "source": "diagnostics_builder",
            "version": formatted["metadata"].get("signals_version", "1.0"),
            "generated_at": formatted["metadata"].get("generated_at"),
            "total_issues": formatted["severity_summary"]["total_issues"]
        }
    }


def get_issue_type_description(issue_type: str) -> str:
    """
    Get human-readable description for an issue type.
    
    This is ONLY for display purposes. Does NOT add interpretation.
    
    Parameters
    ----------
    issue_type : str
        Issue type from diagnostics
    
    Returns
    -------
    str
        Display-friendly description
    """
    descriptions = {
        "high_skew": "Skewed Distribution",
        "missing_values": "Missing Data",
        "outliers": "Outlier Presence",
        "drift": "Data Drift",
        "imbalance": "Class Imbalance",
        "high_cardinality": "High Cardinality",
        "high_correlation": "High Correlation"
    }
    return descriptions.get(issue_type, issue_type.replace("_", " ").title())


def get_severity_label(severity: str) -> str:
    """
    Get display label for severity level.
    
    Parameters
    ----------
    severity : str
        Severity level: 'low', 'medium', or 'high'
    
    Returns
    -------
    str
        Display label
    """
    labels = {
        "high": "⚠️ High Priority",
        "medium": "⚡ Medium Priority",
        "low": "ℹ️ Low Priority"
    }
    return labels.get(severity, severity.upper())


# Example usage (for testing only)
if __name__ == "__main__":
    # Sample diagnostics output
    sample_diagnostics = {
        "diagnostics": [
            {
                "issue_type": "high_skew",
                "affected_columns": ["income"],
                "metric_name": "skewness",
                "metric_value": 2.5,
                "severity": "high"
            },
            {
                "issue_type": "missing_values",
                "affected_columns": ["age"],
                "metric_name": "missing_pct",
                "metric_value": 15.0,
                "severity": "medium"
            }
        ],
        "summary": {
            "total_issues": 2,
            "high_severity_count": 1,
            "medium_severity_count": 1,
            "low_severity_count": 0,
            "issue_types": {
                "high_skew": 1,
                "missing_values": 1,
                "outliers": 0,
                "drift": 0,
                "imbalance": 0,
                "high_cardinality": 0,
                "high_correlation": 0
            }
        },
        "metadata": {
            "signals_version": "1.0",
            "generated_at": "2026-02-06T10:00:00Z",
            "num_columns_analyzed": 5
        }
    }
    
    # Test formatting
    formatted = format_diagnostics_for_explanation(sample_diagnostics)
    print("Formatted Context:")
    print(f"  Has Issues: {formatted['has_issues']}")
    print(f"  Total Issues: {formatted['severity_summary']['total_issues']}")
    print(f"  High Severity: {len(formatted['detected_issues']['high'])}")
    print(f"  Medium Severity: {len(formatted['detected_issues']['medium'])}")
    print(f"  Low Severity: {len(formatted['detected_issues']['low'])}")
    
    # Test context building
    context = build_diagnostics_context_for_prompt(sample_diagnostics)
    print(f"\nContext Scope: {context['scope']}")
    print(f"Context Source: {context['metadata']['source']}")
