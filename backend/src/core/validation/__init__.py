"""
Validation Package - Transformation Impact Evaluation.

This package provides non-agentic validation of transformation execution.
It compares metrics before and after transformations without making decisions.

Exports
-------
ValidationFeedbackLoop : class
    Main validation class for comparing datasets
ValidationReport : dataclass
    Structured validation report
MetricComparison : dataclass
    Single metric comparison result
MetricStatus : Enum
    Status of metric computation
validate_transformation_impact : function
    Convenience function for validation
ValidationResultsStore : class
    Storage for validation results
get_validation_store : function
    Get the global validation store singleton
"""

from src.core.validation.validation_feedback_loop import (
    ValidationFeedbackLoop,
    ValidationReport,
    MetricComparison,
    MetricStatus,
    validate_transformation_impact
)
from src.core.validation.validation_storage import (
    ValidationResultsStore,
    get_validation_store
)

__all__ = [
    'ValidationFeedbackLoop',
    'ValidationReport',
    'MetricComparison',
    'MetricStatus',
    'validate_transformation_impact',
    'ValidationResultsStore',
    'get_validation_store'
]
