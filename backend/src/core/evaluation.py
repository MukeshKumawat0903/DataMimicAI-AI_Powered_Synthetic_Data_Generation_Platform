# src/core/evaluation.py
from sdv.evaluation.single_table import (
    run_diagnostic, evaluate_quality
)

def generate_evaluation_report(real_data, synthetic_data, metadata):
    """Full implementation of your evaluation logic"""
    return {
        "diagnostic": run_diagnostic(real_data, synthetic_data, metadata),
        "quality_report": evaluate_quality(real_data, synthetic_data, metadata)
    }