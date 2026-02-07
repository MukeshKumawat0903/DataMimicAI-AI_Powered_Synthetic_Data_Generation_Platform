"""
Execution Module - Deterministic Execution Layer

This module provides strict, non-agentic execution of approved
transformation plans with zero reasoning or intelligence.
"""

from .deterministic_execution_engine import (
    DeterministicExecutionEngine,
    ExecutionResult,
    ExecutionStatus,
    TransformationVocabulary,
    execute_approved_plan,
)

__all__ = [
    "DeterministicExecutionEngine",
    "ExecutionResult",
    "ExecutionStatus",
    "TransformationVocabulary",
    "execute_approved_plan",
]
