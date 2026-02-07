"""
RAG Package - Retrieval-Augmented Generation for DataMimicAI.

This package provides a minimal, deterministic RAG foundation for augmenting
explanations with system knowledge.

Modules:
- knowledge_loader: Load and validate YAML rule files
- retriever: Simple rule-based retrieval by issue_type

Author: DataMimicAI Team
Date: February 2026
"""

from .knowledge_loader import KnowledgeLoader, get_default_loader, validate_rule
from .retriever import SimpleRetriever, augment_diagnostics_with_rules

__all__ = [
    "KnowledgeLoader",
    "get_default_loader",
    "validate_rule",
    "SimpleRetriever",
    "augment_diagnostics_with_rules"
]
