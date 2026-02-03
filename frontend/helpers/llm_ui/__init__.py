"""
LLM UI Module - Streamlit UI components for AI-powered explanations.

This module provides the UI layer (STEP 7) that connects the LLM pipeline
to the Streamlit interface.
"""

from .explanation_ui import show_explanation_tab

__all__ = ['show_explanation_tab']
