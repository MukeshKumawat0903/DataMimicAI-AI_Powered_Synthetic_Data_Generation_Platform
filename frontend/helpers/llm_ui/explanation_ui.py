"""
Explanation UI - STEP 7: UI Integration for LLM Pipeline.

This module provides the Streamlit UI for AI-powered data explanations.
It orchestrates the full LLM pipeline (STEP 1-6) and displays results.

Author: DataMimicAI Team
Date: February 2026
"""

import streamlit as st
import sys
from pathlib import Path


def show_explanation_tab():
    """
    Display the AI Explanation tab UI and handle pipeline execution (STEP 7).
    
    This is a PRESENTATION LAYER that:
    - Provides UI for triggering explanation generation
    - Orchestrates STEP 1-6 pipeline
    - Displays validated explanations
    - Manages caching and state
    
    All intelligence happens in STEPS 1-6. This function only presents results.
    """
    # STEP 7: UI Integration - Connect full LLM pipeline to Streamlit
    st.markdown("### 🔍 AI-Powered Data Explanation")
    st.markdown("""
    Get intelligent, natural language explanations of your dataset's key characteristics, 
    patterns, and statistical properties. The AI analyzes your data and provides clear 
    insights grounded in computed facts.
    """)
    
    # Check if data is available
    if not hasattr(st.session_state, 'df') or st.session_state.df is None or st.session_state.df.empty:
        st.warning("⚠️ Please upload or load a dataset first to generate explanations.")
        return
    
    # Initialize session state for caching explanations
    if 'llm_explanation' not in st.session_state:
        st.session_state.llm_explanation = None
    if 'llm_explanation_file_id' not in st.session_state:
        st.session_state.llm_explanation_file_id = None
    
    # Display existing explanation if available for current dataset
    current_file_id = st.session_state.get('file_id')
    
    # Check if cached explanation is valid for current dataset
    has_cached_explanation = (
        st.session_state.llm_explanation is not None and 
        current_file_id is not None and
        st.session_state.llm_explanation_file_id == current_file_id
    )
    
    if has_cached_explanation:
        fallback_msg = "The analysis highlights notable patterns in the data, but the explanation could not be confidently validated."
        is_fallback = st.session_state.llm_explanation.strip() == fallback_msg
        
        if is_fallback:
            st.warning("⚠️ Explanation generated, but validation checks failed.")
            st.markdown("---")
            
            # Show the raw LLM output with warning if available
            if hasattr(st.session_state, 'raw_llm_explanation') and st.session_state.raw_llm_explanation:
                with st.expander("🔍 View Generated Explanation (Unvalidated)", expanded=True):
                    st.markdown(st.session_state.raw_llm_explanation)
                    st.caption("⚠️ This output failed validation checks. Some statistics may not match your data.")
            
            # Show validation details if available
            if hasattr(st.session_state, 'validation_report') and st.session_state.validation_report:
                with st.expander("🔬 Validation Details"):
                    st.json(st.session_state.validation_report)
                    st.caption("💡 The LLM may have included numbers not present in your dataset.")
            
            st.markdown("---")
        else:
            st.success("✅ Explanation generated successfully!")
            st.markdown("---")
            st.markdown(st.session_state.llm_explanation)
            st.markdown("---")
        
        st.caption("💡 Click 'Generate New Explanation' to refresh with updated data.")
    
    # Button to trigger explanation generation
    button_label = "Generate New Explanation" if has_cached_explanation else "Generate Explanation"
    
    if st.button(button_label, type="primary", use_container_width=True):
        # Run the full LLM pipeline (STEP 1 → 6)
        _run_llm_pipeline(current_file_id)
    
    # Additional info
    _show_pipeline_info()


def _run_llm_pipeline(current_file_id: str):
    """
    Execute the complete LLM pipeline and display results.
    
    Pipeline Steps:
    1. STEP 1: Extract explainable signals
    2. STEP 2: Select scoped context
    3. STEP 4: Build safe prompt
    4. STEP 5: Run LLaMA inference
    5. STEP 6: Validate output
    
    Parameters
    ----------
    current_file_id : str
        Current dataset identifier for caching
    """
    with st.spinner("🤖 Analyzing your data... This may take 10-30 seconds."):
        try:
            # Add backend to Python path for imports
            backend_src_path = Path(__file__).parent.parent.parent.parent / "backend" / "src"
            backend_src_path = backend_src_path.resolve()
            
            if backend_src_path not in sys.path:
                sys.path.insert(0, str(backend_src_path))
            
            # Import LLM pipeline components
            from core.LLM import (
                build_explainable_signals,
                select_explainable_context,
                build_explanation_prompt,
                run_llama_explanation,
                validate_llm_output,
                get_validation_report
            )
            
            # STEP 1: Extract explainable signals from data
            # This computes all deterministic statistics
            signals = build_explainable_signals(st.session_state.df)
            
            # STEP 2: Scope and filter signals for focused context
            # Using dataset_overview scope for general explanation
            context = select_explainable_context(
                signals, 
                scope="dataset_overview"
            )
            
            # STEP 4: Build safe prompt with constraints
            # Ensures LLM cannot invent statistics
            prompt = build_explanation_prompt(
                context, 
                tone="clear"
            )
            
            # STEP 5: Run LLaMA inference via Groq
            # This is the ONLY step that calls external API
            # Use max_tokens=1500 to ensure all sections are generated
            raw_explanation = run_llama_explanation(
                prompt,
                max_tokens=1500
            )
            
            # Store raw explanation for debugging
            st.session_state.raw_llm_explanation = raw_explanation
            
            # STEP 6: Validate output against source facts
            # This prevents hallucinations from reaching users
            # Critical safety layer - all numbers must match computed facts
            validated_explanation = validate_llm_output(
                raw_explanation, 
                context,
                max_length=3000
            )
            
            # Get validation report for debugging
            validation_report = get_validation_report(raw_explanation, context)
            st.session_state.validation_report = validation_report
            
            # Cache the explanation for this dataset
            st.session_state.llm_explanation = validated_explanation
            st.session_state.llm_explanation_file_id = current_file_id
            
            # Check if this is the fallback message (validation failed)
            fallback_msg = "The analysis highlights notable patterns in the data, but the explanation could not be confidently validated."
            is_fallback = validated_explanation.strip() == fallback_msg
            
            # Display the result
            if is_fallback:
                st.warning("⚠️ Explanation generated, but validation checks failed.")
                st.markdown("---")
                
                # Show the raw LLM output with warning
                with st.expander("🔍 View Generated Explanation (Unvalidated)", expanded=True):
                    st.markdown(raw_explanation)
                    st.caption("⚠️ This output failed validation checks. Some statistics may not match your data.")
                
                # Show validation details
                with st.expander("🔬 Validation Details"):
                    st.json(validation_report)
                    st.caption("💡 The LLM may have included numbers not present in your dataset.")
                
                st.markdown("---")
            else:
                st.success("✅ Explanation generated successfully!")
                st.markdown("---")
                st.markdown(validated_explanation)
                st.markdown("---")
                st.caption("💡 This explanation is based on computed statistics from your data.")
            
        except ImportError as e:
            st.error(f"❌ LLM pipeline not available: {str(e)}")
            st.info("💡 Make sure the backend LLM modules are properly installed.")
        except Exception as e:
            st.error("❌ Failed to generate explanation. Please try again.")
            st.caption(f"Error: {str(e)}")
            # Reset cached explanation on error
            st.session_state.llm_explanation = None
            
            # Show detailed traceback for debugging
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())


def _show_pipeline_info():
    """Display information about how the pipeline works."""
    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
        **The AI Explanation Pipeline:**
        
        1. **Signal Extraction**: Analyzes your data and computes statistical properties
        2. **Context Selection**: Focuses on the most relevant insights for explanation
        3. **Prompt Building**: Creates a structured prompt with safety constraints
        4. **LLM Inference**: Uses Groq's LLaMA model to generate natural language
        5. **Validation**: Verifies all claims against computed facts (hallucination control)
        
        **Why validation matters:**
        - LLMs can sometimes invent plausible-sounding statistics
        - Every number in the explanation is verified against your actual data
        - If validation fails, a safe fallback message is shown instead
        
        **Privacy & Security:**
        - Your data is processed locally for statistics
        - Only anonymized statistics are sent to the AI (never raw data)
        - Requires GROQ_API_KEY in .env file
        """)
