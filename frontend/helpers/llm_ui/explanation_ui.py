"""
Explanation UI - STEP 7: UI Integration for LLM Pipeline.

This module provides the Streamlit UI for AI-powered data explanations.
It connects to the FastAPI LLM endpoint to generate explanations.

Author: DataMimicAI Team
Date: February 2026
"""

import streamlit as st
import requests
import os
import frontend_config as config


def get_api_base():
    """Return per-session override or configured API base."""
    return st.session_state.get('custom_api') or os.getenv("API_URL") or config.API_BASE


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
    Execute the LLM pipeline via FastAPI endpoint.
    
    Parameters
    ----------
    current_file_id : str
        Current dataset identifier for the API call
    """
    with st.spinner("🤖 Analyzing your data... This may take 10-30 seconds."):
        try:
            # Get API base URL
            api_base = get_api_base()
            
            # Prepare request payload
            payload = {
                "file_id": current_file_id,
                "scope": "dataset_overview",
                "tone": "clear",
                "max_tokens": 1500
            }
            
            # Call LLM API endpoint
            response = requests.post(
                f"{api_base}/llm/explain",
                json=payload,
                timeout=60
            )
            
            # Handle response
            if response.status_code == 200:
                data = response.json()
                
                # Extract response data
                validated_explanation = data.get("explanation", "")
                is_validated = data.get("validated", False)
                validation_report = data.get("validation_report", {})
                metadata = data.get("metadata", {})
                
                # Store in session state
                st.session_state.llm_explanation = validated_explanation
                st.session_state.llm_explanation_file_id = current_file_id
                st.session_state.validation_report = validation_report
                
                # Check if validation failed (fallback message)
                fallback_msg = "The analysis highlights notable patterns in the data, but the explanation could not be confidently validated."
                is_fallback = validated_explanation.strip() == fallback_msg
                
                # Display the result
                if is_fallback or not is_validated:
                    st.warning("⚠️ Explanation generated, but validation checks failed.")
                    st.markdown("---")
                    
                    # Show the explanation with warning
                    with st.expander("🔍 View Generated Explanation (Unvalidated)", expanded=True):
                        st.markdown(validated_explanation)
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
                    
                    # Show metadata in expander
                    with st.expander("ℹ️ Generation Details"):
                        st.json(metadata)
                
            elif response.status_code == 404:
                st.error("❌ Dataset not found. Please upload your data again.")
                st.session_state.llm_explanation = None
                
            elif response.status_code == 400:
                error_detail = response.json().get("detail", "Invalid request parameters")
                st.error(f"❌ {error_detail}")
                st.session_state.llm_explanation = None
                
            else:
                error_detail = response.json().get("detail", "Unknown error")
                st.error(f"❌ Failed to generate explanation: {error_detail}")
                st.session_state.llm_explanation = None
                
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The analysis is taking too long. Please try again.")
            st.session_state.llm_explanation = None
            
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the backend API. Make sure the server is running.")
            st.info(f"💡 Trying to connect to: {get_api_base()}")
            st.session_state.llm_explanation = None
            
        except Exception as e:
            st.error("❌ Failed to generate explanation. Please try again.")
            st.caption(f"Error: {str(e)}")
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
        - Your data file is processed by the backend API
        - Only anonymized statistics are sent to the AI (never raw data)
        - Requires GROQ_API_KEY in backend .env file
        
        **API Endpoint:**
        - Using: `{get_api_base()}/llm/explain`
        """)
        
        # Show available scopes
        st.markdown("**Available Analysis Scopes:**")
        try:
            api_base = get_api_base()
            response = requests.get(f"{api_base}/llm/scopes", timeout=5)
            if response.status_code == 200:
                scopes_data = response.json()
                for scope, info in scopes_data.get("scopes", {}).items():
                    st.markdown(f"- **{scope}**: {info['description']}")
        except:
            st.caption("_Could not load available scopes_")
