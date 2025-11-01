# frontend/helpers/progress_ui.py
"""
Shared progress UI components for generation workflows.
Provides consistent progress bars, status messages, and error handling.
"""
import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any


class GenerationProgress:
    """Context manager for handling generation progress UI consistently."""
    
    def __init__(self):
        self.progress_bar = None
        self.status_text = None
        
    def __enter__(self):
        """Initialize progress UI elements."""
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up progress UI on exit."""
        if exc_type is not None:
            # Error occurred, clear progress UI
            if self.progress_bar:
                self.progress_bar.empty()
            if self.status_text:
                self.status_text.empty()
        return False  # Don't suppress exceptions
        
    def update(self, progress: int, message: str):
        """
        Update progress bar and status message.
        
        Args:
            progress: Progress percentage (0-100)
            message: Status message to display
        """
        if self.progress_bar:
            self.progress_bar.progress(progress)
        if self.status_text:
            self.status_text.text(message)
            
    def complete(self, message: str = "✅ Complete!"):
        """Mark progress as complete."""
        if self.progress_bar:
            self.progress_bar.progress(100)
        if self.status_text:
            self.status_text.text(message)
            
    def clear(self):
        """Clear progress UI elements."""
        if self.progress_bar:
            self.progress_bar.empty()
        if self.status_text:
            self.status_text.empty()


def show_generation_progress(
    progress: int,
    message: str,
    progress_bar=None,
    status_text=None
):
    """
    Simple function to update progress bar and status text.
    
    Args:
        progress: Progress percentage (0-100)
        message: Status message
        progress_bar: Streamlit progress bar element
        status_text: Streamlit empty element for status text
    """
    if progress_bar is not None:
        progress_bar.progress(progress)
    if status_text is not None:
        status_text.text(message)


def handle_generation_error(error, error_type: str = "general", progress_bar=None, status_text=None):
    """
    Display formatted error messages with troubleshooting guidance.
    
    Args:
        error: Exception or error message
        error_type: Type of error (timeout, connection, api, general)
        progress_bar: Progress bar to clear
        status_text: Status text to clear
    """
    # Clear progress UI
    if progress_bar:
        progress_bar.empty()
    if status_text:
        status_text.empty()
    
    error_messages = {
        "timeout": """
        ⏱️ **Request Timed Out**
        
        The generation took too long (>5 minutes).
        
        **What to try:**
        - Reduce the number of rows or epochs
        - Use a faster algorithm (e.g., GaussianCopula)
        - Check if the API server is responsive
        """,
        "connection": """
        🔌 **Connection Error**
        
        Could not connect to the API server.
        
        **What to try:**
        - Check API URL in sidebar settings
        - Verify API server is running
        - Test API connection using "Test API" button
        """,
        "api": f"""
        ⚠️ **API Error**
        
        An unexpected error occurred: {str(error)}
        
        **What to try:**
        - Check logs for more details
        - Verify API server health
        - Try again in a few moments
        """,
        "general": f"""
        ❌ **Generation Failed**
        
        {str(error)}
        
        **What to try:**
        - Check your data has enough rows (minimum 100 recommended)
        - Reduce number of epochs or rows to generate
        - Try a different algorithm
        - Verify API connection in sidebar settings
        """
    }
    
    st.error(error_messages.get(error_type, error_messages["general"]))


def update_generation_session_state(df: pd.DataFrame, file_id: str):
    """
    Update session state after successful generation.
    
    Args:
        df: Generated synthetic dataframe
        file_id: File ID for the generation
    """
    st.session_state.generated_file_id = file_id
    st.session_state.data_columns = df.columns.tolist()
    st.session_state.data_history = st.session_state.get('data_history', [])
    st.session_state.data_history.append(df.copy())
    st.session_state.df = df.copy()


def compute_quality_badges(synth_df: pd.DataFrame, real_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Compute lightweight quality badges for generated data.
    
    These are simple heuristics for UI badges and not rigorous metrics.
    
    Args:
        synth_df: Synthetic dataframe
        real_df: Original dataframe (optional)
        
    Returns:
        Dictionary with badge values for fidelity, privacy_risk, mode_collapse
    """
    badges = {
        'fidelity': None,
        'privacy_risk': None,
        'mode_collapse': None
    }
    
    # Fidelity: compare number of unique values in numeric columns to original (if available)
    try:
        if real_df is None:
            badges['fidelity'] = 'unknown'
        else:
            num_cols = real_df.select_dtypes(include=['number']).columns
            if len(num_cols) == 0:
                badges['fidelity'] = 'n/a'
            else:
                ratios = []
                for c in num_cols:
                    r_uniques = real_df[c].nunique()
                    s_uniques = synth_df[c].nunique() if c in synth_df.columns else 0
                    ratios.append(min(1.0, s_uniques / max(1, r_uniques)))
                avg = sum(ratios) / len(ratios)
                if avg > 0.8:
                    badges['fidelity'] = 'high'
                elif avg > 0.5:
                    badges['fidelity'] = 'medium'
                else:
                    badges['fidelity'] = 'low'
    except Exception:
        badges['fidelity'] = 'unknown'

    # Privacy risk: simple heuristic based on presence of highly unique string columns
    try:
        str_cols = synth_df.select_dtypes(include=['object']).columns
        high_cardinality = 0
        for c in str_cols:
            if synth_df[c].nunique() > 0.8 * len(synth_df):
                high_cardinality += 1
        if high_cardinality == 0:
            badges['privacy_risk'] = 'low'
        elif high_cardinality <= 2:
            badges['privacy_risk'] = 'medium'
        else:
            badges['privacy_risk'] = 'high'
    except Exception:
        badges['privacy_risk'] = 'unknown'

    # Mode collapse: look for columns with very low unique counts
    try:
        collapse_count = 0
        for c in synth_df.columns:
            if synth_df[c].nunique() <= 3 and synth_df.shape[0] > 10:
                collapse_count += 1
        if collapse_count == 0:
            badges['mode_collapse'] = 'no'
        elif collapse_count <= 2:
            badges['mode_collapse'] = 'possible'
        else:
            badges['mode_collapse'] = 'likely'
    except Exception:
        badges['mode_collapse'] = 'unknown'

    return badges


def render_quality_badges(badges: Dict[str, str]):
    """
    Render quality badges in a consistent format.
    
    Args:
        badges: Dictionary with badge values
    """
    cols = st.columns(3)
    with cols[0]:
        st.metric("Fidelity", badges.get('fidelity', 'unknown'))
    with cols[1]:
        st.metric("Privacy Risk", badges.get('privacy_risk', 'unknown'))
    with cols[2]:
        st.metric("Mode Collapse", badges.get('mode_collapse', 'unknown'))
