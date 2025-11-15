"""
Transformation Visualization Module
Creates visual comparisons for data transformations.
Part of the EDA transformation analysis system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
PLOT_SAMPLE_LIMIT = 10000  # Maximum points to plot to avoid performance issues


def _sample_for_plotting(data: pd.Series, max_points: int = PLOT_SAMPLE_LIMIT) -> pd.Series:
    """
    Sample data for plotting if it's too large.
    
    Args:
        data: Series to potentially sample
        max_points: Maximum number of points to keep
        
    Returns:
        Sampled or original series
    """
    if len(data) > max_points:
        return data.sample(n=max_points, random_state=42)
    return data


def _clean_series_for_plotting(data: pd.Series) -> pd.Series:
    """
    Clean series by removing inf/-inf and NaN values.
    
    Args:
        data: Series to clean
        
    Returns:
        Cleaned series
    """
    return data.replace([np.inf, -np.inf], np.nan).dropna()


def plot_overlay_histograms(
    column: str,
    original_data: pd.Series,
    transformed_data: pd.Series,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create overlayed histogram and boxplot for before/after comparison.
    
    Generates an interactive Plotly visualization with two subplots:
    1. Overlayed histograms showing distribution changes
    2. Side-by-side boxplots showing outlier changes
    
    Args:
        column: Column name for labeling
        original_data: Original column data (before transformation)
        transformed_data: Transformed column data (after transformation)
        title: Optional custom title for the plot
        
    Returns:
        Dictionary with the following structure:
        {
            "is_numeric": bool,
            "plot_json": str,  # JSON-serialized Plotly figure
            "column": str,
            "sampled": bool,  # Whether data was sampled
            "original_count": int,  # Original data points
            "plotted_count": int,  # Plotted data points
            "error": str  # Only present if error occurred
        }
        
    Examples:
        >>> original = pd.Series(np.random.exponential(2, 1000))
        >>> transformed = pd.Series(np.log1p(original))
        >>> plot_data = plot_overlay_histograms("value", original, transformed)
        >>> plot_data["is_numeric"]
        True
    """
    # Check if numeric
    if not pd.api.types.is_numeric_dtype(original_data) or not pd.api.types.is_numeric_dtype(transformed_data):
        return {
            "error": "Non-numeric data cannot be plotted",
            "is_numeric": False
        }
    
    # Clean data (remove inf/-inf and NaN)
    orig_clean = _clean_series_for_plotting(original_data)
    trans_clean = _clean_series_for_plotting(transformed_data)
    
    # Check minimum data
    if len(orig_clean) < 2 or len(trans_clean) < 2:
        return {
            "error": "Insufficient data for plotting (need at least 2 valid values)",
            "is_numeric": True
        }
    
    # Sample if needed
    original_count = len(orig_clean)
    orig_sampled = _sample_for_plotting(orig_clean)
    trans_sampled = _sample_for_plotting(trans_clean)
    sampled = len(orig_sampled) < original_count
    
    # Create subplots: 1 row, 2 columns (histogram, boxplot)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution Comparison", "Boxplot Comparison"),
        column_widths=[0.7, 0.3]
    )
    
    # Histogram - Original
    fig.add_trace(
        go.Histogram(
            x=orig_sampled,
            name="Original",
            opacity=0.6,
            marker_color="rgb(99, 110, 250)",
            nbinsx=30,
            hovertemplate="<b>Original</b><br>Value: %{x}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Histogram - Transformed
    fig.add_trace(
        go.Histogram(
            x=trans_sampled,
            name="Transformed",
            opacity=0.6,
            marker_color="rgb(239, 85, 59)",
            nbinsx=30,
            hovertemplate="<b>Transformed</b><br>Value: %{x}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Boxplot - Original
    fig.add_trace(
        go.Box(
            y=orig_sampled,
            name="Original",
            marker_color="rgb(99, 110, 250)",
            boxmean='sd',
            hovertemplate="<b>Original</b><br>Value: %{y}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Boxplot - Transformed
    fig.add_trace(
        go.Box(
            y=trans_sampled,
            name="Transformed",
            marker_color="rgb(239, 85, 59)",
            boxmean='sd',
            hovertemplate="<b>Transformed</b><br>Value: %{y}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Update layout
    plot_title = title or f"Before/After Comparison: {column}"
    if sampled:
        plot_title += f" (sampled: {len(orig_sampled):,} of {original_count:,} points)"
    
    fig.update_layout(
        title_text=plot_title,
        showlegend=True,
        height=400,
        barmode='overlay',
        hovermode='closest',
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    
    # Convert to JSON for transmission
    return {
        "is_numeric": True,
        "plot_json": fig.to_json(),
        "column": column,
        "sampled": sampled,
        "original_count": original_count,
        "plotted_count": len(orig_sampled)
    }


def create_qq_plot(
    column: str,
    original_data: pd.Series,
    transformed_data: pd.Series
) -> Dict[str, Any]:
    """
    Create Q-Q plots to assess normality before and after transformation.
    
    Args:
        column: Column name
        original_data: Original column data
        transformed_data: Transformed column data
        
    Returns:
        Dictionary with plot JSON
    """
    from scipy import stats as scipy_stats
    
    if not pd.api.types.is_numeric_dtype(original_data):
        return {"is_numeric": False, "error": "Non-numeric data"}
    
    # Clean data
    orig_clean = _clean_series_for_plotting(original_data)
    trans_clean = _clean_series_for_plotting(transformed_data)
    
    if len(orig_clean) < 3 or len(trans_clean) < 3:
        return {"is_numeric": True, "error": "Insufficient data for Q-Q plot"}
    
    # Sample if needed
    orig_sampled = _sample_for_plotting(orig_clean)
    trans_sampled = _sample_for_plotting(trans_clean)
    
    # Create subplots for Q-Q plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original Q-Q Plot", "Transformed Q-Q Plot")
    )
    
    # Original Q-Q plot
    qq_orig = scipy_stats.probplot(orig_sampled, dist="norm")
    fig.add_trace(
        go.Scatter(
            x=qq_orig[0][0],
            y=qq_orig[0][1],
            mode='markers',
            name='Original',
            marker=dict(color='rgb(99, 110, 250)'),
            hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add reference line for original
    fig.add_trace(
        go.Scatter(
            x=qq_orig[0][0],
            y=qq_orig[1][1] + qq_orig[1][0] * qq_orig[0][0],
            mode='lines',
            name='Normal (Original)',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Transformed Q-Q plot
    qq_trans = scipy_stats.probplot(trans_sampled, dist="norm")
    fig.add_trace(
        go.Scatter(
            x=qq_trans[0][0],
            y=qq_trans[0][1],
            mode='markers',
            name='Transformed',
            marker=dict(color='rgb(239, 85, 59)'),
            hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add reference line for transformed
    fig.add_trace(
        go.Scatter(
            x=qq_trans[0][0],
            y=qq_trans[1][1] + qq_trans[1][0] * qq_trans[0][0],
            mode='lines',
            name='Normal (Transformed)',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text=f"Q-Q Plot Comparison: {column}",
        showlegend=True,
        height=400,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Theoretical Quantiles")
    fig.update_yaxes(title_text="Sample Quantiles")
    
    return {
        "is_numeric": True,
        "plot_json": fig.to_json(),
        "column": column
    }


def create_kde_plot(
    column: str,
    original_data: pd.Series,
    transformed_data: pd.Series
) -> Dict[str, Any]:
    """
    Create Kernel Density Estimate plots for smooth distribution comparison.
    
    Args:
        column: Column name
        original_data: Original column data
        transformed_data: Transformed column data
        
    Returns:
        Dictionary with plot JSON
    """
    from scipy.stats import gaussian_kde
    
    if not pd.api.types.is_numeric_dtype(original_data):
        return {"is_numeric": False, "error": "Non-numeric data"}
    
    # Clean data
    orig_clean = _clean_series_for_plotting(original_data)
    trans_clean = _clean_series_for_plotting(transformed_data)
    
    if len(orig_clean) < 3 or len(trans_clean) < 3:
        return {"is_numeric": True, "error": "Insufficient data for KDE"}
    
    # Sample if needed
    orig_sampled = _sample_for_plotting(orig_clean)
    trans_sampled = _sample_for_plotting(trans_clean)
    
    fig = go.Figure()
    
    # Original KDE
    if len(orig_sampled) > 1:
        try:
            kde_orig = gaussian_kde(orig_sampled)
            x_orig = np.linspace(orig_sampled.min(), orig_sampled.max(), 200)
            y_orig = kde_orig(x_orig)
            
            fig.add_trace(
                go.Scatter(
                    x=x_orig,
                    y=y_orig,
                    mode='lines',
                    name='Original',
                    line=dict(color='rgb(99, 110, 250)', width=2),
                    fill='tozeroy',
                    opacity=0.5,
                    hovertemplate="Value: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>"
                )
            )
        except Exception as e:
            return {"is_numeric": True, "error": f"KDE computation failed: {str(e)}"}
    
    # Transformed KDE
    if len(trans_sampled) > 1:
        try:
            kde_trans = gaussian_kde(trans_sampled)
            x_trans = np.linspace(trans_sampled.min(), trans_sampled.max(), 200)
            y_trans = kde_trans(x_trans)
            
            fig.add_trace(
                go.Scatter(
                    x=x_trans,
                    y=y_trans,
                    mode='lines',
                    name='Transformed',
                    line=dict(color='rgb(239, 85, 59)', width=2),
                    fill='tozeroy',
                    opacity=0.5,
                    hovertemplate="Value: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>"
                )
            )
        except Exception as e:
            return {"is_numeric": True, "error": f"KDE computation failed: {str(e)}"}
    
    fig.update_layout(
        title_text=f"Kernel Density Estimate: {column}",
        xaxis_title="Value",
        yaxis_title="Density",
        showlegend=True,
        height=400,
        template="plotly_white"
    )
    
    return {
        "is_numeric": True,
        "plot_json": fig.to_json(),
        "column": column
    }
