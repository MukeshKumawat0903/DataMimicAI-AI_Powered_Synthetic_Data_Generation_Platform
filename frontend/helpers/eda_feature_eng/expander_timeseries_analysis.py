"""
Time-Series Analysis Expander for DataMimicAI

Provides interactive Streamlit UI for time-series diagnostics:
- ACF/PACF analysis with Plotly visualizations
- STL seasonal decomposition
- Generator configuration hints

Author: DataMimicAI Team
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

API_BASE = os.getenv("API_URL", "http://localhost:8000")


def expander_timeseries_analysis():
    """
    Main time-series analysis expander with detection, ACF/PACF, and decomposition.
    Auto-triggers detection on file upload.
    """
    st.divider()
    with st.expander("⏱️ Time-Series Analysis & Diagnostics", expanded=False):
        st.markdown("""
        Analyze temporal patterns in your dataset to guide time-series synthetic data generation:
        - **ACF/PACF**: Autocorrelation diagnostics for AR/MA order suggestions
        - **STL Decomposition**: Separate trend, seasonality, and residuals
        - **Generator Hints**: Get configuration recommendations for time-series models
        """)
        
        file_id = st.session_state.get("file_id")
        if not file_id:
            st.error("No file uploaded. Please upload a dataset first.")
            return
        
        # Check if detection is needed for this file_id
        current_detection_file = st.session_state.get('timeseries_file_id')
        detection_completed = st.session_state.get('timeseries_detection_completed', False)
        
        # Only trigger detection once per file_id
        if current_detection_file != file_id:
            # New file - reset state and trigger detection
            st.session_state['timeseries_file_id'] = file_id
            st.session_state['timeseries_detection_completed'] = False
            st.session_state.pop('timeseries_detection', None)  # Clear old results
            
            # Trigger detection
            if not detection_completed:
                _run_timeseries_detection(file_id)
                return  # Exit to allow rerun after detection
        
        # Display detection results
        detection = st.session_state.get('timeseries_detection', None)
        
        if not detection:
            # Detection not yet run for this file
            st.info("🔄 Running time-series detection...")
            return
        
        _display_detection_summary(detection)
        
        # If confirmed time-series, show analysis tools
        if detection.get('is_timeseries'):
            st.markdown("---")
            
            # Get datetime and target columns
            datetime_col = detection.get('primary_datetime_column')
            target_cols = detection.get('potential_targets', [])
            
            if not target_cols:
                st.warning("No numeric target columns found for time-series analysis.")
                return
            
            # Tab layout for different analyses
            tab1, tab2, tab3 = st.tabs([
                "📈 ACF/PACF Analysis",
                "🔄 Seasonal Decomposition",
                "💡 Generator Hints"
            ])
            
            with tab1:
                _show_acf_pacf_section(file_id, datetime_col, target_cols)
            
            with tab2:
                _show_decomposition_section(file_id, datetime_col, target_cols)
            
            with tab3:
                _show_generator_hints(detection)
        else:
            st.info("📊 Dataset does not appear to be time-series. Time-series analysis tools are disabled.")


def _run_timeseries_detection(file_id):
    """Run time-series detection API call with timeout."""
    with st.spinner("Detecting time-series characteristics..."):
        try:
            response = requests.post(
                f"{API_BASE}/eda/detect-timeseries/{file_id}",
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state['timeseries_detection'] = result['detection']
                st.session_state['timeseries_detection_completed'] = True
                st.success("✅ Time-series detection completed!")
                st.rerun()
            else:
                st.error(f"Detection failed: {response.text}")
                st.session_state['timeseries_detection_completed'] = True
        except requests.exceptions.Timeout:
            st.error("⏱️ Detection timed out. The dataset may be too large or the server is busy.")
            st.session_state['timeseries_detection_completed'] = True
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            st.session_state['timeseries_detection_completed'] = True


def _display_detection_summary(detection):
    """Display time-series detection results with KPI cards."""
    st.markdown("### 📊 Time-Series Detection Results")
    
    is_ts = detection.get('is_timeseries', False)
    confidence = detection.get('detection_confidence', 'low')
    datetime_cols = detection.get('datetime_columns', [])
    
    # Status banner
    if is_ts:
        st.success(f"✅ **Time-Series Detected** (Confidence: {confidence.upper()})")
    else:
        if len(datetime_cols) == 0:
            st.warning("""
            ⚠️ **No Datetime Columns Detected**
            
            The automatic detection could not identify any datetime columns in your dataset.
            
            **Possible reasons:**
            - Your datetime column may be in an unusual format
            - The column might be numeric (e.g., Unix timestamps)
            - Column values might not be recognized as valid dates
            
            **Next steps:**
            - Check your data has a column with date/time values
            - Ensure dates are in a standard format (YYYY-MM-DD, DD/MM/YYYY, etc.)
            - If you have Unix timestamps, convert them to datetime format first
            - Verify column names don't conflict with numeric data
            """)
        else:
            st.info("ℹ️ **Not Time-Series** - Dataset lacks temporal structure")
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Datetime Columns", len(datetime_cols))
    
    with col2:
        freq = detection.get('frequency', 'Unknown')
        st.metric("Frequency", freq if freq else 'Irregular')
    
    with col3:
        periods = detection.get('periods', 0)
        st.metric("Time Periods", f"{periods:,}")
    
    with col4:
        targets = len(detection.get('potential_targets', []))
        st.metric("Target Columns", targets)
    
    # Detailed info expander
    with st.expander("📋 Detection Details"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Temporal Information:**")
            st.write(f"- Primary datetime column: `{detection.get('primary_datetime_column', 'N/A')}`")
            st.write(f"- Start date: {detection.get('start_date', 'N/A')}")
            st.write(f"- End date: {detection.get('end_date', 'N/A')}")
            st.write(f"- Regular intervals: {'Yes' if detection.get('is_regular') else 'No'}")
        
        with col_b:
            st.markdown("**Available Targets:**")
            for target in detection.get('potential_targets', []):
                st.write(f"- `{target}`")


def _show_acf_pacf_section(file_id, datetime_col, target_cols):
    """ACF/PACF analysis section."""
    st.markdown("### 📊 Autocorrelation Analysis")
    st.markdown("""
    Autocorrelation (ACF) and Partial Autocorrelation (PACF) help identify the order of AR and MA models:
    - **ACF cutoff** → suggests MA order
    - **PACF cutoff** → suggests AR order
    """)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        selected_target = st.selectbox(
            "Select target column",
            options=target_cols,
            key="acf_target_selector"
        )
    
    with col2:
        lags = st.number_input(
            "Number of lags",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            key="acf_lags"
        )
    
    # Run analysis button
    if st.button("📈 Compute ACF/PACF", use_container_width=True, key="run_acf_pacf"):
        _run_acf_pacf_analysis(file_id, datetime_col, selected_target, lags)
    
    # Display results
    acf_result = st.session_state.get('acf_pacf_results', None)
    if acf_result:
        _display_acf_pacf_results(acf_result)


def _run_acf_pacf_analysis(file_id, datetime_col, target_col, lags):
    """Call ACF/PACF API endpoint with timeout."""
    with st.spinner(f"Computing ACF/PACF for '{target_col}'..."):
        try:
            response = requests.post(
                f"{API_BASE}/eda/timeseries-acf-pacf",
                json={
                    "file_id": file_id,
                    "datetime_col": datetime_col,
                    "target_col": target_col,
                    "lags": lags
                },
                timeout=60  # 60 second timeout for analysis
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state['acf_pacf_results'] = result['acf_pacf']
                st.success(f"✅ ACF/PACF analysis completed for '{target_col}'")
                st.rerun()
            else:
                st.error(f"Analysis failed: {response.text}")
        except requests.exceptions.Timeout:
            st.error("⏱️ ACF/PACF computation timed out. Try reducing the number of lags or check server status.")
        except Exception as e:
            st.error(f"Error during ACF/PACF: {str(e)}")


def _display_acf_pacf_results(results):
    """Display ACF/PACF plots and suggestions."""
    st.markdown("#### 📉 ACF & PACF Plots")
    
    acf_data = results.get('acf', {})
    pacf_data = results.get('pacf', {})
    suggestions = results.get('suggestions', {})
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)'),
        horizontal_spacing=0.15
    )
    
    # ACF plot
    acf_lags = acf_data.get('lags', [])
    acf_values = acf_data.get('values', [])
    acf_upper = acf_data.get('upper', [])
    acf_lower = acf_data.get('lower', [])
    
    # ACF bars
    fig.add_trace(
        go.Bar(
            x=acf_lags,
            y=acf_values,
            name='ACF',
            marker_color='#3498db',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # ACF confidence bands (now using absolute bounds)
    fig.add_trace(
        go.Scatter(
            x=acf_lags + acf_lags[::-1],
            y=acf_upper + acf_lower[::-1],
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # PACF plot
    pacf_lags = pacf_data.get('lags', [])
    pacf_values = pacf_data.get('values', [])
    pacf_upper = pacf_data.get('upper', [])
    pacf_lower = pacf_data.get('lower', [])
    
    # PACF bars
    fig.add_trace(
        go.Bar(
            x=pacf_lags,
            y=pacf_values,
            name='PACF',
            marker_color='#e74c3c',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # PACF confidence bands (now using absolute bounds)
    fig.add_trace(
        go.Scatter(
            x=pacf_lags + pacf_lags[::-1],
            y=pacf_upper + pacf_lower[::-1],
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_yaxes(title_text="Partial Correlation", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model suggestions
    st.markdown("#### 💡 Model Suggestions")
    
    model_type = suggestions.get('model_type', 'Unknown')
    explanation = suggestions.get('explanation', 'No clear pattern detected')
    ar_order = suggestions.get('ar_order')
    ma_order = suggestions.get('ma_order')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Suggested Model", model_type)
    
    with col2:
        st.metric("AR Order (p)", ar_order if ar_order is not None else "N/A")
    
    with col3:
        st.metric("MA Order (q)", ma_order if ma_order is not None else "N/A")
    
    st.info(f"📝 **Interpretation:** {explanation}")
    
    # Interpretation guide
    with st.expander("ℹ️ How to Read ACF/PACF Plots"):
        st.markdown("""
        **ACF (Autocorrelation Function):**
        - Measures correlation between observations at different time lags
        - Sharp cutoff suggests MA model
        - Gradual decline suggests AR model
        
        **PACF (Partial Autocorrelation Function):**
        - Measures correlation after removing effects of intermediate lags
        - Sharp cutoff suggests AR model
        - Gradual decline suggests MA model
        
        **Model Selection Guide:**
        - **AR(p)**: PACF cuts off at lag p, ACF decays gradually
        - **MA(q)**: ACF cuts off at lag q, PACF decays gradually
        - **ARMA(p,q)**: Both ACF and PACF decay gradually
        - **White Noise**: All values within confidence bands
        """)


def _show_decomposition_section(file_id, datetime_col, target_cols):
    """STL seasonal decomposition section with irregular series handling."""
    st.markdown("### 🔄 Seasonal Decomposition (STL)")
    st.markdown("""
    Decompose time-series into trend, seasonal, and residual components to understand underlying patterns.
    """)
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_target = st.selectbox(
            "Select target column",
            options=target_cols,
            key="decomp_target_selector"
        )
    
    with col2:
        model_type = st.radio(
            "Decomposition model",
            options=['additive', 'multiplicative'],
            key="decomp_model",
            help="Additive: components add up to original. Multiplicative: seasonal variation increases with level."
        )
    
    with col3:
        period = st.number_input(
            "Seasonal period (optional)",
            min_value=0,
            max_value=365,
            value=0,
            step=1,
            key="decomp_period",
            help="Leave as 0 for auto-detection. Common: 7 (weekly), 12 (monthly), 24 (hourly)"
        )
    
    # Advanced options for irregular series
    with st.expander("⚙️ Advanced Options (for irregular time-series)"):
        st.markdown("**Resampling Options** - Use if your time-series has irregular intervals")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            resample_freq = st.selectbox(
                "Resample frequency",
                options=['Auto', 'D (Daily)', 'H (Hourly)', 'T (Minutely)', 'W (Weekly)', 'M (Monthly)'],
                key="decomp_resample",
                help="Resample irregular series to regular intervals"
            )
        
        with col_b:
            aggregation = st.selectbox(
                "Aggregation method",
                options=['mean', 'sum', 'median', 'first', 'last'],
                key="decomp_agg",
                help="How to aggregate values when resampling"
            )
    
    # Parse resample frequency
    freq_map = {
        'Auto': None,
        'D (Daily)': 'D',
        'H (Hourly)': 'H',
        'T (Minutely)': 'T',
        'W (Weekly)': 'W',
        'M (Monthly)': 'M'
    }
    resample_freq_value = freq_map.get(resample_freq, None)
    
    # Run decomposition button
    if st.button("🔄 Run Decomposition", use_container_width=True, key="run_decomp"):
        _run_decomposition(
            file_id, 
            datetime_col, 
            selected_target, 
            model_type, 
            period if period > 0 else None,
            resample_freq_value,
            aggregation
        )
    
    # Display results
    decomp_result = st.session_state.get('decomposition_results', None)
    if decomp_result:
        _display_decomposition_results(decomp_result)


def _run_decomposition(file_id, datetime_col, target_col, model, period, resample_freq=None, aggregation='mean'):
    """Call decomposition API endpoint with timeout and resampling support."""
    with st.spinner(f"Decomposing '{target_col}'..."):
        try:
            response = requests.post(
                f"{API_BASE}/eda/timeseries-decompose",
                json={
                    "file_id": file_id,
                    "datetime_col": datetime_col,
                    "target_col": target_col,
                    "model": model,
                    "period": period,
                    "resample_freq": resample_freq,
                    "aggregation": aggregation
                },
                timeout=60  # 60 second timeout for decomposition
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state['decomposition_results'] = result['decomposition']
                st.success(f"✅ Decomposition completed for '{target_col}'")
                st.rerun()
            else:
                st.error(f"Decomposition failed: {response.text}")
        except requests.exceptions.Timeout:
            st.error("⏱️ Decomposition timed out. The series may be too long or the server is busy.")
        except Exception as e:
            st.error(f"Error during decomposition: {str(e)}")


def _display_decomposition_results(results):
    """Display decomposition components as stacked line plots."""
    st.markdown("#### 📊 Decomposition Components")
    
    # Extract components
    observed = results.get('observed', {})
    trend = results.get('trend', {})
    seasonal = results.get('seasonal', {})
    residual = results.get('residual', {})
    
    model = results.get('model', 'additive')
    period = results.get('period', 'auto')
    
    # Display metadata
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", model.capitalize())
    
    with col2:
        st.metric("Period", period)
    
    with col3:
        trend_strength = results.get('strength_of_trend', 0)
        st.metric("Trend Strength", f"{trend_strength:.2%}")
    
    with col4:
        seasonal_strength = results.get('strength_of_seasonality', 0)
        st.metric("Seasonality Strength", f"{seasonal_strength:.2%}")
    
    # Create stacked subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual')
    )
    
    # Convert index to datetime
    obs_index = pd.to_datetime(observed.get('index', []))
    
    # Observed
    fig.add_trace(
        go.Scatter(
            x=obs_index,
            y=observed.get('values', []),
            name='Observed',
            line=dict(color='#2c3e50', width=1.5),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Trend
    trend_index = pd.to_datetime(trend.get('index', []))
    fig.add_trace(
        go.Scatter(
            x=trend_index,
            y=trend.get('values', []),
            name='Trend',
            line=dict(color='#3498db', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Seasonal
    seasonal_index = pd.to_datetime(seasonal.get('index', []))
    fig.add_trace(
        go.Scatter(
            x=seasonal_index,
            y=seasonal.get('values', []),
            name='Seasonal',
            line=dict(color='#e74c3c', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Residual
    residual_index = pd.to_datetime(residual.get('index', []))
    fig.add_trace(
        go.Scatter(
            x=residual_index,
            y=residual.get('values', []),
            name='Residual',
            mode='markers',
            marker=dict(color='#95a5a6', size=3),
            showlegend=False
        ),
        row=4, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_layout(
        height=800,
        hovermode='x unified',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("#### 💡 Decomposition Insights")
    
    insights = []
    
    if trend_strength > 0.7:
        insights.append("🔼 **Strong trend component** - Data shows clear directional movement over time")
    elif trend_strength > 0.4:
        insights.append("📈 **Moderate trend** - Some directional pattern present")
    else:
        insights.append("➡️ **Weak/no trend** - Data relatively stable over time")
    
    if seasonal_strength > 0.7:
        insights.append(f"📅 **Strong seasonality** (period={period}) - Clear repeating patterns detected")
    elif seasonal_strength > 0.4:
        insights.append(f"🔄 **Moderate seasonality** (period={period}) - Some cyclical patterns present")
    else:
        insights.append("⚪ **Weak/no seasonality** - No clear repeating patterns")
    
    for insight in insights:
        st.info(insight)


def _show_generator_hints(detection):
    """Show generator configuration hints based on detection results."""
    st.markdown("### 💡 Generator Configuration Hints")
    st.markdown("""
    Based on detected time-series characteristics, here are recommended configurations 
    for synthetic data generation:
    """)
    
    freq = detection.get('frequency')
    is_regular = detection.get('is_regular', False)
    periods = detection.get('periods', 0)
    
    # Create hint table
    hints = []
    
    if is_regular and freq:
        hints.append({
            "Characteristic": "Regular frequency detected",
            "Recommendation": f"Use time-series generators with frequency='{freq}'",
            "Models": "ARIMA, Prophet, TimeGAN"
        })
    else:
        hints.append({
            "Characteristic": "Irregular intervals",
            "Recommendation": "Consider interpolation or event-based models",
            "Models": "Poisson process, Hawkes process"
        })
    
    if periods < 100:
        hints.append({
            "Characteristic": "Short series (< 100 periods)",
            "Recommendation": "Use simpler models to avoid overfitting",
            "Models": "AR(1-3), Simple exponential smoothing"
        })
    elif periods < 1000:
        hints.append({
            "Characteristic": "Medium-length series",
            "Recommendation": "Standard ARIMA/SARIMA models suitable",
            "Models": "ARIMA, SARIMA, Prophet"
        })
    else:
        hints.append({
            "Characteristic": "Long series (> 1000 periods)",
            "Recommendation": "Deep learning models can capture complex patterns",
            "Models": "LSTM, TimeGAN, Transformer-based"
        })
    
    # ACF/PACF suggestions if available
    acf_result = st.session_state.get('acf_pacf_results')
    if acf_result:
        model_type = acf_result.get('suggestions', {}).get('model_type')
        if model_type:
            hints.append({
                "Characteristic": "ACF/PACF analysis",
                "Recommendation": f"Suggested model: {model_type}",
                "Models": model_type
            })
    
    # Decomposition hints if available
    decomp_result = st.session_state.get('decomposition_results')
    if decomp_result:
        seasonal_strength = decomp_result.get('strength_of_seasonality', 0)
        if seasonal_strength > 0.6:
            period = decomp_result.get('period')
            hints.append({
                "Characteristic": f"Strong seasonality (period={period})",
                "Recommendation": f"Enable seasonal components with period={period}",
                "Models": "SARIMA, Prophet with seasonality"
            })
    
    # Display hints table
    if hints:
        df_hints = pd.DataFrame(hints)
        st.dataframe(df_hints, use_container_width=True, hide_index=True)
    
    # Additional tips
    with st.expander("📚 Additional Time-Series Generation Tips"):
        st.markdown("""
        **General Guidelines:**
        
        1. **Preserve Autocorrelation**: Ensure generated data maintains correlation structure
        2. **Match Distribution**: Use distribution analysis to match marginal distributions
        3. **Validate Stationarity**: Check if differencing is needed for non-stationary data
        4. **Handle Missing Values**: Decide on imputation strategy before generation
        5. **Cross-validation**: Use rolling window validation for time-series models
        
        **Model Selection:**
        - **Statistical Models** (ARIMA, Prophet): Good for interpretability and short-term forecasting
        - **Deep Learning** (LSTM, GRU): Better for long sequences and multivariate patterns
        - **GANs** (TimeGAN): Best for capturing complex distributions and dependencies
        
        **Quality Metrics:**
        - Compare ACF/PACF of synthetic vs. real data
        - Visual inspection of decomposed components
        - Distribution similarity tests (KS test, Wasserstein distance)
        """)
