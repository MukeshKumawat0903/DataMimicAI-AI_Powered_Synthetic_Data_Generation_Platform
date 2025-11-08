import streamlit as st
import requests
import pandas as pd
import os
import plotly.graph_objects as go
from .visual_profiling import (
    plot_correlation_heatmap, 
    filter_dataframe,
    load_data_from_api,
    prepare_data_for_viz
)

API_BASE = os.getenv("API_URL", "http://localhost:8000")

def _get_pii_columns():
    """Get list of columns identified as containing PII from session state."""
    pii_scan = st.session_state.get('pii_scan_results', {})
    results = pii_scan.get('results', {})
    detections = results.get('detections', []) if isinstance(results, dict) else []
    
    pii_columns = set()
    for detection in detections:
        pii_columns.add(detection.get('column'))
    
    return pii_columns

def _add_pii_badges(feature_name):
    """Add PII badge to feature name if it contains PII."""
    pii_columns = _get_pii_columns()
    if feature_name in pii_columns:
        return f"🔒 {feature_name}"
    return feature_name

def _call_correlation_api(file_id, top_k=10):
    """Call backend to get correlation and pattern discovery results."""
    with st.spinner("Calculating correlations..."):
        response = requests.post(
            f"{API_BASE}/eda/correlation",
            params={"file_id": file_id, "top_k": top_k}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Correlation analysis failed. Please try again.")
            return None

def _show_corr_heatmap(result):
    """Display the Pearson correlation heatmap if available."""
    st.subheader("📈 Pearson Correlation Heatmap")
    if result.get("corr_heatmap_base64"):
        st.markdown(
            f'<img src="data:image/png;base64,{result["corr_heatmap_base64"]}" style="max-width:100%;">',
            unsafe_allow_html=True
        )
    else:
        st.info("Not enough numeric features for correlation heatmap.")

def _show_top_corr_pairs(result):
    st.subheader("🔥 Top Correlated Feature Pairs")
    top_corrs = result.get("top_corrs", [])
    if top_corrs:
        top_pairs_df = pd.DataFrame(top_corrs, columns=["Feature 1", "Feature 2", "Correlation"])
        # Add PII badges
        top_pairs_df["Feature 1"] = top_pairs_df["Feature 1"].apply(_add_pii_badges)
        top_pairs_df["Feature 2"] = top_pairs_df["Feature 2"].apply(_add_pii_badges)
        st.table(top_pairs_df)
    else:
        st.info("No strong linear correlations detected.")

def _show_nonlinear_corrs(result):
    with st.expander("Show Nonlinear Correlations (Spearman, Kendall)", expanded=False):
        spearman = result.get("spearman_corr_matrix", {})
        kendall = result.get("kendall_corr_matrix", {})
        if spearman:
            st.write("**Spearman:**")
            st.dataframe(pd.DataFrame(spearman))
        else:
            st.info("Not enough numeric features for Spearman.")
        if kendall:
            st.write("**Kendall:**")
            st.dataframe(pd.DataFrame(kendall))
        else:
            st.info("Not enough numeric features for Kendall.")

def _show_categorical_associations(result):
    st.subheader("🔄 Categorical Associations (Cramér's V)")
    cat_df = pd.DataFrame(result.get("categorical_assoc", []))
    if not cat_df.empty:
        colmap = {k: "Feature 1" for k in cat_df.columns if k.lower() in ["col1", "feature 1"]}
        colmap.update({k: "Feature 2" for k in cat_df.columns if k.lower() in ["col2", "feature 2"]})
        colmap.update({k: "CramersV" for k in cat_df.columns if "cramers" in k.lower()})
        cat_df = cat_df.rename(columns=colmap)
        cat_df = cat_df[["Feature 1", "Feature 2", "CramersV"]]
        cat_df = cat_df.sort_values("CramersV", ascending=False)
        
        # Add PII badges
        cat_df["Feature 1"] = cat_df["Feature 1"].apply(_add_pii_badges)
        cat_df["Feature 2"] = cat_df["Feature 2"].apply(_add_pii_badges)
        
        st.dataframe(cat_df.head(10), use_container_width=True)
        st.markdown(
            """
            - **Cramér’s V ≈ 1.0:** Features are almost always found together (strong dependency)
            - **0.4–0.6:** Strong business link—reflects a real market or operational structure
            - **Below 0.2:** Features are mostly independent
            """
        )
    else:
        st.info("No categorical pairs detected for Cramér’s V.")

def _show_data_leakage(result):
    st.subheader("🚨 Potential Data Leakage or Duplicates")
    leakage = result.get("leakage_pairs", [])
    if leakage:
        st.warning("Features with almost perfect correlation detected!")
        leak_df = pd.DataFrame(leakage, columns=["Feature 1", "Feature 2", "Correlation"])
        st.table(leak_df)
    else:
        st.success("No strong data leakage detected.")


def _plot_dependency_network(graph_data):
    """
    Plot functional dependency network using Plotly.
    
    Args:
        graph_data: Dict with 'nodes' and 'edges' from backend
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    if not nodes or not edges:
        st.info("No functional dependencies detected with current settings.")
        return
    
    # Create node position layout (simple circular layout)
    import math
    n = len(nodes)
    node_positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        x = math.cos(angle)
        y = math.sin(angle)
        node_positions[node["id"]] = (x, y)
    
    # Create edge traces
    edge_traces = []
    for edge in edges:
        source_pos = node_positions[edge["source"]]
        target_pos = node_positions[edge["target"]]
        
        # Create arrow annotation
        edge_trace = go.Scatter(
            x=[source_pos[0], target_pos[0], None],
            y=[source_pos[1], target_pos[1], None],
            mode='lines',
            line=dict(
                width=2 + edge["confidence"] * 3,  # Thicker = higher confidence
                color=f'rgba(100, 149, 237, {0.3 + edge["confidence"] * 0.7})'
            ),
            hoverinfo='text',
            text=f"{edge['source']} → {edge['target']}<br>Confidence: {edge['confidence']:.2%}",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in nodes:
        x, y = node_positions[node["id"]]
        node_x.append(x)
        node_y.append(y)
        
        # Node size based on connections
        connections = node.get("as_determinant", 0) + node.get("as_dependent", 0)
        node_size.append(20 + connections * 10)
        
        hover_text = (
            f"<b>{node['id']}</b><br>"
            f"Determines: {node.get('as_determinant', 0)} columns<br>"
            f"Determined by: {node.get('as_dependent', 0)} columns"
        )
        node_text.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node["id"] for node in nodes],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color='#FF6B6B',
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title="Functional Dependency Network<br><sub>Arrows show X → Y relationships (determinant → dependent)</sub>",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _show_dependency_validation_warning(real_file_id, synthetic_file_id, dependencies, min_confidence):
    """
    Show warning banner if synthetic data violates functional dependencies.
    
    Args:
        real_file_id: Real dataset file ID
        synthetic_file_id: Synthetic dataset file ID
        dependencies: Detected dependencies
        min_confidence: Confidence threshold
    """
    with st.spinner("Validating dependencies in synthetic data..."):
        try:
            response = requests.post(
                f"{API_BASE}/eda/validate-dependencies",
                json={
                    "real_file_id": real_file_id,
                    "synthetic_file_id": synthetic_file_id,
                    "min_confidence": min_confidence
                }
            )
            
            if response.status_code == 200:
                validation_result = response.json()
                
                preserved_count = validation_result.get("preserved_count", 0)
                violated_count = validation_result.get("violated_count", 0)
                preservation_rate = validation_result.get("preservation_rate", 0)
                
                if violated_count > 0:
                    st.warning(
                        f"⚠️ **Synthetic Data Quality Alert**: {violated_count} out of {len(dependencies)} "
                        f"functional dependencies were violated in the synthetic data "
                        f"({preservation_rate:.1%} preservation rate). "
                        f"This may indicate that critical data relationships are not being maintained."
                    )
                    
                    with st.expander("🔍 View Violated Dependencies"):
                        violated = validation_result.get("violated", [])
                        if violated:
                            violated_df = pd.DataFrame(violated)
                            st.dataframe(violated_df, use_container_width=True)
                            
                            st.markdown("""
                            **Recommended Actions:**
                            1. Review synthetic data generation parameters
                            2. Consider using conditional generation to preserve dependencies
                            3. Apply post-processing to enforce functional relationships
                            """)
                else:
                    st.success(
                        f"✅ All {preserved_count} functional dependencies are preserved in synthetic data!"
                    )
            
        except Exception as e:
            st.error(f"Could not validate dependencies: {str(e)}")


def _show_functional_dependencies(file_id):
    """
    Display functional dependency analysis section.
    
    Args:
        file_id: Current file ID
    """
    st.markdown("### 🔗 Functional Dependency Detection")
    st.markdown(
        "Detects **X → Y** relationships where one column determines another "
        "(e.g., ZIP Code → City, Product ID → Category)."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_cardinality = st.number_input(
            "Max unique values in determinant",
            min_value=10,
            max_value=10000,
            value=1000,
            step=100,
            key="dep_max_card",
            help="Higher values allow more complex relationships but take longer to compute"
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
            key="dep_min_conf",
            help="Higher confidence = stricter dependency (95% = 95% of X values map to single Y)"
        )
    
    if st.button("🔍 Detect Dependencies", key="detect_deps_btn"):
        with st.spinner("Analyzing functional dependencies..."):
            try:
                response = requests.get(
                    f"{API_BASE}/eda/detect-dependencies/{file_id}",
                    params={
                        "max_determinant_cardinality": max_cardinality,
                        "min_confidence": min_confidence
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state['dependency_result'] = result
                else:
                    st.error(f"Dependency detection failed: {response.text}")
                    return
                    
            except Exception as e:
                st.error(f"Error calling API: {str(e)}")
                return
    
    # Display results if available
    result = st.session_state.get('dependency_result', None)
    if result:
        dependencies = result.get("dependencies", [])
        graph_data = result.get("graph_data", {})
        
        if dependencies:
            st.success(f"Found {len(dependencies)} functional dependencies!")
            
            # Check if synthetic data exists and show validation warning
            synthetic_file_id = st.session_state.get('synthetic_file_id', None)
            if synthetic_file_id:
                _show_dependency_validation_warning(file_id, synthetic_file_id, dependencies, min_confidence)
            
            # Show dependency table
            with st.expander("📋 Dependency Details", expanded=True):
                dep_df = pd.DataFrame(dependencies)
                
                # Format for display
                display_df = dep_df[[
                    "determinant", "dependent", "confidence", "type"
                ]].copy()
                display_df.columns = ["Determinant (X)", "Dependent (Y)", "Confidence", "Type"]
                display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(display_df, use_container_width=True)
                
                st.markdown("""
                **Interpretation:**
                - **1:1** - Perfect mapping (e.g., Employee ID → Employee Name)
                - **M:1** - Many-to-one (e.g., ZIP Code → City, multiple ZIPs to same city)
                - **Confidence** - % of determinant values that map to single dependent value
                """)
            
            # Show network graph
            st.markdown("---")
            _plot_dependency_network(graph_data)
            
        else:
            st.info("No functional dependencies found with current settings. Try lowering the confidence threshold.")


def _display_mi_heatmap(mi_data):
    """
    Display Mutual Information heatmap and analysis results.
    
    Args:
        mi_data: MI computation results from API
    """
    import plotly.graph_objects as go
    import numpy as np
    
    mi_matrix = mi_data.get('mi_matrix', {})
    mi_columns = mi_data.get('mi_columns', [])
    mi_heatmap = mi_data.get('mi_heatmap', [])
    top_pairs = mi_data.get('top_mi_pairs', [])
    nonlinear = mi_data.get('nonlinear_analysis', {})
    
    if not mi_heatmap or not mi_columns:
        st.warning("No MI data available to display.")
        return
    
    # Convert to numpy array for plotting
    mi_array = np.array(mi_heatmap)
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=mi_array,
        x=mi_columns,
        y=mi_columns,
        colorscale='Viridis',
        hovertemplate='MI(%{x}, %{y}) = %{z:.3f}<extra></extra>',
        colorbar=dict(title="MI Score")
    ))
    
    fig.update_layout(
        title="Mutual Information Matrix<br><sub>Higher scores indicate stronger dependencies (linear or non-linear)</sub>",
        xaxis_title="Features",
        yaxis_title="Features",
        height=600,
        width=800,
        xaxis={'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top MI pairs
    if top_pairs:
        st.markdown("#### 🔝 Top MI Pairs")
        
        top_df = pd.DataFrame(top_pairs)
        top_df['mi_score'] = top_df['mi_score'].apply(lambda x: f"{x:.4f}")
        top_df.columns = ['Column 1', 'Column 2', 'MI Score']
        
        st.dataframe(top_df.head(10), use_container_width=True)
    
    # Non-linear relationship analysis
    nonlinear_rels = nonlinear.get('nonlinear_relationships', [])
    
    if nonlinear_rels:
        st.markdown("#### 🔄 Non-Linear Relationships Detected")
        st.info(
            f"Found {len(nonlinear_rels)} column pairs with non-linear dependencies "
            f"(high MI but low Pearson correlation)."
        )
        
        with st.expander("📊 View Non-Linear Pairs", expanded=True):
            nonlinear_df = pd.DataFrame(nonlinear_rels)
            
            # Format for display
            display_df = nonlinear_df[[
                'column1', 'column2', 'mi_score', 'pearson_score', 'difference'
            ]].copy()
            
            display_df.columns = ['Column 1', 'Column 2', 'MI Score', 'Pearson', 'Difference']
            display_df['MI Score'] = display_df['MI Score'].apply(lambda x: f"{x:.3f}")
            display_df['Pearson'] = display_df['Pearson'].apply(lambda x: f"{x:.3f}")
            display_df['Difference'] = display_df['Difference'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - These pairs show **high Mutual Information** but **low Pearson correlation**
            - This indicates **non-linear relationships** that linear models may miss
            - Consider: polynomial features, interaction terms, or non-linear models
            """)
    
    # Recommendations
    recommendations = nonlinear.get('recommendations', [])
    if recommendations:
        st.markdown("#### 💡 Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.info(f"{i}. {rec}")
    
    # Comparison chart
    if nonlinear_rels:
        st.markdown("#### 📈 MI vs Pearson Comparison")
        
        # Create scatter plot comparing MI and Pearson
        fig2 = go.Figure()
        
        mi_scores = [r['mi_score'] for r in nonlinear_rels]
        pearson_scores = [r['pearson_score'] for r in nonlinear_rels]
        labels = [f"{r['column1']} - {r['column2']}" for r in nonlinear_rels]
        
        fig2.add_trace(go.Scatter(
            x=pearson_scores,
            y=mi_scores,
            mode='markers+text',
            text=labels,
            textposition='top center',
            marker=dict(
                size=12,
                color=mi_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="MI Score")
            ),
            hovertemplate='<b>%{text}</b><br>Pearson: %{x:.3f}<br>MI: %{y:.3f}<extra></extra>'
        ))
        
        # Add diagonal line (y=x)
        fig2.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Linear (MI = Pearson)',
            showlegend=True
        ))
        
        fig2.update_layout(
            title="MI vs Pearson Correlation<br><sub>Points above diagonal indicate non-linear relationships</sub>",
            xaxis_title="Pearson Correlation (|r|)",
            yaxis_title="Mutual Information",
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig2, use_container_width=True)


def expander_correlation():
    """Enhanced Streamlit expander for Smart Correlation & Pattern Discovery with interactive heatmap."""
    st.divider()
    with st.expander("🔗 Smart Correlation & Pattern Discovery", expanded=False):
        
        # Load the actual DataFrame for interactive correlation using API
        df = load_data_from_api(st.session_state.file_id)
        
        if df is None:
            st.error("Unable to load dataset. Please upload a file first.")
            return
        
        # Apply sampling for large datasets
        df_viz = prepare_data_for_viz(df, max_rows=50000)
        
        # Apply interactive filters
        filtered_df = filter_dataframe(df_viz, key_prefix="correlation")
        
        # === STEP 1: Interactive Correlation Heatmap ===
        st.markdown("### 📊 Interactive Correlation Analysis")
        st.markdown("Explore correlations between numeric features with Pearson correlation or Mutual Information (MI).")
        
        # Correlation type toggle
        analysis_type = st.radio(
            "Analysis Type",
            options=["Pearson Correlation", "Mutual Information (MI)"],
            horizontal=True,
            key="corr_analysis_type",
            help="Pearson: Linear relationships | MI: Non-linear dependencies"
        )
        
        if analysis_type == "Pearson Correlation":
            # Pearson correlation heatmap (existing functionality)
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                correlation_method = st.selectbox(
                    "Correlation Method",
                    options=["pearson", "spearman", "kendall"],
                    index=0,
                    key="correlation_method",
                    help="Pearson: linear correlation | Spearman: monotonic correlation | Kendall: rank correlation"
                )
            
            with col2:
                show_triangle = st.checkbox(
                    "Upper triangle only",
                    value=False,
                    key="correlation_triangle",
                    help="Show only upper triangle of the correlation matrix"
                )
            
            with col3:
                min_corr = st.slider(
                    "Min |correlation|",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key="correlation_min",
                    help="Filter out correlations below this threshold"
                )
            
            # Display interactive heatmap
            plot_correlation_heatmap(
                filtered_df,
                method=correlation_method,
                show_upper_triangle_only=show_triangle,
                min_correlation=min_corr
            )
        
        else:
            # Mutual Information heatmap (new functionality)
            st.info("💡 Mutual Information captures non-linear dependencies that Pearson correlation may miss.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                normalize_mi = st.checkbox(
                    "Normalize to [0, 1]",
                    value=True,
                    key="mi_normalize",
                    help="Normalize MI scores for easier interpretation"
                )
            
            with col2:
                if st.button("🔄 Compute MI Matrix", key="compute_mi_btn"):
                    with st.spinner("Computing Mutual Information matrix (this may take a moment)..."):
                        try:
                            response = requests.post(
                                f"{API_BASE}/eda/compute-mi-matrix/{st.session_state.file_id}",
                                json={"columns": None, "normalize": normalize_mi}
                            )
                            
                            if response.status_code == 200:
                                mi_data = response.json()
                                st.session_state['mi_result'] = mi_data
                                st.success("✅ MI matrix computed!")
                            else:
                                st.error(f"MI computation failed: {response.text}")
                        except Exception as e:
                            st.error(f"Error computing MI: {str(e)}")
            
            # Display MI results
            if 'mi_result' in st.session_state:
                _display_mi_heatmap(st.session_state['mi_result'])
        
        st.markdown("---")
        
        # === STEP 2: Backend Correlation API (existing functionality) ===
        st.markdown("### 🔬 Advanced Pattern Discovery")
        st.markdown("Run backend analysis for additional insights including categorical associations and data leakage detection.")
        
        run_corr = st.button("Run Advanced Correlation Analysis")
        if run_corr:
            result = _call_correlation_api(st.session_state.file_id)
            if result:
                st.session_state['correlation_result'] = result
        
        result = st.session_state.get('correlation_result', None)
        if result:
            _show_corr_heatmap(result)
            _show_top_corr_pairs(result)
            _show_nonlinear_corrs(result)
            _show_categorical_associations(result)
            _show_data_leakage(result)
        
        st.markdown("---")
        
        # === STEP 3: Functional Dependency Detection ===
        _show_functional_dependencies(st.session_state.file_id)



