import matplotlib.pyplot as plt
import streamlit as st
import requests
import pandas as pd
import os
import io
import numpy as np

API_BASE = os.getenv("API_URL", "http://localhost:8000")

def _call_feature_suggestions(file_id, target_col=None, max_suggestions=5):
    """Call backend API to get AI feature suggestions for a dataset."""
    params = {"file_id": file_id, "max_suggestions": max_suggestions}
    if target_col:
        params["target_col"] = target_col
    response = requests.post(f"{API_BASE}/eda/feature-suggestions", params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to generate feature suggestions.")
        return None

def _show_suggestions(result):
    """Display AI feature suggestions and Python code in Streamlit."""
    st.markdown("### 🔬 AI Feature Suggestions")
    
    suggestions = result.get("suggestions", [])
    explanations = result.get("explanations", [])
    code_blocks = result.get("code_blocks", [])
    
    if not suggestions:
        st.warning("No feature suggestions generated. Try with a different target column or dataset.")
        return
    
    # Display suggestions in an organized format
    for i, (suggestion, explanation) in enumerate(zip(suggestions, explanations), 1):
        with st.container():
            col1, col2 = st.columns([1, 20])
            with col1:
                st.markdown(f"**{i}.**")
            with col2:
                st.markdown(f"**{suggestion}**")
                st.caption(explanation)
            
            # Show corresponding code in expandable section
            if i - 1 < len(code_blocks):
                with st.expander(f"📝 View Python code for suggestion #{i}"):
                    st.code(code_blocks[i - 1], language="python")
            
            st.markdown("")  # Spacing
    
    # Optional: Show all code together
    with st.expander("💻 View All Generated Code", expanded=False):
        st.markdown("**Complete feature engineering pipeline:**")
        all_code = "\n".join(code_blocks)
        st.code(all_code, language="python")
        
        # Add copy button hint
        st.caption("💡 Tip: Click the copy icon (⧉) in the top-right corner of the code block to copy all code.")

def _show_feature_importance(result):
    """Show feature importance table if available."""
    if result.get("feature_importance"):
        st.markdown("### ⭐ Feature Importance Ranking")
        st.caption("Features ranked by their predictive power for the target variable")
        
        fi_data = result["feature_importance"]
        fi_df = pd.DataFrame(fi_data)
        
        # Add visual bars if importance scores are available
        if "importance" in fi_df.columns or "Importance" in fi_df.columns:
            importance_col = "importance" if "importance" in fi_df.columns else "Importance"
            
            # Normalize importance for bar display
            max_importance = fi_df[importance_col].max()
            fi_df['Importance_Pct'] = (fi_df[importance_col] / max_importance * 100).round(1)
            
            # Display with progress bars
            for idx, row in fi_df.head(10).iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    feature_name = row.get('feature', row.get('Feature', f'Feature {idx}'))
                    st.markdown(f"**{feature_name}**")
                with col2:
                    importance_val = row[importance_col]
                    st.caption(f"{importance_val:.4f}")
                with col3:
                    st.progress(row['Importance_Pct'] / 100)
        else:
            # Fallback to table display
            st.dataframe(fi_df.head(10), use_container_width=True)
    else:
        st.info("ℹ️ Feature importance not available. Specify a target column to see importance rankings.")

def _select_features_to_apply(result):
    """Let the user select which features to apply from suggestions."""
    st.markdown("### ✅ Apply Features to Dataset")
    st.caption("Select which feature engineering transformations to apply to your data")
    
    if "selected_feature_indices" not in st.session_state:
        st.session_state.selected_feature_indices = []
    
    suggestions = result.get("suggestions", [])
    if not suggestions:
        return []
    
    # Create better display options
    options = []
    for idx, suggestion in enumerate(suggestions):
        # Truncate long suggestions for better display
        display_text = suggestion if len(suggestion) <= 80 else suggestion[:77] + "..."
        options.append((idx, display_text))
    
    default_selected = [
        opt for opt in options if opt[0] in st.session_state.selected_feature_indices
    ]
    
    # Show helpful message
    if not default_selected:
        st.info("💡 Select one or more feature transformations to apply to your dataset.")
    
    selected = st.multiselect(
        "Choose features to apply:",
        options=options,
        default=default_selected,
        format_func=lambda tup: f"#{tup[0] + 1}: {tup[1]}",
        key="multiselect_suggestions",
        help="Selected features will be added as new columns to your dataset"
    )
    
    selected_indices = [str(idx) for idx, _ in selected]
    st.session_state.selected_feature_indices = [idx for idx, _ in selected]
    
    # Show preview of selected features
    if selected:
        st.success(f"✓ {len(selected)} feature(s) selected")
        with st.expander("📋 Preview selected transformations"):
            for idx, _ in selected:
                if idx < len(result.get("code_blocks", [])):
                    st.code(result["code_blocks"][idx], language="python")
    
    return selected_indices

def _apply_selected_features(selected_indices, result, file_id):
    """Apply selected engineered features to the backend dataset."""
    if not selected_indices:
        st.caption("ℹ️ No features selected. Select features above to apply them.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        apply_btn = st.button(
            f"🚀 Apply {len(selected_indices)} Selected Feature(s)",
            type="primary",
            use_container_width=True,
            key="apply_features_btn"
        )
    
    with col2:
        if st.button("🔄 Reset Selection", use_container_width=True):
            st.session_state.selected_feature_indices = []
            st.rerun()
    
    if apply_btn:
        with st.spinner(f"Applying {len(selected_indices)} feature transformations..."):
            selected_code = [result["code_blocks"][int(idx)] for idx in selected_indices]
            resp = requests.post(
                f"{API_BASE}/eda/apply-features",
                params={"file_id": file_id},
                json={"code_blocks": selected_code}
            )
            
            if resp.status_code == 200:
                st.success("✅ Features successfully applied to your dataset!")
                st.info("💡 **Next steps:** You can now re-run data profiling or download the updated dataset.")
                
                # Show what was applied
                with st.expander("✓ Applied transformations"):
                    for code in selected_code:
                        st.code(code, language="python")
                
                # Clear selection after applying
                st.session_state.selected_feature_indices = []
                
            else:
                st.error(f"❌ Failed to apply features. Status code: {resp.status_code}")
                with st.expander("Error details"):
                    st.code(resp.text)

def _explain_feature_llm(result):
    """Request LLM-based explanation for a selected feature engineering code line."""
    st.markdown("### 💬 Get AI Explanation")
    st.caption("Get a natural language explanation of how a specific feature transformation works")
    
    code_blocks = result.get("code_blocks", [])
    suggestions = result.get("suggestions", [])
    
    if not code_blocks:
        st.info("Generate feature suggestions first to see explanations.")
        return
    
    # Create better selection interface
    feature_options = []
    for idx, suggestion in enumerate(suggestions):
        display = f"#{idx + 1}: {suggestion[:60]}..." if len(suggestion) > 60 else f"#{idx + 1}: {suggestion}"
        feature_options.append((idx, display))
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected = st.selectbox(
            "Select a feature transformation to explain:",
            options=feature_options,
            format_func=lambda x: x[1],
            key="explain_feature_select"
        )
    
    with col2:
        explain_btn = st.button(
            "🤖 Explain",
            type="primary",
            use_container_width=True,
            key="explain_feature_btn"
        )
    
    if explain_btn and selected:
        feat_idx = selected[0]
        code_line = code_blocks[feat_idx]
        
        with st.spinner("🤖 AI is generating explanation..."):
            exp_resp = requests.post(
                f"{API_BASE}/eda/explain-feature",
                params={"feature_code": code_line}
            )
            
            if exp_resp.status_code == 200:
                explanation = exp_resp.json().get("explanation", "No explanation available.")
                
                # Display in a nice format
                st.markdown("#### 💡 AI Explanation")
                st.info(explanation)
                
                st.markdown("#### 📝 Python Code")
                st.code(code_line, language="python")
            else:
                st.error("Failed to generate explanation. Please try again.")
                with st.expander("Error details"):
                    st.code(exp_resp.text)

def _visualize_engineered_features(result, file_id):
    """Visualize the distributions of newly engineered features vs base features."""
    st.markdown("#### 📊 Visualize Feature Distributions (Before/After)")
    engineered_features = []
    suffixes = ["_log", "_squared", "_sqrt", "_inv", "_bin", "_zscore", "_minmax", "_rank", "_outlier", "_len", "_wordcount", "_cubed"]
    for i, code_line in enumerate(result["code_blocks"]):
        if "=" in code_line:
            lhs = code_line.split("=")[0].strip()
            if lhs.startswith("df["):
                new_col = lhs.split("[", 1)[1].split("]", 1)[0].replace("'", "").replace('"', '')
                base_col = None
                for suffix in suffixes:
                    if new_col.endswith(suffix):
                        base_col = new_col.replace(suffix, "")
                        break
                engineered_features.append((new_col, base_col, i))
    feature_options = [f"{new} (vs {base})" if base else new for new, base, idx in engineered_features]
    selected_indices = st.multiselect(
        "Select engineered features to visualize",
        options=[idx for _, _, idx in engineered_features],
        format_func=lambda idx: feature_options[[f[2] for f in engineered_features].index(idx)]
    )
    if st.button("Show selected feature distributions"):
        response = requests.get(
            f"{API_BASE}/eda/download",
            params={"file_id": file_id}
        )
        if response.status_code == 200:
            df = pd.read_csv(io.BytesIO(response.content))
            if selected_indices:
                for idx in selected_indices:
                    new_col, base_col, _ = engineered_features[
                        [f[2] for f in engineered_features].index(idx)
                        ]
                    st.write(
                        f"Distribution for engineered feature: `{new_col}`" + 
                        (f" vs `{base_col}`" if base_col else "")
                        )
                    _plot_distribution(df, base_col, new_col)
            else:
                st.info("Select at least one feature to plot.")
        else:
            st.warning("Could not load data for plotting.")

def _plot_distribution(df, base_col, new_col, top_n_categories=10):
    """
    Plot the distribution of base and new feature columns using matplotlib and Streamlit.

    Args:
        df (pd.DataFrame): The dataframe containing the features.
        base_col (str or None): The original feature column name.
        new_col (str): The engineered feature column name.
        top_n_categories (int): Number of top categories to show for categorical features.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted_any = False

    columns_to_plot = [(base_col, "tab:blue"), (new_col, "tab:orange")]

    for col, color in columns_to_plot:
        if col is None or col not in df.columns:
            continue

        series = df[col].dropna()
        label = f"{col}"

        # Numeric columns
        if np.issubdtype(series.dtype, np.number):
            if len(series) == 0 or series.nunique() == 1:
                continue
            ax.hist(series, bins=30, alpha=0.5, label=label, color=color)
            plotted_any = True

        # Boolean columns
        elif np.issubdtype(series.dtype, np.bool_):
            counts = series.value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values, alpha=0.5, label=label, color=color)
            plotted_any = True

        # Categorical/object columns
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            counts = series.value_counts().head(top_n_categories)
            if len(counts) > 0:
                ax.bar(counts.index.astype(str), counts.values, alpha=0.5, label=label, color=color)
                plotted_any = True

    ax.set_title(f"Distribution: {base_col} vs {new_col}")
    ax.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()

    if plotted_any:
        st.pyplot(fig)
    else:
        st.info("No valid data to plot for selected columns.")

def expander_feature_suggestions():
    """Streamlit expander: AI-Driven Feature Engineering suggestions for the dataset."""
    st.divider()
    with st.expander("🪄 AI-Driven Feature Suggestions", expanded=False):
        
        # Enhanced header with guidance
        st.markdown("""
        **Automatically generate feature engineering suggestions** based on your data patterns.
        The AI analyzes your dataset and recommends transformations like:
        - 📊 Numeric transformations (log, sqrt, polynomial)
        - 🔢 Binning and discretization
        - 📝 Text feature extraction
        - ⏰ Date/time decomposition
        """)
        
        # Smart target column selection
        st.markdown("### 🎯 Configuration")
        
        # Load data to get column list
        try:
            from .visual_profiling import load_data_from_api
            df = load_data_from_api(st.session_state.file_id, sample_size=1000)
            
            if df is not None:
                # Identify potential target columns (numeric columns, or last column)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                all_cols = df.columns.tolist()
                
                # Smart defaults for target column
                suggested_targets = []
                
                # Check for common target column names
                target_keywords = ['target', 'label', 'class', 'outcome', 'prediction', 'y', 'price', 'amount', 'sales', 'revenue']
                for col in all_cols:
                    if any(keyword in col.lower() for keyword in target_keywords):
                        suggested_targets.append(col)
                
                # If no keyword match, suggest numeric columns
                if not suggested_targets and numeric_cols:
                    suggested_targets = numeric_cols[-3:]  # Last 3 numeric columns
                
                # Build options list with "None" option
                target_options = ["(None - General feature suggestions)"] + all_cols
                
                # Determine default index
                if suggested_targets:
                    default_col = suggested_targets[0]
                    default_index = all_cols.index(default_col) + 1  # +1 for "None" option
                else:
                    default_index = 0
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    target_selection = st.selectbox(
                        "Target column (optional, for feature importance ranking)",
                        options=target_options,
                        index=default_index,
                        key="target_col_selector",
                        help="Select a target variable to rank features by importance. Leave as 'None' for general suggestions."
                    )
                
                with col2:
                    max_suggestions = st.number_input(
                        "Max suggestions",
                        min_value=1,
                        max_value=20,
                        value=5,
                        step=1,
                        key="max_suggestions",
                        help="Number of feature suggestions to generate"
                    )
                
                # Parse target column
                target_col = None if target_selection.startswith("(None") else target_selection
                
                # Show smart suggestions
                if suggested_targets and target_col in suggested_targets:
                    st.info(f"💡 **Smart detection**: '{target_col}' looks like a target variable based on its name.")
                
                # Show dataset info
                with st.expander("📋 Dataset Info", expanded=False):
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.metric("Total Columns", len(all_cols))
                    with col_info2:
                        st.metric("Numeric Columns", len(numeric_cols))
                    with col_info3:
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        st.metric("Categorical Columns", len(categorical_cols))
                
            else:
                st.warning("Unable to load dataset. Using manual input.")
                target_col = st.text_input(
                    "Target column (optional, for feature importance):",
                    placeholder="e.g., 'price', 'target', 'label'",
                    help="Enter the name of your target variable, or leave blank for general suggestions"
                )
                max_suggestions = 5
                
        except Exception as e:
            st.warning(f"Could not auto-detect columns: {str(e)}")
            target_col = st.text_input(
                "Target column (optional, for feature importance):",
                placeholder="e.g., 'price', 'target', 'label'",
                help="Enter the name of your target variable, or leave blank for general suggestions"
            )
            max_suggestions = 5
        
        st.markdown("---")
        
        # Generate button with better styling
        run_suggest = st.button(
            "🚀 Generate AI Feature Suggestions",
            type="primary",
            use_container_width=True,
            key="generate_suggestions_btn"
        )
        
        if run_suggest:
            with st.spinner("🤖 AI is analyzing your data and generating feature suggestions..."):
                result = _call_feature_suggestions(st.session_state.file_id, target_col, max_suggestions)
                if result:
                    st.session_state.feature_suggestions = result
                    st.success("✨ Suggestions generated! Review and select features to apply below.")
                    st.balloons()
        
        # Display results
        result = st.session_state.get("feature_suggestions", None)
        if result:
            st.markdown("---")
            _show_suggestions(result)
            _show_feature_importance(result)
            
            st.markdown("---")
            selected_indices = _select_features_to_apply(result)
            _apply_selected_features(selected_indices, result, st.session_state.file_id)
            
            st.markdown("---")
            _explain_feature_llm(result)
            
            st.markdown("---")
            _visualize_engineered_features(result, st.session_state.file_id)
        else:
            # Show placeholder when no results
            st.info("👆 Click the button above to generate AI-powered feature engineering suggestions for your dataset.")
