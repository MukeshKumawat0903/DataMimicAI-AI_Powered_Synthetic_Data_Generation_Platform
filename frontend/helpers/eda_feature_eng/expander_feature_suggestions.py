import matplotlib.pyplot as plt
import streamlit as st
import requests
import pandas as pd
import os
import io
import numpy as np

API_BASE = os.getenv("API_URL", "http://localhost:8000")

def _call_feature_suggestions(file_id, target_col=None):
    """Call backend API to get AI feature suggestions for a dataset."""
    params = {"file_id": file_id}
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
    st.subheader("🔬 AI Feature Suggestions")
    feature_df = pd.DataFrame({
        "Suggestion": result["suggestions"],
        "Explanation": result["explanations"]
    })
    st.dataframe(feature_df)
    st.markdown("#### 🧑‍💻 Suggested Python Code")
    code = "\n".join(result["code_blocks"])
    st.code(code, language="python")

def _show_feature_importance(result):
    """Show feature importance table if available."""
    if result.get("feature_importance"):
        st.subheader("⭐ Top Features by Importance")
        fi = pd.DataFrame(result["feature_importance"])
        st.table(fi)

def _select_features_to_apply(result):
    """Let the user select which features to apply from suggestions."""
    st.markdown("#### ✅ Select features to apply to your data:")
    if "selected_feature_indices" not in st.session_state:
        st.session_state.selected_feature_indices = []
    options = list(enumerate(result["suggestions"]))
    default_selected = [
        opt for opt in options if opt[0] in st.session_state.selected_feature_indices
    ]
    selected = st.multiselect(
        "Select by suggestion",
        options=options,
        default=default_selected,
        format_func=lambda tup: tup[1],
        key="multiselect_suggestions"
    )
    selected_indices = [str(idx) for idx, _ in selected]
    st.session_state.selected_feature_indices = [idx for idx, _ in selected]
    return selected_indices

def _apply_selected_features(selected_indices, result, file_id):
    """Apply selected engineered features to the backend dataset."""
    if selected_indices and st.button("Apply Selected Features"):
        selected_code = [result["code_blocks"][int(idx)] for idx in selected_indices]
        resp = requests.post(
            f"{API_BASE}/eda/apply-features",
            params={"file_id": file_id},
            json={"code_blocks": selected_code}
        )
        if resp.status_code == 200:
            st.success("Selected features applied! You may rerun profiling or download updated data.")
        else:
            st.error("Failed to apply features.")

def _explain_feature_llm(result):
    """Request LLM-based explanation for a selected feature engineering code line."""
    st.markdown("#### 💬 Natural Language Explanation (LLM, demo)")
    if result and result.get("code_blocks") and len(result["code_blocks"]) > 0:
        feat_idx = st.number_input(
            "Feature suggestion index for LLM explanation",
            min_value=0,
            max_value=len(result["code_blocks"]) - 1,
            step=1
        )
        if st.button("Explain Feature with LLM"):
            code_line = result["code_blocks"][feat_idx]
            exp_resp = requests.post(
                f"{API_BASE}/eda/explain-feature",
                params={"feature_code": code_line}
            )
            if exp_resp.status_code == 200:
                st.info(exp_resp.json()["explanation"])
            st.markdown("**Python code for this feature:**")
            st.code(code_line, language="python")
    else:
        st.info("Run 'Generate AI Feature Suggestions' first and select a feature.")

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
        target_col = st.text_input("Target column (optional, for feature importance):")
        run_suggest = st.button("Generate AI Feature Suggestions")
        if run_suggest:
            result = _call_feature_suggestions(st.session_state.file_id, target_col)
            if result:
                st.session_state.feature_suggestions = result
                st.success("Suggestions generated! Now you can select and apply.")
        result = st.session_state.get("feature_suggestions", None)
        if result:
            _show_suggestions(result)
            _show_feature_importance(result)
            selected_indices = _select_features_to_apply(result)
            _apply_selected_features(selected_indices, result, st.session_state.file_id)
            _explain_feature_llm(result)
            _visualize_engineered_features(result, st.session_state.file_id)