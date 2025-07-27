import streamlit as st
import pandas as pd
import requests
import json
import io

# -------------------- UI HELPERS --------------------

def scroll_to_anchor(anchor):
    js = f"""
    <script>
        setTimeout(function() {{
            var target = document.querySelector('{anchor}');
            if (target) {{
                target.scrollIntoView({{
                    behavior: 'smooth',
                    block: 'start'
                }});
            }}
        }}, 200);
    </script>
    """
    st.components.v1.html(js, height=0, width=0)

def show_progress(step: int):
    steps = ["Step 1: Feedback", "Step 2: Generate"]
    progress_html = "".join([
        f"<span style='color: {('#1f77b4' if i < step else '#bbb')}; font-weight: bold;'>{s}</span>"
        + (" ➔ " if i < len(steps)-1 else "")
        for i, s in enumerate(steps)
    ])
    st.markdown(f"<div style='margin-bottom:8px;font-size:1.1em'>{progress_html}</div>", unsafe_allow_html=True)

def export_log_button(transformation_log):
    if transformation_log:
        st.download_button(
            label="📄 Download Transformation Log (JSON)",
            data=json.dumps(transformation_log, indent=2),
            file_name="transformation_log.json",
            mime="application/json",
            use_container_width=True
        )
        st.code(json.dumps(transformation_log, indent=2), language="json")

def confirm_reset():
    return st.session_state.get('reset_confirmed', False) or st.checkbox(
        "Confirm Reset", key="reset_confirmed", help="Are you sure? This will clear all actions and start over."
    )

def highlight_changes(df, preview_df):
    orig_cols = set(df.columns)
    preview_cols = set(preview_df.columns)
    added = preview_cols - orig_cols
    color_map = {}
    for c in preview_df.columns:
        if c in added:
            color_map[c] = 'background-color: #d1ffd1'
        else:
            color_map[c] = ''
    return preview_df.style.apply(lambda _: [color_map.get(c, '') for c in preview_df.columns], axis=1)

def row_preview_selector(df):
    total_rows = len(df)
    row_options = ["Show 10 rows", "Show 100 rows", f"Show all ({total_rows}) rows"]
    default_idx = 0 if total_rows > 100 else 2
    selected_option = st.selectbox(
        "How many rows to preview?", options=row_options, index=default_idx, key="preview_row_selector"
    )
    if "10" in selected_option:
        return df.head(10)
    elif "100" in selected_option:
        return df.head(100)
    else:
        return df

# -------------------- FEEDBACK TAB COMPONENTS --------------------

def _drop_columns(df):
    drop_cols = st.multiselect(
        "🗑️ Columns to Drop",
        options=df.columns.tolist(),
        key='phase1_drop_cols',
        help="Remove columns completely. Dropped columns won't appear in further steps."
    )
    df = df.drop(columns=drop_cols) if drop_cols else df
    return df, drop_cols

def _rename_columns(df):
    rename_cols = st.multiselect(
        "🔤 Columns to Rename",
        options=df.columns.tolist(),
        key='phase1_rename_cols',
        help="Select columns to rename. You will enter new names below."
    )
    rename_mapping = {}
    if rename_cols:
        st.markdown("#### ✏️ Enter New Names for Selected Columns")
        for col in rename_cols:
            col_key = f'phase1_rename_{col}'
            if col_key not in st.session_state:
                st.session_state[col_key] = col
            input_col1, input_col2 = st.columns([3, 1])
            with input_col1:
                new_name = st.text_input(
                    f"New name for column: **{col}**",
                    value=st.session_state[col_key],
                    key=col_key,
                    placeholder=f"Enter new name for {col}",
                    help="Choose a unique new column name. " \
                    "Use only alphanumeric and underscores."
                )
            with input_col2:
                if new_name.strip() and new_name.strip() != col:
                    st.success(f"✅ {col} → {new_name.strip()}")
                else:
                    st.info(f"📝 {col}")
            if new_name.strip() and new_name.strip() != col:
                rename_mapping[col] = new_name.strip()
    df = df.rename(columns=rename_mapping) if rename_mapping else df
    return df, rename_mapping

def _impute_columns(df):
    numeric_na_cols = [
        col for col in df.select_dtypes(include='number').columns if df[col].isna().any()
        ]
    impute_methods = {}
    if numeric_na_cols:
        st.markdown("#### 🩹 Handle Missing Values")
        for col in numeric_na_cols:
            missing_count = df[col].isna().sum()
            total_count = len(df[col])
            missing_pct = (missing_count / total_count) * 100
            method = st.selectbox(
                f"**{col}** ({missing_count}/{total_count} missing, {missing_pct:.1f}%)",
                ["None", "mean", "median", "mode"],
                key=f'phase1_impute_{col}',
                help="How to fill missing values: 'mean' uses average, " \
                "'median' uses middle value, 'mode' uses most common value."
            )
            if method != "None":
                impute_methods[col] = method
    else:
        st.info("✅ No numeric columns with missing values found.")
    for col, method in impute_methods.items():
        if method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'mode':
            mode_series = df[col].mode()
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series[0])
            else:
                st.warning(f"⚠️ Could not compute mode for '{col}' - skipping imputation")
    return df, impute_methods

def _encode_columns(df):
    categorical_cols = [col for col in df.select_dtypes(include='object').columns]
    encode_cols = []
    if categorical_cols:
        st.markdown("#### 🎯 Categorical Encoding")
        encode_cols = st.multiselect(
            "One-hot encode these categorical columns",
            options=categorical_cols,
            key='phase1_encode_cols',
            help="One-hot encoding splits each selected column into multiple binary columns."
        )
    else:
        st.info("✅ No categorical columns available for encoding.")
    if encode_cols:
        df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
    return df, encode_cols

def render_transformation_summary(
        orig_df, drop_cols, rename_mapping, 
        impute_methods, encode_cols
        ):
    st.markdown("---")
    st.markdown("#### 📝 Transformations Applied")
    if drop_cols:
        st.write(f"• Dropped {len(drop_cols)} columns: {', '.join(drop_cols)}")
    if rename_mapping:
        st.write("• Renamed columns: " + ", ".join([f"{k}→{v}" for k, v in rename_mapping.items()]))
    if impute_methods:
        st.write("• Imputed: " + ", ".join([f"{col}({method})" for col, method in impute_methods.items()]))
    if encode_cols:
        st.write(f"• One-hot encoded: {', '.join(encode_cols)}")
    export_log_button({
        "drop": drop_cols,
        "rename": rename_mapping,
        "impute": impute_methods,
        "encode": encode_cols,
    })

def render_preview_section(orig_df, preview_df):
    st.markdown("<div id='preview-anchor'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Data Shape:**")
        st.info(f"📊 {orig_df.shape[0]} rows × {orig_df.shape[1]} columns")
    with col2:
        st.markdown("**Preview Data Shape:**")
        st.info(f"📊 {preview_df.shape[0]} rows × {preview_df.shape[1]} columns")
    # Column comparison
    original_cols = set(orig_df.columns)
    preview_cols = set(preview_df.columns)
    if original_cols != preview_cols:
        st.markdown("#### 🔄 Column Changes")
        c1, c2 = st.columns(2)
        with c1:
            removed_cols = original_cols - preview_cols
            if removed_cols:
                st.markdown("**Removed Columns:**")
                for col in sorted(removed_cols):
                    st.write(f"❌ {col}")
        with c2:
            added_cols = preview_cols - original_cols
            if added_cols:
                st.markdown("**Added Columns:**")
                for col in sorted(added_cols):
                    st.write(f"✅ {col}")
    # Preview
    st.markdown("#### 🔍 Preview of Transformed Data")
    preview_for_table = row_preview_selector(preview_df)
    styled_preview = highlight_changes(orig_df, preview_for_table)
    st.dataframe(styled_preview, use_container_width=True, height=400)
    if st.session_state.get('scroll_to_preview', False):
        scroll_to_anchor("#preview-anchor")
        st.session_state['scroll_to_preview'] = False

# -------------------- FEEDBACK TAB LOGIC --------------------

def render_feedback_tab(df):
    st.markdown("### 🛠️ Configure Data Transformations")
    working_df, drop_cols = _drop_columns(df)
    working_df, rename_mapping = _rename_columns(working_df)
    working_df, impute_methods = _impute_columns(working_df)
    working_df, encode_cols = _encode_columns(working_df)
    with st.form("phase1_feedback_form", clear_on_submit=False):
        st.markdown("---")
        phase1_submit = st.form_submit_button("👀 Preview Changes", use_container_width=True)
        if phase1_submit:
            st.session_state['preview_df'] = working_df
            st.session_state['feedback_changes_submitted'] = True
            st.session_state['transformation_log'] = {
                "drop": drop_cols,
                "rename": rename_mapping,
                "impute": impute_methods,
                "encode": encode_cols,
            }
            st.session_state['scroll_to_preview'] = True
            st.toast("✅ Preview generated!", icon="✅")
    if st.session_state.get('feedback_changes_submitted', False):
        render_transformation_summary(df, drop_cols, rename_mapping, impute_methods, encode_cols)
        render_preview_section(df, working_df)

# -------------------- GENERATION TAB COMPONENTS --------------------

def _select_generation_algorithm():
    return st.selectbox(
        "🔧 Synthesis Algorithm",
        ["CTGAN", "GaussianCopula", "TVAE", "PARS"],
        key='phase2_algorithm',
        help="Choose the algorithm for generating synthetic data"
    )

def _input_generation_parameters(algorithm):
    num_rows = st.number_input(
        "📈 Number of Rows to Generate",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100,
        key='phase2_num_rows',
        help="Choose how many synthetic rows to generate."
    )
    num_sequences, sequence_length = None, None
    if algorithm == "PARS":
        num_sequences = st.number_input(
            "🔢 Number of Sequences",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key='phase2_num_sequences',
            help="For PARS: Number of sequence patterns to generate."
        )
        sequence_length = st.number_input(
            "📏 Sequence Length",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            key='phase2_sequence_length',
            help="For PARS: How long each sequence will be."
        )
    return num_rows, num_sequences, sequence_length

def _build_feedback_payload(df):
    drop_cols = st.session_state.get('phase1_drop_cols', [])
    rename_mapping = {}
    rename_cols = st.session_state.get('phase1_rename_cols', [])
    for col in rename_cols:
        new_name = st.session_state.get(f'phase1_rename_{col}', col)
        if new_name and new_name.strip() and new_name.strip() != col:
            rename_mapping[col] = new_name.strip()
    numeric_na_cols = [
        col for col in df.select_dtypes(include='number').columns
        if df[col].isna().any() and col not in drop_cols
    ]
    impute_methods = {}
    for col in numeric_na_cols:
        method = st.session_state.get(f'phase1_impute_{col}', "None")
        if method != "None":
            impute_methods[col] = method
    encode_cols = st.session_state.get('phase1_encode_cols', [])
    feedback_payload = []
    if drop_cols:
        feedback_payload.extend([{"action": "drop", "column": c} for c in drop_cols])
    if rename_mapping:
        feedback_payload.extend([{"action": "rename", "column": c, "new_name": n} for c, n in rename_mapping.items()])
    if impute_methods:
        feedback_payload.extend([{"action": "impute", "column": c, "method": m} for c, m in impute_methods.items()])
    if encode_cols:
        feedback_payload.extend([{"action": "encode", "column": c} for c in encode_cols])
    return feedback_payload

def _review_generation_request(feedback_payload, generator_config):
    with st.expander("🔍 Review Request Details", expanded=False):
        st.markdown("**Feedback Actions:**")
        st.json(feedback_payload)
        st.markdown("**Generator Configuration:**")
        st.json(generator_config)

def _handle_generation_request(api_base, file_id, feedback_payload, generator_config):
    try:
        with st.spinner("🔄 Generating synthetic data... This may take a few minutes."):
            response = requests.post(
                f"{api_base}/feedback-generation/generate_and_download",
                params={"file_id": file_id},
                json={
                    "feedback": feedback_payload,
                    "generator_config": generator_config
                },
                timeout=300
            )
        if response.status_code == 200:
            st.toast("🎉 Synthetic data generated successfully!", icon="🎉")
            synthetic_df = pd.read_csv(io.BytesIO(response.content))
            for col in synthetic_df.select_dtypes(include='object').columns:
                synthetic_df[col] = synthetic_df[col].astype(str)
            st.markdown("### 📊 Synthetic Data Preview")
            st.dataframe(synthetic_df.head(10), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Generated Rows", len(synthetic_df))
            c2.metric("Generated Columns", len(synthetic_df.columns))
            st.download_button(
                label="📥 Download Synthetic Data CSV",
                data=response.content,
                file_name=f"synthetic_data_{generator_config['algorithm']}_{generator_config['num_rows']}rows.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown("#### ⭐ Rate This Generation (Optional)")
            st.slider("How satisfied are you with the synthetic data?", min_value=1, max_value=5, value=5, help="For future improvement. No data stored unless you agree.")
        else:
            st.toast(f"Generation failed: {response.text}", icon="❌")
    except requests.exceptions.Timeout:
        st.toast("⏱️ Request timed out. Please try again or reduce rows.", icon="⏱️")
    except requests.exceptions.RequestException as e:
        st.toast(f"❌ Request failed: {str(e)}", icon="❌")
    except Exception as e:
        st.toast(f"❌ Unexpected error: {str(e)}", icon="❌")

def _reset_generation():
    st.markdown("---")
    if st.button("🔄 Reset for New Generation", help="Start fresh with all feedback and params cleared."):
        if confirm_reset():
            for k in list(st.session_state.keys()):
                if k.startswith('phase1_') or k.startswith('phase2_') or k in (
                    'feedback_changes_submitted','preview_df','transformation_log',
                    'previous_rename_cols','advanced_show_full'
                ):
                    del st.session_state[k]
            st.toast("Session reset. Please upload/select data again.", icon="🧹")
            st.rerun()
        else:
            st.warning("Please check 'Confirm Reset' above before resetting.")

# -------------------- GENERATION TAB LOGIC --------------------

def render_generation_tab(file_id, api_base, df):
    if not st.session_state.get('feedback_changes_submitted', False):
        st.info("👆 Please complete Step 1 first to define and preview your data transformations.")
        st.markdown("---")
        st.markdown("**Why this step is required:**")
        st.write("• Step 1 processes your feedback and creates a preview of the transformed data")
        st.write("• Step 2 uses this processed data to generate synthetic data with your specifications")
        st.write("• This ensures the synthetic data matches your exact requirements")
        return
    st.markdown("### 🚀 Generate Synthetic Data")
    algorithm = _select_generation_algorithm()
    if st.button("⬅️ Back to Step 1"):
        st.session_state['feedback_changes_submitted'] = False
        st.toast("Go back and adjust feedback actions.", icon="↩️")
        st.experimental_rerun()
    with st.form("phase2_synthetic_form"):
        num_rows, num_sequences, sequence_length = _input_generation_parameters(algorithm)
        st.markdown("---")
        phase2_submit = st.form_submit_button("✨ Generate Synthetic Data", use_container_width=True)
    if phase2_submit:
        feedback_payload = _build_feedback_payload(df)
        generator_config = {
            "algorithm": algorithm,
            "num_rows": num_rows
        }
        if algorithm == "PARS":
            generator_config["num_sequences"] = num_sequences
            generator_config["sequence_length"] = sequence_length
        _review_generation_request(feedback_payload, generator_config)
        _handle_generation_request(api_base, file_id, feedback_payload, generator_config)
    _reset_generation()

# -------------------- MAIN ENTRYPOINT --------------------

def show_feedback_loop(file_id: str, api_base: str, df: pd.DataFrame):
    st.markdown("## 🔁 EDA-Driven Feedback Loop: Refine & Re-Generate Synthetic Data")
    step_complete = int(st.session_state.get('feedback_changes_submitted', False))
    show_progress(step_complete + 1)
    tab1, tab2 = st.tabs([
        "🛠️ Step 1: Define Feedback Actions",
        "🚀 Step 2: Generate Synthetic Data"
    ])
    with tab1:
        render_feedback_tab(df)
    with tab2:
        render_generation_tab(file_id, api_base, df)