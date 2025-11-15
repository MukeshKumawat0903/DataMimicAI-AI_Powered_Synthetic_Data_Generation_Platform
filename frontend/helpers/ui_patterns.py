import streamlit as st
import pandas as pd

def get_step_status(step_index, current_step):
    """Determine the status of a step based on session state."""
    if step_index < current_step:
        return "completed"
    elif step_index == current_step:
        return "in-progress"
    else:
        return "not-started"

def status_badge(label: str, status: str):
    """
    Display a visual status badge with theme-aware colors.
    status: 'active', 'pending', 'error', 'complete', 'warning'
    """
    colors = {
        'active': '#5fca75',
        'pending': '#FFA500',
        'error': '#FF4B4B',
        'complete': '#4B9BFF',
        'warning': '#FFD700'
    }
    icons = {
        'active': '▶️',
        'pending': '⏳',
        'error': '❌',
        'complete': '✅',
        'warning': '⚠️'
    }
    color = colors.get(status, '#808080')
    icon = icons.get(status, '•')
    st.markdown(f"""
        <span style="background:{color};color:white;padding:4px 12px;
                     border-radius:16px;font-size:0.9rem;font-weight:600;
                     display:inline-flex;align-items:center;gap:6px;
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            {icon} {label}
        </span>
    """, unsafe_allow_html=True)

def show_new_feature_badge(feature_name: str):
    """Shows a pulsing 'NEW' badge for recent features"""
    st.markdown(f"""
        <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.6; }}
        }}
        .new-badge {{
            background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7em;
            font-weight: 700;
            animation: pulse 2s infinite;
            display: inline-block;
            margin-left: 8px;
            vertical-align: middle;
        }}
        </style>
        <span class="new-badge">🆕 NEW</span> <span style="vertical-align:middle;">{feature_name}</span>
    """, unsafe_allow_html=True)

def show_smart_recommendations(step: int):
    """Display context-aware recommendations based on current step."""
    recommendations = {
        0: {
            "icon": "💡",
            "tip": "Use 'Smart Preview' to check data quality before moving forward",
            "action": None
        },
        1: {
            "icon": "💡", 
            "tip": "Explore your data thoroughly - understanding it now will help you make better generation choices in Step 2",
            "action": None
        },
        2: {
            "icon": "💡",
            "tip": "Use insights from Step 1 (Data Exploration) to choose the right algorithm and parameters",
            "action": None
        },
        3: {
            "icon": "💡",
            "tip": "Compare original vs synthetic data quality to validate results",
            "action": None
        }
    }
    
    if step in recommendations:
        rec = recommendations[step]
        st.info(f"{rec['icon']} **Tip:** {rec['tip']}")

def sidebar_stepper(current_step):
    """Display a sidebar stepper for navigation between steps, with modern look and strong highlight."""
    steps = [
        ("Data Upload", "📁", "upload"),
        ("Data Exploration", "🔍", "exploration"),  # Renamed from "EDA & Feature Eng."
        ("Generate Synthetic Data", "⚙️", "generation"),  # Renamed for clarity
        ("Validate & Refine", "✅", "validation"),  # Renamed from "Visualization"
        # ("Roadmap", "🧩", "roadmap"),
    ]
    st.markdown(
        """
        <style>
        .stepper-container {
            margin-bottom: 4px;
            padding-left: 0;
        }
        .stepper-row {
            display: flex;
            align-items: center;
            margin-bottom: 3px;
            border-radius: 7px;
            transition: background 0.18s;
        }
        .stepper-step {
            flex: 1 1 auto;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 1.04rem;
            font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
            font-weight: 400;
            padding: 7px 0 7px 15px;
            border-left: 5px solid transparent;
            background: transparent;
        }
        .stepper-step.active {
            background: #20294A;
            border-left: 5px solid #3A7DF4;
            font-weight: 700;
            color: #3A7DF4;
        }
        .stepper-step.completed {
            color: #16CC6B;
            border-left: 5px solid #16CC6B;
            background: #182B18;
            font-weight: 500;
        }
        .stepper-step.pending {
            color: #bdbdbd;
        }
        .stepper-icon {
            font-size: 1.12rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.4em;
        }
        .jump-btn {
            border: none;
            background: none;
            color: #3A7DF4;
            font-size: 1.16rem;
            margin-right: 5px;
            cursor: pointer;
            border-radius: 4px;
            height: 27px; width: 27px;
            display: flex; align-items: center; justify-content: center;
            transition: background 0.15s;
        }
        .jump-btn:hover {
            background: #213e6533;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="stepper-container">', unsafe_allow_html=True)
    for i, (step, icon, step_key) in enumerate(steps):
        # Determine step status with better logic
        status = get_step_status(i, current_step)
        
        # Check session state for more granular status
        if i < current_step:
            state_class = "stepper-step completed"
            step_icon = "✅"
            status_text = "Completed"
        elif i == current_step:
            state_class = "stepper-step active"
            step_icon = icon
            status_text = "In Progress"
        else:
            state_class = "stepper-step pending"
            step_icon = "•"
            status_text = "Not Started"

        cols = st.columns([10, 1], gap="small")
        with cols[0]:
            st.markdown(
                f'<div class="stepper-row" title="{status_text}">'
                f'<div class="{state_class}">'
                f'<span class="stepper-icon">{step_icon}</span>'
                f'{step}'
                f'</div></div>',
                unsafe_allow_html=True
            )
        with cols[1]:
            if i != current_step:
                if st.button("➡️", key=f"jump_{i}", help=f"Jump to {step}", use_container_width=True):
                    st.session_state.current_step = i
                    st.rerun()
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def apply_feature_suggestions(suggestions, source_df_key="uploaded_df", target_df_key="df"):
    """
    Apply a list of feature-suggestion actions to the dataframe stored in session_state.
    Stays in current step and shows inline preview - does NOT force navigation.
    
    Args:
        suggestions: list of dicts with transformation actions
        source_df_key: fallback dataframe key if target doesn't exist
        target_df_key: primary working dataframe key to update
    
    Returns:
        tuple: (success: bool, message: str, preview_df: DataFrame or None)
    """
    df = st.session_state.get(target_df_key) or st.session_state.get(source_df_key)
    if df is None:
        return False, "No dataset available to apply suggestions.", None
    
    if df.empty:
        return False, "Dataset is empty. Cannot apply transformations.", None
    
    if not suggestions or len(suggestions) == 0:
        return False, "No suggestions provided to apply.", None

    original_shape = df.shape
    changed_columns = set()
    
    try:
        # Track changes for preview
        for suggestion in suggestions:
            action = suggestion.get("action", "")
            
            if action == "drop":
                cols = [c for c in suggestion.get("cols", []) if c in df.columns]
                if cols:
                    df = df.drop(columns=cols)
                    changed_columns.update(cols)
                    
            elif action == "fillna":
                col = suggestion.get("col")
                strategy = suggestion.get("strategy", "mean")
                value = suggestion.get("value")
                
                if col in df.columns:
                    if value is not None:
                        df[col] = df[col].fillna(value)
                    elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == "mode":
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
                    elif strategy == "forward":
                        df[col] = df[col].fillna(method='ffill')
                    elif strategy == "backward":
                        df[col] = df[col].fillna(method='bfill')
                    changed_columns.add(col)
                    
            elif action == "rename":
                mapping = suggestion.get("mapping", {})
                valid_mapping = {k: v for k, v in mapping.items() if k in df.columns}
                if valid_mapping:
                    df = df.rename(columns=valid_mapping)
                    changed_columns.update(valid_mapping.keys())
                    
            elif action == "cast":
                col = suggestion.get("col")
                dtype = suggestion.get("dtype")
                if col in df.columns and dtype:
                    try:
                        df[col] = df[col].astype(dtype)
                        changed_columns.add(col)
                    except Exception as e:
                        st.warning(f"Could not cast {col} to {dtype}: {str(e)}")
                        
            elif action == "encode":
                col = suggestion.get("col")
                method = suggestion.get("method", "label")
                if col in df.columns:
                    if method == "label":
                        df[col] = pd.Categorical(df[col]).codes
                    elif method == "onehot":
                        dummies = pd.get_dummies(df[col], prefix=col)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    changed_columns.add(col)
                    
            elif action == "derive":
                new_col = suggestion.get("new_col")
                expression = suggestion.get("expression")
                if new_col and expression:
                    try:
                        # Safe eval with limited scope (only column names allowed)
                        df[new_col] = eval(expression, {"__builtins__": {}}, df.to_dict('series'))
                        changed_columns.add(new_col)
                    except Exception as e:
                        st.warning(f"Could not derive {new_col}: {str(e)}")

        # Verify dataframe is still valid
        if df is None or df.empty:
            return False, "Transformations resulted in an empty dataframe.", None

        # Initialize or update data history for undo functionality
        if "data_history" not in st.session_state:
            st.session_state["data_history"] = []
            orig = st.session_state.get(source_df_key)
            if orig is not None:
                st.session_state["data_history"].append(orig.copy())

        # Add snapshot to history
        st.session_state["data_history"].append(df.copy())
        
        # Update working dataframe
        st.session_state[target_df_key] = df.copy()
        st.session_state["features_applied"] = True
        st.session_state["last_changed_columns"] = list(changed_columns)
        
        # Build success message
        new_shape = df.shape
        changes_msg = f"Applied {len(suggestions)} transformation(s). "
        changes_msg += f"Shape changed from {original_shape} to {new_shape}. "
        if changed_columns:
            changes_msg += f"Modified columns: {', '.join(list(changed_columns)[:5])}"
            if len(changed_columns) > 5:
                changes_msg += f" and {len(changed_columns) - 5} more"
        
        # Return preview (first 10 rows)
        preview_df = df.head(10).copy()
        return True, changes_msg, preview_df
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error details: {error_details}")
        return False, f"Error applying suggestions: {str(e)}", None

def sticky_action_bar(
    apply_label=None,
    on_apply=None,
    show_preview=True,
    on_preview=None,
    show_undo=True,
    on_undo=None,
    help_text=None,
    key_prefix="action",
    return_to_preview=False
):
    """
    Display a sticky action bar at the bottom of the page.
    
    Args:
        apply_label: Text for apply button (None to hide)
        on_apply: Callback for apply action
        show_preview: Show preview button
        on_preview: Callback for preview (inline preview, not navigation)
        show_undo: Show undo button
        on_undo: Callback for undo
        help_text: Help text to show
        key_prefix: Unique prefix for button keys
        return_to_preview: If True, navigate to Smart Preview after apply (default: False)
    """
    st.markdown(
        """
        <style>
        .sticky-bar {
            position: fixed; 
            bottom: 0; 
            left: 0; 
            width: 100%;
            background: linear-gradient(180deg, rgba(24, 28, 41, 0.95) 0%, rgba(24, 28, 41, 0.98) 100%);
            backdrop-filter: blur(10px);
            padding: 12px 0; 
            z-index: 999;
            border-top: 1px solid rgba(58, 125, 244, 0.2);
            box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.15);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        st.markdown('<div class="sticky-bar">', unsafe_allow_html=True)
        cols = st.columns([1, 2, 2, 2, 1])

        # Preview button (inline preview modal)
        if show_preview and on_preview:
            if cols[1].button("👁️ Quick Preview", key=f"{key_prefix}_preview", use_container_width=True):
                on_preview()

        # Apply button (stays in current step by default)
        if apply_label:
            if cols[2].button(f"✅ {apply_label}", key=f"{key_prefix}_apply", use_container_width=True, type="primary"):
                if on_apply:
                    on_apply()
                # Only navigate if explicitly requested
                if return_to_preview:
                    st.session_state.current_step = 0  # Go to Upload/Smart Preview
                    st.rerun()

        # Undo button
        if show_undo and on_undo:
            if cols[3].button("↩️ Undo Last", key=f"{key_prefix}_undo", use_container_width=True):
                on_undo()

        # Help button
        if help_text:
            if cols[4].button("❓", key=f"{key_prefix}_help", use_container_width=True):
                st.info(help_text)
                
        st.markdown('</div>', unsafe_allow_html=True)

def preview_modal(changes_summary, preview_df, changed_cols=None, on_open_full=None, on_close=None, key_prefix="preview"):
    """
    Display an inline modal showing preview of changes.
    Highlights changed columns if provided.
    """
    st.markdown("---")
    st.markdown("### 🔍 Preview of Changes")
    
    # Show summary in info box
    st.info(changes_summary)
    
    # Show dataframe with optional highlighting
    if preview_df is not None and not preview_df.empty:
        if changed_cols and len(changed_cols) > 0:
            st.markdown(f"**Highlighted columns:** {', '.join(changed_cols[:10])}")
            try:
                styled_df = highlight_changes(preview_df, changed_cols)
                st.dataframe(styled_df, use_container_width=True)
            except Exception as e:
                # Fallback to regular dataframe if styling fails
                st.dataframe(preview_df, use_container_width=True)
        else:
            st.dataframe(preview_df, use_container_width=True)
        
        # Show shape info
        st.caption(f"Showing {len(preview_df)} rows × {len(preview_df.columns)} columns (preview)")
    else:
        st.warning("⚠️ No preview data available. The dataframe might be empty.")
    
    # Add option to navigate to Smart Preview for full analysis
    st.markdown("")  # Spacing
    button_cols = st.columns([1, 2, 2, 1])
    with button_cols[1]:
        if on_open_full:
            st.button(
                "📊 Open Full Smart Preview",
                key=f"{key_prefix}_open_full",
                use_container_width=True,
                on_click=on_open_full
            )
    with button_cols[2]:
        if on_close:
            st.button(
                "✖ Close Preview",
                key=f"{key_prefix}_close",
                use_container_width=True,
                on_click=on_close
            )
    
    st.markdown("---")

def undo_last_change():
    """Revert to previous state in data history."""
    if "data_history" in st.session_state and len(st.session_state.data_history) > 1:
        st.session_state.data_history.pop()
        st.session_state.df = st.session_state.data_history[-1].copy()
        st.session_state["features_applied"] = False
        st.session_state.pop("last_changed_columns", None)
        st.success("✅ Reverted to previous state.")
        st.rerun()
    else:
        st.warning("⚠️ No changes to undo.")

def highlight_changes(df, changed_cols=None):
    """Apply yellow highlighting to changed columns in dataframe."""
    if not changed_cols:
        return df
    
    def highlight_col(col):
        highlight_style = 'background-color: #FFF59D; color: #111111;'
        return [highlight_style if col.name in changed_cols else '' for _ in col]
    
    return df.style.apply(highlight_col, axis=0)

def show_data_change_notification():
    """
    Show a notification banner when data has been modified via feature suggestions.
    This appears in Smart Preview to inform users that data has changed.
    """
    if st.session_state.get("features_applied"):
        changed_cols = st.session_state.get("last_changed_columns", [])
        
        st.markdown(
            """
            <style>
            .data-changed-banner {
                background: linear-gradient(135deg, #3A7DF4 0%, #5B8FFF 100%);
                color: white;
                padding: 16px 20px;
                border-radius: 12px;
                margin: 16px 0;
                display: flex;
                align-items: center;
                gap: 12px;
                font-weight: 500;
                box-shadow: 0 4px 12px rgba(58, 125, 244, 0.3);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        cols_text = f" (Modified: {', '.join(changed_cols[:5])})" if changed_cols else ""
        
        st.markdown(
            f"""
            <div class="data-changed-banner">
                <span style="font-size: 1.5rem;">🔄</span>
                <div>
                    <strong>Data Updated!</strong><br>
                    Feature transformations have been applied{cols_text}. 
                    The preview below reflects your latest changes.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def onboarding_tour():
    with st.expander("🚀 Take a Quick Tour!", expanded=True):
        st.markdown("""
        # 👋 Welcome to **DataMimicAI**

        Get from raw data to high‑quality synthetic datasets in a few focused steps.

        ---
        ## 1) 📁 Upload or Try Demo Data
        - Use the **Data Upload** tab to add your CSV
        - Or switch on **Demo Mode** in the sidebar to explore instantly

        ## 2) 🧐 Smart Preview (Auto‑analysis)
        - Instant overview: shape, column types, missing values, basic stats
        - Quickly spot schema issues before moving ahead

        ## 3) 🔍 Explore, Engineer, Analyze
        Inside **Data Exploration** you'll find:
        - 📄 **Data Profiling** — summary, types, distributions
        - 🧠 **Feature Suggestions** — smart ideas for transformations
        - 📊 **Correlation** — relationships at a glance
        - ⚠️ **Outliers & Drift** — detect anomalies and potential drift

        ## 4) ⚙️ Generate Synthetic Data
        Choose the approach that fits your needs:
        - 🚀 **Standard Models (SDV)** — CTGAN, TVAE, GaussianCopula
        - 💎 **Advanced / AutoML (SynthCity)** — single‑model tuning or best‑model selection
        - ✍️ **LLM‑Powered** — prompt/schema‑guided generation (experimental)

        ## 5) ✅ Validate & Refine
        - Quality report and visual comparisons (original vs synthetic)
        - Track versions, review scores, and **refine iteratively** with recommendations

        ---
        ### 💡 Tips
        - Use the **sidebar stepper** to jump between steps anytime
        - The **sticky action bar** shows context‑aware actions with **inline previews**
        - **Apply suggestions** stays in Data Exploration - preview changes inline before moving forward
        - **Quick Actions** in the sidebar let you download original or synthetic data
        - Try **Demo Mode** for fast trials — no upload required
        """, unsafe_allow_html=True)

        # Quick CTA buttons to jump to key steps
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Go to Upload", key="tour_go_upload"):
            st.session_state.current_step = 0
            st.rerun()
        if c2.button("Explore Data", key="tour_go_explore"):
            st.session_state.current_step = 1
            st.rerun()
        if c3.button("Generate Data", key="tour_go_generate"):
            st.session_state.current_step = 2
            st.rerun()
        if c4.button("Validate & Refine", key="tour_go_validate"):
            st.session_state.current_step = 3
            st.rerun()


def sticky_section_header(title, icon=None):
    """Render a sticky top header with consistent spacing for main page sections."""
    icon_html = f"{icon} " if icon else ""

    st.markdown(
        f"""
        <div class="sticky-top">
            <h1 class="main-page-heading">{icon_html}{title}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_feature_highlights():
    """Display a 'What's New' section showcasing platform features."""
    with st.expander("✨ What's New in DataMimicAI", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            show_new_feature_badge("Inline Feature Preview")
            st.write("Apply suggestions and preview changes without leaving Data Exploration")
            
            st.markdown("---")
            show_new_feature_badge("Advanced AutoML Models")
            st.write("Use SynthCity's best-model selector for optimal results")
            if st.button("Try Now →", key="try_automl"):
                st.session_state.current_step = 1
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 🎯 Smart Preview")
            st.write("Get instant data quality insights before generation")
            
        with col2:
            st.markdown("### 📊 Enhanced Visualizations")
            st.write("Compare distributions and validate synthetic data quality")
            
            st.markdown("---")
            st.markdown("### 🔄 Seamless Workflow")
            st.write("Stay in context with inline previews and smart navigation")

def quick_actions_panel():
    """Display quick action shortcuts in sidebar."""
    st.markdown("### 🚀 Quick Actions")
    
    # Download Original Data
    if st.session_state.get('file_id') and st.session_state.get('uploaded_df') is not None:
        df = st.session_state.uploaded_df
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Original Data",
            data=csv,
            file_name="original_data.csv",
            mime="text/csv",
            key="download_original",
            help="Download the original uploaded dataset"
        )
    
    # Download Synthetic Data
    if st.session_state.get('generated_file_id') and st.session_state.get('df') is not None:
        df_synthetic = st.session_state.df
        csv_synthetic = df_synthetic.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Synthetic Data",
            data=csv_synthetic,
            file_name="synthetic_data.csv",
            mime="text/csv",
            key="download_synthetic",
            help="Download the generated synthetic dataset"
        )
    
    # Jump to Results
    if st.session_state.get('generated_file_id'):
        if st.button("📊 Jump to Visualization", key="jump_to_viz", help="View visualization results"):
            st.session_state.current_step = 3
            st.rerun()
    
    # Recently Used (placeholder for future implementation)
    st.markdown("---")
    st.markdown("**Recently Used:**")
    recent_files = st.session_state.get('recent_files', [])
    if not recent_files:
        st.caption("No recent files")
    else:
        for file in recent_files[:3]:
            st.caption(f"• {file}")

def platform_settings_panel():
    """Display platform settings and feature toggles."""
    with st.expander("⚙️ Platform Settings", expanded=False):
        st.markdown("### Feature Toggles")
        
        # Initialize settings in session state
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'experimental_features': False,
                'advanced_params_default': False,
                'auto_save_session': False,
                'show_recommendations': True
            }
        
        settings = st.session_state.settings
        
        settings['experimental_features'] = st.checkbox(
            "Enable experimental features",
            value=settings.get('experimental_features', False),
            help="Activate beta features and new capabilities"
        )
        
        settings['advanced_params_default'] = st.checkbox(
            "Show advanced parameters by default",
            value=settings.get('advanced_params_default', False),
            help="Always expand advanced configuration options"
        )
        
        settings['auto_save_session'] = st.checkbox(
            "Auto-save session state",
            value=settings.get('auto_save_session', False),
            help="Automatically preserve your work between sessions"
        )
        
        settings['show_recommendations'] = st.checkbox(
            "Show smart recommendations",
            value=settings.get('show_recommendations', True),
            help="Display contextual tips and suggestions"
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        if col1.button("💾 Save Settings", key="save_settings"):
            st.session_state.settings = settings
            st.success("✅ Settings saved!")
        
        if col2.button("🔄 Reset to Defaults", key="reset_settings"):
            st.session_state.settings = {
                'experimental_features': False,
                'advanced_params_default': False,
                'auto_save_session': False,
                'show_recommendations': True
            }
            st.success("✅ Reset to defaults!")
            st.rerun()

def smart_preview_section(df, file_id):
    """Enhanced Smart Preview with data change notifications."""
    # Show notification if data was modified
    show_data_change_notification()
    
    if file_id and df is not None:
        st.markdown("### Quick Data Overview")
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            st.markdown("**Shape:**")
            st.code(f"{df.shape[0]} rows × {df.shape[1]} columns", language='markdown')
            st.markdown("**Columns:**")
            st.code(", ".join(df.columns[:8]) + (", ..." if len(df.columns) > 8 else ""), language='markdown')
        with col2:
            st.markdown("**Detected Types:**")
            # Convert dtype objects to strings so PyArrow/Streamlit can serialize the table
            try:
                types_df = df.dtypes.astype(str).rename('Type').to_frame()
            except Exception:
                # Fallback: coerce by mapping str() to each value
                types_df = df.dtypes.rename('Type').to_frame()
                types_df['Type'] = types_df['Type'].map(lambda v: str(v))
            st.dataframe(types_df.head(8), use_container_width=True)
            st.markdown("*Full types shown on EDA page*")
        with col3:
            st.markdown("**Missing Values:**")
            miss = df.isnull().sum().sum()
            st.metric("Total missing", miss)
            st.markdown("**Duplicates:**")
            dups = df.duplicated().sum()
            st.metric("Duplicates", dups)

        # --- QUICK QUALITY FEEDBACK ---
        feedback = []
        if df.isnull().sum().sum() > 0:
            feedback.append("⚠️ Data has missing values.")
        if df.duplicated().sum() > 0:
            feedback.append("⚠️ Duplicate rows found.")
        if df.select_dtypes(include='object').nunique().max() == 1:
            feedback.append("ℹ️ Some columns might be constants.")

        if feedback:
            # Show warnings if there are issues
            st.info("\n".join(feedback))
        else:
            # 🎉 Extra celebratory callout for perfect data!
            st.markdown(
                """
                <style>
                .datamimic-perfect {
                    margin-top: 1em;
                    padding: 1.25rem;
                    border-radius: 1.25rem;
                    border: 1px solid #5fca75;
                    font-weight: 500;
                    font-size: 1.15rem;
                    background-color: rgba(90,210,120,0.10);
                    color: inherit;
                }
                @media (prefers-color-scheme: dark) {
                  .datamimic-perfect {
                    background-color: rgba(90,210,120,0.15);
                    color: #fff;
                  }
                }
                </style>
                <div class="datamimic-perfect">
                  <span style="font-size:1.5rem;">🎉</span> <b>Ready to Generate!</b><br>
                  Your data looks perfect. You can move to the next step to generate synthetic data.
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")
        with st.expander("🔎 Show Data Sample", expanded=False):
            # Highlight changed columns if features were applied
            changed_cols = st.session_state.get("last_changed_columns", [])
            if changed_cols and len(changed_cols) > 0:
                styled_df = highlight_changes(df.head(10), changed_cols)
                st.dataframe(styled_df, use_container_width=True)
                st.caption(f"Yellow highlight shows modified columns: {', '.join(changed_cols[:5])}")
            else:
                st.dataframe(df.head(10), use_container_width=True)

        st.info("➡️ Switch to **Generation** to create synthetic data.")
    elif file_id and df is None:
        st.error("Error: File uploaded but no data found! Please re-upload or try a different file.")
    else:
        st.info("Please upload your dataset in the **Data Upload** tab.")