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
            "tip": "Compare original vs synthetic data quality. Use the feedback loop to iteratively improve results",
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

def sticky_action_bar(
    apply_label=None,
    on_apply=None,
    show_preview=True,
    on_preview=None,
    show_undo=True,
    on_undo=None,
    help_text=None,
    key_prefix="action"
):
    """Display a sticky action bar at the bottom of the page."""
    st.markdown(
        """
        <style>
        .sticky-bar {
            position: fixed; bottom: 0; left: 0; width: 100%;
            background: #181C29CC; padding: 8px 0; z-index: 99;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        st.markdown('<div class="sticky-bar">', unsafe_allow_html=True)
        cols = st.columns(4)

    if apply_label:
        if cols[1].button(f"✅ {apply_label}", key=f"{key_prefix}_apply"):
            if on_apply: on_apply()

    # Add preview and undo as needed (uncomment if you want those features visible)
    # if show_preview and on_preview:
    #     if cols[0].button("👁️ Preview", key=f"{key_prefix}_preview"):
    #         on_preview()
    # if show_undo and on_undo:
    #     if cols[2].button("↩️ Undo", key=f"{key_prefix}_undo"):
    #         on_undo()
    # if help_text:
    #     if cols[3].button("❓ Help", key=f"{key_prefix}_help"):
    #         st.info(help_text)
    st.markdown('</div>', unsafe_allow_html=True)

def preview_modal(changes_summary, preview_df):
    st.markdown("### 🔍 Preview of Changes")
    st.info(changes_summary)
    st.dataframe(preview_df)

def undo_last_change():
    if "data_history" in st.session_state and len(st.session_state.data_history) > 1:
        st.session_state.data_history.pop()
        st.session_state.df = st.session_state.data_history[-1].copy()
        st.success("Reverted last change.")
    else:
        st.warning("No changes to undo.")

def highlight_changes(df, changed_cols=None):
    if not changed_cols:
        return df
    def highlight_col(col):
        return ['background-color: #FFF59D' if col.name in changed_cols else '' for _ in col]
    return df.style.apply(highlight_col, axis=0)

def onboarding_tour():
    with st.expander("🚀 Take a Quick Tour!", expanded=True):
        st.markdown("""
        # 👋 Welcome to **DataMimicAI!**

        **Start your journey in just a few clicks.**  
        *Here’s how it works:*

        - **📁 Upload or Try Demo Data**  
        Use the **Data Upload** tab to upload your CSV file — or explore instantly with demo data.

        - **🧐 Smart Preview**  
        Instantly preview your data’s shape, column types, and spot issues before generation.

        - **⚡ Generate Synthetic Data**  
        Go to the next step to create high-quality synthetic datasets, fast!

        - **🔬 Explore, Engineer, Analyze**  
        Dive into your data with built-in feature tools and easy visualizations.

        - **📊 Visualize Results**  
        Create quick charts and tables for deeper insights.

        # - **🗺️ Roadmap**  
        # See what’s coming next, and suggest your ideas!

        ---
        ### 💡 **Tips for a Smooth Experience**
        - Use the **sidebar** to jump between steps at any time.
        - The **sticky action bar** always shows your next options.
        - **Demo Mode:** Great for quick trials—no data needed.

        ---

        **Ready?**  
        👉 Head to the [Data Upload](#) tab and get started!
        """, unsafe_allow_html=True)


def sticky_section_header(title, subtitle=None, icon=None):
    """Render a sticky top header (title + optional subtitle) for main page sections."""
    # CSS block (only inject once per session)
    if not hasattr(st, "_sticky_header_css"):
        st.markdown("""
            <style>
            .sticky-top {
                position: -webkit-sticky;
                position: sticky;
                top: 0;
                z-index: 999;
                background: #181a20;
                padding: 0.5rem 0 0.4rem;
                border-bottom: 1px solid #2a2d34;
            }
            </style>
        """, unsafe_allow_html=True)
        st._sticky_header_css = True

    icon_html = f"{icon} " if icon else ""
    subtitle_html = f'<p style="margin:0;color:#a0a0a0;font-size:1.04rem;">{subtitle}</p>' if subtitle else ""

    st.markdown(
        f"""
        <div class="sticky-top">
            <h1 style="margin:0;">{icon_html}{title}</h1>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_feature_highlights():
    """Display a 'What's New' section showcasing platform features."""
    with st.expander("✨ What's New in DataMimicAI", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            show_new_feature_badge("Advanced AutoML Models")
            st.write("Use SynthCity's best-model selector for optimal results")
            if st.button("Try Now →", key="try_automl"):
                st.session_state.current_step = 1
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 🎯 Smart Preview")
            st.write("Get instant data quality insights before generation")
            
        with col2:
            show_new_feature_badge("EDA Feedback Loop")
            st.write("Iteratively refine synthetic data based on analysis")
            if st.button("Explore EDA →", key="explore_eda"):
                st.session_state.current_step = 2
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 📊 Enhanced Visualizations")
            st.write("Compare distributions and validate synthetic data quality")

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
            st.dataframe(df.head(10), use_container_width=True)

        st.info("➡️ Switch to **Generation** to create synthetic data.")
    elif file_id and df is None:
        st.error("Error: File uploaded but no data found! Please re-upload or try a different file.")
    else:
        st.info("Please upload your dataset in the **Data Upload** tab.")