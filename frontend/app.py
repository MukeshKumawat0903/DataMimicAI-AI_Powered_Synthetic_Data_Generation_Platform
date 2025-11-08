import streamlit as st
import pandas as pd
import os
import base64
from pathlib import Path

# UI Patterns
from helpers.ui_patterns import (
    sidebar_stepper,
    sticky_action_bar,
    preview_modal,
    undo_last_change,
    highlight_changes,
    onboarding_tour,
    sticky_section_header,
    smart_preview_section,
    quick_actions_panel,
    platform_settings_panel,
    status_badge,
    show_new_feature_badge
)

from helpers.file_upload import handle_demo_mode, handle_file_upload
from helpers.generation import show_generation_controls
from helpers.visualization import show_visualization
from helpers.validation import show_validation_and_refinement
# from helpers.roadmap import show_feature_placeholders
from helpers.eda_feature_eng.expander_data_profiling import expander_data_profiling
from helpers.eda_feature_eng.expander_correlation import expander_correlation
from helpers.eda_feature_eng.expander_feature_suggestions import expander_feature_suggestions
from helpers.eda_feature_eng.expander_outlier_and_drift import expander_outlier_and_drift
from helpers.eda_feature_eng.expander_privacy_audit import expander_privacy_audit
from helpers.advanced_generation import show_advanced_generation_controls

from frontend_config import API_BASE as CONFIG_API_BASE

# Use a runtime API base which can be overridden per-session via the UI
API_BASE = os.getenv("API_URL", CONFIG_API_BASE)

def set_step(n):
    st.session_state.current_step = n
    st.rerun()

def show_eda_and_feature_engineering():
    # Reordered tabs for better workflow: Profile → Suggest → Correlate → Validate → Privacy
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Data Profiling",
        "💡 Feature Suggestions", 
        "📊 Correlation",
        "⚠️ Outlier & Drift",
        "🔒 Privacy Audit"
    ])
    with tab1:
        expander_data_profiling()
    with tab2:
        expander_feature_suggestions()
    with tab3:
        expander_correlation()
    with tab4:
        expander_outlier_and_drift()
    with tab5:
        expander_privacy_audit()

def main():
    st.set_page_config(page_title="DataMimicAI Synthetic Data Platform", layout="wide")

    # --- Session state initialization ---
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'file_id' not in st.session_state:
        st.session_state.file_id = None
    if 'generated_file_id' not in st.session_state:
        st.session_state.generated_file_id = None
    if 'data_columns' not in st.session_state:
        st.session_state.data_columns = []
    if 'original_columns' not in st.session_state:
        st.session_state.original_columns = []
    if 'data_history' not in st.session_state:
        st.session_state.data_history = []
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    # allow per-session override of API base (editable in sidebar)
    if 'custom_api' not in st.session_state:
        st.session_state.custom_api = None
    if 'edit_api' not in st.session_state:
        st.session_state.edit_api = False
    
    # Refinement engine state (Phase 4 & 5)
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'current_generation_version' not in st.session_state:
        st.session_state.current_generation_version = 0
    if 'quality_scores_history' not in st.session_state:
        st.session_state.quality_scores_history = []
    if 'refinement_recommendations' not in st.session_state:
        st.session_state.refinement_recommendations = []
    if 'generation_parameters' not in st.session_state:
        st.session_state.generation_parameters = {}
    if 'parameter_changes_log' not in st.session_state:
        st.session_state.parameter_changes_log = []

    st.markdown("""
    <style>
    .sticky-top {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #181a20;
        padding: 0.3rem 0 0.2rem;
        border-bottom: 1px solid #23242b;
    }
    .sticky-top-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
        display: flex;
        align-items: center;
        gap: 0.6em;
    }
    .sticky-top-caption {
        font-size: 1.01rem;
        margin: 0;
        color: #a0a0a0;
    }
    </style>
    """, unsafe_allow_html=True)


    # --- Sidebar ---
    # 1️⃣  Either paste the data-URI directly …
    EMOJI_PNG = Path(__file__).parent / "logo_DataMimicAI.png"

    def png_to_data_uri(png_path: Path) -> str:
        """Return a data:image/png;base64,… URI for the given file."""
        with png_path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # Build the data-URI, or fallback to a simple emoji if the file is missing
    try:
        DATA_URI = png_to_data_uri(EMOJI_PNG)
        logo_html = f'<img src="{DATA_URI}" width="56" height="56" style="margin:0;" />'

    except FileNotFoundError:
        logo_html = "🧠"   # fallback icon

    # Sidebar branding
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:28px;">
                {logo_html}
                <span style="font-weight:900;font-size:1.35rem;letter-spacing:0.5px;">
                    DataMimicAI
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        sidebar_stepper(st.session_state.current_step)
        st.divider()
        
        # Quick Actions Panel
        quick_actions_panel()
        st.divider()
        
        st.header("Configuration")
        demo_mode = st.checkbox("Use Demo Data", help="Load sample data for quick testing")
        st.markdown("---")
        st.subheader("API")
        current_api = st.session_state.custom_api or API_BASE
        st.write("Current API URL:")
        st.code(current_api)
        if st.button("Edit API URL"):
            st.session_state.edit_api = True
        if st.session_state.edit_api:
            new_api = st.text_input("API URL", value=current_api)
            cols = st.columns([1,1])
            if cols[0].button("Save API URL"):
                st.session_state.custom_api = new_api.strip() if new_api else None
                st.session_state.edit_api = False
                st.experimental_rerun()
            if cols[1].button("Cancel"):
                st.session_state.edit_api = False
        if st.button("Reset App", key="reset_app", help="Clear all data and restart from Step 0"):
            for key in [
                'file_id', 'generated_file_id', 'data_columns', 'original_columns',
                'data_history', 'df'
            ]:
                st.session_state[key] = None if 'id' in key or key == 'df' else []
            st.session_state.current_step = 0
            st.rerun()
        
        # Quick API health check tool
        st.markdown("---")
        st.subheader("API Health")
        cols_api = st.columns([3,1])
        if cols_api[1].button("Test", key="test_api", help="Check if the API is reachable"):
            api_to_test = st.session_state.custom_api or API_BASE
            try:
                import requests
                with st.spinner("Testing API connection..."):
                    resp = requests.get(f"{api_to_test.rstrip('/')}/health", timeout=5)
                if resp.status_code == 200:
                    status_badge("API Online", "complete")
                else:
                    status_badge(f"Status {resp.status_code}", "warning")
            except Exception as e:
                status_badge("API Offline", "error")
                st.error(f"Connection failed: {str(e)}")
        
        # Platform Settings Panel
        st.markdown("---")
        platform_settings_panel()

    step = st.session_state.current_step
    step_names = [
        "Data Exploration",  # Renamed from "EDA & Feature Eng."
        "Generate Synthetic Data",  # Renamed from "Synthetic Data Generation"
        "Validate & Refine",  # Renamed from "Visualization"
        # "Roadmap"
    ]
    n_steps = len(step_names) + 1  # +1 for Roadmap/final

    # ---- Step 0: DATA UPLOAD ----
    if step == 0:
        sticky_section_header(
            "Data Upload",
            subtitle="Upload your dataset to get started.",
            icon="📁"
        )
        st.divider()

        # Put Tour last so upload tab is always first after rerun
        tab_titles = [
            "📁 Data Upload",
            "🧐 Smart Preview (Auto-analysis)",
            "🚀 Take a Quick Tour!"
            ]
        
        upload_tab, preview_tab, tour_tab = st.tabs(tab_titles)

        # --- 1. Data Upload Tab ---
        with upload_tab:
            if demo_mode:
                handle_demo_mode()
            else:
                handle_file_upload()
            st.caption("Accepted: CSV. Try demo mode for sample data.")
            if st.session_state.get('uploaded_df') is not None:
                st.success("File uploaded successfully! " \
                "👉 Now click the **Smart Preview** tab above to analyze your data.")
            else:
                st.info("➡️ Switch to **Smart Preview** to analyze your data.")

        # --- 2. Smart Preview Tab ---
        with preview_tab:
            file_id = st.session_state.get('file_id')
            df = st.session_state.get('uploaded_df')
            smart_preview_section(df, file_id)

        # --- 3. Quick Tour Tab ---
        with tour_tab:
            onboarding_tour()
            st.info("➡️ Switch to **Data Upload** to begin.")

    # ---- Step 1: DATA EXPLORATION (Phase 2: Swapped with old Step 2) ----
    elif step == 1:
        st.markdown('<div id="custom-main-scroll">', unsafe_allow_html=True)
        sticky_section_header(
            "Data Exploration",
            subtitle="Understand your data to make informed generation choices",
            icon="🔍"
        )
        st.divider()
        
        if not st.session_state.file_id:
            st.warning("📁 Please upload your dataset in Step 0 (Data Upload) first.")
        else:
            # Show EDA and Feature Engineering
            show_eda_and_feature_engineering()
            if st.session_state.df is not None:
                st.dataframe(highlight_changes(st.session_state.df, changed_cols=None))
        
        sticky_action_bar(
            apply_label=None,
            on_apply=None,
            show_preview=True,
            on_preview=lambda: preview_modal(
                "Preview of EDA results.", 
                st.session_state.df.head() if st.session_state.df is not None else pd.DataFrame()
            ),
            show_undo=True,
            on_undo=undo_last_change,
            help_text="Explore, transform, and analyze your data here.",
            key_prefix="eda"
        )

        st.markdown('</div>', unsafe_allow_html=True)


    # ---- Step 2: GENERATE SYNTHETIC DATA (Phase 2: Swapped with old Step 1) ----
    elif step == 2:
        st.markdown('<div id="custom-main-scroll">', unsafe_allow_html=True)
        sticky_section_header(
            "Generate Synthetic Data",
            subtitle="Create high-quality synthetic datasets based on your data insights",
            icon="⚙️"
        )

        st.divider()
        
        if not st.session_state.file_id:
            st.warning("📁 Please complete Step 1 (Data Exploration) first to understand your data.")
        else:
            # Show the generation controls in a tabbed layout
            tabs = st.tabs([
                "🚀 Standard Models",
                "💎 Advanced / AutoML",
                # "🕒 Time Series Generator",
                # "🏭 Industry Simulators",
                "✍️ LLM-Powered"
            ])

            # --- 🚀 Standard Models ---
            with tabs[0]:
                st.caption(
                    "Use **SDV** for fast, reliable generation of tabular synthetic data. "
                    "Recommended for most users and general-purpose datasets.\n\n"
                    "**Models included:** CTGAN, TVAE, GaussianCopula"
                )
                show_generation_controls()
            
            # --- 💎 Advanced / AutoML ---
            with tabs[1]:
                st.caption(
                    "⚡ Unlock advanced synthetic data generation for tabular datasets with **SynthCity** models. "
                    "Benefit from high fidelity, privacy protection, and leading-edge AI. "
                    "Choose between single-model tuning or effortless **AutoML** for best-model selection.\n\n"
                    "**Available models:** TabDDPM, CTGAN, TVAE, PrivBayes, DP-GAN, PATE-GAN, ARF, and more."
                )

                show_advanced_generation_controls()

            
            # # --- 🕒 Time Series Generator ---
            # with tabs[2]:
            #     st.caption(
            #         "Dedicated tools for generating sequential or temporal data. "
            #         "Ideal for IoT, finance, healthcare, and any time-dependent datasets.\n\n"
            #         "**Models included:** TimeGAN, FourierFlows, TimeVAE, AR, VAR"
            #     )
            #     # show_time_series_controls(models=["TimeGAN", "FourierFlows", "TimeVAE", "AR", "VAR"])
            
            # # --- 🏭 Industry Templates ---
            # with tabs[3]:
            #     st.caption(
            #         "Domain-specific simulators for creating realistic, multi-table datasets.\n\n"
            #         "**Available Domains:**\n"
            #         "🏥 Healthcare (Synthea)\n"
            #         "💰 Finance (FinSynth)\n"
            #         "🛒 Retail (RetailSim)\n"
            #         "🚗 Automotive (Telematics)\n"
            #         "🏦 Banking (Credit scoring)\n"
            #         "🎓 Education (Student records)\n"
            #         "⚡ Energy (Smart meters)\n"
            #         "📶 Telecom (CDRs, call logs)"
            #     )
            #     st.markdown("🚧 *Coming Soon: Industry Templates with domain-specific wizards and presets*")

            
            # --- ✍️ LLM-Powered ---
            with tabs[2]:
                st.caption(
                    "Use Large Language Models (LLMs) like **TableGPT** and **OpenAI GPT-4o** to generate data "
                    "from natural language prompts or schema definitions.\n\n"
                    "Ideal for prototyping or generating novel datasets from text.\n"
                    "🚧 *Experimental Feature: Validate results carefully.*"
                )
                st.markdown("🚧 *Coming Soon: AI-powered prompt-based data generation*")

        sticky_action_bar(
            apply_label=None,
            on_apply=None,
            show_preview=True,
            on_preview=lambda: preview_modal(
                "Preview of generated data.",
                st.session_state.df.head() if st.session_state.df is not None else pd.DataFrame()
            ),
            show_undo=True,
            on_undo=undo_last_change,
            help_text="After generating, review your synthetic data here.",
            key_prefix="generation"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Step 3: VALIDATION & REFINEMENT (Phase 3: Restructured with 3-tab layout) ----
    elif step == 3:
        sticky_section_header(
            "Validate & Refine",
            subtitle="Compare, validate, and iteratively improve your synthetic data",
            icon="✅"
        )
        st.divider()
        
        # Use new 3-tab validation structure: Quality Report | Detailed Analysis | Iterative Refinement
        show_validation_and_refinement()
        
        sticky_action_bar(
            apply_label=None,
            on_apply=None,
            show_preview=False,
            show_undo=False,
            help_text="Assess quality, analyze deeply, and refine iteratively.",
            key_prefix="validation"
        )

    # # ---- Step 4: ROADMAP ----
    # elif step == 4:
    #     # st.header("🧩 Roadmap & Coming Soon")
    #     sticky_section_header(
    #         "Roadmap & Coming Soon",
    #         subtitle="Explore upcoming features and provide feedback.",
    #         icon="🧩"
    #     )

    #     st.divider()
    #     show_feature_placeholders()
    #     sticky_action_bar(
    #         apply_label=None,
    #         on_apply=None,
    #         show_preview=False,
    #         show_undo=False,
    #         help_text="Want to start over? Use the button above.",
    #         key_prefix="roadmap"
    #     )


    # ---- Sticky Previous (bottom left) ----
    if st.session_state.current_step > 0:
        st.markdown('<div id="nav-prev">', unsafe_allow_html=True)
        if st.button("⬅️ Previous", key="nav_prev_btn"):
            set_step(st.session_state.current_step - 1)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---- Sticky Next (bottom right, with step name) ----
    # Show Next if NOT on last step (Roadmap)
    if st.session_state.current_step < n_steps - 1:
        st.markdown('<div id="nav-next">', unsafe_allow_html=True)
        next_label = f"Next: {step_names[st.session_state.current_step]} ➡️"
        if st.button(next_label, key="nav_next_btn"):
            set_step(st.session_state.current_step + 1)
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer: (API display removed — API is available in Sidebar → Configuration)

if __name__ == "__main__":
    main()