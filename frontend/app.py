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
    smart_preview_section
)

from helpers.file_upload import handle_demo_mode, handle_file_upload
from helpers.generation import show_generation_controls
from helpers.visualization import show_visualization
# from helpers.roadmap import show_feature_placeholders
from helpers.eda_feature_eng.expander_data_profiling import expander_data_profiling
from helpers.eda_feature_eng.expander_correlation import expander_correlation
from helpers.eda_feature_eng.expander_feature_suggestions import expander_feature_suggestions
from helpers.eda_feature_eng.expander_outlier_and_drift import expander_outlier_and_drift
from helpers.eda_feature_eng.expander_eda_feedback_loop import expander_eda_feedback_loop
from helpers.advanced_generation import show_advanced_generation_controls

API_BASE = os.getenv("API_URL", "http://localhost:8000")

def set_step(n):
    st.session_state.current_step = n
    st.rerun()

def show_eda_and_feature_engineering():
    # Add sticky tabs using st.tabs (with sticky CSS set below)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Data Profiling", 
        "🔗 Correlation", 
        "💡 Feature Suggestions", 
        "⚠️ Outlier & Drift", 
        "🔁 Feedback Loop"
    ])
    with tab1:
        expander_data_profiling()
    with tab2:
        expander_correlation()
    with tab3:
        expander_feature_suggestions()
    with tab4:
        expander_outlier_and_drift()
    with tab5:
        expander_eda_feedback_loop()

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
        st.header("Configuration")
        demo_mode = st.checkbox("Use Demo Data")
        if st.button("Reset App", key="reset_app"):
            for key in [
                'file_id', 'generated_file_id', 'data_columns', 'original_columns',
                'data_history', 'df'
            ]:
                st.session_state[key] = None if 'id' in key or key == 'df' else []
            st.session_state.current_step = 0
            st.rerun()

    step = st.session_state.current_step
    step_names = [
        "Synthetic Data Generation",
        "EDA & Feature Eng.",
        "Visualization",
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

    # ---- Step 1: GENERATION ----
    elif step == 1:
        st.markdown('<div id="custom-main-scroll">', unsafe_allow_html=True)
        sticky_section_header(
            "Synthetic Data Generation",
            subtitle="Generate high-quality synthetic datasets tailored to your needs.",
            icon="⚙️"
        )
        st.divider()
        if not st.session_state.file_id:
            st.warning("Please upload or select a dataset in the 'Data Upload' step first.")
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


    # ---- Step 2: EDA & FEATURE ENGINEERING ----
    elif step == 2:
        st.markdown('<div id="custom-main-scroll">', unsafe_allow_html=True)
        sticky_section_header(
            "EDA & Feature Engineering",
            subtitle="AI-Powered Exploratory Data Analysis and Feature Engineering Tools",
            icon="🧠"
        )

        st.divider()
        if not st.session_state.file_id:
            st.warning("Upload data first in the 'Data Upload' step.")
        else:
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

    # ---- Step 3: VISUALIZATION ----
    elif step == 3:
        sticky_section_header(
            "Visualization",
            subtitle="Visualize your (or synthetic) data.",
            icon="📊"
        )
        st.divider()
        if not st.session_state.generated_file_id:
            st.info("Generate synthetic data first in the 'Generation' step.")
        else:
            show_visualization()  # Calls the tabbed UI as above
        sticky_action_bar(
            apply_label=None,
            on_apply=None,
            show_preview=False,
            show_undo=False,
            help_text="Visualize trends, compare real vs. synthetic data, and more.",
            key_prefix="visualization"
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

if __name__ == "__main__":
    main()