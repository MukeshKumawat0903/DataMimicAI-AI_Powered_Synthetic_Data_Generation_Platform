import streamlit as st
import pandas as pd
import requests
import io
import os
import frontend_config as config

API_BASE = os.getenv("API_URL", "http://localhost:8000")

def show_advanced_generation_controls():
    """
    Main entry point for Advanced/AutoML (SynthCity) tab.
    Keeps SDV-style dropdown with short user-friendly algorithm names, handles backend code mapping,
    and ensures the AutoML Best Model workflow works for both evaluation and final generation.
    """
    _show_synthcity_algorithm_selector()
    algo_code = st.session_state.get("advanced_selected_algorithm", None)  # always internal code e.g. "ctgan", "ddpm", or "AutoML Best Model"
    if not algo_code:
        st.info("Select a SynthCity algorithm to continue.")
        return

    _show_synthcity_algorithm_info(algo_code)
    gen_params = _show_synthcity_algorithm_parameters(algo_code)
    _handle_synthcity_generation_request(algo_code, gen_params)

def _show_synthcity_algorithm_selector():
    synthcity_algos = list(config.ALGORITHM_INFO_SYNTHCITY.keys())
    display_options = ["AutoML Best Model"] + [config.FRIENDLY_ALGO_LABELS[k] for k in synthcity_algos]
    algo_code_map = {config.FRIENDLY_ALGO_LABELS[k]: k for k in synthcity_algos}
    algo_code_map["AutoML Best Model"] = "AutoML Best Model"

    # On startup, fallback to first
    if "advanced_selected_algorithm_label" not in st.session_state:
        st.session_state.advanced_selected_algorithm_label = display_options[0]

    selected_label = st.selectbox(
        "SynthCity Generator",
        display_options,
        index=display_options.index(st.session_state.advanced_selected_algorithm_label)
    )
    st.session_state.advanced_selected_algorithm_label = selected_label
    st.session_state.advanced_selected_algorithm = algo_code_map[selected_label]
    st.divider()

def _show_synthcity_algorithm_info(algo_code):
    """Info/compare/performance blocks, SDV-style."""
    if algo_code == "AutoML Best Model":
        st.info("AutoML Best Model: Finds the optimal SynthCity generator for your dataset using robust statistical ranking.")
    else:
        info = config.ALGORITHM_INFO_SYNTHCITY[algo_code]
        st.info(f"**{algo_code.upper()}:** {info['desc']}\n\n*{info['use']}*")
    with st.expander("🔍 Compare All Advanced (SynthCity) Algorithms", expanded=False):
        st.table(pd.DataFrame(config.ALGORITHM_INFO_SYNTHCITY).T)
    with st.expander("💡 Performance Tips & System Requirements", expanded=False):
        st.info(
            "**Advanced Model Performance Guide:**\n\n"
            "🔧 **Resource Usage**: These models use significant CPU/GPU and memory\n\n"
            "⏱️ **Training Time**: Expect 1-10 minutes depending on epochs and data size\n\n"
            "🎯 **Epoch Recommendations:**\n"
            "• **Quick testing**: 10-50 epochs (faster, good quality)\n"
            "• **Production use**: 100-300 epochs (high quality)\n"
            "• **Best results**: 300+ epochs (longest time, best quality)\n\n"
            "💻 **System Tips**: Close other heavy applications for best performance"
        )
    st.divider()

def _show_synthcity_algorithm_parameters(algo_code):
    params = {}
    col1, col2 = st.columns(2)
    with col1:
        params["num_rows"] = st.number_input(
            "Number of Rows to Generate",
            min_value=100, value=1000, step=500, format="%d",
            key="advanced_num_rows"
        )
    with col2:
        params["epochs"] = st.slider(
            "Epochs", min_value=10, max_value=1000, value=100, step=10,
            key="advanced_epochs",
            help="Lower=faster, Higher=better quality"
        )
    return params

def _handle_synthcity_generation_request(algo_code, gen_params):
    if gen_params is None:
        return
    if st.button(f"Generate {'Best Model' if algo_code == 'AutoML Best Model' else algo_code.upper()}"):
        file_id = st.session_state.get("file_id", None)
        if not file_id:
            st.error("Upload or select a dataset first.")
            return

        if algo_code == "AutoML Best Model":
            # AutoML: Evaluate, then call .generate using best model code
            with st.spinner(f"Evaluating all advanced models with {gen_params['epochs']} epochs..."):
                params_eval = {
                    "file_id": file_id,
                    "target_column": st.session_state.get("target_column", "Survived"),
                    "epochs": gen_params["epochs"]
                }
                try:
                    r_eval = requests.post(f"{API_BASE}/evaluate_models", params=params_eval)
                    if r_eval.status_code != 200:
                        st.error(f"AutoML evaluation failed: {r_eval.text}")
                        return
                    result = r_eval.json()
                    st.success("AutoML: Best model selected!")
                    best_model_display = result["best_model"]
                    # Map possibly display-name best model to code (for generation)
                    best_model_code = config.DISPLAY_NAME_TO_ALGORITHM.get(best_model_display, best_model_display)
                    st.markdown(f"**AutoML picked:** `{best_model_code}` (trained with {gen_params['epochs']} epochs)")
                    st.table(pd.DataFrame([result["best_model_metrics"]]).T.rename(columns={0: "Score"}))
                except Exception as e:
                    st.error(f"Evaluation API error: {e}")
                    return

                with st.spinner(f"Generating data using **{best_model_code}**..."):
                    gen_params_gen = {
                        "file_id": file_id,
                        "algorithm": best_model_code,
                        "num_rows": gen_params["num_rows"],
                        "epochs": gen_params["epochs"]
                    }
                    try:
                        r_gen = requests.post(f"{API_BASE}/generate", params=gen_params_gen)
                        if r_gen.status_code == 200:
                            df = pd.read_csv(io.StringIO(r_gen.content.decode("utf-8")))
                            st.session_state.generated_file_id = file_id
                            st.session_state.data_columns = df.columns.tolist()
                            st.success("Best-model generation complete!")
                            st.download_button(
                                "Download AutoML Synthetic Data",
                                data=r_gen.content,
                                file_name="synthetic_data.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error(f"Generation failed: {r_gen.text}")
                    except Exception as e:
                        st.error(f"Generation API error: {e}")
        else:
            # Single algorithm case
            st.info(f"Generating data with {algo_code.upper()} using {gen_params['epochs']} epochs...")
            params = {
                "file_id": file_id,
                "algorithm": algo_code,
                "num_rows": gen_params["num_rows"],
                "epochs": gen_params["epochs"]
            }
            try:
                resp = requests.post(f"{API_BASE}/generate", params=params)
                if resp.status_code == 200:
                    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
                    st.session_state.generated_file_id = file_id
                    st.session_state.data_columns = df.columns.tolist()
                    st.success("Generation complete!")
                    st.download_button(
                        "Download Synthetic Data",
                        data=resp.content,
                        file_name="synthetic_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Generation failed: {resp.text}")
            except Exception as e:
                st.error(f"API Error: {e}")