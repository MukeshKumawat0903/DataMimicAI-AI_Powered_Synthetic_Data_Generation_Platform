# frontend/helpers/generation.py
""""
Generation tab for DataMimicAI platform.
Handles algorithm selection, parameter input, and synthetic data generation.
"""
import streamlit as st
import pandas as pd
import requests
import io
import os
import frontend_config as config

API_BASE = os.getenv("API_URL", "http://localhost:8000") 

def show_generation_controls():
    """
    Entry point for the Generation tab. 
    Displays algorithm selector, parameter inputs, and handles data generation.
    """
    # st.header("Generation Settings")
    _show_algorithm_selector()
    selected_algorithm = st.session_state.get("selected_algorithm", None)
    if not selected_algorithm:
        st.info("Please select a synthesis algorithm to continue.")
        return

    _show_algorithm_info(selected_algorithm)
    gen_params = _show_algorithm_parameters(selected_algorithm)
    _handle_generation_request(selected_algorithm, gen_params)


def _show_algorithm_selector():
    """Internal: Render algorithm selection box and save choice in session state."""
    if "selected_algorithm" not in st.session_state:
        st.session_state.selected_algorithm = list(config.ALGORITHM_INFO.keys())[1]

    algorithm = st.selectbox(
        "Synthesis Algorithm",
        list(config.ALGORITHM_INFO.keys()),
        index=list(config.ALGORITHM_INFO.keys()).index(st.session_state.selected_algorithm),
        format_func=lambda x: f"{x} - {config.ALGORITHM_INFO[x]['use']}"
    )
    st.session_state.selected_algorithm = algorithm
    st.divider()


def _show_algorithm_info(algorithm):
    """Internal: Show algorithm description and comparison table."""
    info = config.ALGORITHM_INFO[algorithm]
    st.info(f"**{algorithm}:** {info['desc']}  \n*{info['use']}*")
    with st.expander("🔍 Compare All Algorithms", expanded=False):
        st.table(pd.DataFrame(config.ALGORITHM_INFO).T)
    st.divider()


def _show_algorithm_parameters(algorithm):
    """
    Internal: Show and collect parameters needed for the selected algorithm.
    Returns a dict of parameter values.
    """
    params = {}
    if algorithm == "PARS":
        sequence_key_candidates = ['Symbol']
        valid_sequence_columns = [
            col for col in st.session_state.get("original_columns", [])
            if col in sequence_key_candidates
        ]

        if not valid_sequence_columns:
            st.error(
                "PAR requires one of these columns as sequence key: " +
                ", ".join(sequence_key_candidates)
            )
            return None

        params["num_sequences"] = st.number_input(
            "Number of Sequences",
            min_value=1,
            value=1,
            step=1
        )
        params["sequence_length"] = st.number_input(
            "Sequence Length",
            min_value=100,
            value=5000,
            step=100
        )
    else:
        params["num_rows"] = st.number_input(
            "Number of Rows to Generate",
            min_value=100,
            value=1000,
            step=500,
            format="%d"
        )
    return params


def _handle_generation_request(algorithm, gen_params):
    """
    Internal: Handle the Generate button logic, perform API call, and handle response.
    """
    if gen_params is None:
        return

    if st.button("Generate Synthetic Data", key="generate_btn"):
        with st.spinner(_get_generation_message(algorithm, gen_params)):
            try:
                params = {
                    "file_id": st.session_state.file_id,
                    "algorithm": algorithm
                }
                params.update(gen_params)
                response = requests.post(
                    f"{API_BASE}/generate",
                    params=params
                )

                if response.status_code == 200:
                    st.session_state.generated_file_id = st.session_state.file_id
                    synthetic_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                    st.session_state.data_columns = synthetic_df.columns.tolist()
                    st.success("Generation completed!")

                    st.divider()
                    st.download_button(
                        "Download Synthetic Data",
                        data=response.content,
                        file_name="synthetic_data.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Generation failed: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {str(e)}")


def _get_generation_message(algorithm, gen_params):
    """
    Internal: Returns an informative message for the generation progress spinner.
    """
    if algorithm == "PARS":
        return f"Generating {gen_params['num_sequences']} sequences of length {gen_params['sequence_length']}..."
    return f"Generating {gen_params['num_rows']} rows with {algorithm}..."
