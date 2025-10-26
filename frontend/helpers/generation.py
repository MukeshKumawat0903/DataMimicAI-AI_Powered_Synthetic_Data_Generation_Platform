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
import json
import frontend_config as config


def get_api_base():
    """Return per-session override or configured API base."""
    return st.session_state.get('custom_api') or os.getenv("API_URL") or config.API_BASE

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

    # Presets and novice/advanced toggle
    if "gen_preset" not in st.session_state:
        st.session_state.gen_preset = "Balanced"
    preset_options = ["Quick (small)", "Balanced", "High Fidelity"]
    preset = st.radio("Preset", preset_options, index=preset_options.index(st.session_state.gen_preset))
    st.session_state.gen_preset = preset

    advanced = st.checkbox("Show advanced options", value=False, key="gen_advanced")

    gen_params = _show_algorithm_parameters(selected_algorithm, preset=preset, advanced=advanced)
    _client_side_sanity_check(gen_params)
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


def _show_algorithm_parameters(algorithm, preset="Balanced", advanced=False):
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
        # Preset defaults
        preset_defaults = {
            "Quick (small)": 200,
            "Balanced": 1000,
            "High Fidelity": 10000
        }
        default_rows = preset_defaults.get(preset, 1000)
        params["num_rows"] = st.number_input(
            "Number of Rows to Generate",
            min_value=10,
            value=default_rows,
            step=max(10, int(default_rows/10)),
            format="%d"
        )
        if advanced:
            params["epochs"] = st.number_input("Epochs", min_value=1, value=100, step=10)
    return params


def _client_side_sanity_check(gen_params):
    """Estimate memory and warn user for very large generations."""
    try:
        num_rows = gen_params.get('num_rows') or gen_params.get('num_sequences') or 0
        # rough columns estimate from session
        n_cols = max(1, len(st.session_state.get('original_columns', [])))
        # assume avg 16 bytes per cell (very rough)
        est_bytes = int(num_rows) * int(n_cols) * 16
        est_mb = est_bytes / (1024*1024)
        if est_mb > 500:
            st.warning(f"Estimated memory for generation ~{est_mb:.0f} MB. This may be slow or fail on low-memory hosts. Consider using a smaller preset.")
        else:
            st.info(f"Estimated memory for generation ~{est_mb:.0f} MB")
    except Exception:
        pass


def _handle_generation_request(algorithm, gen_params):
    """
    Internal: Handle the Generate button logic, perform API call, and handle response.
    """
    if gen_params is None:
        return

    if st.button("Generate Synthetic Data", key="generate_btn"):
        with st.spinner(_get_generation_message(algorithm, gen_params)):
            try:
                api_base = get_api_base()
                params = {
                    "file_id": st.session_state.file_id,
                    "algorithm": algorithm
                }
                params.update(gen_params)
                response = requests.post(
                    f"{api_base}/generate",
                    params=params
                )

                if response.status_code == 200:
                    st.session_state.generated_file_id = st.session_state.file_id
                    synthetic_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                    st.session_state.data_columns = synthetic_df.columns.tolist()
                    st.session_state.data_history = st.session_state.get('data_history', [])
                    st.session_state.data_history.append(synthetic_df.copy())
                    st.session_state.df = synthetic_df.copy()
                    st.success("Generation completed!")

                    # Compute lightweight heuristics for badges
                    badges = _compute_badges(synthetic_df, st.session_state.get('uploaded_df'))
                    _render_badges(badges)

                    st.divider()
                    # Export options
                    csv_bytes = response.content
                    st.download_button(
                        "Download as CSV",
                        data=csv_bytes,
                        file_name="synthetic_data.csv",
                        mime="text/csv"
                    )

                    # JSON export
                    json_bytes = synthetic_df.to_json(orient='records').encode('utf-8')
                    st.download_button(
                        "Download as JSON",
                        data=json_bytes,
                        file_name="synthetic_data.json",
                        mime="application/json"
                    )

                    # Parquet export (if pyarrow available)
                    try:
                        import pyarrow  # noqa: F401
                        buf = io.BytesIO()
                        synthetic_df.to_parquet(buf, index=False)
                        buf.seek(0)
                        st.download_button(
                            "Download as Parquet",
                            data=buf,
                            file_name="synthetic_data.parquet",
                            mime="application/octet-stream"
                        )
                    except Exception:
                        # Parquet unavailable; show info
                        st.info("Parquet export not available (pyarrow not installed).")

                    # Optional S3 upload UI
                    if config.S3_UPLOAD_ENABLED:
                        with st.expander("Upload to S3"):
                            s3_key = st.text_input("S3 Key (path/filename)", value=f"synthetic_data_{st.session_state.file_id}.csv")
                            if st.button("Upload to S3"):
                                _upload_to_s3(csv_bytes, s3_key)

                else:
                    st.error(f"Generation failed: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {str(e)}")


def _compute_badges(synth_df, real_df=None):
    """Lightweight heuristics for fidelity, privacy risk, and mode collapse.

    These are simple heuristics for UI badges and not rigorous metrics.
    """
    badges = {
        'fidelity': None,
        'privacy_risk': None,
        'mode_collapse': None
    }
    # Fidelity: compare number of unique values in numeric columns to original (if available)
    try:
        if real_df is None:
            # Without real data, default to 'unknown'
            badges['fidelity'] = 'unknown'
        else:
            num_cols = real_df.select_dtypes(include=['number']).columns
            if len(num_cols) == 0:
                badges['fidelity'] = 'n/a'
            else:
                ratios = []
                for c in num_cols:
                    r_uniques = real_df[c].nunique()
                    s_uniques = synth_df[c].nunique() if c in synth_df.columns else 0
                    ratios.append(min(1.0, s_uniques / max(1, r_uniques)))
                avg = sum(ratios) / len(ratios)
                if avg > 0.8:
                    badges['fidelity'] = 'high'
                elif avg > 0.5:
                    badges['fidelity'] = 'medium'
                else:
                    badges['fidelity'] = 'low'
    except Exception:
        badges['fidelity'] = 'unknown'

    # Privacy risk: simple heuristic based on presence of highly unique string columns
    try:
        str_cols = synth_df.select_dtypes(include=['object']).columns
        high_cardinality = 0
        for c in str_cols:
            if synth_df[c].nunique() > 0.8 * len(synth_df):
                high_cardinality += 1
        if high_cardinality == 0:
            badges['privacy_risk'] = 'low'
        elif high_cardinality <= 2:
            badges['privacy_risk'] = 'medium'
        else:
            badges['privacy_risk'] = 'high'
    except Exception:
        badges['privacy_risk'] = 'unknown'

    # Mode collapse: look for columns with very low unique counts
    try:
        collapse_count = 0
        for c in synth_df.columns:
            if synth_df[c].nunique() <= 3 and synth_df.shape[0] > 10:
                collapse_count += 1
        if collapse_count == 0:
            badges['mode_collapse'] = 'no'
        elif collapse_count <= 2:
            badges['mode_collapse'] = 'possible'
        else:
            badges['mode_collapse'] = 'likely'
    except Exception:
        badges['mode_collapse'] = 'unknown'

    return badges


def _render_badges(badges: dict):
    cols = st.columns(3)
    with cols[0]:
        st.metric("Fidelity", badges.get('fidelity', 'unknown'))
    with cols[1]:
        st.metric("Privacy Risk", badges.get('privacy_risk', 'unknown'))
    with cols[2]:
        st.metric("Mode Collapse", badges.get('mode_collapse', 'unknown'))


def _upload_to_s3(data_bytes: bytes, key: str):
    """Simple client-side request to backend S3 presign/upload endpoint.

    Expects the backend to expose a /s3-upload or pre-signed URL endpoint; this
    function will call a presumed endpoint. If your backend does not provide
    this, adjust accordingly.
    """
    api_base = get_api_base()
    try:
        # Ask backend for pre-signed URL
        resp = requests.post(f"{api_base}/s3/presign", json={"key": key})
        if resp.status_code != 200:
            st.error(f"Failed to get presigned URL: {resp.text}")
            return
        data = resp.json()
        presigned_url = data.get('url')
        if not presigned_url:
            st.error("Presigned URL not returned by backend")
            return
        # Upload directly to S3
        upload_resp = requests.put(presigned_url, data=data_bytes)
        if upload_resp.status_code in (200, 201):
            st.success("Uploaded to S3 successfully")
        else:
            st.error(f"S3 upload failed: {upload_resp.status_code}")
    except Exception as e:
        st.error(f"S3 upload error: {str(e)}")


def _get_generation_message(algorithm, gen_params):
    """
    Internal: Returns an informative message for the generation progress spinner.
    """
    if algorithm == "PARS":
        return f"Generating {gen_params['num_sequences']} sequences of length {gen_params['sequence_length']}..."
    return f"Generating {gen_params['num_rows']} rows with {algorithm}..."
