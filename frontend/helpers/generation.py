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
from helpers.progress_ui import (
    GenerationProgress,
    handle_generation_error,
    update_generation_session_state,
    compute_quality_badges,
    render_quality_badges
)


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
    if "saved_presets" not in st.session_state:
        st.session_state.saved_presets = {}
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        preset_options = ["Quick (small)", "Balanced", "High Fidelity"] + list(st.session_state.saved_presets.keys())
        preset = st.radio(
            "Preset", 
            preset_options, 
            index=preset_options.index(st.session_state.gen_preset) if st.session_state.gen_preset in preset_options else 1,
            help="Choose a configuration preset or load a saved configuration"
        )
        st.session_state.gen_preset = preset
    
    with col2:
        st.markdown("**💾 Presets**")
        with st.expander("Save/Load"):
            preset_name = st.text_input("Preset Name", key="preset_name_input", help="Name for your custom preset")
            if st.button("💾 Save Current", key="save_preset_btn", help="Save current configuration as a preset"):
                if preset_name:
                    # Use session state value since selected_algorithm is in scope
                    st.session_state.saved_presets[preset_name] = {
                        'algorithm': st.session_state.selected_algorithm,
                        'preset': preset
                    }
                    st.success(f"✅ Saved preset: {preset_name}")
                else:
                    st.warning("Please enter a preset name")
            
            if st.session_state.saved_presets:
                st.markdown("**Saved Presets:**")
                for name in list(st.session_state.saved_presets.keys()):
                    col_load, col_del = st.columns([3, 1])
                    with col_load:
                        if st.button(f"📂 {name}", key=f"load_{name}"):
                            saved = st.session_state.saved_presets[name]
                            st.session_state.selected_algorithm = saved['algorithm']
                            st.session_state.gen_preset = saved['preset']
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"del_{name}", help="Delete preset"):
                            del st.session_state.saved_presets[name]
                            st.rerun()

    advanced = st.checkbox(
        "Show advanced options", 
        value=st.session_state.get('settings', {}).get('advanced_params_default', False), 
        key="gen_advanced",
        help="Display advanced configuration parameters"
    )

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
            step=1,
            help="Number of time series sequences to generate"
        )
        params["sequence_length"] = st.number_input(
            "Sequence Length",
            min_value=100,
            value=5000,
            step=100,
            help="Length of each generated sequence"
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
            format="%d",
            help="Total number of synthetic rows to create. More rows = better quality but slower generation"
        )
        if advanced:
            params["epochs"] = st.number_input(
                "Epochs", 
                min_value=1, 
                value=100, 
                step=10,
                help="Training iterations. More epochs = better quality but longer generation time"
            )
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

    if st.button("Generate Synthetic Data", key="generate_btn", help="Start generating synthetic data with selected parameters"):
        with GenerationProgress() as progress:
            try:
                progress.update(10, "🔧 Preparing generation request...")
                
                api_base = get_api_base()
                params = {
                    "file_id": st.session_state.file_id,
                    "algorithm": algorithm
                }
                params.update(gen_params)
                
                progress.update(30, f"⚙️ Generating synthetic data with {algorithm}...")
                
                response = requests.post(
                    f"{api_base}/generate",
                    params=params,
                    timeout=300  # 5 minute timeout
                )
                
                progress.update(90, "✅ Processing results...")

                if response.status_code == 200:
                    progress.complete("✅ Generation complete!")
                    
                    synthetic_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                    update_generation_session_state(synthetic_df, st.session_state.file_id)
                    st.success("✅ Generation completed successfully!")

                    # Compute lightweight heuristics for badges
                    badges = compute_quality_badges(synthetic_df, st.session_state.get('uploaded_df'))
                    render_quality_badges(badges)

                    st.divider()
                    # Export options
                    csv_bytes = response.content
                    st.download_button(
                        "Download as CSV",
                        data=csv_bytes,
                        file_name="synthetic_data.csv",
                        mime="text/csv",
                        help="Download synthetic data in CSV format"
                    )

                    # JSON export
                    json_bytes = synthetic_df.to_json(orient='records').encode('utf-8')
                    st.download_button(
                        "Download as JSON",
                        data=json_bytes,
                        file_name="synthetic_data.json",
                        mime="application/json",
                        help="Download synthetic data in JSON format"
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
                            mime="application/octet-stream",
                            help="Download synthetic data in Parquet format (compressed)"
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
                    # Enhanced error messaging
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        error_data = {"detail": response.text}
                    
                    error_msg = f"""**What went wrong:** {error_data.get('detail', 'Unknown error')}
                    
**What to try:**
- Check your data has enough rows (minimum 100 recommended)
- Reduce number of epochs or rows to generate
- Try a different algorithm (e.g., GaussianCopula for smaller datasets)
- Verify API connection in sidebar settings

**Status Code:** {response.status_code}"""
                    handle_generation_error(error_msg, "general", progress.progress_bar, progress.status_text)

            except requests.exceptions.Timeout:
                handle_generation_error("", "timeout", progress.progress_bar, progress.status_text)
                
            except requests.exceptions.ConnectionError:
                handle_generation_error("", "connection", progress.progress_bar, progress.status_text)
                
            except requests.exceptions.RequestException as e:
                handle_generation_error(str(e), "api", progress.progress_bar, progress.status_text)



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
