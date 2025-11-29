"""
DataMimicAI File Upload and Demo Data Helpers.
Handles file uploads, demo data loading, and session state management for the DataMimicAI platform.
This module provides functions to upload CSV files, load demo datasets, and manage the state of uploaded.
"""

import streamlit as st
import pandas as pd
import requests
import io
import os

import frontend_config as config


def get_api_base():
    return st.session_state.get('custom_api') or os.getenv("API_URL") or config.API_BASE

def handle_demo_mode():
    """
    Handles the demo mode: lets the user select a demo algorithm,
    loads demo data from the backend, and updates Streamlit session state.
    """
    st.subheader("Demo Mode Settings")
    algorithm = st.selectbox(
        "Select Algorithm for Demo",
        ["CTGAN", "GaussianCopula", "TVAE", "PARS"],
        index=0
    )

    if st.button("Load Demo Data"):
        try:
            with st.spinner(f"Loading {algorithm} demo..."):
                response = requests.post(
                    f"{get_api_base()}/load-demo",
                    params={"algorithm": algorithm},
                    timeout=120  # 2 minutes for cold start + processing
                )
                if response.status_code == 200:
                    data = response.json()
                    # Store original columns and update session state
                    st.session_state.original_columns = data['columns']
                    st.session_state.file_id = data["file_id"]
                    st.session_state.generated_file_id = None
                    st.session_state.data_columns = data["columns"]

                    # Preview the loaded data
                    with st.expander("Demo Data Preview"):
                        st.write(f"**Dataset:** {data['algorithm']} Example")
                        st.write(f"**Rows:** {data['num_rows']:,}")
                        st.write(f"**Columns ({len(data['columns'])}):**")
                        df = pd.DataFrame(data['sample_data'])
                        st.session_state.uploaded_df = df
                        st.session_state.df = df.copy()
                        st.session_state.data_history = [df.copy()]
                        st.session_state.features_applied = False
                        st.session_state.last_changed_columns = []
                        st.session_state.feature_preview_df = None
                        st.session_state.feature_preview_message = None
                        st.session_state.last_applied_summary = None
                        st.session_state.quick_preview_visible = False
                        st.session_state.quick_preview_df = None
                        st.session_state.quick_preview_message = "Preview of EDA results."
                        st.session_state.quick_preview_changed_cols = []
                        st.dataframe(
                            df,
                            column_config={
                                col: st.column_config.Column(
                                    col,
                                    help=f"Demo column from {data['algorithm']} dataset"
                                ) for col in data['columns']
                            },
                            use_container_width=True
                        )
                    st.success("Demo data loaded successfully!")
                else:
                    try:
                        err = response.json().get('detail', response.text)
                    except Exception:
                        err = response.text
                    st.error(f"Failed to load demo: {err}")
        except requests.exceptions.ConnectionError:
            st.error(
                "❌ **Cannot connect to backend API server!**\n\n"
                f"The server at `{get_api_base()}` is not running.\n\n"
                "**To fix this:**\n"
                "1. Open a new terminal\n"
                "2. Navigate to the `backend` directory\n"
                "3. Run: `uvicorn src.api.main:app --reload --port 8000`"
            )
        except Exception as e:
            st.error(f"Error loading demo: {str(e)}")

def handle_file_upload():
    """
    Handles uploading a CSV file, sending it to the backend,
    and updating Streamlit session state with uploaded file info and columns.
    """
    # Only show uploader if no dataset is selected
    if st.session_state.file_id is None:
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV)", 
            type=["csv"],
            key="file_uploader"
        )

        MAX_UPLOAD_SIZE = int(os.environ.get('MAX_UPLOAD_SIZE', 10 * 1024 * 1024))  # 10 MB default

        if uploaded_file is not None:
            # Basic client-side size validation
            try:
                raw = uploaded_file.getvalue()
                size = len(raw)
                if size > MAX_UPLOAD_SIZE:
                    st.error(f"File too large ({size/1024/1024:.2f} MB). Max allowed is {MAX_UPLOAD_SIZE/1024/1024:.1f} MB.")
                    return

                # Show spinner while uploading to backend
                with st.spinner("Uploading dataset to API..."):
                    # Send file bytes to backend as multipart/form-data
                    files = {"file": (uploaded_file.name, raw, "text/csv")}
                    try:
                        response = requests.post(
                            f"{get_api_base()}/upload",
                            files=files,
                            timeout=300  # 5 minutes to handle cold start (60s) + upload + processing
                        )
                    except requests.exceptions.ConnectionError:
                        st.error(
                            "❌ **Cannot connect to backend API server!**\n\n"
                            f"The server at `{get_api_base()}` is not running.\n\n"
                            "**To fix this:**\n"
                            "1. Open a new terminal\n"
                            "2. Navigate to the `backend` directory\n"
                            "3. Run: `uvicorn src.api.main:app --reload --port 8000`\n\n"
                            "Or use **Demo Mode** below to test without uploading."
                        )
                        return
                    except requests.exceptions.RequestException as e:
                        st.error(f"Upload error: {str(e)}")
                        return

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.file_id = data.get("file_id")
                    st.session_state.generated_file_id = None
                    st.session_state.data_columns = []

                    # Store original columns for later preview
                    try:
                        df = pd.read_csv(io.BytesIO(raw))
                        st.session_state.original_columns = df.columns.tolist()
                        st.session_state.uploaded_df = df
                        st.session_state.df = df.copy()
                        st.session_state.data_history = [df.copy()]
                        st.session_state.features_applied = False
                        st.session_state.last_changed_columns = []
                        st.session_state.feature_preview_df = None
                        st.session_state.feature_preview_message = None
                        st.session_state.last_applied_summary = None
                        st.session_state.quick_preview_visible = False
                        st.session_state.quick_preview_df = None
                        st.session_state.quick_preview_message = "Preview of EDA results."
                        st.session_state.quick_preview_changed_cols = []
                    except Exception:
                        # If local parsing fails, still rely on backend metadata
                        st.session_state.original_columns = data.get('columns', [])
                        st.session_state.df = None
                        st.session_state.data_history = []
                        st.session_state.features_applied = False
                        st.session_state.last_changed_columns = []
                        st.session_state.feature_preview_df = None
                        st.session_state.feature_preview_message = None
                        st.session_state.last_applied_summary = None
                        st.session_state.quick_preview_visible = False
                        st.session_state.quick_preview_df = None
                        st.session_state.quick_preview_message = "Preview of EDA results."
                        st.session_state.quick_preview_changed_cols = []

                    st.success("File uploaded successfully!")
                    st.rerun()

                else:
                    # Try to show a helpful error from backend
                    try:
                        err = response.json().get('detail', response.text)
                    except Exception:
                        err = response.text
                    st.error(f"Upload failed: {err}")

            except Exception as e:
                st.error(f"Upload error: {str(e)}")
    else:
        st.info("Dataset uploaded. Proceed with generation.")