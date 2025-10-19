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
                    timeout=10
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
            st.error("Could not connect to API server. Check API URL in sidebar.")
        except Exception as e:
            st.error(f"Error loading demo: {str(e)}")

def handle_file_upload():
    """
    Handles uploading a CSV file, sending it to the backend,
    and updating Streamlit session state with uploaded file info and columns.
    """
    if st.session_state.file_id is None:
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV)", 
            type=["csv"],
            key="file_uploader"
        )
        if uploaded_file is not None:
            try:
                response = requests.post(
                    f"{get_api_base()}/upload",
                    files={"file": uploaded_file.getvalue()}
                )
                if response.status_code == 200:
                    st.session_state.file_id = response.json().get("file_id")
                    st.session_state.generated_file_id = None
                    st.session_state.data_columns = []

                    # Store original columns for later
                    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                    st.session_state.original_columns = df.columns.tolist()
                    st.session_state.uploaded_df = df

                    st.success("File uploaded successfully!")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Upload error: {str(e)}")
    else:
        st.info("Dataset uploaded. Proceed with generation.")