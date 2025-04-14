import streamlit as st
import requests
import os
import pandas as pd
import io
from typing import Optional

# Configure API URL
API_BASE = os.getenv("API_URL", "http://localhost:8000")   # Will be set in Render

def main():
    st.title("Synthetic Data Platform")
    
    # Initialize session state
    if 'file_id' not in st.session_state:
        st.session_state.file_id = None
    if 'generated_file_id' not in st.session_state:
        st.session_state.generated_file_id = None
    if 'data_columns' not in st.session_state:
        st.session_state.data_columns = []
    if 'original_columns' not in st.session_state:
        st.session_state.original_columns = []
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Configuration")
        demo_mode = st.checkbox("Use Demo Data")
        if st.button("Reset App"):
            st.session_state.file_id = None
            st.session_state.generated_file_id = None
            st.session_state.data_columns = []
            st.rerun()
    
    # Main content area
    if demo_mode:
        handle_demo_mode()
    else:
        handle_file_upload()
    
    if st.session_state.file_id:
        show_generation_controls()
        show_visualization()

def handle_demo_mode():
    """Demo mode with algorithm-specific datasets"""
    st.subheader("Demo Mode Settings")
    
    # Add algorithm selection
    algorithm = st.selectbox(
        "Select Algorithm for Demo",
        ["CTGAN", "GaussianCopula", "TVAE", "PARS"],
        index=0
    )    

    if st.button("Load Demo Data"):
        try:
            with st.spinner(f"Loading {algorithm} demo..."):
                response = requests.post(
                    f"{API_BASE}/load-demo",
                    params={"algorithm": algorithm},  # Send as query parameter
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Store original columns immediately
                    st.session_state.original_columns = data['columns']
                    
                    # Store other metadata
                    st.session_state.file_id = data["file_id"]
                    st.session_state.generated_file_id = None
                    st.session_state.data_columns = data["columns"]

                    # Show preview with actual data
                    with st.expander("Demo Data Preview"):
                        st.write(f"**Dataset:** {data['algorithm']} Example")
                        st.write(f"**Rows:** {data['num_rows']:,}")
                        st.write(f"**Columns ({len(data['columns'])}):**")
                        
                        # Create DataFrame from sample data dict
                        df = pd.DataFrame(data['sample_data'])
                        
                        # Display interactive table
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
                    st.error(f"Failed to load demo: {response.json().get('detail', 'Unknown error')}")
                    
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to API server")
        except Exception as e:
            st.error(f"Error loading demo: {str(e)}")

def handle_file_upload():
    """Handle file upload workflow"""
    if st.session_state.file_id is None:
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV)", 
            type=["csv"],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                response = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": uploaded_file.getvalue()}
                )
                if response.status_code == 200:
                    st.session_state.file_id = response.json().get("file_id")
                    st.session_state.generated_file_id = None
                    st.session_state.data_columns = []
            
                    # Store original columns immediately after upload
                    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                    st.session_state.original_columns = df.columns.tolist()

                    st.success("File uploaded successfully!")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Upload error: {str(e)}")
    else:
        st.info("Dataset uploaded. Proceed with generation.")

def show_generation_controls():
    """Show generation parameters and controls"""
    st.header("Generation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.selectbox(
            "Synthesis Algorithm",
            # ["CTGAN", "GaussianCopula", "TVAE", "HMAS", "PARS"],
            ["CTGAN", "GaussianCopula", "TVAE", "PARS"],
            index=1
        )
    
    # In show_generation_controls()
    with col2:
        if algorithm == "PARS":
            # Check for valid sequence key columns
            sequence_key_candidates = ['Symbol']
            valid_sequence_columns = [
                col for col in st.session_state.original_columns 
                if col in sequence_key_candidates
            ]

            if not valid_sequence_columns:
                st.error(
                    "PAR requires one of these columns as sequence key: " +
                    ", ".join(sequence_key_candidates)
                )
                return  # Exit early to prevent parameter display

            num_sequences = st.number_input(
                "Number of Sequences",
                min_value=1,
                value=1,
                step=1
            )
            sequence_length = st.number_input(
                "Sequence Length",
                min_value=100,
                value=5000,
                step=100
            )
        else:
            num_rows = st.number_input(
                "Number of Rows to Generate",
                min_value=100,
                value=1000,
                step=500,
                format="%d"
            )

    # with col2:
    #     num_rows = st.number_input(
    #         "Number of Rows to Generate",
    #         min_value=100,
    #         value=1000,
    #         step=500,
    #         format="%d"
    #     )  

    if st.button("Generate Synthetic Data", key="generate_btn"):
        # with st.spinner(f"Generating {num_rows} rows with {algorithm}..."):
        with st.spinner(get_generation_message(algorithm, locals())):
            try:
                # url = f"{API_BASE}/generate?file_id={st.session_state.file_id}&algorithm={algorithm}&num_rows={num_rows}"
                # response = requests.post(url, json=[])

                # In generation button click handler:
                params = {
                    "file_id": st.session_state.file_id,
                    "algorithm": algorithm
                }

                if algorithm == "PARS":
                    params.update({
                        "num_sequences": num_sequences,
                        "sequence_length": sequence_length
                    })
                else:
                    params["num_rows"] = num_rows

                response = requests.post(
                    f"{API_BASE}/generate",
                    params=params
                )

                if response.status_code == 200:
                    # Store generated file ID and columns
                    st.session_state.generated_file_id = st.session_state.file_id
                    
                    # Extract columns from generated data
                    synthetic_df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
                    st.session_state.data_columns = synthetic_df.columns.tolist()
                    
                    st.success("Generation completed!")
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

def get_generation_message(algorithm, local_vars):
    """Generate appropriate progress message"""
    if algorithm == "PARS":
        return f"Generating {local_vars['num_sequences']} sequences of length {local_vars['sequence_length']}..."
    return f"Generating {local_vars['num_rows']} rows with {algorithm}..."

def show_visualization():
    """Show data visualization controls"""
    st.header("Data Visualization")
    
    if not st.session_state.generated_file_id:
        st.warning("Generate synthetic data first to enable visualization")
        return
    
    if not st.session_state.data_columns:
        st.error("No columns available for visualization")
        return
    
    # Column selection UI
    col1, col2 = st.columns(2)
    
    with col1:
        selected_column = st.selectbox(
            "Select column for distribution plot",
            options=st.session_state.data_columns,
            index=0
        )
    
    with col2:
        selected_pairs = st.multiselect(
            "Select columns for pair plot (2-3 recommended)",
            options=st.session_state.data_columns,
            default=st.session_state.data_columns[:2] if len(st.session_state.data_columns) >= 2 else []
        )
    
    if st.button("Generate Visualizations", key="visualize_btn"):
        if not selected_pairs:
            st.warning("Please select at least one pair of columns for the pair plot")
            return
            
        with st.spinner("Creating visualizations..."):
            try:
                # Convert list to comma-separated string for API
                pair_str = ",".join(selected_pairs)
                
                response = requests.get(
                    f"{API_BASE}/visualize",
                    params={
                        "file_id": st.session_state.generated_file_id,
                        "column": selected_column,
                        "pair_columns": pair_str
                    }
                )
                
                if response.status_code == 200:
                    st.subheader("Data Distribution Comparison")
                    st.components.v1.html(
                        response.content, 
                        height=800,
                        scrolling=True
                    )
                else:
                    st.error(f"Visualization failed: {response.text}")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Visualization error: {str(e)}")

if __name__ == "__main__":
    main()