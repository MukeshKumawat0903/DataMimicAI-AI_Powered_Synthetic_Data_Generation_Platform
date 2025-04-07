from fastapi import FastAPI, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
# from core.data_processing import load_data, detect_metadata
from core.data_processing import (
    load_data,
    detect_metadata,
    prepare_training_data as data_loader
)
from core.visualization import generate_visualizations
from core.synthesizers import create_synthesizer
from typing import Optional, List  # Import Optional and List
import pandas as pd
import uuid
import os
import io
import logging

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Synthetic Data API!"}

logger = logging.getLogger(__name__)

# Temporary storage for demo (replace with DB in production)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")


# Global cache to store DataFrames (replace with a database in production)
DATA_CACHE = {}

@app.post("/upload")
async def upload_dataset(file: UploadFile):
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Read uploaded CSV into DataFrame
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Store DataFrame in cache
        DATA_CACHE[file_id] = df
        
        # Optional: Save raw file to disk (if needed)
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"Uploaded file {file.filename} as {file_id}")
        return {"file_id": file_id}
    
    except pd.errors.ParserError as e:
        logger.error(f"Invalid CSV: {str(e)}")
        raise HTTPException(400, "Invalid CSV file")
    
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, "File upload failed")

@app.post("/generate")
async def generate_data(
    file_id: str,
    algorithm: str = "CTGAN",
    num_rows: int = 1000,
    num_sequences: Optional[int] = 3,  # New PAR parameter
    sequence_length: Optional[int] = 100,  # New PAR parameter
    context_columns: Optional[List[str]] = None
):
    """Generate synthetic data endpoint"""
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        
        if not os.path.exists(file_path):
            raise HTTPException(404, "File not found")

        # Load and prepare data
        logger.info(f"Loading data for file ID: {file_id}")
        data_dict = load_data(file_path)
        # training_data = prepare_training_data(data_dict)
        data = data_loader(data_dict)
        
        # Process metadata
        logger.info("Detecting metadata")
        metadata = detect_metadata(data_dict)

        # --- ADD VALIDATION HERE ---
        if algorithm == "PARS":
            if not metadata.sequence_key:
                raise HTTPException(
                    400, 
                    "PAR requires sequence key. Columns like 'Symbol' or 'id' must exist."
                )
            seq_key_type = metadata.columns[metadata.sequence_key]['sdtype']
            if seq_key_type != 'id':
                raise HTTPException(
                    400,
                    f"Sequence key column '{metadata.sequence_key}' must be type 'id' (current: {seq_key_type})"
                )
        # --- END VALIDATION ---       
        
        # Create synthesizer
        logger.info(f"Creating {algorithm} synthesizer")
        synthesizer = create_synthesizer(
            real_data=data,
            metadata=metadata,
            sdv_algorithm=algorithm,
            context_columns=context_columns or []
        )
        
        # Generate synthetic data
        logger.info(f"Generating {num_rows} synthetic rows")

        if algorithm == "PARS":
            synthetic_data = synthesizer.sample(
                num_sequences=num_sequences, 
                sequence_length = sequence_length
                )
        else:
            synthetic_data = synthesizer.sample(num_rows=num_rows)
        
        # Save results
        output_path = os.path.join(UPLOAD_DIR, f"syn_{file_id}.csv")
        synthetic_data.to_csv(output_path, index=False)

        logger.info(f"Generation complete for file ID: {file_id}")
        return FileResponse(
            output_path,
            filename="synthetic_data.csv",
            media_type="text/csv"
        )
    
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(500, f"Data generation failed: {str(e)}")

# Visualization
@app.get("/visualize", response_class=HTMLResponse)
async def visualize_data(
    file_id: str,
    column: Optional[str] = None,
    pair_columns: Optional[str] = None  # Comma-separated string
):
    try:
        syn_path = os.path.join(UPLOAD_DIR, f"syn_{file_id}.csv")
        orig_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")

        synthetic_data = pd.read_csv(syn_path)
        original_data = pd.read_csv(orig_path)
        
        # Process column parameters
        pair_cols = pair_columns.split(",") if pair_columns else None
        
        visualizations = generate_visualizations(
            real_data=original_data,
            synthetic_data=synthetic_data,
            metadata=detect_metadata({"main_table": original_data}),
            column_name=column,
            pair_columns=pair_cols
        )
        
        return visualizations['column_plot'] + visualizations['pair_plot']
    
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        raise HTTPException(500, f"Visualization failed: {str(e)}")
