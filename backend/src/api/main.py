# Standard library
import os
import io
import uuid
import json
import logging

# Third-party libraries
import pandas as pd
import numpy as np
from typing import Optional, List
from fastapi import FastAPI, UploadFile, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sdv.datasets.demo import download_demo

# Local application imports
from src.core.database import get_async_db, Base
from src.core.models import UploadedDataset
from src.core.data_processing import (
    load_data,
    detect_metadata,
    prepare_training_data as data_loader,
    process_demo_data,
)
from src.core.visualization import generate_visualizations
from src.core.synthesizers import create_synthesizer
from src.core.database import create_tables  

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Synthetic Data API!"}

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

## For render postgress database connection

# @app.on_event("startup")
# async def on_startup():
#     await create_tables()
#     logging.info("Database tables created (if not exist).")

# @app.post("/upload")
# async def upload_dataset(file: UploadFile, db: AsyncSession = Depends(get_async_db)):
#     try:
#         # Read file content (bytes)
#         file_content = await file.read()
#         print("file_content2.............", file_content)

#         # Validate size (e.g., 10MB limit)
#         max_size = 10 * 1024 * 1024
#         if len(file_content) > max_size:
#             raise HTTPException(status_code=413, detail="File exceeds 10MB limit")

#         # Parse file content with Pandas
#         try:
#             df = pd.read_csv(io.BytesIO(file_content))
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"CSV parse error: {str(e)}")

#         # Extract metadata
#         record_count = len(df)
#         sample_data = df.head(5).to_dict(orient="records")
#         sample_json = json.dumps(sample_data)

#         # Create DB entry
#         dataset = UploadedDataset(
#             id=str(uuid.uuid4()),
#             filename=file.filename,
#             record_count=record_count,
#             sample_data=sample_json,
#             original_data=file_content
#         )

#         db.add(dataset)
#         await db.commit()
#         await db.refresh(dataset)

#         return {"file_id": dataset.id, "preview_rows": sample_data}

#     except SQLAlchemyError as e:
#         await db.rollback()
#         logger.error(f"Database error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Database error")

#     except Exception as e:
#         logger.error(f"Upload failed: {str(e)}")
#         raise HTTPException(status_code=500, detail="Unexpected error")

#     finally:
#         await file.close()


# Global cache to store DataFrames (replace with a database in production)
DATA_CACHE = {}

@app.post("/upload")
async def upload_dataset(file: UploadFile):
    """Handle CSV file upload and cache it for processing"""
    try:
        file_id = str(uuid.uuid4())
        content = await file.read()

        df = pd.read_csv(io.BytesIO(content))

        DATA_CACHE[file_id] = df

        # Optional save
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Uploaded file '{file.filename}' as '{file_id}'")

        return {"file_id": file_id}

    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid CSV format")

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")
    
def sanitize_json(data):
    """Convert numpy types and handle NaNs for JSON serialization"""
    try:
        return json.loads(
            json.dumps(
                data,
                default=lambda x: x.item() if isinstance(x, np.generic) else None,
                allow_nan=False
            )
        )
    except Exception as e:
        logger.error(f"JSON sanitization failed: {str(e)}")
        raise

@app.post("/load-demo")
async def load_demo(
        algorithm: str = Query(
            default="CTGAN",
            title="Demo Algorithm",
            description="Select algorithm to load appropriate demo dataset",
            enum=["CTGAN", "GaussianCopula", "TVAE", "PARS"]
        )
    ):
    """Load demo data based on selected algorithm"""
    try:
        file_id = str(uuid.uuid4())
        
        # Dataset selection logic
        if algorithm == "PARS":
            data, metadata = download_demo(
                modality='sequential',    
                dataset_name='nasdaq100_2019'
            )
        else:
            data, metadata = download_demo(
                modality='single_table',
                dataset_name='fake_hotel_guests'
            )

        # Process and store demo data
        dataset_info = process_demo_data(data, metadata, file_id, algorithm)

        sample_data = sanitize_json(
            data.head().replace({np.nan: None}).to_dict(orient='records')
        )
        
        return {
            "file_id": file_id,
            "columns": dataset_info["columns"],
            "algorithm": algorithm,
            "dataset_type": dataset_info["type"],
            "num_rows": len(data),
            "sample_data": sample_data
        }
    except Exception as e:
        raise HTTPException("Data processing failed: ", 500, detail=str(e))

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
        print("file_path...............1", file_path)
        
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
