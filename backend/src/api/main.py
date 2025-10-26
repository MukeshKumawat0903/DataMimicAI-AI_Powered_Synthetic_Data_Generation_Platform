# Standard library
import os
import io
import uuid
import json
import time
import logging
import psutil
from synthcity.plugins import Plugins

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
from backend.src.core.database import get_async_db, Base
from backend.src.core.data_processing import (
    load_data,
    detect_metadata,
    prepare_training_data as data_loader,
    process_demo_data,
)
from backend.src.core.visualization import DataVisualizer
from backend.src.core.database import create_tables

from backend.src.api.eda_feature_api import router as eda_feature_router
from backend.src.core.synth.generator import SDVSyntheticGenerator
from backend.src.core.synth.synthcity_generator import SynthCitySyntheticGenerator
from backend.src.api.feedback_generate_api import router as feedback_router
from backend.src.core.synth.model_selection import synthcity_model_comparison, select_best_model, get_memory_usage_mb

from backend.src.core.synth.config import advanced_models, metric_cols

app = FastAPI()
app.include_router(eda_feature_router)
app.include_router(feedback_router)

@app.get("/")
def home():
    return {"message": "Welcome to the Synthetic Data API!"}

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def check_system_resources():
    """Check if system has enough resources for training"""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    if memory.percent > 80:
        raise HTTPException(503, f"System memory too high ({memory.percent}%). Please try again later.")
    
    if cpu_percent > 85:
        raise HTTPException(503, f"System CPU too high ({cpu_percent}%). Please try again later.")

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
    algorithm: str = Query("CTGAN", min_length=1),
    num_rows: int = Query(1000, ge=1, le=5_000_000),
    num_sequences: Optional[int] = Query(3, ge=1, le=10000),
    sequence_length: Optional[int] = Query(100, ge=1, le=10_000_000),
    context_columns: Optional[List[str]] = None,
    epochs: Optional[int] = Query(100, ge=1, le=100000),
    target_column: str = Query("Survived")
):
    """Generate synthetic data endpoint with fixed SynthCity parameter handling"""
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        logger.info(f"file_path: {file_path}")

        if not os.path.exists(file_path):
            raise HTTPException(404, "File not found")

        # Load and prepare data
        logger.info(f"Loading data for file ID: {file_id}")
        data_dict = load_data(file_path)
        data = data_loader(data_dict)

        # Detect metadata (for SDV/PARS)
        logger.info("Detecting metadata")
        metadata = detect_metadata(data_dict)

        # --- SYNTHCITY PARAMETER MAPPING ---
        ITERATION_PARAMS = {
            'ddpm': 'n_iter',
            'ctgan': 'n_iter',
            'tvae': 'n_iter', 
            'dpgan': 'n_iter', 
            'pategan': 'n_iter', 
            'privbayes': None,  # No iteration parameter
            'arf': None         # No iteration parameter
        }

        # --- Handle PARS Validation (still SDV logic) ---
        if algorithm == "PARS":
            if not metadata.sequence_key:
                raise HTTPException(400, "PAR requires sequence key. Columns like 'Symbol' or 'id' must exist.")
            seq_key_type = metadata.columns[metadata.sequence_key]['sdtype']
            if seq_key_type != 'id':
                raise HTTPException(400, f"Sequence key column '{metadata.sequence_key}' must be type 'id' (current: {seq_key_type})")
        
        # --- SynthCity Section (FIXED: Conservative Parameter Handling) ---
        if algorithm in advanced_models.values():  # e.g., "ddpm", "ctgan", etc.
            logger.info(f"Creating SynthCity generator for: {algorithm} with epochs={epochs}")
            
            # Build plugin_kwargs with correct parameter name (conservative approach)
            synthcity_plugin_kwargs = {}
            iteration_param_name = ITERATION_PARAMS.get(algorithm)
            if iteration_param_name:
                synthcity_plugin_kwargs[iteration_param_name] = epochs
                logger.info(f"Applying '{iteration_param_name}={epochs}' for {algorithm} model.")
            else:
                logger.info(f"Using default parameters for {algorithm} (no epochs parameter supported)")

            generator = SynthCitySyntheticGenerator(
                algorithm=algorithm,
                target_column=target_column,
                plugin_kwargs=synthcity_plugin_kwargs
            )
            synthetic_data = generator.generate(real_data=data, num_rows=num_rows)

        # --- SDV/PARS Section (Existing) ---
        elif algorithm == "PARS":
            generator = SDVSyntheticGenerator(
                algorithm=algorithm,
                metadata=metadata,
                context_columns=context_columns or {},
                epochs=epochs  # Pass epochs to SDV generator too
            )
            synthetic_data = generator.generate(
                real_data=data,
                num_sequences=num_sequences,
                sequence_length=sequence_length,
            )

        else:
            # Regular SDV models
            generator = SDVSyntheticGenerator(
                algorithm=algorithm,
                metadata=metadata,
                context_columns=context_columns or {},
                epochs=epochs
            )
            synthetic_data = generator.generate(real_data=data, num_rows=num_rows)

        # --- Save results with improved data type handling ---
        try:
            # Fix any problematic data types before saving
            for col in synthetic_data.columns:
                if synthetic_data[col].dtype == 'object':
                    synthetic_data[col] = synthetic_data[col].astype(str)
            
            output_path = os.path.join(UPLOAD_DIR, f"syn_{file_id}.csv")
            synthetic_data.to_csv(
                output_path, 
                index=False,
                encoding='utf-8',
                float_format='%.6f'  # Prevent float precision issues
            )
            
        except Exception as save_error:
            logger.error(f"Save operation failed: {str(save_error)}")
            # Try alternative save method
            safe_df = synthetic_data.copy()
            for col in safe_df.select_dtypes(include=['object']).columns:
                safe_df[col] = safe_df[col].astype(str)
            safe_df.to_csv(output_path, index=False)
            logger.info("Alternative save method succeeded")

        logger.info(f"Generation complete for file ID: {file_id}")
        return FileResponse(
            output_path,
            filename="synthetic_data.csv",
            media_type="text/csv"
        )

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(500, f"Data generation failed: {str(e)}")


@app.get("/health")
def health_check():
    """Lightweight health endpoint used by frontend to verify API connectivity."""
    try:
        # Quick resource probe (non-blocking)
        mem = psutil.virtual_memory()
        return {
            "status": "ok",
            "memory_percent": mem.percent,
            "uptime": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(500, "Health check failed")

@app.post("/evaluate_models")
async def evaluate_models(
    file_id: str,
    target_column: str = "Survived",
    epochs: int = 100
):
    """
    Model evaluation with pandas compatibility fixes.
    """
    start_memory = get_memory_usage_mb()
    start_time = time.time()
    
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(404, f"File not found: {file_id}")

        real_df = pd.read_csv(file_path)
        if real_df.empty:
            raise HTTPException(400, "Dataset is empty")
        
        # Check if target column exists
        if target_column and target_column not in real_df.columns:
            logger.warning(f"Target column '{target_column}' not found, proceeding without target")
            target_column = None

        logger.info(f"Starting model evaluation for {len(real_df)} samples with {epochs} epochs")

        # Run model comparison
        df_metrics = synthcity_model_comparison(
            real_df, 
            epochs=epochs, 
            target_column=target_column
        )

        # Select best model
        best_model_name, best_model_row = select_best_model(df_metrics, metric_cols)

        # ✅ FIXED: Pandas-compatible JSON-safe results
        def make_json_safe(value):
            """Convert value to JSON-safe format with pandas compatibility"""
            if value is None or pd.isna(value):
                return None
            if isinstance(value, (int, float)):
                # Use numpy instead of pandas for isfinite check
                if not np.isfinite(value):
                    return None
                return float(value)
            return str(value)

        # Clean the dataframe for JSON serialization
        metrics_clean = []
        for _, row in df_metrics.iterrows():
            clean_row = {}
            for col, value in row.items():
                clean_row[col] = make_json_safe(value)
            metrics_clean.append(clean_row)

        # Prepare best model metrics
        best_metrics = {}
        if best_model_row is not None:
            for col in metric_cols:
                if col in best_model_row:
                    best_metrics[col] = make_json_safe(best_model_row[col])

        total_time = time.time() - start_time
        successful_models = df_metrics[df_metrics['Status'] == 'Success']
        
        result = {
            "summary": {
                "total_models": len(df_metrics),
                "successful_models": len(successful_models),
                "failed_models": len(df_metrics) - len(successful_models),
                "total_evaluation_time": round(total_time, 2),
                "best_model": str(best_model_name) if best_model_name else "None",
                "memory_usage_mb": round(get_memory_usage_mb() - start_memory, 2)
            },
            "metrics": metrics_clean,
            "best_model": str(best_model_name) if best_model_name else "None",
            "best_model_metrics": best_metrics
        }

        logger.info(f"Model evaluation completed in {total_time:.1f}s")
        return result

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Model evaluation failed after {total_time:.1f}s: {str(e)}")
        raise HTTPException(500, f"Model evaluation failed: {str(e)}")

# Visualization
@app.get("/visualize")
async def visualize_data(
    file_id: str,
    tab: str = Query(..., description="Which visualization to render"),
    column: str = None,
    columns: str = None,
    plot_type: str = "Histogram",
    overlay: bool = True,
    show_type: str = "Compare Both"
):
    """
    Generic visualization endpoint for all visualization tabs.
    Params depend on `tab` type:
      - tab: "distribution", "pairplot", "real_vs_synth", "drift", "correlation"
      - column: column name (distribution)
      - columns: comma-separated columns (pairplot, real_vs_synth, drift)
      - plot_type: Histogram/KDE/Boxplot (distribution)
      - overlay: bool (distribution/pairplot)
      - show_type: Real/Synthetic/Compare Both (correlation)
    """
    try:
        syn_path = os.path.join(UPLOAD_DIR, f"syn_{file_id}.csv")
        orig_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")

        if not os.path.exists(syn_path) or not os.path.exists(orig_path):
            raise HTTPException(404, "File not found")

        synthetic_data = pd.read_csv(syn_path)
        real_data = pd.read_csv(orig_path)
        metadata = detect_metadata({"main_table": real_data})

        visualizer = DataVisualizer(real_data, synthetic_data, metadata)

        if tab == "distribution":
            if not column:
                column = real_data.columns[0]
            html = visualizer.distribution_plot(column, plot_type, overlay)
        elif tab == "pairplot":
            if not columns:
                columns = ",".join(real_data.columns[:2])
            cols = [c.strip() for c in columns.split(",")]
            html = visualizer.pair_plot(cols, overlay)
        elif tab == "real_vs_synth":
            if not columns:
                columns = ",".join(real_data.columns[:2])
            cols = [c.strip() for c in columns.split(",")]
            html = visualizer.real_vs_synth_summary(cols)
        elif tab == "drift":
            if not columns:
                columns = ",".join(real_data.columns[:2])
            cols = [c.strip() for c in columns.split(",")]
            html = visualizer.drift_detection(cols)
        elif tab == "correlation":
            html = visualizer.correlation_heatmap(show_type)
        else:
            raise HTTPException(400, f"Unknown visualization tab: {tab}")

        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        raise HTTPException(500, f"Visualization failed: {str(e)}")


    @app.get("/metrics")
    def compute_metrics(file_id: str, columns: str = Query(None, description="Comma-separated columns to evaluate")):
        """Compute lightweight similarity metrics between real and synthetic for selected columns.

        Returns KS statistic for numeric columns and chi-square p-value for categorical.
        """
        try:
            syn_path = os.path.join(UPLOAD_DIR, f"syn_{file_id}.csv")
            orig_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
            if not os.path.exists(syn_path) or not os.path.exists(orig_path):
                raise HTTPException(404, "File not found")

            synthetic_data = pd.read_csv(syn_path)
            real_data = pd.read_csv(orig_path)

            if not columns:
                cols = list(set(real_data.columns).intersection(set(synthetic_data.columns)))
            else:
                cols = [c.strip() for c in columns.split(',') if c.strip()]

            results = {}
            for c in cols:
                try:
                    if pd.api.types.is_numeric_dtype(real_data[c]):
                        # KS-test
                        try:
                            from scipy.stats import ks_2samp
                            stat, p = ks_2samp(real_data[c].dropna(), synthetic_data[c].dropna())
                            results[c] = {"type": "numeric", "ks_stat": float(stat), "p_value": float(p)}
                        except Exception:
                            # fallback: compare means and std
                            r_mean = float(real_data[c].dropna().mean())
                            s_mean = float(synthetic_data[c].dropna().mean())
                            results[c] = {"type": "numeric", "mean_real": r_mean, "mean_synth": s_mean}
                    else:
                        # categorical: chi-square on value counts
                        try:
                            from scipy.stats import chi2_contingency
                            r_counts = real_data[c].fillna('___NA___').astype(str).value_counts()
                            s_counts = synthetic_data[c].fillna('___NA___').astype(str).value_counts()
                            all_index = list(set(r_counts.index).union(set(s_counts.index)))
                            r_arr = [r_counts.get(i, 0) for i in all_index]
                            s_arr = [s_counts.get(i, 0) for i in all_index]
                            table = np.array([r_arr, s_arr])
                            chi2, p, dof, ex = chi2_contingency(table)
                            results[c] = {"type": "categorical", "chi2": float(chi2), "p_value": float(p)}
                        except Exception:
                            results[c] = {"type": "categorical", "note": "chi2 unavailable"}
                except Exception as e:
                    results[c] = {"error": str(e)}

            return {"file_id": file_id, "metrics": results}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Metrics computation failed: {str(e)}")
            raise HTTPException(500, f"Metrics computation failed: {str(e)}")