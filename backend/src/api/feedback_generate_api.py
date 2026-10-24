# src/api/feedback_generate_api.py

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import os
import io
import re
import shutil
import logging
from datetime import datetime

from src.core.feedback_engine import EDAFeedbackEngine
# LAZY LOADING: SDVSyntheticGenerator imported inside function to avoid loading
# SDV library at module import time (saves 200MB+ at startup)
from src.core.data_processing import detect_metadata

# Resolve UPLOAD_DIR to absolute path
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", None)
if not UPLOAD_DIR:
    current_file = os.path.abspath(__file__)
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    UPLOAD_DIR = os.path.join(backend_dir, "uploads")
else:
    UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)

router = APIRouter(prefix="/feedback-generation", tags=["feedback-generation"])

@router.post("/generate_and_download")
def generate_and_download(
    file_id: str,
    feedback: list = Body(...),
    generator_config: dict = Body(...)
):
    """
    Applies feedback, generates synthetic data, and streams the result as a downloadable CSV file.
    Use this endpoint for preview+download combo in frontend.
    """
    # LAZY LOAD: Import generators only when generating data
    # This prevents loading 200MB+ of SDV/SynthCity dependencies at startup
    from src.core.synth.config import advanced_models
    from src.core.synth.generator import SDVSyntheticGenerator
    from src.core.synth.synthcity_generator import InsufficientResourcesError, create_synthcity_generator

    algorithm = generator_config.get("algorithm", "CTGAN")
    num_rows = generator_config.get("num_rows", 1000)
    epochs = generator_config.get("epochs")

    try:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")
        df = pd.read_csv(file_path)
        # 1. Apply feedback
        feedback_engine = EDAFeedbackEngine(df, feedback)
        df_fixed, log = feedback_engine.apply_feedback(return_log=True)
        # 2. Generate synthetic data
        if algorithm in advanced_models.values():
            gen = create_synthcity_generator(algorithm, epochs=epochs)
            synth_df = gen.generate(real_data=df_fixed, num_rows=num_rows)
        else:
            metadata = detect_metadata({"main_table": df_fixed})
            # Only pass epochs when given, so each model keeps its own default otherwise
            sdv_kwargs = {"epochs": epochs} if epochs else {}
            gen = SDVSyntheticGenerator(
                algorithm=algorithm,
                metadata=metadata,
                context_columns=generator_config.get("context_columns", []),
                **sdv_kwargs
            )
            synth_df = gen.generate(
                real_data=df_fixed,
                num_rows=num_rows,
                num_sequences=generator_config.get("num_sequences"),
                sequence_length=generator_config.get("sequence_length")
            )
        # Log the synthetic file path after saving
        syn_file_path = os.path.join(UPLOAD_DIR, f"syn_{file_id}.csv")
        logging.info(f"Synthetic file saved at: {syn_file_path}")

        # Save synthetic data, plus a versioned copy so earlier generations can be restored
        synth_df.to_csv(syn_file_path, index=False)
        version_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        shutil.copyfile(syn_file_path, os.path.join(UPLOAD_DIR, f"syn_{file_id}__{version_id}.csv"))

        # 4. Clean DataFrame for NaN/inf
        synth_df = synth_df.replace([np.inf, -np.inf], np.nan)
        if synth_df.isnull().any().any():
            synth_df = synth_df.fillna(0)   # Or you can use "" or -1 or any other sensible default

        # 5. Stream result CSV to client (no file-system write required)
        csv_bytes = io.BytesIO()
        synth_df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)

        return StreamingResponse(
            csv_bytes,
            media_type='text/csv',
            headers={
                "Content-Disposition": f"attachment; filename=synthetic_data_{file_id}.csv",
                "X-Synthetic-Version": version_id,
            }
        )
    except InsufficientResourcesError as e:
        raise HTTPException(503, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.post("/restore-version")
def restore_version(file_id: str, version: str):
    """Make an earlier generation (saved by generate_and_download) the current synthetic dataset."""
    # Both values become part of a file name, so only allow plain identifiers
    if not re.fullmatch(r"[A-Za-z0-9-]+", file_id) or not re.fullmatch(r"\d+", version):
        raise HTTPException(400, detail="Invalid file_id or version")
    versioned = os.path.join(UPLOAD_DIR, f"syn_{file_id}__{version}.csv")
    if not os.path.exists(versioned):
        raise HTTPException(404, detail=f"Version {version} not found")
    shutil.copyfile(versioned, os.path.join(UPLOAD_DIR, f"syn_{file_id}.csv"))
    return {"status": "restored", "file_id": file_id, "version": version}
