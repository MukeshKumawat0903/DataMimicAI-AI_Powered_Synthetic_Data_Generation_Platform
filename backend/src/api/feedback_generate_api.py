# src/api/feedback_generate_api.py

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import os
import io

from backend.src.core.feedback_engine import EDAFeedbackEngine
from backend.src.core.synth.generator import SDVSyntheticGenerator
from backend.src.core.data_processing import detect_metadata

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
    try:
        file_path = os.path.join(os.getenv("UPLOAD_DIR", "uploads"), f"{file_id}.csv")
        df = pd.read_csv(file_path)
        # 1. Apply feedback
        feedback_engine = EDAFeedbackEngine(df, feedback)
        df_fixed, log = feedback_engine.apply_feedback(return_log=True)
        # 2. Detect metadata
        metadata = detect_metadata({"main_table": df_fixed})
        # 3. Generate synthetic data
        gen = SDVSyntheticGenerator(
            algorithm=generator_config.get("algorithm", "CTGAN"),
            metadata=metadata,
            context_columns=generator_config.get("context_columns", [])
        )
        synth_df = gen.generate(
            real_data=df_fixed,
            num_rows=generator_config.get("num_rows", 1000),
            num_sequences=generator_config.get("num_sequences"),
            sequence_length=generator_config.get("sequence_length")
        )
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
                "Content-Disposition": f"attachment; filename=synthetic_data_{file_id}.csv"
            }
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))
