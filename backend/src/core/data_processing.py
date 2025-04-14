from sdv.datasets.local import load_csvs
from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata, MultiTableMetadata
from pathlib import Path
import pandas as pd
import logging
import os


logger = logging.getLogger(__name__)

def load_data(file_path: str, demo_fallback: bool = True):
    """Load data from uploaded file or demo data"""
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            # Maintain consistent single-table naming
            return {"main_table": data}
        else:
            data = load_csvs(folder_name=file_path)
            logger.info(f"Loaded multi-table data: {list(data.keys())}")
            return data
    except Exception as e:
        logger.warning(f"Loading failed: {str(e)}")
        if demo_fallback:
            logger.info("Using demo data fallback")
            return download_demo(modality='single_table', dataset_name='fake_hotels')
        raise

def process_demo_data(data, metadata, file_id, algorithm):
    """Process and store demo dataset"""
    try:
        # Create uploads directory if not exists
        upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
        upload_dir.mkdir(exist_ok=True)
        
        # Save dataset
        file_path = upload_dir / f"{file_id}.csv"
        
        if algorithm == "PARS":
            # Handle sequential data
            data.to_csv(file_path, index=False)
            columns = list(data.columns)
            dataset_type = "sequential"
        else:
            # Handle single table data
            data.to_csv(file_path, index=False)
            columns = list(data.columns)
            dataset_type = "single_table"

        return {
            "columns": columns,
            "type": dataset_type,
            "path": str(file_path)
        }
    except Exception as e:
        raise RuntimeError(f"Data processing failed: {str(e)}")

def detect_metadata(data: dict):
    """Detect metadata type based on data structure"""
    if len(data) == 1 and 'main_table' in data:
        return _handle_single_table_metadata(data['main_table'])
    return _handle_multi_table_metadata(data)

def _handle_single_table_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """Enhanced metadata processing with explicit ID typing"""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # Explicitly configure Symbol as ID
    if 'Symbol' in df.columns:
        metadata.update_column(
            column_name='Symbol',
            sdtype='id',  # <-- Critical fix
            regex_format=r'[A-Z0-9]+'  # Customize based on your symbol format
        )
        metadata.set_sequence_key('Symbol')

    if 'Date' in df.columns:
        metadata.set_sequence_index('Date')
    
    # Validate before return
    try:
        metadata.validate()
    except Exception as e:
        raise ValueError(f"Metadata validation failed: {str(e)}")
    
    return metadata

def _handle_multi_table_metadata(data: dict) -> MultiTableMetadata:
    """Existing multi-table processing"""
    metadata = MultiTableMetadata()
    metadata.detect_from_dataframes(data)
    metadata.validate()
    return metadata

def prepare_training_data(data: dict) -> pd.DataFrame:
    """Universal data preparation"""
    if len(data) == 1:
        return next(iter(data.values()))
    raise ValueError("Multi-table data requires different processing")