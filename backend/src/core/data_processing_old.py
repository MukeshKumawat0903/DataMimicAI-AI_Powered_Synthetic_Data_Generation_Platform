from sdv.datasets.local import load_csvs
from sdv.datasets.demo import download_demo
import pandas as pd
import logging
from sdv.metadata import Metadata

logger = logging.getLogger(__name__)

def load_data(file_path: str, demo_fallback: bool = True):
    """Load data from uploaded file or demo data"""
    try:
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
            data_dict = {"global_power_plant_database_s1": data}
            return data_dict
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

def detect_metadata(data: dict) -> Metadata:
    """Detect and validate metadata from loaded data"""
    try:
        metadata = Metadata.detect_from_dataframes(data)

        print("Metadata....", metadata)

        # Auto-detect sequence key if PAR is expected
        if 'id' in metadata.columns:
            metadata.update_column('id', sdtype='id')
            metadata.set_sequence_key('id')
        if 'time' in metadata.columns:
            metadata.set_sequence_index('time')

        metadata.validate()
        metadata.validate_data(data)
        logger.info("Metadata validated successfully")
        return metadata
    except Exception as e:
        logger.error(f"Metadata validation failed: {str(e)}")
        raise

def prepare_training_data(data: dict) -> pd.DataFrame:
    """Prepare single-table data for training"""
    if not isinstance(data, dict):
        raise ValueError("Input data must be a dictionary of DataFrames")
    
    table_name = list(data.keys())[0]
    df = data[table_name]
    
    logger.info(f"Using table '{table_name}' with shape {df.shape}")
    return df
