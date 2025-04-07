# src/core/synthesizers.py
from sdv.single_table import (
    CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
)
from sdv.multi_table import HMASynthesizer
from sdv.sequential import PARSynthesizer

def create_synthesizer(real_data, metadata, sdv_algorithm = "CTGAN", context_columns=['Sector', 'Industry']):
    """Create and return the appropriate SDV synthesizer based on the chosen algorithm."""
    if sdv_algorithm == "CTGAN":
        custom_synthesizer = CTGANSynthesizer(
            metadata, # required
            enforce_rounding=False,
            epochs=500,
            verbose=True
            )
    
    elif sdv_algorithm == "GaussianCopula":
        custom_synthesizer = GaussianCopulaSynthesizer(
            metadata
            # epochs=1000,
            # verbose=True
        )

    elif sdv_algorithm == "TVAE":
        custom_synthesizer = TVAESynthesizer(
            metadata, # required
            enforce_min_max_values=True,
            enforce_rounding=False,
            epochs=500
        )  

    elif sdv_algorithm == "HMAS":  
        custom_synthesizer = HMASynthesizer(
            metadata,
            # epochs=1000,
            # verbose=True
        )

    elif sdv_algorithm == "PARS":     
        custom_synthesizer = PARSynthesizer(
            metadata,
            # epochs=1000,
            context_columns=context_columns,
            verbose=True
        )
    else:
        raise ValueError("Invalid synthesizer algorithm chosen.")
    
    print(f"Applied Algorithm: {sdv_algorithm}")
    custom_synthesizer.fit(real_data)

    return custom_synthesizer

