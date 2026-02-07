# src/core/synth/generator.py

import pandas as pd

# LAZY LOADING: Heavy SDV imports moved inside fit() method
# This prevents loading all synthesizers at module import time
# Synthesizers are imported only when generator.fit() is called

class SDVSyntheticGenerator:
    """
    Modular SDV-based synthetic data generator.
    - Call fit() with real data and metadata.
    - Call sample() to get synthetic DataFrame.
    """
    def __init__(self, algorithm="CTGAN", metadata=None, context_columns=None, **kwargs):
        self.algorithm = algorithm
        self.metadata = metadata
        self.context_columns = context_columns or []
        self.kwargs = kwargs
        self.synthesizer = None

    def fit(self, real_data):
        """Fit synthesizer on real_data DataFrame."""
        # LAZY LOAD: Import synthesizers only when fit() is called
        # This keeps app startup fast - SDV libraries load on-demand
        from sdv.single_table import (
            CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
        )
        from sdv.multi_table import HMASynthesizer
        from sdv.sequential import PARSynthesizer
        
        algo = self.algorithm
        if algo == "CTGAN":
            self.synthesizer = CTGANSynthesizer(
                self.metadata,
                enforce_rounding=False,
                epochs=self.kwargs.get("epochs", 500),
                verbose=True
            )

        elif algo == "GaussianCopula":
            self.synthesizer = GaussianCopulaSynthesizer(
                self.metadata
                )
            
        elif algo == "TVAE":
            self.synthesizer = TVAESynthesizer(
                self.metadata,
                enforce_min_max_values=True,
                enforce_rounding=False,
                epochs=self.kwargs.get("epochs", 500)
            )

        elif algo == "HMAS":
            self.synthesizer = HMASynthesizer(
                self.metadata
                )
            
        elif algo == "PARS":
            self.synthesizer = PARSynthesizer(
                self.metadata,
                context_columns=self.context_columns,
                verbose=True
            )

        else:
            raise ValueError(
                f"Invalid synthesizer algorithm chosen: {algo}"
                )
        
        print(f"Fitting SDV synthesizer: {algo}")
        self.synthesizer.fit(real_data)
        return self

    def sample(self, num_rows=1000, num_sequences=None, sequence_length=None):
        """Sample synthetic data from trained synthesizer."""
        algo = self.algorithm
        if algo == "PARS" and num_sequences and sequence_length:
            return self.synthesizer.sample(
                num_sequences=num_sequences,
                sequence_length=sequence_length
            )
        else:
            return self.synthesizer.sample(num_rows=num_rows)

    def generate(self, real_data, **sample_kwargs):
        """
        Fit and sample in one call. Returns DataFrame.
        sample_kwargs: num_rows, num_sequences, sequence_length
        """
        self.fit(real_data)
        return self.sample(**sample_kwargs)