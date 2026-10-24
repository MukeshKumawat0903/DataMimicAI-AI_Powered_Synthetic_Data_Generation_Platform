advanced_models = {
    "SynthCity - Diffusion (DDPM)": "ddpm",
    "SynthCity - CTGAN": "ctgan",
    "SynthCity - TVAE": "tvae",
    "SynthCity - PrivBayes": "privbayes",
    "SynthCity - DP-GAN": "dpgan",
    "SynthCity - PATE-GAN": "pategan",
    "SynthCity - AutoML (Fallback to ARF)": "arf"
}

# Name of each SynthCity plugin's training-iterations parameter (None = no such parameter)
SYNTHCITY_ITERATION_PARAMS = {
    'ddpm': 'n_iter',
    'ctgan': 'n_iter',
    'tvae': 'n_iter',
    'dpgan': 'n_iter',
    'pategan': 'n_iter',
    'privbayes': None,
    'arf': None
}

metric_cols = [
    'KolmogorovSmirnov',
    'JensenShannon',
    'Wasserstein',
    'MaxMeanDiscrep'
]
