advanced_models = {
    "SynthCity - Diffusion (DDPM)": "ddpm",
    "SynthCity - CTGAN": "ctgan",
    "SynthCity - TVAE": "tvae",
    "SynthCity - PrivBayes": "privbayes",
    "SynthCity - DP-GAN": "dpgan",
    "SynthCity - PATE-GAN": "pategan",
    "SynthCity - AutoML (Fallback to ARF)": "arf"
}

metric_cols = [
    'KolmogorovSmirnov',
    'JensenShannon',
    'Wasserstein',
    'MaxMeanDiscrep'
]
