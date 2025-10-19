import os

# API base used by frontend. Can be overridden by setting API_URL env var.
API_BASE = os.getenv("API_URL", "http://localhost:8000")

# Optional S3 upload configuration. These are used only if enabled in the
# environment. Frontend will only attempt S3 uploads when S3_UPLOAD_ENABLED is
# set to a truthy value (1/true/yes).
S3_UPLOAD_ENABLED = str(os.getenv("S3_UPLOAD_ENABLED", "false")).lower() in ("1", "true", "yes")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_REGION = os.getenv("S3_REGION", "")

ALGORITHM_INFO = {
    "CTGAN": {
        "desc": "Generates high-quality synthetic data for large, complex, and mixed-type tabular datasets.",
        "use": "Best for: Large datasets, categorical + numerical columns, data with imbalanced classes."
    },
    "GaussianCopula": {
        "desc": "Fast, robust algorithm for mostly numeric tabular data with moderate size.",
        "use": "Best for: Small to medium datasets, mostly numeric columns."
    },
    "TVAE": {
        "desc": "Neural network-based method for generating synthetic data; handles complex relationships.",
        "use": "Best for: Tabular data where deep learning-based synthesis is needed."
    },
    "PARS": {
        "desc": "Pattern-based synthesis for sequential/time-series data.",
        "use": "Best for: Time-series or sequence data (e.g., stock prices, sensor readings)."
    }
}

ALGORITHM_INFO_SYNTHCITY = {
    "ddpm": {
        "desc": "State-of-the-art Diffusion Model for highly realistic, privacy-aware tabular data. Robust even for complex and imbalanced datasets.",
        "use": "Best for: High-fidelity synthesis, enterprise datasets, privacy-critical scenarios."
    },
    "ctgan": {
        "desc": "GAN-based algorithm for mixed-type tabular data with strong deep feature modeling.",
        "use": "Best for: Data with rare categorical events, imbalanced classes, healthcare, or finance."
    },
    "tvae": {
        "desc": "Neural variational autoencoder for tabular data. Balances speed and data utility.",
        "use": "Best for: General-purpose tables and benchmarking against VAE/SDV."
    },
    "privbayes": {
        "desc": "Generates differentially private synthetic data using Bayesian networks.",
        "use": "Best for: Privacy-first generation, small tabular datasets with regulatory constraints."
    },
    "dpgan": {
        "desc": "Differentially private GAN for strong privacy plus high utility.",
        "use": "Best for: Synthetic data in compliance contexts or regulated industries."
    },
    "pategan": {
        "desc": "PATE-GAN architecture for the strongest privacy-utility tradeoffs.",
        "use": "Best for: Highly sensitive datasets, e.g. medical records."
    },
    "arf": {
        "desc": "AutoML Random Forest fallback. A non-deep-learning baseline model.",
        "use": "Best for: When neural methods are overkill or infeasible."
    }
}

DISPLAY_NAME_TO_ALGORITHM = {
    "SynthCity - Diffusion (DDPM)": "ddpm",
    "SynthCity - CTGAN": "ctgan", 
    "SynthCity - TVAE": "tvae",
    "SynthCity - PrivBayes": "privbayes",
    "SynthCity - DP-GAN": "dpgan",
    "SynthCity - PATE-GAN": "pategan",
    "SynthCity - AutoML (Fallback to ARF)": "arf"
}


FRIENDLY_ALGO_LABELS = {
    "ddpm":      "DDPM: Complex & privacy-safe data",
    "ctgan":     "CTGAN: Imbalanced/tabular data",
    "tvae":      "TVAE: General-purpose tables",
    "privbayes": "PrivBayes: Privacy-first, small data",
    "dpgan":     "DP-GAN: Compliance-critical use",
    "pategan":   "PATE-GAN: Medical/sensitive data",
    "arf":       "ARF: Fast non-DL fallback"
}
