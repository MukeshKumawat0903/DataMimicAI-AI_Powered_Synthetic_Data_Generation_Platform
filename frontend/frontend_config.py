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
