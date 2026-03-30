/**
 * Algorithm metadata — mirrors frontend/frontend_config.py ALGORITHM_INFO
 */

export interface AlgorithmInfo {
  desc: string;
  use: string;
}

export const SDV_ALGORITHM_INFO: Record<string, AlgorithmInfo> = {
  CTGAN: {
    desc: "Generates high-quality synthetic data for large, complex, and mixed-type tabular datasets.",
    use: "Best for: Large datasets, categorical + numerical columns, data with imbalanced classes.",
  },
  GaussianCopula: {
    desc: "Fast, robust algorithm for mostly numeric tabular data with moderate size.",
    use: "Best for: Small to medium datasets, mostly numeric columns.",
  },
  TVAE: {
    desc: "Neural network-based method for generating synthetic data; handles complex relationships.",
    use: "Best for: Tabular data where deep learning-based synthesis is needed.",
  },
  PARS: {
    desc: "Pattern-based synthesis for sequential/time-series data.",
    use: "Best for: Time-series or sequence data (e.g., stock prices, sensor readings).",
  },
};

export const SYNTHCITY_ALGORITHM_INFO: Record<string, AlgorithmInfo> = {
  ddpm: {
    desc: "State-of-the-art Diffusion Model for highly realistic, privacy-aware tabular data. Robust even for complex and imbalanced datasets.",
    use: "Best for: High-fidelity synthesis, enterprise datasets, privacy-critical scenarios.",
  },
  ctgan: {
    desc: "GAN-based algorithm for mixed-type tabular data with strong deep feature modeling.",
    use: "Best for: Data with rare categorical events, imbalanced classes, healthcare, or finance.",
  },
  tvae: {
    desc: "Neural variational autoencoder for tabular data. Balances speed and data utility.",
    use: "Best for: General-purpose tables and benchmarking against VAE/SDV.",
  },
  privbayes: {
    desc: "Generates differentially private synthetic data using Bayesian networks.",
    use: "Best for: Privacy-first generation, small tabular datasets with regulatory constraints.",
  },
  dpgan: {
    desc: "Differentially private GAN for strong privacy plus high utility.",
    use: "Best for: Synthetic data in compliance contexts or regulated industries.",
  },
  pategan: {
    desc: "PATE-GAN architecture for the strongest privacy-utility tradeoffs.",
    use: "Best for: Highly sensitive datasets, e.g. medical records.",
  },
  arf: {
    desc: "AutoML Random Forest fallback. A non-deep-learning baseline model.",
    use: "Best for: When neural methods are overkill or infeasible.",
  },
};

export const SDV_ALGORITHMS = Object.keys(SDV_ALGORITHM_INFO) as string[];
export const SYNTHCITY_ALGORITHMS = Object.keys(SYNTHCITY_ALGORITHM_INFO) as string[];

export const PRESET_CONFIGS: Record<string, { num_rows: number; epochs: number }> = {
  "Quick (small)": { num_rows: 200, epochs: 100 },
  Balanced: { num_rows: 1000, epochs: 300 },
  "High Fidelity": { num_rows: 10000, epochs: 500 },
};
