"""
src/core/feature_engineering/feature_suggester.py

Modular AI-driven feature suggestion for DataMimicAI.
Breaks suggestion logic into helper methods for clarity and extensibility.

Usage:
    from src.core.feature_engineering.feature_suggester import (
        FeatureSuggester, FeatureEngConfig, FeatureEngError
    )
    suggester = FeatureSuggester(df, config=FeatureEngConfig())
    result = suggester.suggest(target_col="target")
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ===========================
# 1. Custom Error Type
# ===========================

class FeatureEngError(Exception):
    """Raised when an error occurs in feature suggestion/engineering."""

# ===========================
# 2. Config Class
# ===========================

@dataclass(frozen=True)
class FeatureEngConfig:
    max_pairs: int = 5        # Max top pairs for numeric interactions
    top_n_importance: int = 5 # Top N features by importance (RF)
    max_per_type: int = 4     # Max features per type per column

# ===========================
# 3. Main FeatureSuggester Class
# ===========================

class FeatureSuggester:
    """
    Modular AI-powered feature suggestion and application engine.
    """

    def __init__(self, df: pd.DataFrame, config: Optional[FeatureEngConfig] = None):
        if df is None or not isinstance(df, pd.DataFrame):
            raise FeatureEngError("Input must be a pandas DataFrame.")
        self.df = df
        self.config = config or FeatureEngConfig()

    def suggest(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        suggestions, code_blocks, explanations = [], [], []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = self.df.select_dtypes(include=['object', 'string']).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Suggest for each feature type
        self._suggest_numeric_features(numeric_cols, suggestions, code_blocks, explanations)
        self._suggest_numeric_interactions(numeric_cols, suggestions, code_blocks, explanations)
        self._suggest_categorical_features(cat_cols, suggestions, code_blocks, explanations)
        self._suggest_text_features(object_cols, suggestions, code_blocks, explanations)
        importance_scores = self._suggest_feature_importance(target_col, suggestions, code_blocks, explanations)

        return {
            "suggestions": suggestions,
            "code_blocks": code_blocks,
            "explanations": explanations,
            "feature_importance": importance_scores
        }

    def _suggest_numeric_features(self, numeric_cols, suggestions, code_blocks, explanations):
        max_per_type = self.config.max_per_type
        for i, col in enumerate(numeric_cols):
            col_skew = self.df[col].skew()
            col_min = self.df[col].min()
            col_nuniq = self.df[col].nunique()

            if col_min > 0 and abs(col_skew) > 0.5:
                suggestions.append(f"Log-transform '{col}' (skewness: {col_skew:.2f})")
                code_blocks.append(f"df['{col}_log'] = np.log1p(df['{col}'])")
                explanations.append(f"Log-transforming '{col}' can reduce skewness and improve linear model performance.")

            if col_min >= 0:
                suggestions.append(f"Square-root transform '{col}'")
                code_blocks.append(f"df['{col}_sqrt'] = np.sqrt(df['{col}'])")
                explanations.append(f"Square-root can help stabilize variance, especially for count/frequency features.")

            if (self.df[col] != 0).all():
                suggestions.append(f"Inverse transform '1/{col}'")
                code_blocks.append(f"df['{col}_inv'] = 1.0 / df['{col}']")
                explanations.append(f"Inverse transformation can highlight differences between large values of '{col}'.")

            suggestions.append(f"Z-score standardize '{col}'")
            code_blocks.append(f"df['{col}_zscore'] = (df['{col}'] - df['{col}'].mean()) / df['{col}'].std()")
            explanations.append(f"Z-score normalization centers and scales '{col}' to improve ML stability.")

            suggestions.append(f"Min-max normalize '{col}'")
            code_blocks.append(f"df['{col}_minmax'] = (df['{col}'] - df['{col}'].min()) / (df['{col}'].max() - df['{col}'].min())")
            explanations.append(f"Min-max normalization rescales '{col}' between 0 and 1 for comparability.")

            if col_nuniq > 5:
                suggestions.append(f"Add '{col}^2' (quadratic feature)")
                code_blocks.append(f"df['{col}_squared'] = df['{col}'] ** 2")
                explanations.append(f"Quadratic features can help capture nonlinear effects in '{col}'.")

                suggestions.append(f"Add '{col}^3' (cubic feature)")
                code_blocks.append(f"df['{col}_cubed'] = df['{col}'] ** 3")
                explanations.append(f"Cubic features can help capture complex nonlinear relationships.")

            suggestions.append(f"Bin '{col}' into quartiles")
            code_blocks.append(f"df['{col}_bin'] = pd.qcut(df['{col}'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])")
            explanations.append(f"Binning '{col}' can reduce noise and help tree-based models.")

            suggestions.append(f"Rank encode '{col}'")
            code_blocks.append(f"df['{col}_rank'] = df['{col}'].rank(method='average')")
            explanations.append(f"Rank encoding transforms '{col}' into its ordinal rank, useful for monotonic trends.")

            suggestions.append(f"Flag outliers in '{col}' (z-score > 3)")
            code_blocks.append(f"df['{col}_outlier'] = ((df['{col}'] - df['{col}'].mean()) / df['{col}'].std()).abs() > 3")
            explanations.append(f"Flagging outliers helps models avoid being overly influenced by extreme values.")

            if i + 1 >= max_per_type:
                break

    def _suggest_numeric_interactions(self, numeric_cols, suggestions, code_blocks, explanations):
        if len(numeric_cols) > 1:
            corr = self.df[numeric_cols].corr().abs()
            pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
            )
            for (col1, col2), val in pairs.head(self.config.max_pairs).items():
                if val > 0.4:
                    feat_name = f"{col1}_x_{col2}"
                    suggestions.append(f"Interaction: {col1} × {col2} (corr: {val:.2f})")
                    code_blocks.append(f"df['{feat_name}'] = df['{col1}'] * df['{col2}']")
                    explanations.append(f"Combining '{col1}' and '{col2}' can capture important interactions.")

    def _suggest_categorical_features(self, cat_cols, suggestions, code_blocks, explanations):
        max_per_type = self.config.max_per_type
        for i, col in enumerate(cat_cols):
            nunique = self.df[col].nunique()
            if nunique <= 12:
                suggestions.append(f"One-hot encode '{col}'")
                code_blocks.append(f"df = pd.get_dummies(df, columns=['{col}'], drop_first=True)")
                explanations.append(f"One-hot encoding '{col}' allows ML models to use categorical information.")
            else:
                suggestions.append(f"Label encode '{col}' (many categories)")
                code_blocks.append(f"df['{col}_le'] = df['{col}'].astype('category').cat.codes")
                explanations.append(f"Label encoding transforms many-category '{col}' into numbers.")

            suggestions.append(f"Frequency encode '{col}'")
            code_blocks.append(f"df['{col}_freq'] = df['{col}'].map(df['{col}'].value_counts(normalize=True))")
            explanations.append(f"Frequency encoding represents '{col}' by the frequency of each category.")

            if i + 1 >= max_per_type:
                break

    def _suggest_text_features(self, object_cols, suggestions, code_blocks, explanations):
        max_per_type = self.config.max_per_type
        for i, col in enumerate(object_cols):
            suggestions.append(f"Add length of '{col}' strings")
            code_blocks.append(f"df['{col}_len'] = df['{col}'].str.len()")
            explanations.append(f"Length of '{col}' can be a useful feature for ML (e.g., longer text might mean ...).")

            suggestions.append(f"Add word count for '{col}'")
            code_blocks.append(f"df['{col}_wordcount'] = df['{col}'].str.split().apply(len)")
            explanations.append(f"Word count in '{col}' may reflect information density.")

            if self.df[col].str.len().mean() > 15:
                suggestions.append(f"Embed '{col}' using NLP models")
                code_blocks.append(
                    f"# TODO: Embed '{col}' with TF-IDF or transformer models\n"
                    f"# df['{col}_emb'] = embed_text_column(df['{col}'])"
                )
                explanations.append(f"Text embeddings convert '{col}' into useful numeric vectors for ML.")

            if i + 1 >= max_per_type:
                break

    def _suggest_feature_importance(self, target_col, suggestions, code_blocks, explanations):
        importance_scores = []
        if target_col and target_col in self.df.columns:
            X = self.df.dropna(subset=[target_col]).drop(target_col, axis=1)
            y = self.df.dropna(subset=[target_col])[target_col]
            X = pd.get_dummies(X.select_dtypes(include=[np.number, 'category', 'object']), drop_first=True)
            if X.shape[1] > 0 and len(y) > 10:
                try:
                    if y.nunique() <= 10:
                        model = RandomForestClassifier(n_estimators=30, random_state=0)
                    else:
                        model = RandomForestRegressor(n_estimators=30, random_state=0)
                    model.fit(X, y)
                    importances = pd.Series(model.feature_importances_, index=X.columns)
                    importances = importances.sort_values(ascending=False).head(self.config.top_n_importance)
                    for feat, val in importances.items():
                        importance_scores.append({"feature": feat, "importance": float(val)})
                    explanations.append(
                        "Feature importance is estimated via Random Forest; higher = more predictive for your target."
                    )
                except Exception as e:
                    explanations.append(f"Feature importance estimation failed: {e}")
        return importance_scores

    def apply(self, code_blocks: List[str]) -> pd.DataFrame:
        """
        Applies a list of Python code blocks to a copy of the DataFrame.
        """
        import copy
        df = self.df.copy()
        safe_globals = {"np": np, "pd": pd}
        safe_locals = {"df": df}
        applied = []
        errors = []
        for line in code_blocks:
            if not line.strip().startswith("#"):
                try:
                    exec(line, safe_globals, safe_locals)
                    df = safe_locals.get("df", df)
                    applied.append(line)
                except Exception as e:
                    errors.append((line, str(e)))
        if errors:
            raise FeatureEngError(f"Feature application errors: {errors}")
        return df

    def export_pipeline_code(self, code_blocks: List[str]) -> str:
        """
        Export all selected feature code as a ready-to-use function.
        """
        pipeline_code = (
            "def feature_engineering_pipeline(df):\n"
            "    '''Apply AI-suggested feature engineering to a DataFrame.'''\n"
            "    import numpy as np\n"
            "    import pandas as pd\n"
            "    # --- AI-generated feature engineering ---\n"
        )
        for line in code_blocks:
            for l in line.split('\n'):
                pipeline_code += "    " + l + "\n"
        pipeline_code += "    return df"
        return pipeline_code

    def explain_feature(self, feature_code: str) -> str:
        """
        Provide a simple natural language explanation for a code block.
        """
        if "np.log" in feature_code or "log1p" in feature_code:
            return "Log-transforming a numeric feature reduces skew and can help models handle outliers."
        if "** 2" in feature_code or "_squared" in feature_code:
            return "Polynomial (squared) features help capture nonlinear effects for regression and classification."
        if "get_dummies" in feature_code:
            return "One-hot encoding converts categorical variables into binary columns, making them usable for ML."
        return "This feature was engineered to enhance model learning or handle special data characteristics."