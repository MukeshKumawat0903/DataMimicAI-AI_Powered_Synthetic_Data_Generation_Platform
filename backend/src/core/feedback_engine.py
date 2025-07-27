# src/core/feedback_engine.py

import pandas as pd

class EDAFeedbackEngine:
    """
    Applies EDA/feature feedback to a DataFrame before synthetic generation.
    Extendable: supports 'impute', 'drop', 'encode', 'bin', and more.
    """

    def __init__(self, df: pd.DataFrame, feedback: list):
        self.df = df.copy()
        self.feedback = feedback   # List of feedback dicts

    def apply_feedback(self, return_log=False):
        log = []
        for fb in self.feedback:
            action = fb.get("action")
            col = fb.get("column")
            if action == "impute":
                method = fb.get("method", "mean")
                val = None
                if method == "mean":
                    val = self.df[col].mean()
                elif method == "median":
                    val = self.df[col].median()
                elif method == "mode":
                    val = self.df[col].mode()[0]
                self.df[col].fillna(val, inplace=True)
                log.append(f"Imputed {col} with {method} ({val})")
            elif action == "drop":
                self.df.drop(columns=[col], inplace=True)
                log.append(f"Dropped column {col}")
            elif action == "encode":
                self.df = pd.get_dummies(self.df, columns=[col], drop_first=True)
                log.append(f"One-hot encoded {col}")
            elif action == "bin":
                bins = fb.get("bins", 4)
                self.df[f"{col}_bin"] = pd.cut(self.df[col], bins)
                log.append(f"Binned {col} into {bins} bins")
            # Add more actions here as needed!
        return (self.df, log) if return_log else self.df
