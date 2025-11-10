"""
Privacy-Driven Feature Transformation Suggester

Analyzes PII detection and k-anonymity risk to recommend privacy-enhancing transformations (PETs):
- Redaction/Anonymization for direct PII
- Generalization/Binning for quasi-identifiers
- Suppression for high re-identification risk
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PrivacySuggestion:
    """Represents a privacy-driven transformation suggestion."""
    column: str
    transformation: str
    reason: str
    risk_level: str  # "high", "medium", "low"
    pii_types: List[str]
    params: Dict[str, Any]
    metrics: Dict[str, Any]


class PrivacySuggester:
    """
    Generates privacy-driven transformation suggestions based on PII and k-anonymity analysis.
    """
    
    def __init__(self, df: pd.DataFrame, pii_report: Optional[Dict] = None, k_anonymity_report: Optional[Dict] = None):
        """
        Initialize privacy suggester.
        
        Args:
            df: Input DataFrame
            pii_report: PII detection results (from pii_scan.py)
            k_anonymity_report: k-anonymity risk assessment (from privacy.py)
        """
        self.df = df
        self.pii_report = pii_report or {}
        self.k_anonymity_report = k_anonymity_report or {}
        self.suggestions = []
    
    def suggest_privacy_transforms(self) -> List[Dict[str, Any]]:
        """
        Generate all privacy-driven transformation suggestions.
        
        Returns:
            List of suggestion dictionaries
        """
        self.suggestions = []
        
        # Process PII detections
        if self.pii_report and "detections" in self.pii_report:
            for detection in self.pii_report["detections"]:
                pii_suggestions = self._suggest_for_pii(detection)
                self.suggestions.extend(pii_suggestions)
        
        # Process k-anonymity risks
        if self.k_anonymity_report and "potential_qis" in self.k_anonymity_report:
            for qi_info in self.k_anonymity_report["potential_qis"]:
                qi_suggestions = self._suggest_for_quasi_identifier(qi_info)
                self.suggestions.extend(qi_suggestions)
        
        # Deduplicate suggestions (prefer higher risk)
        self.suggestions = self._deduplicate_suggestions(self.suggestions)
        
        return [self._suggestion_to_dict(s) for s in self.suggestions]
    
    def _suggest_for_pii(self, detection: Dict[str, Any]) -> List[PrivacySuggestion]:
        """
        Generate suggestions for detected PII columns.
        
        Args:
            detection: PII detection dictionary
            
        Returns:
            List of PrivacySuggestion objects
        """
        suggestions = []
        column = detection.get("column")
        pii_types = detection.get("pii_types", [])
        confidence = detection.get("confidence", 0.0)
        
        if not column or not pii_types:
            return suggestions
        
        # Direct identifiers - recommend redaction/anonymization
        direct_pii = ["EMAIL", "SSN", "PHONE", "CREDIT_CARD", "PHONE_US"]
        
        if any(pii_type in direct_pii for pii_type in pii_types):
            risk_level = "high"
            
            # Suggest redaction
            suggestions.append(PrivacySuggestion(
                column=column,
                transformation="redact",
                reason=f"Direct PII detected ({', '.join(pii_types)}). Redaction eliminates re-identification risk by removing sensitive data.",
                risk_level=risk_level,
                pii_types=pii_types,
                params={"replacement": "[REDACTED]"},
                metrics={"confidence": confidence, "pii_count": len(pii_types)}
            ))
            
            # Also suggest hashing as alternative
            suggestions.append(PrivacySuggestion(
                column=column,
                transformation="hash",
                reason=f"Direct PII detected ({', '.join(pii_types)}). One-way hashing preserves uniqueness while protecting privacy.",
                risk_level=risk_level,
                pii_types=pii_types,
                params={"algorithm": "sha256", "salt": True},
                metrics={"confidence": confidence, "pii_count": len(pii_types)}
            ))
        
        # Indirect identifiers - recommend masking
        elif any(pii_type in ["NAME", "ADDRESS", "URL", "IP_ADDRESS"] for pii_type in pii_types):
            risk_level = "medium"
            
            suggestions.append(PrivacySuggestion(
                column=column,
                transformation="mask",
                reason=f"Indirect PII detected ({', '.join(pii_types)}). Partial masking retains format while reducing identifiability.",
                risk_level=risk_level,
                pii_types=pii_types,
                params={"mask_percentage": 0.5, "preserve_format": True},
                metrics={"confidence": confidence, "pii_count": len(pii_types)}
            ))
        
        return suggestions
    
    def _suggest_for_quasi_identifier(self, qi_info: Dict[str, Any]) -> List[PrivacySuggestion]:
        """
        Generate suggestions for quasi-identifier columns based on k-anonymity risk.
        
        Args:
            qi_info: Quasi-identifier information dictionary
            
        Returns:
            List of PrivacySuggestion objects
        """
        suggestions = []
        column = qi_info.get("column")
        risk_level = qi_info.get("risk_level", "Low").lower()
        uniqueness_ratio = qi_info.get("uniqueness_ratio", 0.0)
        unique_values = qi_info.get("unique_values", 0)
        
        if not column:
            return suggestions
        
        # Only suggest for medium/high risk QIs
        if risk_level not in ["medium", "high"]:
            return suggestions
        
        col_data = self.df[column] if column in self.df.columns else None
        if col_data is None:
            return suggestions
        
        # Determine optimal binning strategy based on data type and cardinality
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        if is_numeric:
            # For numeric QIs, suggest binning
            n_bins = self._calculate_optimal_bins(unique_values, risk_level)
            
            suggestions.append(PrivacySuggestion(
                column=column,
                transformation="bin",
                reason=f"Quasi-identifier with {risk_level} re-identification risk (uniqueness: {uniqueness_ratio:.1%}). Binning into {n_bins} groups reduces granularity and risk.",
                risk_level=risk_level,
                pii_types=["QUASI_IDENTIFIER"],
                params={"n_bins": n_bins, "strategy": "quantile"},
                metrics={"uniqueness_ratio": uniqueness_ratio, "unique_values": unique_values}
            ))
        else:
            # For categorical QIs, suggest generalization
            suggestions.append(PrivacySuggestion(
                column=column,
                transformation="generalize",
                reason=f"Categorical quasi-identifier with {risk_level} re-identification risk (uniqueness: {uniqueness_ratio:.1%}). Generalization groups rare values to reduce risk.",
                risk_level=risk_level,
                pii_types=["QUASI_IDENTIFIER"],
                params={"min_frequency": 5, "replacement": "Other"},
                metrics={"uniqueness_ratio": uniqueness_ratio, "unique_values": unique_values}
            ))
        
        # For high-risk QIs, also suggest suppression
        if risk_level == "high" and uniqueness_ratio > 0.5:
            suggestions.append(PrivacySuggestion(
                column=column,
                transformation="suppress",
                reason=f"Very high re-identification risk (uniqueness: {uniqueness_ratio:.1%}). Suppression removes the column entirely to eliminate risk.",
                risk_level=risk_level,
                pii_types=["QUASI_IDENTIFIER"],
                params={},
                metrics={"uniqueness_ratio": uniqueness_ratio, "unique_values": unique_values}
            ))
        
        return suggestions
    
    def _calculate_optimal_bins(self, unique_values: int, risk_level: str) -> int:
        """
        Calculate optimal number of bins based on data characteristics and risk level.
        
        Args:
            unique_values: Number of unique values in the column
            risk_level: Privacy risk level
            
        Returns:
            Optimal number of bins
        """
        # Base bins on risk level
        base_bins = {
            "high": 3,
            "medium": 5,
            "low": 7
        }
        
        n_bins = base_bins.get(risk_level, 5)
        
        # Adjust based on cardinality
        if unique_values < 10:
            n_bins = min(n_bins, 3)
        elif unique_values < 20:
            n_bins = min(n_bins, 4)
        
        return n_bins
    
    def _deduplicate_suggestions(self, suggestions: List[PrivacySuggestion]) -> List[PrivacySuggestion]:
        """
        Remove duplicate suggestions, keeping the one with highest risk.
        
        Args:
            suggestions: List of suggestions
            
        Returns:
            Deduplicated list
        """
        column_suggestions = {}
        risk_priority = {"high": 3, "medium": 2, "low": 1}
        
        for suggestion in suggestions:
            col = suggestion.column
            
            if col not in column_suggestions:
                column_suggestions[col] = suggestion
            else:
                # Keep suggestion with higher risk level
                current_priority = risk_priority.get(column_suggestions[col].risk_level, 0)
                new_priority = risk_priority.get(suggestion.risk_level, 0)
                
                if new_priority > current_priority:
                    column_suggestions[col] = suggestion
        
        return list(column_suggestions.values())
    
    def _suggestion_to_dict(self, suggestion: PrivacySuggestion) -> Dict[str, Any]:
        """Convert PrivacySuggestion to dictionary."""
        return {
            "column": suggestion.column,
            "transformation": suggestion.transformation,
            "reason": suggestion.reason,
            "risk_level": suggestion.risk_level,
            "pii_types": suggestion.pii_types,
            "params": suggestion.params,
            "metrics": suggestion.metrics,
            "category": "privacy"
        }
    
    def get_transform_code(self, suggestion: Dict[str, Any]) -> str:
        """
        Generate Python code snippet for applying a privacy transformation.
        
        Args:
            suggestion: Suggestion dictionary
            
        Returns:
            Python code string
        """
        col = suggestion["column"]
        transform = suggestion["transformation"]
        params = suggestion.get("params", {})
        
        code_templates = {
            "redact": f"df['{col}'] = '{params.get('replacement', '[REDACTED]')}'",
            "hash": f"import hashlib\ndf['{col}_hashed'] = df['{col}'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())",
            "mask": f"df['{col}_masked'] = df['{col}'].apply(lambda x: x[:len(x)//2] + '*' * (len(x) - len(x)//2) if isinstance(x, str) else x)",
            "bin": f"df['{col}_binned'] = pd.qcut(df['{col}'], q={params.get('n_bins', 5)}, duplicates='drop')",
            "generalize": f"value_counts = df['{col}'].value_counts()\nrare_values = value_counts[value_counts < {params.get('min_frequency', 5)}].index\ndf['{col}_generalized'] = df['{col}'].apply(lambda x: '{params.get('replacement', 'Other')}' if x in rare_values else x)",
            "suppress": f"df = df.drop(columns=['{col}'])"
        }
        
        return code_templates.get(transform, f"# Privacy transform: {transform}")
