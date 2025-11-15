"""
Privacy Risk Assessment Module
Implements k-anonymity analysis for re-identification risk assessment.
"""
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class KAnonymityAnalyzer:
    """
    Analyzes re-identification risk using k-anonymity metrics.
    k-anonymity: Each record is indistinguishable from at least k-1 other records
    based on quasi-identifiers (QIs).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize k-anonymity analyzer.
        
        Args:
            df: Input DataFrame
        """
        self.df = df
        self.qi_columns = []
        self.k_values = None
        self.risk_analysis = None
    
    def identify_potential_qis(self, 
                               max_cardinality: int = 100,
                               min_cardinality: int = 2) -> List[Dict]:
        """
        Identify columns that could be quasi-identifiers.
        
        QI candidates are typically:
        - Categorical or ordinal columns
        - Numeric columns with reasonable cardinality
        - Not unique identifiers (too high cardinality)
        - Not constants (cardinality = 1)
        
        Args:
            max_cardinality: Maximum unique values for QI candidate
            min_cardinality: Minimum unique values for QI candidate
        
        Returns:
            List of potential QI column info
        """
        potential_qis = []
        
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            
            # Skip if outside cardinality range
            if unique_count < min_cardinality or unique_count > max_cardinality:
                continue
            
            # Calculate uniqueness ratio
            uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Categorize risk level based on uniqueness
            if uniqueness_ratio < 0.01:
                risk_level = "Low"
            elif uniqueness_ratio < 0.1:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            potential_qis.append({
                "column": col,
                "unique_values": int(unique_count),
                "uniqueness_ratio": round(uniqueness_ratio, 4),
                "risk_level": risk_level,
                "dtype": str(self.df[col].dtype),
                "sample_values": self._get_sample_values(col, n=5)
            })
        
        # Sort by risk level (High first) and uniqueness ratio
        risk_order = {"High": 3, "Medium": 2, "Low": 1}
        potential_qis.sort(
            key=lambda x: (risk_order[x["risk_level"]], x["uniqueness_ratio"]),
            reverse=True
        )
        
        return potential_qis
    
    def _get_sample_values(self, column: str, n: int = 5) -> List[str]:
        """Get sample values from a column."""
        sample = self.df[column].dropna().head(n).astype(str).tolist()
        return sample
    
    def compute_k_anonymity(self, quasi_identifiers: List[str]) -> Dict:
        """
        Compute k-anonymity for each record based on selected QIs.
        
        Args:
            quasi_identifiers: List of column names to use as QIs
        
        Returns:
            Dict with k-values and risk statistics
        """
        self.qi_columns = quasi_identifiers
        
        # Validate QI columns exist
        missing_cols = [col for col in quasi_identifiers if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Quasi-identifier columns not found: {missing_cols}")
        
        # Group by QI combination and count group sizes
        qi_groups = self.df.groupby(quasi_identifiers, dropna=False).size()
        
        # Create a mapping of QI combination → group size (k-value)
        k_mapping = qi_groups.to_dict()
        
        # Assign k-value to each record
        k_values = []
        for idx, row in self.df.iterrows():
            qi_tuple = tuple(row[qi] for qi in quasi_identifiers)
            k = k_mapping.get(qi_tuple, 1)
            k_values.append(k)
        
        self.k_values = pd.Series(k_values, index=self.df.index)
        
        # Calculate risk statistics
        self.risk_analysis = self._analyze_risk()
        
        return {
            "k_values": self.k_values.tolist(),
            "risk_analysis": self.risk_analysis,
            "qi_columns": quasi_identifiers
        }
    
    def _analyze_risk(self) -> Dict:
        """
        Analyze re-identification risk based on k-values.
        
        Returns:
            Dict with risk metrics and distribution
        """
        if self.k_values is None:
            raise ValueError("Must call compute_k_anonymity first")
        
        total_records = len(self.k_values)
        
        # Calculate risk for different k thresholds
        k_thresholds = [1, 2, 3, 5, 10, 20, 50, 100]
        risk_by_k = []
        
        for k_threshold in k_thresholds:
            at_risk_count = (self.k_values < k_threshold).sum()
            at_risk_pct = (at_risk_count / total_records * 100) if total_records > 0 else 0
            
            risk_by_k.append({
                "k": k_threshold,
                "at_risk_count": int(at_risk_count),
                "at_risk_percentage": round(at_risk_pct, 2)
            })
        
        # Overall statistics
        stats = {
            "min_k": int(self.k_values.min()),
            "max_k": int(self.k_values.max()),
            "mean_k": round(float(self.k_values.mean()), 2),
            "median_k": int(self.k_values.median()),
            "unique_records": int((self.k_values == 1).sum()),
            "unique_percentage": round((self.k_values == 1).sum() / total_records * 100, 2)
        }
        
        # Risk severity classification
        if stats["unique_percentage"] > 10:
            severity = "CRITICAL"
        elif stats["unique_percentage"] > 5:
            severity = "HIGH"
        elif stats["unique_percentage"] > 1:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        return {
            "statistics": stats,
            "risk_by_k": risk_by_k,
            "severity": severity,
            "distribution": self._get_k_distribution()
        }
    
    def _get_k_distribution(self) -> List[Dict]:
        """
        Get distribution of k-values.
        
        Returns:
            List of k-value counts
        """
        if self.k_values is None:
            return []
        
        value_counts = self.k_values.value_counts().sort_index()
        
        distribution = []
        for k, count in value_counts.items():
            distribution.append({
                "k": int(k),
                "count": int(count),
                "percentage": round(count / len(self.k_values) * 100, 2)
            })
        
        return distribution[:20]  # Limit to top 20 for visualization
    
    def get_vulnerable_records(self, k_threshold: int = 5) -> pd.DataFrame:
        """
        Get records that are vulnerable (k < threshold).
        
        Args:
            k_threshold: K-value threshold for vulnerability
        
        Returns:
            DataFrame of vulnerable records with their k-values
        """
        if self.k_values is None:
            raise ValueError("Must call compute_k_anonymity first")
        
        vulnerable_mask = self.k_values < k_threshold
        vulnerable_df = self.df[vulnerable_mask].copy()
        vulnerable_df['k_value'] = self.k_values[vulnerable_mask]
        
        return vulnerable_df
    
    def generate_risk_report(self) -> Dict:
        """
        Generate comprehensive privacy risk report.
        
        Returns:
            Dict with complete risk assessment
        """
        if self.risk_analysis is None:
            raise ValueError("Must call compute_k_anonymity first")
        
        stats = self.risk_analysis["statistics"]
        
        # Generate recommendations
        recommendations = []
        
        if stats["unique_percentage"] > 5:
            recommendations.append({
                "priority": "HIGH",
                "issue": f"{stats['unique_percentage']:.1f}% of records are unique (k=1)",
                "recommendation": "Consider removing or generalizing high-cardinality quasi-identifiers"
            })
        
        if stats["mean_k"] < 10:
            recommendations.append({
                "priority": "MEDIUM",
                "issue": f"Average k-anonymity is only {stats['mean_k']:.1f}",
                "recommendation": "Apply data suppression or generalization to increase k-values"
            })
        
        if stats["min_k"] == 1:
            recommendations.append({
                "priority": "HIGH",
                "issue": "Some records are completely unique",
                "recommendation": "Remove or generalize identifying attributes"
            })
        
        return {
            "summary": {
                "quasi_identifiers": self.qi_columns,
                "total_records": len(self.df),
                "severity": self.risk_analysis["severity"],
                "unique_records": stats["unique_records"],
                "average_k": stats["mean_k"]
            },
            "statistics": stats,
            "risk_by_k": self.risk_analysis["risk_by_k"],
            "distribution": self.risk_analysis["distribution"],
            "recommendations": recommendations
        }
    
    def save_to_metadata(self, file_id: str, metadata_path: str = None) -> str:
        """
        Save k-anonymity analysis to metadata storage for generation constraints.
        
        Args:
            file_id: Unique identifier for the dataset
            metadata_path: Optional path to metadata directory
        
        Returns:
            Path to saved metadata file
        """
        if self.risk_analysis is None:
            logger.warning(f"No k-anonymity analysis to save for file_id: {file_id}")
            return None
        
        if metadata_path is None:
            # Default to workspace directory
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            metadata_path = os.path.join(backend_dir, "workspace", "metadata")
        
        import os
        os.makedirs(metadata_path, exist_ok=True)
        
        # Prepare metadata structure
        metadata = {
            "file_id": file_id,
            "timestamp": datetime.now().isoformat(),
            "quasi_identifiers": self.qi_columns,
            "risk_report": self.generate_risk_report(),
            "generation_constraints": self._generate_privacy_constraints()
        }
        
        # Save to JSON file
        output_file = os.path.join(metadata_path, f"k_anonymity_{file_id}.json")
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved k-anonymity analysis (QIs: {self.qi_columns}) to {output_file}")
        return output_file
    
    def _generate_privacy_constraints(self) -> Dict:
        """
        Generate privacy constraints for synthetic data generation.
        
        Returns:
            Dict of constraints that can guide generation
        """
        stats = self.risk_analysis["statistics"]
        
        constraints = {
            "type": "privacy_preservation",
            "quasi_identifiers": self.qi_columns,
            "min_k_target": max(5, stats["median_k"]),  # Target at least k=5 or current median
            "suppress_unique_combinations": stats["unique_percentage"] > 5,
            "generalization_level": self._suggest_generalization_level(),
            "rules": [
                {
                    "rule": "maintain_k_anonymity",
                    "description": f"Ensure k-anonymity >= {max(5, stats['median_k'])} for QI combinations"
                },
                {
                    "rule": "avoid_unique_records",
                    "description": "Suppress or generalize unique QI combinations (k=1)"
                }
            ]
        }
        
        return constraints
    
    def _suggest_generalization_level(self) -> str:
        """Suggest generalization level based on current risk."""
        stats = self.risk_analysis["statistics"]
        
        if stats["unique_percentage"] > 10:
            return "HIGH"  # Aggressive generalization needed
        elif stats["unique_percentage"] > 5:
            return "MEDIUM"
        elif stats["mean_k"] < 5:
            return "MEDIUM"
        else:
            return "LOW"


def load_k_anonymity_metadata(file_id: str, metadata_path: str = None) -> Optional[Dict]:
    """
    Load saved k-anonymity metadata for a dataset.
    
    Args:
        file_id: Dataset identifier
        metadata_path: Optional metadata directory path
    
    Returns:
        Metadata dict or None if not found
    """
    if metadata_path is None:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        metadata_path = os.path.join(backend_dir, "workspace", "metadata")
    
    import os
    metadata_file = os.path.join(metadata_path, f"k_anonymity_{file_id}.json")
    
    if not os.path.exists(metadata_file):
        logger.warning(f"No k-anonymity metadata found for file_id: {file_id}")
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded k-anonymity metadata for file_id: {file_id}")
    return metadata


def compute_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: List[str]
) -> Dict:
    """
    Convenience function to compute k-anonymity.
    
    Args:
        df: Input DataFrame
        quasi_identifiers: List of QI column names
    
    Returns:
        K-anonymity analysis results
    """
    analyzer = KAnonymityAnalyzer(df)
    return analyzer.compute_k_anonymity(quasi_identifiers)


def identify_quasi_identifiers(
    df: pd.DataFrame,
    max_cardinality: int = 100
) -> List[Dict]:
    """
    Convenience function to identify potential QIs.
    
    Args:
        df: Input DataFrame
        max_cardinality: Max unique values for QI candidate
    
    Returns:
        List of potential QI columns
    """
    analyzer = KAnonymityAnalyzer(df)
    return analyzer.identify_potential_qis(max_cardinality=max_cardinality)
