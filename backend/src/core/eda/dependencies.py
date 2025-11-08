"""
Functional Dependency Detection Module
Detects column-level dependencies (1:1, M:1 mappings) to identify data relationships.
"""
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DependencyDetector:
    """
    Detects functional dependencies between columns in a DataFrame.
    A functional dependency X → Y means X determines Y uniquely or with high confidence.
    """
    
    def __init__(self, df: pd.DataFrame, min_confidence: float = 0.95):
        """
        Initialize dependency detector.
        
        Args:
            df: Input DataFrame
            min_confidence: Minimum confidence threshold for dependency (0-1)
        """
        self.df = df
        self.min_confidence = min_confidence
        self.dependencies = []
    
    def find_dependencies(self, max_determinant_cardinality: int = 1000) -> List[Dict]:
        """
        Find functional dependencies in the dataset.
        
        Args:
            max_determinant_cardinality: Max unique values in determinant column to check
        
        Returns:
            List of dependency dicts with structure:
            [{"determinant": "Zip", "dependent": "City", "confidence": 1.0, "type": "1:1"}]
        """
        dependencies = []
        columns = self.df.columns.tolist()
        
        for i, det_col in enumerate(columns):
            # Skip if determinant has too many unique values (performance)
            if self.df[det_col].nunique() > max_determinant_cardinality:
                continue
            
            for j, dep_col in enumerate(columns):
                if i == j:  # Skip same column
                    continue
                
                # Check for functional dependency
                dependency = self._check_dependency(det_col, dep_col)
                if dependency:
                    dependencies.append(dependency)
        
        self.dependencies = dependencies
        return dependencies
    
    def _check_dependency(self, determinant: str, dependent: str) -> Optional[Dict]:
        """
        Check if determinant → dependent is a functional dependency.
        
        Args:
            determinant: Column that determines the dependent
            dependent: Column that is determined
        
        Returns:
            Dependency dict or None if no dependency found
        """
        # Remove rows with missing values in either column
        df_clean = self.df[[determinant, dependent]].dropna()
        
        if len(df_clean) == 0:
            return None
        
        # Group by determinant and count unique dependent values
        grouped = df_clean.groupby(determinant)[dependent].nunique()
        
        # Calculate confidence: % of determinant values that map to exactly 1 dependent value
        one_to_one_count = (grouped == 1).sum()
        total_determinant_values = len(grouped)
        
        if total_determinant_values == 0:
            return None
        
        confidence = one_to_one_count / total_determinant_values
        
        # Only return if confidence meets threshold
        if confidence < self.min_confidence:
            return None
        
        # Determine dependency type
        unique_determinants = df_clean[determinant].nunique()
        unique_dependents = df_clean[dependent].nunique()
        
        if unique_determinants == unique_dependents and confidence == 1.0:
            dep_type = "1:1"  # One-to-one mapping
        elif confidence == 1.0:
            dep_type = "M:1"  # Many-to-one mapping
        else:
            dep_type = "M:1 (partial)"  # Partial many-to-one
        
        # Calculate additional statistics
        avg_dependents_per_determinant = df_clean.groupby(determinant)[dependent].nunique().mean()
        
        return {
            "determinant": determinant,
            "dependent": dependent,
            "confidence": round(confidence, 4),
            "type": dep_type,
            "determinant_cardinality": unique_determinants,
            "dependent_cardinality": unique_dependents,
            "avg_dependents_per_determinant": round(avg_dependents_per_determinant, 2),
            "sample_mapping": self._get_sample_mapping(df_clean, determinant, dependent)
        }
    
    def _get_sample_mapping(self, df: pd.DataFrame, determinant: str, 
                           dependent: str, n_samples: int = 3) -> List[Dict]:
        """
        Get sample mappings to illustrate the dependency.
        
        Args:
            df: DataFrame subset
            determinant: Determinant column
            dependent: Dependent column
            n_samples: Number of samples to return
        
        Returns:
            List of sample mappings
        """
        samples = []
        grouped = df.groupby(determinant)[dependent].apply(list).head(n_samples)
        
        for det_val, dep_vals in grouped.items():
            samples.append({
                str(determinant): str(det_val),
                str(dependent): str(dep_vals[0]) if len(dep_vals) == 1 else str(dep_vals[:3])
            })
        
        return samples
    
    def get_dependency_graph_data(self, dependencies: Optional[List[Dict]] = None) -> Dict:
        """
        Prepare data for network graph visualization.
        
        Args:
            dependencies: Optional list of dependencies to visualize. If None, uses self.dependencies
        
        Returns:
            Dict with nodes and edges for graph visualization
        """
        deps_to_use = dependencies if dependencies is not None else self.dependencies
        
        if not deps_to_use:
            return {"nodes": [], "edges": []}
        
        # Extract unique columns involved in dependencies
        columns = set()
        for dep in deps_to_use:
            columns.add(dep["determinant"])
            columns.add(dep["dependent"])
        
        # Create nodes with connection counts
        node_stats = {}
        for col in columns:
            node_stats[col] = {"as_determinant": 0, "as_dependent": 0}
        
        for dep in deps_to_use:
            node_stats[dep["determinant"]]["as_determinant"] += 1
            node_stats[dep["dependent"]]["as_dependent"] += 1
        
        # Create nodes
        nodes = []
        for col in columns:
            nodes.append({
                "id": col,
                "label": col,
                "as_determinant": node_stats[col]["as_determinant"],
                "as_dependent": node_stats[col]["as_dependent"]
            })
        
        # Create edges
        edges = []
        for dep in deps_to_use:
            edges.append({
                "source": dep["determinant"],
                "target": dep["dependent"],
                "confidence": dep["confidence"],
                "type": dep["type"],
                "label": f"{dep['type']} ({dep['confidence']:.2f})"
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "total_dependencies": len(deps_to_use),
                "columns_involved": len(columns),
                "strict_dependencies": len([d for d in deps_to_use if d["confidence"] == 1.0])
            }
        }
    
    def validate_against_synthetic(self, synthetic_df: pd.DataFrame) -> List[Dict]:
        """
        Validate that dependencies are preserved in synthetic data.
        
        Args:
            synthetic_df: Generated synthetic DataFrame
        
        Returns:
            List of validation results for each dependency
        """
        validation_results = []
        
        for dep in self.dependencies:
            det_col = dep["determinant"]
            dep_col = dep["dependent"]
            
            # Check if columns exist in synthetic data
            if det_col not in synthetic_df.columns or dep_col not in synthetic_df.columns:
                validation_results.append({
                    "dependency": f"{det_col} → {dep_col}",
                    "status": "MISSING_COLUMNS",
                    "original_confidence": dep["confidence"],
                    "synthetic_confidence": 0.0
                })
                continue
            
            # Recalculate confidence in synthetic data
            synth_dependency = self._check_dependency_on_df(
                synthetic_df, det_col, dep_col
            )
            
            if synth_dependency:
                synth_confidence = synth_dependency["confidence"]
                status = "PRESERVED" if synth_confidence >= self.min_confidence else "VIOLATED"
            else:
                synth_confidence = 0.0
                status = "VIOLATED"
            
            validation_results.append({
                "dependency": f"{det_col} → {dep_col}",
                "status": status,
                "original_confidence": dep["confidence"],
                "synthetic_confidence": synth_confidence,
                "confidence_drop": round(dep["confidence"] - synth_confidence, 4)
            })
        
        return validation_results
    
    def _check_dependency_on_df(self, df: pd.DataFrame, determinant: str, 
                                dependent: str) -> Optional[Dict]:
        """Check dependency on a specific DataFrame (helper for validation)."""
        df_clean = df[[determinant, dependent]].dropna()
        
        if len(df_clean) == 0:
            return None
        
        grouped = df_clean.groupby(determinant)[dependent].nunique()
        one_to_one_count = (grouped == 1).sum()
        total = len(grouped)
        
        if total == 0:
            return None
        
        confidence = one_to_one_count / total
        
        return {"confidence": confidence}
    
    def save_to_metadata(self, file_id: str, metadata_path: str = None) -> str:
        """
        Save detected dependencies to metadata storage for generation constraints.
        
        Args:
            file_id: Unique identifier for the dataset
            metadata_path: Optional path to metadata directory (defaults to workspace/)
        
        Returns:
            Path to saved metadata file
        """
        if not self.dependencies:
            logger.warning(f"No dependencies to save for file_id: {file_id}")
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
            "total_dependencies": len(self.dependencies),
            "dependencies": self.dependencies,
            "graph_data": self.get_dependency_graph_data(),
            "generation_constraints": self._generate_constraints()
        }
        
        # Save to JSON file
        output_file = os.path.join(metadata_path, f"dependencies_{file_id}.json")
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved {len(self.dependencies)} dependencies to {output_file}")
        return output_file
    
    def _generate_constraints(self) -> List[Dict]:
        """
        Generate constraints for synthetic data generation from dependencies.
        
        Returns:
            List of constraint rules that can be applied during generation
        """
        constraints = []
        
        for dep in self.dependencies:
            if dep["confidence"] >= 0.99:  # Only strict dependencies
                constraint = {
                    "type": "functional_dependency",
                    "determinant": dep["determinant"],
                    "dependent": dep["dependent"],
                    "confidence": dep["confidence"],
                    "dependency_type": dep["type"],
                    "rule": f"preserve_mapping({dep['determinant']} -> {dep['dependent']})",
                    "description": f"Maintain {dep['type']} mapping from {dep['determinant']} to {dep['dependent']}"
                }
                constraints.append(constraint)
        
        return constraints


def load_dependencies_metadata(file_id: str, metadata_path: str = None) -> Optional[Dict]:
    """
    Load saved dependency metadata for a dataset.
    
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
    metadata_file = os.path.join(metadata_path, f"dependencies_{file_id}.json")
    
    if not os.path.exists(metadata_file):
        logger.warning(f"No dependency metadata found for file_id: {file_id}")
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded dependency metadata for file_id: {file_id}")
    return metadata


def detect_functional_dependencies(
    df: pd.DataFrame,
    min_confidence: float = 0.95,
    max_determinant_cardinality: int = 1000
) -> List[Dict]:
    """
    Convenience function to detect functional dependencies.
    
    Args:
        df: Input DataFrame
        min_confidence: Minimum confidence threshold
        max_determinant_cardinality: Max unique values in determinant
    
    Returns:
        List of detected dependencies
    """
    # Optional adapter: try to use Microsoft Fabric / sem.py if explicitly requested
    # This keeps backward compatibility while allowing deployment in environments
    # that have the Fabric semantic API available.
    def _try_fabric_adapter():
        # Try common module names, but don't require them.
        try:
            import sem  # type: ignore
            sem_module = sem
        except Exception:
            sem_module = None

        try:
            import fabric  # type: ignore
            fabric_module = fabric
        except Exception:
            fabric_module = None

        # Prefer sem module if present, otherwise fabric. Both are optional.
        if sem_module is not None:
            # sem.find_dependencies is an assumed API; adapt if needed.
            try:
                if hasattr(sem_module, 'find_dependencies'):
                    logger.info('Using sem.find_dependencies adapter for functional dependency detection')
                    return sem_module.find_dependencies(df, min_confidence=min_confidence,
                                                        max_cardinality=max_determinant_cardinality)
            except Exception as e:
                logger.warning(f'sem adapter failed: {e}', exc_info=True)

        if fabric_module is not None:
            try:
                if hasattr(fabric_module, 'find_dependencies'):
                    logger.info('Using fabric.find_dependencies adapter for functional dependency detection')
                    return fabric_module.find_dependencies(df, min_confidence=min_confidence,
                                                           max_cardinality=max_determinant_cardinality)
            except Exception as e:
                logger.warning(f'fabric adapter failed: {e}', exc_info=True)

        return None

    # If an environment provides Fabric/sem.py, attempt to delegate to it. If it
    # isn't present or the adapter call fails, fall back to the built-in detector.
    adapter_result = _try_fabric_adapter()
    if adapter_result is not None:
        return adapter_result

    detector = DependencyDetector(df, min_confidence)
    return detector.find_dependencies(max_determinant_cardinality)
