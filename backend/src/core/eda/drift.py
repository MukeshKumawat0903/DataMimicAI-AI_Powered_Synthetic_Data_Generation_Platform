"""
Drift detection module for comparing datasets.
Supports statistical tests (KS, Chi-square, PSI) and ML-based drift detection.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class DriftDetectionError(Exception):
    """Raised when drift detection fails."""
    pass


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    ks_alpha: float = 0.05  # Significance level for KS test
    chi_alpha: float = 0.05  # Significance level for Chi-square
    psi_threshold: float = 0.2  # PSI thresholds: <0.1 (no drift), 0.1-0.2 (moderate), >0.2 (significant)
    classifier_threshold: float = 0.7  # AUC threshold for drift classifier


class DriftDetector:
    """
    Comprehensive drift detection between reference and current datasets.
    """
    def __init__(self, df_reference: pd.DataFrame, df_current: pd.DataFrame, config: DriftConfig = None):
        if df_reference.empty or df_current.empty:
            raise DriftDetectionError("Reference or current DataFrame is empty.")
        self.df_ref = df_reference.copy()
        self.df_cur = df_current.copy()
        self.config = config or DriftConfig()
        self.metadata_dir = Path("workspace/metadata")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    # ============= STATISTICAL TESTS =============
    
    def kolmogorov_smirnov_test(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov test for numerical columns.
        Tests if two samples come from the same distribution.
        """
        if columns is None:
            # Get numeric columns that exist in both datasets
            ref_numeric = set(self.df_ref.select_dtypes(include=[np.number]).columns)
            cur_numeric = set(self.df_cur.select_dtypes(include=[np.number]).columns)
            columns = list(ref_numeric & cur_numeric)
        
        results = []
        
        for col in columns:
            if col not in self.df_ref.columns or col not in self.df_cur.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df_ref[col]) or not pd.api.types.is_numeric_dtype(self.df_cur[col]):
                continue
            
            ref_data = self.df_ref[col].dropna()
            cur_data = self.df_cur[col].dropna()
            
            if len(ref_data) < 3 or len(cur_data) < 3:
                continue
            
            # Perform KS test
            statistic, pvalue = stats.ks_2samp(ref_data, cur_data)
            
            drift_detected = pvalue < self.config.ks_alpha
            
            results.append({
                "column": col,
                "method": "Kolmogorov-Smirnov",
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "alpha": self.config.ks_alpha,
                "drift_detected": bool(drift_detected),
                "ref_mean": float(ref_data.mean()),
                "cur_mean": float(cur_data.mean()),
                "ref_std": float(ref_data.std()),
                "cur_std": float(cur_data.std())
            })
        
        return {"results": results}

    def chi_square_test(self, columns: Optional[List[str]] = None, bins: int = 10) -> Dict[str, Any]:
        """
        Chi-square test for categorical columns (or binned numerical columns).
        Tests independence of categorical distributions.
        """
        if columns is None:
            # Get categorical columns that exist in both datasets
            ref_cat = set(self.df_ref.select_dtypes(include=['object', 'category']).columns)
            cur_cat = set(self.df_cur.select_dtypes(include=['object', 'category']).columns)
            columns = list(ref_cat & cur_cat)
        
        results = []
        
        for col in columns:
            if col not in self.df_ref.columns or col not in self.df_cur.columns:
                continue
            
            ref_data = self.df_ref[col].dropna()
            cur_data = self.df_cur[col].dropna()
            
            if len(ref_data) < 5 or len(cur_data) < 5:
                continue
            
            # For numerical columns, bin them first
            if pd.api.types.is_numeric_dtype(ref_data):
                # Create bins based on reference data
                _, bin_edges = pd.cut(ref_data, bins=bins, retbins=True, duplicates='drop')
                ref_binned = pd.cut(ref_data, bins=bin_edges, include_lowest=True)
                cur_binned = pd.cut(cur_data, bins=bin_edges, include_lowest=True)
            else:
                ref_binned = ref_data
                cur_binned = cur_data
            
            # Get value counts
            ref_counts = ref_binned.value_counts()
            cur_counts = cur_binned.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(cur_counts.index)
            ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories])
            cur_freq = np.array([cur_counts.get(cat, 0) for cat in all_categories])
            
            # Remove categories with zero counts in both
            mask = (ref_freq > 0) | (cur_freq > 0)
            ref_freq = ref_freq[mask]
            cur_freq = cur_freq[mask]
            
            if len(ref_freq) < 2:
                continue
            
            # Perform chi-square test
            try:
                contingency_table = np.array([ref_freq, cur_freq])
                statistic, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
                
                drift_detected = pvalue < self.config.chi_alpha
                
                results.append({
                    "column": col,
                    "method": "Chi-Square",
                    "statistic": float(statistic),
                    "p_value": float(pvalue),
                    "degrees_of_freedom": int(dof),
                    "alpha": self.config.chi_alpha,
                    "drift_detected": bool(drift_detected),
                    "n_categories": len(ref_freq)
                })
            except Exception as e:
                results.append({
                    "column": col,
                    "method": "Chi-Square",
                    "error": str(e)
                })
        
        return {"results": results}

    def population_stability_index(self, columns: Optional[List[str]] = None, bins: int = 10) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI) for numerical columns.
        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        Thresholds: <0.1 (no drift), 0.1-0.2 (moderate), >0.2 (significant)
        """
        if columns is None:
            ref_numeric = set(self.df_ref.select_dtypes(include=[np.number]).columns)
            cur_numeric = set(self.df_cur.select_dtypes(include=[np.number]).columns)
            columns = list(ref_numeric & cur_numeric)
        
        results = []
        
        for col in columns:
            if col not in self.df_ref.columns or col not in self.df_cur.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df_ref[col]) or not pd.api.types.is_numeric_dtype(self.df_cur[col]):
                continue
            
            ref_data = self.df_ref[col].dropna()
            cur_data = self.df_cur[col].dropna()
            
            if len(ref_data) < 10 or len(cur_data) < 10:
                continue
            
            # Create bins based on reference data quantiles
            try:
                _, bin_edges = pd.qcut(ref_data, q=bins, retbins=True, duplicates='drop')
            except:
                # Fallback to equal-width bins
                _, bin_edges = pd.cut(ref_data, bins=bins, retbins=True, duplicates='drop')
            
            # Bin both datasets
            ref_binned = pd.cut(ref_data, bins=bin_edges, include_lowest=True)
            cur_binned = pd.cut(cur_data, bins=bin_edges, include_lowest=True)
            
            # Get proportions
            ref_proportions = ref_binned.value_counts(normalize=True).sort_index()
            cur_proportions = cur_binned.value_counts(normalize=True).sort_index()
            
            # Align bins
            all_bins = ref_proportions.index.union(cur_proportions.index)
            ref_prop = np.array([ref_proportions.get(b, 0) for b in all_bins])
            cur_prop = np.array([cur_proportions.get(b, 0) for b in all_bins])
            
            # Add small constant to avoid log(0)
            epsilon = 1e-10
            ref_prop = np.maximum(ref_prop, epsilon)
            cur_prop = np.maximum(cur_prop, epsilon)
            
            # Calculate PSI
            psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
            
            # Determine drift level
            if psi < 0.1:
                drift_level = "No drift"
            elif psi < 0.2:
                drift_level = "Moderate drift"
            else:
                drift_level = "Significant drift"
            
            drift_detected = psi >= self.config.psi_threshold
            
            results.append({
                "column": col,
                "method": "PSI",
                "psi_value": float(psi),
                "threshold": self.config.psi_threshold,
                "drift_detected": bool(drift_detected),
                "drift_level": drift_level,
                "n_bins": len(all_bins)
            })
        
        return {"results": results}

    # ============= ML-BASED DRIFT DETECTION =============
    
    def drift_classifier(self, columns: Optional[List[str]] = None, test_size: float = 0.3) -> Dict[str, Any]:
        """
        Train a classifier to distinguish between reference and current datasets.
        High accuracy indicates drift. Returns feature importance for drift analysis.
        """
        if columns is None:
            ref_numeric = set(self.df_ref.select_dtypes(include=[np.number]).columns)
            cur_numeric = set(self.df_cur.select_dtypes(include=[np.number]).columns)
            columns = list(ref_numeric & cur_numeric)
        
        if len(columns) == 0:
            raise DriftDetectionError("No common numeric columns for drift classifier.")
        
        # Prepare data
        df_ref_subset = self.df_ref[columns].dropna()
        df_cur_subset = self.df_cur[columns].dropna()
        
        if len(df_ref_subset) < 10 or len(df_cur_subset) < 10:
            raise DriftDetectionError("Insufficient data for drift classifier (need at least 10 samples each).")
        
        # Create labels: 0 for reference, 1 for current
        df_ref_subset['drift_label'] = 0
        df_cur_subset['drift_label'] = 1
        
        # Combine datasets
        df_combined = pd.concat([df_ref_subset, df_cur_subset], ignore_index=True)
        
        # Shuffle
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X = df_combined[columns]
        y = df_combined['drift_label']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Drift detected if classifier can distinguish datasets well
        drift_detected = auc >= self.config.classifier_threshold
        
        result = {
            "method": "Drift Classifier",
            "accuracy": float(accuracy),
            "auc_roc": float(auc),
            "threshold": self.config.classifier_threshold,
            "drift_detected": bool(drift_detected),
            "feature_importance": feature_importance.to_dict('records'),
            "top_drift_features": feature_importance.head(5)['feature'].tolist(),
            "n_features": len(columns),
            "ref_samples": len(df_ref_subset),
            "cur_samples": len(df_cur_subset)
        }
        
        return result

    # ============= UNIFIED DRIFT ANALYSIS =============
    
    def analyze_all(self, columns: Optional[List[str]] = None, methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run multiple drift detection methods.
        methods: ['ks', 'chi_square', 'psi', 'classifier']
        """
        if methods is None:
            methods = ['ks', 'psi']
        
        all_results = {}
        
        for method in methods:
            try:
                if method == 'ks':
                    result = self.kolmogorov_smirnov_test(columns)
                    all_results['ks_test'] = result
                elif method == 'chi_square':
                    result = self.chi_square_test(columns)
                    all_results['chi_square'] = result
                elif method == 'psi':
                    result = self.population_stability_index(columns)
                    all_results['psi'] = result
                elif method == 'classifier':
                    result = self.drift_classifier(columns)
                    all_results['classifier'] = result
            except Exception as e:
                all_results[method] = {"error": str(e)}
        
        # Summary: count drift detections
        drift_summary = {
            "total_columns_analyzed": len(columns) if columns else 0,
            "methods_used": methods,
            "drift_detected_by_method": {}
        }
        
        for method_key, method_result in all_results.items():
            if 'error' in method_result:
                continue
            
            if 'results' in method_result:
                drifted_cols = [r['column'] for r in method_result['results'] if r.get('drift_detected')]
                drift_summary['drift_detected_by_method'][method_key] = drifted_cols
            elif method_key == 'classifier' and method_result.get('drift_detected'):
                drift_summary['drift_detected_by_method'][method_key] = method_result.get('top_drift_features', [])
        
        return {
            "results": all_results,
            "summary": drift_summary
        }

    # ============= METADATA PERSISTENCE =============
    
    def save_metadata(self, file_id_ref: str, file_id_cur: str, analysis_results: Dict[str, Any]) -> str:
        """Save drift analysis metadata to JSON."""
        metadata = {
            "file_id_reference": file_id_ref,
            "file_id_current": file_id_cur,
            "timestamp": datetime.now().isoformat(),
            "analysis_results": analysis_results,
            "config": {
                "ks_alpha": self.config.ks_alpha,
                "chi_alpha": self.config.chi_alpha,
                "psi_threshold": self.config.psi_threshold,
                "classifier_threshold": self.config.classifier_threshold
            }
        }
        
        metadata_path = self.metadata_dir / f"drift_{file_id_ref}_vs_{file_id_cur}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(metadata_path)

    def load_metadata(self, file_id_ref: str, file_id_cur: str) -> Optional[Dict[str, Any]]:
        """Load drift analysis metadata from JSON."""
        metadata_path = self.metadata_dir / f"drift_{file_id_ref}_vs_{file_id_cur}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None


# ============= CONVENIENCE FUNCTIONS FOR API =============

def analyze_drift_comprehensive(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    file_id_ref: str,
    file_id_cur: str,
    methods: List[str] = None,
    columns: List[str] = None,
    config: DriftConfig = None
) -> Dict[str, Any]:
    """
    Comprehensive drift analysis with multiple methods.
    Returns combined results and saves metadata.
    """
    detector = DriftDetector(df_reference, df_current, config)
    results = detector.analyze_all(columns=columns, methods=methods)
    
    # Save metadata
    metadata_path = detector.save_metadata(file_id_ref, file_id_cur, results)
    results["metadata_path"] = metadata_path
    
    return results


def get_drift_summary(df_reference: pd.DataFrame, df_current: pd.DataFrame, methods: List[str] = None) -> pd.DataFrame:
    """Get a summary DataFrame of drift analysis results."""
    detector = DriftDetector(df_reference, df_current)
    results = detector.analyze_all(methods=methods or ['ks', 'psi'])
    
    # Flatten results into DataFrame
    summary_rows = []
    
    for method_key, method_result in results.get('results', {}).items():
        if 'results' in method_result:
            for r in method_result['results']:
                summary_rows.append(r)
    
    if summary_rows:
        return pd.DataFrame(summary_rows)
    return pd.DataFrame()


# ============= STANDALONE METADATA LOADERS (for API use) =============

def load_drift_metadata_file(file_id_ref: str, file_id_cur: str) -> Optional[Dict[str, Any]]:
    """
    Load drift analysis metadata from JSON without instantiating detector.
    Use this in API endpoints to avoid empty DataFrame errors.
    """
    metadata_dir = Path("workspace/metadata")
    metadata_path = metadata_dir / f"drift_{file_id_ref}_vs_{file_id_cur}.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return None
    return None

