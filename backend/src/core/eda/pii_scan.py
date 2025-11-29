"""
PII Detection Module
Implements fast regex-based and deep AI-powered PII detection for privacy auditing.
"""
import os
import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

# PII Detection Patterns (Fast Scan)
PII_PATTERNS = {
    "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "PHONE_US": r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
    "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
    "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    "IP_ADDRESS": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    "URL": r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)',
    "DATE_OF_BIRTH": r'\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b',
    "ZIP_CODE": r'\b\d{5}(?:-\d{4})?\b',
}

# Common PII column name patterns
PII_COLUMN_NAMES = {
    "NAME": ["name", "first_name", "last_name", "full_name", "firstname", "lastname", "username"],
    "EMAIL": ["email", "e_mail", "email_address", "mail"],
    "PHONE": ["phone", "telephone", "mobile", "cell", "phone_number"],
    "ADDRESS": ["address", "street", "city", "state", "zip", "postal", "location"],
    "SSN": ["ssn", "social_security", "social_security_number"],
    "DOB": ["dob", "birth_date", "date_of_birth", "birthday"],
    "ID": ["id", "customer_id", "user_id", "employee_id", "account_id"],
}


class PIIScanner:
    """
    Detects Personally Identifiable Information (PII) in datasets.
    Supports both fast regex-based scanning and deep AI-powered analysis.
    
    NOTE: Deep scanning (Presidio) is currently DISABLED to reduce memory usage on deployment.
    Only fast regex-based scanning is active.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize PII scanner.
        
        Args:
            df: Input DataFrame to scan
        """
        self.df = df
        self.fast_results = {}
        self.deep_results = {}
        self.presidio_available = False
        
        # FEATURE DISABLED: Presidio deep scanning commented out to save memory (~100-200MB)
        # Uncomment the block below to re-enable deep AI-powered PII detection
        
        # # Try to import Presidio for deep scanning
        # try:
        #     from presidio_analyzer import AnalyzerEngine
        #     from presidio_analyzer.nlp_engine import NlpEngineProvider

        #     # Suppress Presidio's internal configuration warnings
        #     presidio_logger = logging.getLogger("presidio-analyzer")
        #     presidio_logger.setLevel(logging.ERROR)

        #     # Force Presidio to use a specific spaCy pipeline to prevent runtime downloads.
        #     model_name = os.getenv("PRESIDIO_SPACY_MODEL", "en_core_web_sm")
        #     nlp_config = {
        #         "nlp_engine_name": "spacy",
        #         "models": [{"lang_code": "en", "model_name": model_name}],
        #     }
        #     provider = NlpEngineProvider(nlp_configuration=nlp_config)
        #     
        #     # Initialize AnalyzerEngine
        #     self.analyzer = AnalyzerEngine(
        #         nlp_engine=provider.create_engine(),
        #         default_score_threshold=0.5,
        #         supported_languages=["en"]
        #     )
        #     self.presidio_available = True
        #     logger.info("Presidio analyzer loaded successfully with %s", model_name)
        # except ImportError:
        #     logger.warning("Presidio not available. Deep PII scanning will be limited.")
        #     self.analyzer = None
        # except Exception as exc:
        #     logger.warning("Failed to initialize Presidio analyzer: %s", exc)
        #     self.analyzer = None
        
        self.analyzer = None
        logger.info("PII Scanner initialized in FAST-ONLY mode (Deep scan disabled)")
    
    def run_fast_scan(self, sample_size: int = 1000) -> Dict:
        """
        Fast asynchronous PII scan using regex patterns and column name heuristics.
        
        Args:
            sample_size: Number of rows to sample for pattern matching
        
        Returns:
            Dict with detected PII entities by column
        """
        results = {
            "summary": {
                "total_columns": len(self.df.columns),
                "pii_columns_detected": 0,
                "scan_type": "fast",
                "sample_size": min(sample_size, len(self.df))
            },
            "detections": []
        }
        
        pii_count = 0
        
        for column in self.df.columns:
            column_results = {
                "column": column,
                "pii_types": [],
                "confidence": 0.0,
                "match_count": 0,
                "detection_method": []
            }
            
            # Check column name patterns
            name_pii = self._check_column_name(column)
            if name_pii:
                column_results["pii_types"].extend(name_pii)
                column_results["detection_method"].append("column_name")
                column_results["confidence"] = 0.7
            
            # Check data patterns (sample for performance)
            if pd.api.types.is_object_dtype(self.df[column]) or pd.api.types.is_string_dtype(self.df[column]):
                sample_data = self.df[column].dropna().astype(str).head(sample_size)
                pattern_matches = self._check_data_patterns(sample_data)
                
                if pattern_matches:
                    for pii_type, count in pattern_matches.items():
                        if pii_type not in column_results["pii_types"]:
                            column_results["pii_types"].append(pii_type)
                        column_results["match_count"] += count
                        column_results["detection_method"].append("pattern_match")
                    
                    # Increase confidence if patterns match
                    match_ratio = sum(pattern_matches.values()) / len(sample_data)
                    column_results["confidence"] = max(column_results["confidence"], min(match_ratio, 0.95))
            
            # Add to results if PII detected
            if column_results["pii_types"]:
                pii_count += 1
                results["detections"].append(column_results)
        
        results["summary"]["pii_columns_detected"] = pii_count
        self.fast_results = results
        
        logger.info(f"Fast PII scan completed. Found {pii_count} columns with potential PII.")
        return results
    
    def _check_column_name(self, column: str) -> List[str]:
        """Check if column name matches known PII patterns."""
        detected = []
        column_lower = column.lower()
        
        for pii_type, keywords in PII_COLUMN_NAMES.items():
            if any(keyword in column_lower for keyword in keywords):
                detected.append(pii_type)
        
        return detected
    
    def _check_data_patterns(self, series: pd.Series) -> Dict[str, int]:
        """Check data values against PII regex patterns."""
        matches = {}
        
        # Combine all values into text for faster scanning
        combined_text = " | ".join(series.astype(str).tolist())
        
        for pii_type, pattern in PII_PATTERNS.items():
            found = re.findall(pattern, combined_text)
            if found:
                matches[pii_type] = len(found)
        
        return matches
    
    def run_deep_scan(self, 
                     columns: Optional[List[str]] = None,
                     sample_size: int = 100,
                     language: str = "en") -> Dict:
        """
        Deep PII scan using Microsoft Presidio AI analyzer.
        
        FEATURE DISABLED: This feature is currently disabled to reduce memory usage.
        Automatically falls back to fast scan.
        
        Args:
            columns: Specific columns to scan (None = all text columns)
            sample_size: Number of rows to analyze per column
            language: Language code for analysis
        
        Returns:
            Dict with detailed PII detections including positions and confidence
        """
        # DISABLED: Deep scan feature commented out to save ~100-200MB memory
        logger.info("Deep scan is currently disabled. Using fast scan instead.")
        return self.run_fast_scan(sample_size)
        
        # Original deep scan logic (COMMENTED OUT)
        # if not self.presidio_available:
        #     logger.warning("Presidio not available. Falling back to fast scan.")
        #     return self.run_fast_scan(sample_size)
        
        results = {
            "summary": {
                "total_columns_scanned": 0,
                "pii_entities_found": 0,
                "scan_type": "deep",
                "sample_size": sample_size
            },
            "detections": []
        }
        
        # Select columns to scan
        if columns is None:
            columns = [col for col in self.df.columns 
                      if pd.api.types.is_object_dtype(self.df[col]) or 
                         pd.api.types.is_string_dtype(self.df[col])]
        
        total_entities = 0
        
        for column in columns:
            try:
                # Sample data for analysis
                sample_data = self.df[column].dropna().astype(str).head(sample_size).tolist()
                
                if not sample_data:
                    continue
                
                column_detections = []
                entity_types = defaultdict(int)
                
                # Analyze each value
                for idx, text in enumerate(sample_data):
                    if len(text) < 3:  # Skip very short texts
                        continue
                    
                    # Run Presidio analysis
                    analysis_results = self.analyzer.analyze(
                        text=text,
                        language=language,
                        entities=None  # Detect all entity types
                    )
                    
                    for result in analysis_results:
                        entity_types[result.entity_type] += 1
                        column_detections.append({
                            "row_index": idx,
                            "entity_type": result.entity_type,
                            "confidence": round(result.score, 3),
                            "start": result.start,
                            "end": result.end,
                            "text_snippet": text[max(0, result.start-10):result.end+10]
                        })
                
                if column_detections:
                    total_entities += len(column_detections)
                    
                    # Aggregate results for column
                    results["detections"].append({
                        "column": column,
                        "total_detections": len(column_detections),
                        "entity_types": dict(entity_types),
                        "confidence_avg": round(
                            sum(d["confidence"] for d in column_detections) / len(column_detections),
                            3
                        ),
                        "sample_detections": column_detections[:10]  # Limit to 10 examples
                    })
            
            except Exception as e:
                logger.error(f"Error analyzing column {column}: {str(e)}")
                continue
        
        results["summary"]["total_columns_scanned"] = len(columns)
        results["summary"]["pii_entities_found"] = total_entities
        
        self.deep_results = results
        
        logger.info(f"Deep PII scan completed. Found {total_entities} PII entities.")
        return results
    
    def get_pii_summary(self) -> Dict:
        """
        Get summary of all PII detections.
        
        Returns:
            Combined summary from fast and deep scans with flattened top-level KPIs
        """
        # Get nested summaries
        fast_summary = self.fast_results.get("summary", {})
        deep_summary = self.deep_results.get("summary", {})
        
        # Identify high-risk columns (detected in both scans or high confidence)
        fast_cols = {d["column"] for d in self.fast_results.get("detections", [])}
        deep_cols = {d["column"] for d in self.deep_results.get("detections", [])}
        
        high_risk = fast_cols.intersection(deep_cols) if deep_cols else fast_cols
        
        # Collect all columns with PII
        all_pii_cols = fast_cols.union(deep_cols) if deep_cols else fast_cols
        
        # Determine which scan was used
        scan_type = "none"
        sample_size = 0
        total_columns = 0
        
        if self.fast_results:
            scan_type = fast_summary.get("scan_type", "fast")
            sample_size = fast_summary.get("sample_size", 0)
            total_columns = fast_summary.get("total_columns", 0)
        
        if self.deep_results:
            scan_type = deep_summary.get("scan_type", "deep")
            sample_size = deep_summary.get("sample_size", 0)
            total_columns = max(total_columns, deep_summary.get("total_columns_scanned", 0))
        
        # Generate recommendations
        recommendations = []
        if high_risk:
            recommendations.append({
                "priority": "HIGH",
                "issue": f"{len(high_risk)} columns contain PII",
                "recommendation": "Consider anonymization, pseudonymization, or removal before sharing data"
            })
            
            # Check for specific PII types
            all_types = set()
            for detection in self.fast_results.get("detections", []):
                all_types.update(detection["pii_types"])
            
            if "EMAIL" in all_types or "SSN" in all_types:
                recommendations.append({
                    "priority": "CRITICAL",
                    "issue": "Direct identifiers (email/SSN) detected",
                    "recommendation": "Hash, encrypt, or remove these columns before data sharing"
                })
        
        # Return flattened summary with top-level KPIs for frontend compatibility
        summary = {
            # Top-level KPIs (for frontend metrics)
            "total_columns": total_columns,
            "columns_with_pii": len(all_pii_cols),
            "high_risk_columns": len(high_risk),
            "high_risk_column_list": list(high_risk),
            "scan_type": scan_type,
            "sample_size": sample_size,
            "recommendations": recommendations,
            
            # Nested detailed summaries (for backward compatibility)
            "fast_scan": fast_summary,
            "deep_scan": deep_summary
        }
        
        return summary
    
    def save_results(self, file_id: str, metadata_path: str = None) -> str:
        """
        Save PII scan results to metadata storage.
        
        Args:
            file_id: Dataset identifier
            metadata_path: Optional metadata directory path
        
        Returns:
            Path to saved metadata file
        """
        if metadata_path is None:
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            metadata_path = os.path.join(backend_dir, "workspace", "metadata")
        
        import os
        from datetime import datetime
        os.makedirs(metadata_path, exist_ok=True)
        
        # Prepare metadata structure
        metadata = {
            "file_id": file_id,
            "timestamp": datetime.now().isoformat(),
            "fast_scan": self.fast_results,
            "deep_scan": self.deep_results,
            "summary": self.get_pii_summary()
        }
        
        # Save to JSON file
        output_file = os.path.join(metadata_path, f"pii_scan_{file_id}.json")
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved PII scan results to {output_file}")
        return output_file


def run_pii_scan_fast(df: pd.DataFrame, sample_size: int = 1000) -> Dict:
    """
    Convenience function for fast PII scan.
    
    Args:
        df: Input DataFrame
        sample_size: Number of rows to sample
    
    Returns:
        PII detection results
    """
    scanner = PIIScanner(df)
    return scanner.run_fast_scan(sample_size)


def run_pii_scan_deep(df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      sample_size: int = 100) -> Dict:
    """
    Convenience function for deep PII scan.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to scan
        sample_size: Number of rows to analyze
    
    Returns:
        Detailed PII detection results
    """
    scanner = PIIScanner(df)
    return scanner.run_deep_scan(columns, sample_size)


def load_pii_scan_results(file_id: str, metadata_path: str = None) -> Optional[Dict]:
    """
    Load saved PII scan results.
    
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
    metadata_file = os.path.join(metadata_path, f"pii_scan_{file_id}.json")
    
    if not os.path.exists(metadata_file):
        logger.warning(f"No PII scan results found for file_id: {file_id}")
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded PII scan results for file_id: {file_id}")
    return metadata
