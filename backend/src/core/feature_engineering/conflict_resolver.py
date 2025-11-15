"""
Conflict Detection and Resolution Engine

Identifies and manages conflicts between utility-driven and privacy-driven suggestions.
Provides decision framework for users to make informed trade-offs.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Conflict:
    """Represents a conflict between utility and privacy suggestions."""
    column: str
    utility_suggestion: Dict[str, Any]
    privacy_suggestion: Dict[str, Any]
    conflict_type: str  # "direct", "indirect"
    severity: str  # "high", "medium", "low"
    recommendation: str


class ConflictResolver:
    """
    Detects and manages conflicts between utility and privacy transformation suggestions.
    """
    
    def __init__(self):
        """Initialize conflict resolver."""
        self.conflicts = []
        self.resolutions = []
    
    def detect_conflicts(
        self,
        utility_suggestions: List[Dict[str, Any]],
        privacy_suggestions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between utility and privacy suggestions.
        
        Args:
            utility_suggestions: List of utility-driven suggestions
            privacy_suggestions: List of privacy-driven suggestions
            
        Returns:
            List of conflict dictionaries
        """
        self.conflicts = []
        
        # Group suggestions by column
        utility_by_col = {s["column"]: s for s in utility_suggestions}
        privacy_by_col = {s["column"]: s for s in privacy_suggestions}
        
        # Find columns with both types of suggestions
        conflicting_columns = set(utility_by_col.keys()) & set(privacy_by_col.keys())
        
        for column in conflicting_columns:
            utility_sug = utility_by_col[column]
            privacy_sug = privacy_by_col[column]
            
            # Determine conflict type and severity
            conflict = self._analyze_conflict(column, utility_sug, privacy_sug)
            self.conflicts.append(conflict)
        
        return [self._conflict_to_dict(c) for c in self.conflicts]
    
    def _analyze_conflict(
        self,
        column: str,
        utility_sug: Dict[str, Any],
        privacy_sug: Dict[str, Any]
    ) -> Conflict:
        """
        Analyze a conflict and determine its characteristics.
        
        Args:
            column: Column name
            utility_sug: Utility suggestion
            privacy_sug: Privacy suggestion
            
        Returns:
            Conflict object
        """
        # Determine conflict type
        conflict_type = self._determine_conflict_type(utility_sug, privacy_sug)
        
        # Calculate severity
        severity = self._calculate_severity(utility_sug, privacy_sug)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            column, utility_sug, privacy_sug, conflict_type, severity
        )
        
        return Conflict(
            column=column,
            utility_suggestion=utility_sug,
            privacy_suggestion=privacy_sug,
            conflict_type=conflict_type,
            severity=severity,
            recommendation=recommendation
        )
    
    def _determine_conflict_type(
        self,
        utility_sug: Dict[str, Any],
        privacy_sug: Dict[str, Any]
    ) -> str:
        """
        Determine the type of conflict.
        
        Args:
            utility_sug: Utility suggestion
            privacy_sug: Privacy suggestion
            
        Returns:
            Conflict type: "direct" or "indirect"
        """
        # Direct conflicts: transformations that are mutually exclusive
        destructive_privacy = ["redact", "suppress", "hash"]
        
        if privacy_sug["transformation"] in destructive_privacy:
            return "direct"
        
        # Indirect conflicts: both modify the same column but can coexist with caution
        return "indirect"
    
    def _calculate_severity(
        self,
        utility_sug: Dict[str, Any],
        privacy_sug: Dict[str, Any]
    ) -> str:
        """
        Calculate conflict severity based on risk and confidence.
        
        Args:
            utility_sug: Utility suggestion
            privacy_sug: Privacy suggestion
            
        Returns:
            Severity level: "high", "medium", "low"
        """
        # Get privacy risk level
        risk_level = privacy_sug.get("risk_level", "low")
        
        # Get utility confidence
        utility_confidence = utility_sug.get("confidence", 0.5)
        
        # High severity: high privacy risk
        if risk_level == "high":
            return "high"
        
        # Medium severity: medium privacy risk or high utility confidence
        if risk_level == "medium" or utility_confidence > 0.8:
            return "medium"
        
        # Low severity: low privacy risk and low utility confidence
        return "low"
    
    def _generate_recommendation(
        self,
        column: str,
        utility_sug: Dict[str, Any],
        privacy_sug: Dict[str, Any],
        conflict_type: str,
        severity: str
    ) -> str:
        """
        Generate a recommendation for resolving the conflict.
        
        Args:
            column: Column name
            utility_sug: Utility suggestion
            privacy_sug: Privacy suggestion
            conflict_type: Type of conflict
            severity: Severity level
            
        Returns:
            Recommendation text
        """
        risk_level = privacy_sug.get("risk_level", "low")
        
        if severity == "high":
            if conflict_type == "direct":
                return f"⚠️ CRITICAL: Privacy risk is HIGH. Strongly recommend accepting the privacy transformation ('{privacy_sug['transformation']}') to protect sensitive data. Utility improvements are secondary to privacy compliance."
            else:
                return f"⚠️ HIGH PRIORITY: Consider privacy transformation first, then apply utility transformation if data remains usable."
        
        elif severity == "medium":
            if conflict_type == "direct":
                return f"⚡ MODERATE: Balance needed. Privacy risk is {risk_level.upper()}. Consider if utility gain justifies the privacy trade-off. You might apply privacy transform to a subset of data."
            else:
                return f"⚡ MODERATE: Both transformations have merit. Consider applying privacy transformation first, then evaluate utility transformation on protected data."
        
        else:  # low severity
            return f"ℹ️ LOW IMPACT: Privacy risk is {risk_level.upper()}. You can prioritize utility transformation, but keep privacy enhancement in mind for production deployment."
    
    def _conflict_to_dict(self, conflict: Conflict) -> Dict[str, Any]:
        """Convert Conflict to dictionary."""
        return {
            "column": conflict.column,
            "utility_suggestion": conflict.utility_suggestion,
            "privacy_suggestion": conflict.privacy_suggestion,
            "conflict_type": conflict.conflict_type,
            "severity": conflict.severity,
            "recommendation": conflict.recommendation,
            "utility_transform": conflict.utility_suggestion["transformation"],
            "privacy_transform": conflict.privacy_suggestion["transformation"],
            "utility_reason": conflict.utility_suggestion["reason"],
            "privacy_reason": conflict.privacy_suggestion["reason"]
        }
    
    def resolve_conflict(
        self,
        column: str,
        chosen_category: str,
        chosen_transformation: str,
        user_note: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a user's resolution of a conflict.
        
        Args:
            column: Column name
            chosen_category: "utility" or "privacy"
            chosen_transformation: The transformation chosen
            user_note: Optional user explanation
            
        Returns:
            Resolution record
        """
        resolution = {
            "column": column,
            "chosen_category": chosen_category,
            "chosen_transformation": chosen_transformation,
            "user_note": user_note,
            "timestamp": datetime.now().isoformat(),
            "resolution_id": f"{column}_{datetime.now().timestamp()}"
        }
        
        self.resolutions.append(resolution)
        return resolution
    
    def get_non_conflicting_suggestions(
        self,
        utility_suggestions: List[Dict[str, Any]],
        privacy_suggestions: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Separate conflicting from non-conflicting suggestions.
        
        Args:
            utility_suggestions: List of utility suggestions
            privacy_suggestions: List of privacy suggestions
            
        Returns:
            Tuple of (non_conflicting_utility, non_conflicting_privacy)
        """
        # Get conflicting columns
        conflicting_columns = {c.column for c in self.conflicts}
        
        # Filter out conflicting suggestions
        non_conflicting_utility = [
            s for s in utility_suggestions if s["column"] not in conflicting_columns
        ]
        non_conflicting_privacy = [
            s for s in privacy_suggestions if s["column"] not in conflicting_columns
        ]
        
        return non_conflicting_utility, non_conflicting_privacy
    
    def generate_conflict_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all detected conflicts.
        
        Returns:
            Summary dictionary
        """
        if not self.conflicts:
            return {
                "total_conflicts": 0,
                "message": "No conflicts detected between utility and privacy suggestions."
            }
        
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        type_counts = {"direct": 0, "indirect": 0}
        
        for conflict in self.conflicts:
            severity_counts[conflict.severity] += 1
            type_counts[conflict.conflict_type] += 1
        
        return {
            "total_conflicts": len(self.conflicts),
            "by_severity": severity_counts,
            "by_type": type_counts,
            "conflicting_columns": [c.column for c in self.conflicts],
            "high_priority_conflicts": [
                c.column for c in self.conflicts if c.severity == "high"
            ]
        }
    
    def get_resolution_history(self) -> List[Dict[str, Any]]:
        """
        Get all recorded conflict resolutions.
        
        Returns:
            List of resolution records
        """
        return self.resolutions
