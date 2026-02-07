"""
Diagnostics Interpreter Agent (STEP 4 - Read-Only Reasoning)

This agent analyzes diagnostics output to identify cross-cutting patterns
and assess overall dataset stability. It does NOT provide recommendations,
actions, or transformations.

STRICT SCOPE:
- Identifies relationships between detected issues
- Groups related issues into logical patterns
- Assesses overall dataset stability qualitatively
- Provides read-only interpretation only

NOT IN SCOPE:
- Suggesting fixes or transformations
- Recommending actions
- Prioritizing issues
- Executing any changes
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PATTERN DEFINITIONS (Deterministic Logic)
# ============================================================================

PATTERN_RULES = {
    "skew_and_outliers": {
        "required_types": {"high_skew", "outliers"},
        "description": "Columns exhibit both skewness and outlier presence"
    },
    "high_feature_redundancy": {
        "required_types": {"high_correlation"},
        "min_count": 3,
        "description": "Multiple columns show high correlation (potential redundancy)"
    },
    "data_completeness_risk": {
        "required_types": {"missing_values"},
        "severity_threshold": "high",
        "description": "Significant missing data that may impact analysis"
    },
    "distribution_instability": {
        "required_types": {"high_skew", "high_kurtosis"},
        "min_count": 2,
        "description": "Multiple distribution issues detected"
    },
    "sparse_data_patterns": {
        "required_types": {"missing_values", "low_variance"},
        "description": "Combination of missingness and low variability"
    },
    "correlation_network": {
        "required_types": {"high_correlation"},
        "min_count": 5,
        "description": "Dense correlation network across features"
    },
}


STABILITY_RULES = {
    "stable": {
        "max_high_severity": 0,
        "max_medium_severity": 2,
        "max_total_issues": 5,
        "description": "Dataset is ready for modeling with minimal preprocessing"
    },
    "moderately_unstable": {
        "max_high_severity": 2,
        "max_medium_severity": 5,
        "max_total_issues": 10,
        "description": "Dataset has notable issues but is workable"
    },
    "unstable": {
        "description": "Dataset has significant quality concerns"
    }
}


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class InterpretationResult:
    """Structured interpretation result from the agent."""
    overall_assessment: str
    dominant_issue_patterns: List[str]
    supporting_evidence: List[Dict[str, Any]]
    confidence: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "interpretation": {
                "overall_assessment": self.overall_assessment,
                "dominant_issue_patterns": self.dominant_issue_patterns,
                "supporting_evidence": self.supporting_evidence,
                "confidence": self.confidence
            }
        }


# ============================================================================
# DIAGNOSTICS INTERPRETER AGENT
# ============================================================================

class DiagnosticsInterpreterAgent:
    """
    Read-only agent that interprets diagnostics to identify patterns
    and assess overall dataset stability.
    
    This agent performs STEP 4 reasoning:
    - Identifies cross-cutting patterns across diagnostics
    - Groups related issues logically
    - Assesses overall dataset stability
    
    IMPORTANT: This agent does NOT provide recommendations, actions,
    or transformations. It only interprets what has been detected.
    
    Parameters
    ----------
    rag_context : str, optional
        Read-only RAG context for pattern naming consistency.
        Agent works without RAG. RAG does not drive logic.
    
    Examples
    --------
    >>> diagnostics_input = {
    ...     "diagnostics": [
    ...         {
    ...             "issue_type": "high_skew",
    ...             "severity": "high",
    ...             "column": "Volume",
    ...             "metrics": {"skewness": 6.9051}
    ...         },
    ...         {
    ...             "issue_type": "outliers",
    ...             "severity": "high",
    ...             "column": "Volume",
    ...             "metrics": {"outlier_percentage": 12.7}
    ...         }
    ...     ],
    ...     "summary": {"total_issues": 2, "high_severity_count": 2},
    ...     "metadata": {"timestamp": "2026-02-06"}
    ... }
    >>> 
    >>> agent = DiagnosticsInterpreterAgent()
    >>> result = agent.interpret(diagnostics_input)
    >>> print(result.overall_assessment)
    'moderately_unstable'
    >>> print(result.dominant_issue_patterns)
    ['skew_and_outliers']
    """
    
    def __init__(self, rag_context: Optional[str] = None):
        """
        Initialize the DiagnosticsInterpreterAgent.
        
        Parameters
        ----------
        rag_context : str, optional
            Read-only context for pattern naming. Does not drive logic.
        """
        self.rag_context = rag_context
        logger.info("DiagnosticsInterpreterAgent initialized (read-only mode)")
    
    def interpret(self, diagnostics_input: Dict[str, Any]) -> InterpretationResult:
        """
        Interpret diagnostics to identify patterns and assess stability.
        
        This is the main entry point for the agent. It performs deterministic
        analysis of diagnostics without any recommendations or actions.
        
        Parameters
        ----------
        diagnostics_input : dict
            Diagnostics output from diagnostics_builder.py containing:
            - diagnostics: List of detected issues
            - summary: Summary statistics
            - metadata: Tracking information
        
        Returns
        -------
        InterpretationResult
            Structured interpretation with patterns and assessment
        
        Raises
        ------
        ValueError
            If diagnostics_input is missing required fields
        """
        # Validate input
        self._validate_input(diagnostics_input)
        
        diagnostics = diagnostics_input.get("diagnostics", [])
        summary = diagnostics_input.get("summary", {})
        
        # STEP 1: Identify patterns (deterministic)
        patterns = self._identify_patterns(diagnostics)
        
        # STEP 2: Build supporting evidence
        evidence = self._build_evidence(patterns, diagnostics)
        
        # STEP 3: Assess overall stability (deterministic)
        assessment = self._assess_stability(summary, diagnostics)
        
        # STEP 4: Calculate confidence (deterministic)
        confidence = self._calculate_confidence(summary)
        
        result = InterpretationResult(
            overall_assessment=assessment,
            dominant_issue_patterns=sorted(list(patterns)),  # Sort for determinism
            supporting_evidence=evidence,
            confidence=confidence
        )
        
        logger.info(
            f"Interpretation complete: {assessment} with {len(patterns)} patterns "
            f"(confidence: {confidence})"
        )
        
        return result
    
    def _validate_input(self, diagnostics_input: Dict[str, Any]) -> None:
        """Validate that diagnostics_input has required structure."""
        if not isinstance(diagnostics_input, dict):
            raise ValueError("diagnostics_input must be a dictionary")
        
        if "diagnostics" not in diagnostics_input:
            raise ValueError("diagnostics_input must contain 'diagnostics' key")
        
        if not isinstance(diagnostics_input["diagnostics"], list):
            raise ValueError("diagnostics must be a list")
    
    def _identify_patterns(self, diagnostics: List[Dict[str, Any]]) -> Set[str]:
        """
        Identify dominant issue patterns based on diagnostics.
        
        This is deterministic pattern matching based on:
        - Issue types present
        - Issue counts
        - Severity levels
        
        Parameters
        ----------
        diagnostics : list
            List of diagnostic issues
        
        Returns
        -------
        set
            Set of detected pattern names
        """
        patterns = set()
        
        # Extract issue types and group by column
        issue_types = {d.get("issue_type") for d in diagnostics if d.get("issue_type")}
        issue_by_column = {}
        
        for diag in diagnostics:
            col = diag.get("column", diag.get("columns", "unknown"))
            if isinstance(col, list):
                col = tuple(sorted(col))  # For correlation issues
            
            if col not in issue_by_column:
                issue_by_column[col] = []
            issue_by_column[col].append(diag)
        
        # Check each pattern rule
        for pattern_name, rule in PATTERN_RULES.items():
            required_types = rule.get("required_types", set())
            min_count = rule.get("min_count", 1)
            severity_threshold = rule.get("severity_threshold")
            
            # Check if required types are present
            if required_types and not required_types.issubset(issue_types):
                continue
            
            # Check if minimum count is met
            matching_issues = [
                d for d in diagnostics
                if d.get("issue_type") in required_types
            ]
            
            if len(matching_issues) < min_count:
                continue
            
            # Check severity threshold if specified
            if severity_threshold:
                high_severity_count = sum(
                    1 for d in matching_issues
                    if d.get("severity") == severity_threshold
                )
                if high_severity_count == 0:
                    continue
            
            # Pattern detected
            patterns.add(pattern_name)
        
        return patterns
    
    def _build_evidence(
        self,
        patterns: Set[str],
        diagnostics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build supporting evidence for detected patterns.
        
        Parameters
        ----------
        patterns : set
            Detected pattern names
        diagnostics : list
            List of diagnostic issues
        
        Returns
        -------
        list
            List of evidence dictionaries
        """
        evidence = []
        
        for pattern_name in sorted(patterns):  # Sort for deterministic order
            rule = PATTERN_RULES.get(pattern_name, {})
            required_types = rule.get("required_types", set())
            
            # Find issues matching this pattern
            matching_issues = [
                d for d in diagnostics
                if d.get("issue_type") in required_types
            ]
            
            # Extract affected columns
            affected_columns = set()
            for issue in matching_issues:
                col = issue.get("column", issue.get("columns"))
                if isinstance(col, list):
                    affected_columns.update(col)
                elif col:
                    affected_columns.add(col)
            
            evidence.append({
                "pattern": pattern_name,
                "linked_issue_types": sorted(list(required_types)),  # Sort for determinism
                "affected_columns": sorted(list(affected_columns))
            })
        
        return evidence
    
    def _assess_stability(
        self,
        summary: Dict[str, Any],
        diagnostics: List[Dict[str, Any]]
    ) -> str:
        """
        Assess overall dataset stability based on severity counts.
        
        This uses deterministic rules to classify stability level.
        
        Parameters
        ----------
        summary : dict
            Summary statistics from diagnostics
        diagnostics : list
            List of diagnostic issues
        
        Returns
        -------
        str
            One of: "stable", "moderately_unstable", "unstable"
        """
        # Count issues by severity
        high_severity_count = sum(
            1 for d in diagnostics
            if d.get("severity") == "high"
        )
        medium_severity_count = sum(
            1 for d in diagnostics
            if d.get("severity") == "medium"
        )
        total_issues = len(diagnostics)
        
        # Apply stability rules (order matters - most strict first)
        stable_rule = STABILITY_RULES["stable"]
        if (high_severity_count <= stable_rule["max_high_severity"] and
            medium_severity_count <= stable_rule["max_medium_severity"] and
            total_issues <= stable_rule["max_total_issues"]):
            return "stable"
        
        moderate_rule = STABILITY_RULES["moderately_unstable"]
        if (high_severity_count <= moderate_rule["max_high_severity"] and
            medium_severity_count <= moderate_rule["max_medium_severity"] and
            total_issues <= moderate_rule["max_total_issues"]):
            return "moderately_unstable"
        
        return "unstable"
    
    def _calculate_confidence(self, summary: Dict[str, Any]) -> str:
        """
        Calculate confidence level based on data completeness.
        
        Confidence is HIGH when diagnostics are comprehensive,
        MEDIUM when some information is missing, LOW when sparse.
        
        Parameters
        ----------
        summary : dict
            Summary statistics from diagnostics
        
        Returns
        -------
        str
            One of: "high", "medium", "low"
        """
        total_issues = summary.get("total_issues", 0)
        columns_analyzed = summary.get("columns_analyzed", 0)
        
        # Confidence based on analysis depth
        if total_issues >= 5 and columns_analyzed >= 5:
            return "high"
        elif total_issues >= 2 and columns_analyzed >= 3:
            return "medium"
        else:
            return "low"


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def interpret_diagnostics(
    diagnostics_input: Dict[str, Any],
    rag_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to interpret diagnostics.
    
    This is the recommended way to use the agent for one-off interpretations.
    
    Parameters
    ----------
    diagnostics_input : dict
        Diagnostics output from diagnostics_builder.py
    rag_context : str, optional
        Read-only RAG context for pattern naming
    
    Returns
    -------
    dict
        Interpretation result as dictionary (JSON-serializable)
    
    Examples
    --------
    >>> diagnostics = {
    ...     "diagnostics": [...],
    ...     "summary": {...},
    ...     "metadata": {...}
    ... }
    >>> result = interpret_diagnostics(diagnostics)
    >>> print(result["interpretation"]["overall_assessment"])
    """
    agent = DiagnosticsInterpreterAgent(rag_context=rag_context)
    interpretation = agent.interpret(diagnostics_input)
    return interpretation.to_dict()
