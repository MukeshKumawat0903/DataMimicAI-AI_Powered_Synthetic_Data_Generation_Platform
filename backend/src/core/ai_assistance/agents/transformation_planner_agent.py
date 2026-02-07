"""
Transformation Planner Agent (STEP 5 - Proposal-Only Planning)

This agent proposes transformation plans based on diagnostics and interpretation
results. It does NOT execute transformations or recommend a "best" plan.

STRICT SCOPE:
- Maps issue patterns to possible transformation strategies
- Proposes transformations at a conceptual level
- Groups transformations into coherent plans
- Provides rationale and risk assessment

NOT IN SCOPE:
- Executing transformations
- Ranking or prioritizing plans
- Selecting hyperparameters
- Modifying pipelines
- Recommending a "best" plan
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# TRANSFORMATION VOCABULARY (FIXED - DO NOT MODIFY)
# ============================================================================

ALLOWED_TRANSFORMATIONS = {
    "log_transform",
    "sqrt_transform",
    "scaling",
    "winsorization",
    "imputation",
    "encoding",
    "feature_deduplication",
    "dimensionality_reduction",
}


# ============================================================================
# PATTERN-TO-TRANSFORMATION MAPPING (Deterministic Rules)
# ============================================================================

PATTERN_TRANSFORMATION_RULES = {
    "skew_and_outliers": {
        "transformations": ["log_transform", "sqrt_transform", "winsorization"],
        "rationale_template": "Addresses heavy skew and outlier presence in numeric columns",
        "assumptions": ["Numeric columns are strictly positive (for log/sqrt)"],
        "risks": ["May reduce interpretability", "Log/sqrt require positive values"]
    },
    "high_feature_redundancy": {
        "transformations": ["feature_deduplication", "dimensionality_reduction"],
        "rationale_template": "Reduces feature redundancy from high correlations",
        "assumptions": ["Correlation indicates genuine redundancy, not domain relationships"],
        "risks": ["May discard features with distinct business meaning"]
    },
    "data_completeness_risk": {
        "transformations": ["imputation"],
        "rationale_template": "Addresses missing data patterns",
        "assumptions": ["Missing data mechanism is MAR or MCAR"],
        "risks": ["Imputation may introduce bias if data is MNAR"]
    },
    "distribution_instability": {
        "transformations": ["scaling", "log_transform", "sqrt_transform"],
        "rationale_template": "Stabilizes non-normal distributions",
        "assumptions": ["Transformations can approximate normality"],
        "risks": ["May not fully address heavy tails or multimodality"]
    },
    "sparse_data_patterns": {
        "transformations": ["imputation", "encoding"],
        "rationale_template": "Handles sparse data with missingness and low variance",
        "assumptions": ["Sparse patterns are data quality issues, not signal"],
        "risks": ["May introduce artificial density"]
    },
    "correlation_network": {
        "transformations": ["dimensionality_reduction", "feature_deduplication"],
        "rationale_template": "Manages dense correlation networks",
        "assumptions": ["PCA or feature selection can preserve information"],
        "risks": ["Reduced interpretability of transformed features"]
    },
}


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TransformationProposal:
    """A single transformation proposal."""
    transformation: str
    target_columns: List[str]
    rationale: str


@dataclass
class TransformationPlan:
    """A complete transformation plan."""
    plan_id: str
    applicable_issue_patterns: List[str]
    proposed_transformations: List[Dict[str, Any]]
    assumptions: List[str]
    risks: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "plan_id": self.plan_id,
            "applicable_issue_patterns": self.applicable_issue_patterns,
            "proposed_transformations": self.proposed_transformations,
            "assumptions": self.assumptions,
            "risks": self.risks
        }


@dataclass
class PlannerResult:
    """Complete result from the transformation planner."""
    proposed_plans: List[TransformationPlan]
    confidence: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "proposed_plans": [plan.to_dict() for plan in self.proposed_plans],
            "confidence": self.confidence
        }


# ============================================================================
# TRANSFORMATION PLANNER AGENT
# ============================================================================

class TransformationPlannerAgent:
    """
    Proposal-only agent that suggests transformation plans based on
    diagnostics and interpretation results.
    
    This agent performs STEP 5 reasoning:
    - Maps patterns to transformation strategies
    - Proposes transformations at a conceptual level
    - Groups transformations into coherent plans
    
    IMPORTANT: This agent does NOT execute transformations, rank plans,
    or recommend a "best" approach. It only proposes possibilities.
    
    Parameters
    ----------
    rag_context : str, optional
        Read-only RAG context for transformation naming standardization.
        Agent works without RAG. RAG does not drive logic.
    
    Examples
    --------
    >>> planner_input = {
    ...     "diagnostics": {...},
    ...     "interpretation": {
    ...         "dominant_issue_patterns": ["skew_and_outliers"],
    ...         "supporting_evidence": [...]
    ...     }
    ... }
    >>> 
    >>> agent = TransformationPlannerAgent()
    >>> result = agent.plan(planner_input)
    >>> print(len(result.proposed_plans))
    1
    >>> print(result.proposed_plans[0].plan_id)
    'TP-001'
    """
    
    def __init__(self, rag_context: Optional[str] = None):
        """
        Initialize the TransformationPlannerAgent.
        
        Parameters
        ----------
        rag_context : str, optional
            Read-only context for naming standardization. Does not drive logic.
        """
        self.rag_context = rag_context
        logger.info("TransformationPlannerAgent initialized (proposal-only mode)")
    
    def plan(self, planner_input: Dict[str, Any]) -> PlannerResult:
        """
        Generate transformation plans based on diagnostics and interpretation.
        
        This is the main entry point for the agent. It performs deterministic
        planning without execution or prioritization.
        
        Parameters
        ----------
        planner_input : dict
            Input containing:
            - diagnostics: Diagnostics output from diagnostics_builder
            - interpretation: Interpretation from DiagnosticsInterpreterAgent
        
        Returns
        -------
        PlannerResult
            Structured transformation plans with rationale and risks
        
        Raises
        ------
        ValueError
            If planner_input is missing required fields
        """
        # Validate input
        self._validate_input(planner_input)
        
        interpretation = planner_input.get("interpretation", {})
        diagnostics = planner_input.get("diagnostics", {})
        
        # STEP 1: Extract patterns and evidence
        patterns = interpretation.get("dominant_issue_patterns", [])
        evidence = interpretation.get("supporting_evidence", [])
        
        # STEP 2: Generate plans for each pattern (deterministic)
        # Pass full planner_input so _generate_plans can access signals
        plans = self._generate_plans(patterns, evidence, planner_input)
        
        # STEP 3: Calculate confidence (deterministic)
        confidence = self._calculate_confidence(interpretation, diagnostics)
        
        result = PlannerResult(
            proposed_plans=plans,
            confidence=confidence
        )
        
        logger.info(
            f"Planning complete: {len(plans)} plan(s) proposed "
            f"(confidence: {confidence})"
        )
        
        return result
    
    def _validate_input(self, planner_input: Dict[str, Any]) -> None:
        """Validate that planner_input has required structure."""
        if not isinstance(planner_input, dict):
            raise ValueError("planner_input must be a dictionary")
        
        if "interpretation" not in planner_input:
            raise ValueError("planner_input must contain 'interpretation' key")
        
        if "diagnostics" not in planner_input:
            raise ValueError("planner_input must contain 'diagnostics' key")
    
    def _generate_plans(
        self,
        patterns: List[str],
        evidence: List[Dict[str, Any]],
        planner_input: Dict[str, Any]
    ) -> List[TransformationPlan]:
        """
        Generate transformation plans for detected patterns.
        
        Parameters
        ----------
        patterns : list
            Dominant issue patterns from interpretation
        evidence : list
            Supporting evidence from interpretation
        planner_input : dict
            Full planner input containing diagnostics and signals
        
        Returns
        -------
        list
            List of TransformationPlan objects
        """
        plans = []
        
        # Sort patterns for determinism
        for i, pattern in enumerate(sorted(patterns), 1):
            # Check if we have transformation rules for this pattern
            if pattern not in PATTERN_TRANSFORMATION_RULES:
                logger.warning(f"No transformation rules for pattern: {pattern}")
                continue
            
            rule = PATTERN_TRANSFORMATION_RULES[pattern]
            
            # Find evidence for this pattern
            pattern_evidence = next(
                (e for e in evidence if e.get("pattern") == pattern),
                None
            )
            
            if not pattern_evidence:
                logger.warning(f"No evidence found for pattern: {pattern}")
                continue
            
            # Extract affected columns
            affected_columns = pattern_evidence.get("affected_columns", [])
            
            # Get column statistics for smart filtering (if available)
            column_stats = self._get_column_statistics(planner_input, affected_columns)
            
            # Build transformation proposals
            proposals = []
            for transformation in sorted(rule["transformations"]):  # Sort for determinism
                # SMART FILTERING: Skip transformations incompatible with column values
                filtered_columns = self._filter_compatible_columns(
                    transformation, 
                    affected_columns, 
                    column_stats
                )
                
                # Only add transformation if at least one column is compatible
                if filtered_columns:
                    proposals.append({
                        "transformation": transformation,
                        "target_columns": sorted(filtered_columns),  # Sort for determinism
                        "rationale": rule["rationale_template"]
                    })
            
            # Create plan
            plan = TransformationPlan(
                plan_id=f"TP-{i:03d}",
                applicable_issue_patterns=[pattern],
                proposed_transformations=proposals,
                assumptions=rule["assumptions"],
                risks=rule["risks"]
            )
            
            plans.append(plan)
        
        return plans
    
    def _get_column_statistics(
        self,
        planner_input: Dict[str, Any],
        affected_columns: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract column statistics from signals (if available).
        
        Parameters
        ----------
        planner_input : dict
            Full planner input containing diagnostics and signals
        affected_columns : list
            Columns to get statistics for
        
        Returns
        -------
        dict
            Dictionary mapping column names to their statistics (min, max, etc.)
        """
        column_stats = {}
        
        # Check if signals are available in planner input
        signals = planner_input.get("signals", {})
        if not signals:
            logger.debug("No signals available for column statistics")
            return column_stats
        
        columns_info = signals.get("columns", {})
        
        for col in affected_columns:
            if col in columns_info:
                col_info = columns_info[col]
                if isinstance(col_info, dict):
                    column_stats[col] = {
                        "min": col_info.get("min"),
                        "max": col_info.get("max"),
                        "type": col_info.get("type"),
                        "negative_count": col_info.get("negative_count", 0),
                        "zeros_count": col_info.get("zeros_count", 0)
                    }
        
        return column_stats
    
    def _filter_compatible_columns(
        self,
        transformation: str,
        columns: List[str],
        column_stats: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Filter columns that are compatible with the given transformation.
        
        This prevents proposing transformations that will fail during execution.
        
        Parameters
        ----------
        transformation : str
            Transformation type (e.g., "log_transform", "sqrt_transform")
        columns : list
            Candidate columns for transformation
        column_stats : dict
            Column statistics (min, max, etc.)
        
        Returns
        -------
        list
            Filtered list of compatible columns
        """
        # If no statistics available, return all columns (fail-safe behavior)
        if not column_stats:
            return columns
        
        compatible_columns = []
        
        for col in columns:
            stats = column_stats.get(col, {})
            min_val = stats.get("min")
            
            # Check transformation compatibility
            is_compatible = True
            
            # log_transform requires strictly positive values (min > 0)
            if transformation == "log_transform":
                if min_val is not None and min_val <= 0:
                    logger.info(
                        f"Skipping log_transform for column '{col}': "
                        f"contains non-positive values (min={min_val})"
                    )
                    is_compatible = False
            
            # sqrt_transform requires non-negative values (min >= 0)
            elif transformation == "sqrt_transform":
                if min_val is not None and min_val < 0:
                    logger.info(
                        f"Skipping sqrt_transform for column '{col}': "
                        f"contains negative values (min={min_val})"
                    )
                    is_compatible = False
            
            # Add other transformation-specific rules here as needed
            # (e.g., winsorization, scaling are generally safe for all numeric columns)
            
            if is_compatible:
                compatible_columns.append(col)
        
        return compatible_columns
    
    def _calculate_confidence(
        self,
        interpretation: Dict[str, Any],
        diagnostics: Dict[str, Any]
    ) -> str:
        """
        Calculate confidence level for proposed plans.
        
        Confidence is based on interpretation confidence and data completeness.
        
        Parameters
        ----------
        interpretation : dict
            Interpretation from DiagnosticsInterpreterAgent
        diagnostics : dict
            Diagnostics output
        
        Returns
        -------
        str
            One of: "high", "medium", "low"
        """
        interp_confidence = interpretation.get("confidence", "low")
        patterns_count = len(interpretation.get("dominant_issue_patterns", []))
        
        # High confidence: high interpretation confidence + multiple patterns
        if interp_confidence == "high" and patterns_count >= 2:
            return "high"
        
        # Medium confidence: medium+ interpretation or some patterns
        if interp_confidence in ["high", "medium"] or patterns_count >= 1:
            return "medium"
        
        # Low confidence: low interpretation or no patterns
        return "low"


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def plan_transformations(
    planner_input: Dict[str, Any],
    rag_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to plan transformations.
    
    This is the recommended way to use the agent for one-off planning.
    
    Parameters
    ----------
    planner_input : dict
        Input containing diagnostics and interpretation
    rag_context : str, optional
        Read-only RAG context for naming standardization
    
    Returns
    -------
    dict
        Transformation plans as dictionary (JSON-serializable)
    
    Examples
    --------
    >>> planner_input = {
    ...     "diagnostics": {...},
    ...     "interpretation": {...}
    ... }
    >>> result = plan_transformations(planner_input)
    >>> print(len(result["proposed_plans"]))
    """
    agent = TransformationPlannerAgent(rag_context=rag_context)
    result = agent.plan(planner_input)
    return result.to_dict()
