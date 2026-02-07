"""
DeterministicExecutionEngine - Non-Agentic Execution Layer

This module is NOT an agent. It contains ZERO reasoning or intelligence.
It executes ONLY human-approved transformation plans.

PURPOSE:
    - Verify plan approval status
    - Execute approved transformations deterministically
    - Return structured execution results

CONSTRAINTS:
    - No LLM calls
    - No reasoning or decision-making
    - No parameter tuning or optimization
    - No modification of unapproved plans
    - No partial execution
    - Deterministic behavior only
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass


class ExecutionStatus(str, Enum):
    """Fixed execution states - no additions allowed."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TransformationVocabulary(str, Enum):
    """
    Allowed transformation names - exact match required.
    No dynamic dispatch or reflection allowed.
    """
    LOG_TRANSFORM = "log_transform"
    SQRT_TRANSFORM = "sqrt_transform"
    SCALING = "scaling"
    NORMALIZATION = "normalization"
    WINSORIZATION = "winsorization"
    IMPUTATION = "imputation"
    ENCODING = "encoding"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    FEATURE_DEDUPLICATION = "feature_deduplication"
    OUTLIER_REMOVAL = "outlier_removal"


@dataclass
class ExecutionResult:
    """Structured execution result."""
    plan_id: str
    execution_status: str
    applied_transformations: List[str]
    error: Optional[str]
    transformed_data: Optional[pd.DataFrame]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding transformed_data for serialization)."""
        return {
            "plan_id": self.plan_id,
            "execution_status": self.execution_status,
            "applied_transformations": self.applied_transformations,
            "error": self.error
        }


class DeterministicExecutionEngine:
    """
    Non-agentic execution layer for approved transformation plans.
    
    This is an EXECUTOR, not a decision maker.
    
    Responsibilities:
        - Verify plan approval status
        - Map transformations to deterministic functions
        - Execute transformations in order
        - Return structured results
    
    Non-Responsibilities (NEVER):
        - Infer approvals
        - Modify plans
        - Tune parameters
        - Choose execution order
        - Optimize transformations
        - Call LLMs or agents
    """
    
    def __init__(self):
        """Initialize the execution engine."""
        # No configuration needed - purely deterministic
        pass
    
    def execute(
        self,
        approval_record: Dict[str, Any],
        plan: Dict[str, Any],
        data: pd.DataFrame
    ) -> ExecutionResult:
        """
        Execute an APPROVED transformation plan.
        
        This method EXECUTES only. It does NOT:
            - Infer approvals
            - Modify plans
            - Make decisions
            - Call external services
        
        Args:
            approval_record: Approval record from PlanReviewAndApprovalGate
            plan: Transformation plan from TransformationPlannerAgent
            data: Input dataframe to transform
        
        Returns:
            ExecutionResult with status, applied transformations, and result data
        
        Raises:
            ValueError: If preconditions not met (not approved, invalid schema, etc.)
        """
        try:
            # PRECONDITION 1: Verify plan schema (must be first to get valid plan_id)
            self._verify_plan_schema(plan)
            
            plan_id = plan.get("plan_id")  # Now guaranteed to exist
            
            # PRECONDITION 2: Verify approval status
            self._verify_approval_status(approval_record, plan_id)
            
            # PRECONDITION 3: Verify data is valid
            self._verify_data(data)
            
            # Execute transformations in order
            transformed_data = data.copy()  # Work on a copy
            applied_transformations = []
            
            transformations = plan.get("proposed_transformations", [])
            
            for idx, transformation_spec in enumerate(transformations):
                transformation_name = transformation_spec.get("transformation")
                target_columns = transformation_spec.get("target_columns", [])
                
                # Verify transformation is in allowed vocabulary
                self._verify_transformation_vocabulary(transformation_name)
                
                # Verify target columns exist in data
                self._verify_columns_exist(transformed_data, target_columns)
                
                # Execute transformation
                transformed_data = self._execute_transformation(
                    transformed_data,
                    transformation_name,
                    target_columns
                )
                
                applied_transformations.append(transformation_name)
            
            # SUCCESS
            return ExecutionResult(
                plan_id=plan_id,
                execution_status=ExecutionStatus.SUCCESS.value,
                applied_transformations=applied_transformations,
                error=None,
                transformed_data=transformed_data
            )
        
        except Exception as e:
            # FAILURE - return error without partial results
            # Try to get plan_id from plan, fallback to "UNKNOWN"
            error_plan_id = plan.get("plan_id", "UNKNOWN") if isinstance(plan, dict) else "UNKNOWN"
            
            return ExecutionResult(
                plan_id=error_plan_id,
                execution_status=ExecutionStatus.FAILED.value,
                applied_transformations=[],
                error=str(e),
                transformed_data=None
            )
    
    # =====================================================================
    # PRECONDITION VERIFICATION (STRICT)
    # =====================================================================
    
    def _verify_approval_status(
        self,
        approval_record: Dict[str, Any],
        plan_id: str
    ) -> None:
        """
        Verify plan has APPROVED status.
        
        Raises:
            ValueError: If plan is not approved
        """
        if not isinstance(approval_record, dict):
            raise ValueError(
                f"Approval record must be a dictionary for plan {plan_id}"
            )
        
        status = approval_record.get("status")
        if status != "APPROVED":
            raise ValueError(
                f"Plan {plan_id} is not approved. Status: {status}. "
                "Only APPROVED plans can be executed."
            )
        
        # Verify plan_id matches
        record_plan_id = approval_record.get("plan_id")
        if record_plan_id != plan_id:
            raise ValueError(
                f"Approval record plan_id '{record_plan_id}' does not match "
                f"plan plan_id '{plan_id}'"
            )
    
    def _verify_plan_schema(self, plan: Dict[str, Any]) -> None:
        """
        Verify plan has required structure.
        
        Raises:
            ValueError: If plan schema is invalid
        """
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a dictionary")
        
        # Required fields
        required_fields = ["plan_id", "proposed_transformations"]
        for field in required_fields:
            if field not in plan:
                raise ValueError(f"Plan missing required field: {field}")
        
        # Verify plan_id
        plan_id = plan.get("plan_id")
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("plan_id must be a non-empty string")
        
        # Verify proposed_transformations
        transformations = plan.get("proposed_transformations")
        if not isinstance(transformations, list):
            raise ValueError("proposed_transformations must be a list")
        
        # Verify each transformation
        for idx, transformation in enumerate(transformations):
            if not isinstance(transformation, dict):
                raise ValueError(
                    f"Transformation at index {idx} must be a dictionary"
                )
            
            required_transform_fields = ["transformation", "target_columns"]
            for field in required_transform_fields:
                if field not in transformation:
                    raise ValueError(
                        f"Transformation at index {idx} missing field: {field}"
                    )
    
    def _verify_data(self, data: pd.DataFrame) -> None:
        """
        Verify data is a valid DataFrame.
        
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"Data must be a pandas DataFrame, got: {type(data)}"
            )
        
        if data.empty:
            raise ValueError("Data is empty - cannot execute transformations")
    
    def _verify_transformation_vocabulary(self, transformation_name: str) -> None:
        """
        Verify transformation is in allowed vocabulary.
        
        Raises:
            ValueError: If transformation not in vocabulary
        """
        allowed_values = [t.value for t in TransformationVocabulary]
        
        if transformation_name not in allowed_values:
            raise ValueError(
                f"Transformation '{transformation_name}' not in allowed vocabulary. "
                f"Allowed: {sorted(allowed_values)}"
            )
    
    def _verify_columns_exist(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> None:
        """
        Verify target columns exist in data.
        
        Raises:
            ValueError: If columns don't exist
        """
        missing_columns = [col for col in target_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(
                f"Target columns not found in data: {missing_columns}. "
                f"Available columns: {sorted(data.columns.tolist())}"
            )
    
    # =====================================================================
    # TRANSFORMATION EXECUTION (DETERMINISTIC MAPPING)
    # =====================================================================
    
    def _execute_transformation(
        self,
        data: pd.DataFrame,
        transformation_name: str,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Execute a single transformation deterministically.
        
        This is a FIXED MAPPING - no dynamic dispatch.
        
        Args:
            data: Input dataframe
            transformation_name: Name of transformation (from vocabulary)
            target_columns: Columns to transform
        
        Returns:
            Transformed dataframe
        
        Raises:
            ValueError: If transformation fails
        """
        # FIXED MAPPING - no reflection, no dynamic dispatch
        if transformation_name == TransformationVocabulary.LOG_TRANSFORM.value:
            return self._apply_log_transform(data, target_columns)
        
        elif transformation_name == TransformationVocabulary.SQRT_TRANSFORM.value:
            return self._apply_sqrt_transform(data, target_columns)
        
        elif transformation_name == TransformationVocabulary.SCALING.value:
            return self._apply_scaling(data, target_columns)
        
        elif transformation_name == TransformationVocabulary.NORMALIZATION.value:
            return self._apply_normalization(data, target_columns)
        
        elif transformation_name == TransformationVocabulary.WINSORIZATION.value:
            return self._apply_winsorization(data, target_columns)
        
        elif transformation_name == TransformationVocabulary.IMPUTATION.value:
            return self._apply_imputation(data, target_columns)
        
        elif transformation_name == TransformationVocabulary.ENCODING.value:
            return self._apply_encoding(data, target_columns)
        
        elif transformation_name == TransformationVocabulary.DIMENSIONALITY_REDUCTION.value:
            return self._apply_dimensionality_reduction(data, target_columns)
        
        elif transformation_name == TransformationVocabulary.FEATURE_DEDUPLICATION.value:
            return self._apply_feature_deduplication(data, target_columns)
        
        elif transformation_name == TransformationVocabulary.OUTLIER_REMOVAL.value:
            return self._apply_outlier_removal(data, target_columns)
        
        else:
            # Should never reach here due to vocabulary validation
            raise ValueError(
                f"Transformation '{transformation_name}' not mapped to implementation"
            )
    
    # =====================================================================
    # DETERMINISTIC TRANSFORMATION FUNCTIONS
    # =====================================================================
    
    def _apply_log_transform(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply log transformation to target columns.
        
        Deterministic: log(x + 1) to handle zeros.
        """
        result = data.copy()
        
        for col in target_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric for log transform"
                )
            
            if (result[col] < 0).any():
                raise ValueError(
                    f"Column '{col}' contains negative values - cannot apply log transform"
                )
            
            result[col] = np.log1p(result[col])  # log(1 + x)
        
        return result
    
    def _apply_sqrt_transform(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply square root transformation to target columns.
        
        Deterministic: sqrt(x) with validation.
        """
        result = data.copy()
        
        for col in target_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric for sqrt transform"
                )
            
            if (result[col] < 0).any():
                raise ValueError(
                    f"Column '{col}' contains negative values - cannot apply sqrt transform"
                )
            
            result[col] = np.sqrt(result[col])
        
        return result
    
    def _apply_scaling(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply min-max scaling to target columns.
        
        Deterministic: (x - min) / (max - min).
        """
        result = data.copy()
        
        for col in target_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric for scaling"
                )
            
            min_val = result[col].min()
            max_val = result[col].max()
            
            if min_val == max_val:
                # Constant column - set to 0
                result[col] = 0.0
            else:
                result[col] = (result[col] - min_val) / (max_val - min_val)
        
        return result
    
    def _apply_normalization(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply z-score normalization to target columns.
        
        Deterministic: (x - mean) / std.
        """
        result = data.copy()
        
        for col in target_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric for normalization"
                )
            
            mean_val = result[col].mean()
            std_val = result[col].std()
            
            if std_val == 0:
                # Constant column - set to 0
                result[col] = 0.0
            else:
                result[col] = (result[col] - mean_val) / std_val
        
        return result
    
    def _apply_winsorization(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply winsorization to target columns (cap at 1st and 99th percentiles).
        
        Deterministic: clip values at percentile boundaries.
        """
        result = data.copy()
        
        for col in target_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric for winsorization"
                )
            
            lower_bound = result[col].quantile(0.01)
            upper_bound = result[col].quantile(0.99)
            
            result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        
        return result
    
    def _apply_imputation(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply mean/mode imputation to target columns.
        
        Deterministic: mean for numeric, mode for categorical.
        """
        result = data.copy()
        
        for col in target_columns:
            if result[col].isna().sum() == 0:
                # No missing values - skip
                continue
            
            if pd.api.types.is_numeric_dtype(result[col]):
                # Impute with mean
                result[col] = result[col].fillna(result[col].mean())
            else:
                # Impute with mode
                mode_value = result[col].mode()
                if len(mode_value) > 0:
                    result[col] = result[col].fillna(mode_value[0])
                else:
                    # No mode - fill with placeholder
                    result[col] = result[col].fillna("MISSING")
        
        return result
    
    def _apply_encoding(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply label encoding to target columns.
        
        Deterministic: sorted label encoding (alphabetical order).
        """
        result = data.copy()
        
        for col in target_columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                # Already numeric - skip
                continue
            
            # Get unique values in sorted order (deterministic)
            unique_values = sorted(result[col].dropna().unique())
            
            # Create mapping
            encoding_map = {val: idx for idx, val in enumerate(unique_values)}
            
            # Apply encoding
            result[col] = result[col].map(encoding_map)
        
        return result
    
    def _apply_dimensionality_reduction(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply dimensionality reduction (keep first column, drop rest).
        
        Deterministic: simple column selection (not PCA - would require sklearn).
        """
        result = data.copy()
        
        if len(target_columns) <= 1:
            # Nothing to reduce
            return result
        
        # Keep only the first target column
        columns_to_drop = target_columns[1:]
        result = result.drop(columns=columns_to_drop)
        
        return result
    
    def _apply_feature_deduplication(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply feature deduplication (remove duplicate columns).
        
        Deterministic: keep first occurrence, drop rest.
        """
        result = data.copy()
        
        # Find columns with identical values
        columns_to_drop = []
        
        for i in range(len(target_columns)):
            for j in range(i + 1, len(target_columns)):
                col1 = target_columns[i]
                col2 = target_columns[j]
                
                if col1 in result.columns and col2 in result.columns:
                    if result[col1].equals(result[col2]):
                        # Duplicate found - mark second column for removal
                        if col2 not in columns_to_drop:
                            columns_to_drop.append(col2)
        
        if columns_to_drop:
            result = result.drop(columns=columns_to_drop)
        
        return result
    
    def _apply_outlier_removal(
        self,
        data: pd.DataFrame,
        target_columns: List[str]
    ) -> pd.DataFrame:
        """
        Apply outlier removal (IQR method).
        
        Deterministic: remove rows where any target column has outliers.
        """
        result = data.copy()
        
        mask = pd.Series([True] * len(result))
        
        for col in target_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric for outlier removal"
                )
            
            Q1 = result[col].quantile(0.25)
            Q3 = result[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Update mask - keep rows within bounds
            mask &= (result[col] >= lower_bound) & (result[col] <= upper_bound)
        
        result = result[mask]
        
        return result


# =====================================================================
# CONVENIENCE FUNCTION
# =====================================================================

def execute_approved_plan(
    approval_record: Dict[str, Any],
    plan: Dict[str, Any],
    data: pd.DataFrame
) -> ExecutionResult:
    """
    Convenience function to execute an approved plan.
    
    Args:
        approval_record: Approval record from PlanReviewAndApprovalGate
        plan: Transformation plan from TransformationPlannerAgent
        data: Input dataframe
    
    Returns:
        ExecutionResult
    """
    engine = DeterministicExecutionEngine()
    return engine.execute(approval_record, plan, data)
