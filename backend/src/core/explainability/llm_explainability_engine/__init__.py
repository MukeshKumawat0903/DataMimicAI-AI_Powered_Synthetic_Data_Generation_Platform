"""
LLM Explainability Engine - Sub-package for LLM-based data explanations.

This sub-package provides a complete pipeline:
1. STEP 1 (explainable_signals): Extract all signals from data
2. STEP 2 (signal_selector): Filter and select focused contexts
3. STEP 3: (Optional) Retrieve RAG context
4. STEP 4 (prompt_builder): Build safe, structured prompts for LLM
5. STEP 5 (llama_inference): Execute LLM inference and return output
6. STEP 6 (output_validator): Validate and sanitize LLM output
"""

from .explainable_signals import (
    build_explainable_signals,
    ExplainableSignalsExtractor,
    SignalConfig,
    ColumnTypeInferrer,
    NumericColumnAnalyzer,
    CategoricalColumnAnalyzer,
    DatetimeColumnAnalyzer,
    TextColumnAnalyzer,
    CorrelationAnalyzer
)

from .signal_selector import (
    select_explainable_context,
    SignalContextSelector,
    SelectorConfig,
    DatasetOverviewSelector,
    ColumnAnalysisSelector,
    CorrelationAnalysisSelector,
    OutlierAnalysisSelector,
    TimeSeriesAnalysisSelector
)

from .prompt_builder import (
    build_explanation_prompt,
    ExplanationPromptBuilder,
    PromptConfig,
    SystemPromptBuilder,
    UserPromptBuilder,
    PromptMetadataBuilder
)

from .llama_inference import (
    run_llama_explanation,
    LLaMAInferenceEngine,
    GroqInferenceEngine,
    GroqAPIHandler,
    InferenceConfig
)

from .output_validator import (
    validate_llm_output,
    get_validation_report,
    OutputValidator,
    ValidatorConfig,
    HallucinationDetector,
    ScopeAlignmentChecker,
    LengthValidator,
    LanguageSanitizer,
    NumericExtractor
)

from .baseline_guard import (
    BaselineGuard,
    BaselineSnapshot,
    capture_signals_snapshot,
    assert_signals_structure,
    assert_context_matches_signals,
    get_snapshot_summary,
    enable_baseline_guard,
    disable_baseline_guard,
    is_baseline_guard_enabled
)

from .diagnostics_builder import (
    build_diagnostics,
    DiagnosticsThresholds,
    get_diagnostics_by_severity,
    get_diagnostics_by_type,
    get_affected_columns,
    classify_skewness_severity,
    classify_missing_severity,
    classify_outlier_severity,
    classify_drift_ks_severity,
    classify_drift_psi_severity,
    classify_imbalance_severity,
    classify_cardinality_severity,
    classify_correlation_severity
)

from .diagnostics_explainer import (
    format_diagnostics_for_explanation,
    build_diagnostics_context_for_prompt,
    get_issue_type_description,
    get_severity_label
)

__all__ = [
    # STEP 1: Signal extraction
    'build_explainable_signals',
    'ExplainableSignalsExtractor',
    'SignalConfig',
    'ColumnTypeInferrer',
    'NumericColumnAnalyzer',
    'CategoricalColumnAnalyzer',
    'DatetimeColumnAnalyzer',
    'TextColumnAnalyzer',
    'CorrelationAnalyzer',
    
    # STEP 2: Context selection
    'select_explainable_context',
    'SignalContextSelector',
    'SelectorConfig',
    'DatasetOverviewSelector',
    'ColumnAnalysisSelector',
    'CorrelationAnalysisSelector',
    'OutlierAnalysisSelector',
    'TimeSeriesAnalysisSelector',
    
    # STEP 4: Prompt building
    'build_explanation_prompt',
    'ExplanationPromptBuilder',
    'PromptConfig',
    'SystemPromptBuilder',
    'UserPromptBuilder',
    'PromptMetadataBuilder',
    
    # STEP 5: LLM inference
    'run_llama_explanation',
    'LLaMAInferenceEngine',
    'GroqInferenceEngine',
    'GroqAPIHandler',
    'InferenceConfig',
    
    # STEP 6: Output validation
    'validate_llm_output',
    'get_validation_report',
    'OutputValidator',
    'ValidatorConfig',
    'HallucinationDetector',
    'ScopeAlignmentChecker',
    'LengthValidator',
    'LanguageSanitizer',
    'NumericExtractor',
    
    # Baseline Guard: Protection for existing behavior
    'BaselineGuard',
    'BaselineSnapshot',
    'capture_signals_snapshot',
    'assert_signals_structure',
    'assert_context_matches_signals',
    'get_snapshot_summary',
    'enable_baseline_guard',
    'disable_baseline_guard',
    'is_baseline_guard_enabled',
    
    # Diagnostics Builder: Deterministic issue classification
    'build_diagnostics',
    'DiagnosticsThresholds',
    'get_diagnostics_by_severity',
    'get_diagnostics_by_type',
    'get_affected_columns',
    'classify_skewness_severity',
    'classify_missing_severity',
    'classify_outlier_severity',
    'classify_drift_ks_severity',
    'classify_drift_psi_severity',
    'classify_imbalance_severity',
    'classify_cardinality_severity',
    'classify_correlation_severity',
    
    # Diagnostics Explainer: Presentation layer for diagnostics
    'format_diagnostics_for_explanation',
    'build_diagnostics_context_for_prompt',
    'get_issue_type_description',
    'get_severity_label'
]
