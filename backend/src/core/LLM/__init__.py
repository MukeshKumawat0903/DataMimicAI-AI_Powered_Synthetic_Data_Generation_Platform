"""
LLM Module - Utilities for LLM-based data explanations.
This is a wrapper for the llm_explainability_engine sub-package.
"""

from .llm_explainability_engine import (
    # STEP 1: Signal extraction
    build_explainable_signals,
    ExplainableSignalsExtractor,
    SignalConfig,
    ColumnTypeInferrer,
    NumericColumnAnalyzer,
    CategoricalColumnAnalyzer,
    DatetimeColumnAnalyzer,
    TextColumnAnalyzer,
    CorrelationAnalyzer,
    
    # STEP 1.5: Diagnostics building
    build_diagnostics,
    DiagnosticsThresholds,
    
    # STEP 2: Context selection
    select_explainable_context,
    SignalContextSelector,
    SelectorConfig,
    DatasetOverviewSelector,
    ColumnAnalysisSelector,
    CorrelationAnalysisSelector,
    OutlierAnalysisSelector,
    TimeSeriesAnalysisSelector,
    
    # STEP 3: Diagnostics formatting for explanation
    build_diagnostics_context_for_prompt,
    format_diagnostics_for_explanation,
    
    # STEP 4: Prompt building
    build_explanation_prompt,
    ExplanationPromptBuilder,
    PromptConfig,
    SystemPromptBuilder,
    UserPromptBuilder,
    PromptMetadataBuilder,
    
    # STEP 5: LLM inference
    run_llama_explanation,
    LLaMAInferenceEngine,
    GroqInferenceEngine,
    GroqAPIHandler,
    InferenceConfig,
    
    # STEP 6: Output validation
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

__all__ = [
    'build_explainable_signals',
    'ExplainableSignalsExtractor',
    'SignalConfig',
    'ColumnTypeInferrer',
    'NumericColumnAnalyzer',
    'CategoricalColumnAnalyzer',
    'DatetimeColumnAnalyzer',
    'TextColumnAnalyzer',
    'CorrelationAnalyzer',
    'build_diagnostics',
    'DiagnosticsThresholds',
    'select_explainable_context',
    'SignalContextSelector',
    'SelectorConfig',
    'DatasetOverviewSelector',
    'ColumnAnalysisSelector',
    'CorrelationAnalysisSelector',
    'OutlierAnalysisSelector',
    'TimeSeriesAnalysisSelector',
    'build_diagnostics_context_for_prompt',
    'format_diagnostics_for_explanation',
    'build_explanation_prompt',
    'ExplanationPromptBuilder',
    'PromptConfig',
    'SystemPromptBuilder',
    'UserPromptBuilder',
    'PromptMetadataBuilder',
    'run_llama_explanation',
    'LLaMAInferenceEngine',
    'GroqInferenceEngine',
    'GroqAPIHandler',
    'InferenceConfig',
    'validate_llm_output',
    'get_validation_report',
    'OutputValidator',
    'ValidatorConfig',
    'HallucinationDetector',
    'ScopeAlignmentChecker',
    'LengthValidator',
    'LanguageSanitizer',
    'NumericExtractor',
]

