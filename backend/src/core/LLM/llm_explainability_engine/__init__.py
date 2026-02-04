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
]
