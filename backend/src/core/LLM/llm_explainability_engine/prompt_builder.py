"""
Prompt Builder for LLM-Based Data Explanations (STEP 4).

This module takes scoped contexts from STEP 2 and builds safe, structured
prompts for LLM consumption. It acts as the SAFETY GATE before LLM inference,
ensuring that the LLM only explains provided facts and never invents statistics.

Key Principle: The LLM must NEVER compute or invent values. It should ONLY
explain the significance and implications of the provided facts.

Author: DataMimicAI Team
Date: February 2026
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PromptConfig:
    """Configuration for prompt building behavior."""
    
    # Available tones
    valid_tones: tuple = ("clear", "concise", "technical", "beginner-friendly", "detailed")
    
    # Default tone
    default_tone: str = "clear"
    
    # System prompt components
    model_role: str = "data science assistant"
    max_system_prompt_length: int = 2000
    max_user_prompt_length: int = 8000
    
    # Output formatting
    include_json_in_prompt: bool = True
    pretty_print_json: bool = True


class SystemPromptBuilder:
    """Builds system prompts that define the LLM's role and constraints."""
    
    def __init__(self, config: PromptConfig):
        """Initialize with configuration."""
        self.config = config
    
    def build(self, tone: str, scope: str) -> str:
        """
        Build a system prompt based on tone and scope.
        
        Parameters
        ----------
        tone : str
            Desired explanation tone
        scope : str
            Analysis scope from STEP 2
        
        Returns
        -------
        str
            System prompt text
        """
        # Base role definition
        system_prompt = self._get_base_role()
        
        # Add strict constraints
        system_prompt += "\n\n" + self._get_constraints()
        
        # Add tone-specific instructions
        system_prompt += "\n\n" + self._get_tone_instructions(tone)
        
        # Add scope-specific guidance
        system_prompt += "\n\n" + self._get_scope_guidance(scope)
        
        return system_prompt.strip()
    
    def _get_base_role(self) -> str:
        """Define the LLM's base role."""
        return (
            f"You are an expert {self.config.model_role} specializing in data quality analysis "
            "and exploratory data analysis (EDA). Your role is to provide clear, actionable "
            "explanations of statistical findings to help data scientists and ML engineers "
            "understand their datasets better."
        )
    
    def _get_constraints(self) -> str:
        """Define strict constraints for the LLM."""
        return (
            "CRITICAL CONSTRAINTS:\n"
            "- You must NEVER compute, calculate, or invent any numerical values\n"
            "- You must ONLY explain the facts and statistics that are explicitly provided\n"
            "- You must NEVER make assumptions about data not shown to you\n"
            "- You must reference specific numbers from the provided facts when explaining\n"
            "- If a fact is missing or unclear, acknowledge the limitation rather than guessing\n"
            "- Focus on explaining WHY patterns matter and WHAT actions to consider"
        )
    
    def _get_tone_instructions(self, tone: str) -> str:
        """Get instructions based on desired tone."""
        tone_instructions = {
            "clear": (
                "TONE: Provide clear, straightforward explanations. Use plain language while "
                "maintaining technical accuracy. Explain complex concepts in accessible terms."
            ),
            "concise": (
                "TONE: Be brief and to the point. Focus on the most important insights. "
                "Use short sentences and avoid unnecessary elaboration."
            ),
            "technical": (
                "TONE: Use precise technical terminology. Assume the reader has a strong "
                "statistical background. Reference specific statistical concepts and methodologies."
            ),
            "beginner-friendly": (
                "TONE: Explain concepts as if to someone new to data science. Define technical "
                "terms when used. Provide context and analogies where helpful."
            ),
            "detailed": (
                "TONE: Provide comprehensive explanations with thorough analysis. Cover multiple "
                "perspectives and implications. Discuss potential causes and recommendations."
            )
        }
        
        return tone_instructions.get(tone, tone_instructions["clear"])
    
    def _get_scope_guidance(self, scope: str) -> str:
        """Get guidance specific to the analysis scope."""
        scope_guidance = {
            "dataset_overview": (
                "SCOPE: You are explaining high-level dataset characteristics. Focus on overall "
                "data quality, size considerations, and general patterns. Highlight any red flags "
                "that would require immediate attention before modeling."
            ),
            "column_analysis": (
                "SCOPE: You are explaining specific column characteristics. For each column, "
                "discuss data type appropriateness, distribution patterns, missing data implications, "
                "and potential feature engineering opportunities."
            ),
            "correlation_analysis": (
                "SCOPE: You are explaining relationships between features. Discuss the strength "
                "and direction of correlations, potential multicollinearity issues, and implications "
                "for feature selection and model performance."
            ),
            "outlier_analysis": (
                "SCOPE: You are explaining outlier patterns. Discuss whether outliers are likely "
                "errors or valid extreme values, their potential impact on modeling, and suggested "
                "handling strategies (removal, transformation, robust methods)."
            ),
            "time_series_analysis": (
                "SCOPE: You are explaining temporal patterns. Discuss trends, seasonality indicators, "
                "data frequency, and considerations for time-based splitting and feature engineering."
            )
        }
        
        return scope_guidance.get(scope, "SCOPE: Provide general data analysis insights.")


class UserPromptBuilder:
    """Builds user prompts containing the task, facts, and context."""
    
    def __init__(self, config: PromptConfig):
        """Initialize with configuration."""
        self.config = config
    
    def build(
        self,
        scoped_context: Dict[str, Any],
        rag_context: Optional[str] = None
    ) -> str:
        """
        Build a user prompt with task instructions and facts.
        
        Parameters
        ----------
        scoped_context : dict
            Scoped context from STEP 2
        rag_context : str, optional
            Retrieved background knowledge
        
        Returns
        -------
        str
            User prompt text
        """
        prompt_parts = []
        
        # Task instruction
        prompt_parts.append(self._get_task_instruction(scoped_context["scope"]))
        
        # Facts section
        prompt_parts.append("\n\n" + self._format_facts(scoped_context))
        
        # RAG context if provided
        if rag_context:
            prompt_parts.append("\n\n" + self._format_rag_context(rag_context))
        
        # Output expectations
        prompt_parts.append("\n\n" + self._get_output_expectations())
        
        return "".join(prompt_parts).strip()
    
    def _get_task_instruction(self, scope: str) -> str:
        """Generate task instruction based on scope."""
        task_instructions = {
            "dataset_overview": (
                "Please provide a comprehensive explanation of the overall dataset characteristics. "
                "Analyze the data quality, size, and composition. Identify any immediate concerns "
                "or notable patterns that would impact downstream analysis or modeling."
            ),
            "column_analysis": (
                "Please analyze the provided column statistics in detail. For each column, explain "
                "the distribution patterns, data quality issues, and potential implications for "
                "feature engineering or modeling. Suggest appropriate handling strategies."
            ),
            "correlation_analysis": (
                "Please explain the correlations found between features. Discuss the strength and "
                "significance of these relationships, potential multicollinearity concerns, and "
                "implications for feature selection and model interpretation."
            ),
            "outlier_analysis": (
                "Please analyze the outlier patterns in the dataset. For each column with outliers, "
                "discuss likely causes, potential impact on analysis, and recommend appropriate "
                "handling strategies (removal, transformation, or robust methods)."
            ),
            "time_series_analysis": (
                "Please examine the temporal characteristics of the dataset. Explain any trends, "
                "patterns, or irregularities. Discuss implications for time-based feature engineering "
                "and model validation strategies."
            )
        }
        
        return task_instructions.get(
            scope,
            "Please analyze and explain the provided statistical findings."
        )
    
    def _format_facts(self, scoped_context: Dict[str, Any]) -> str:
        """Format facts for inclusion in prompt."""
        facts = scoped_context.get("facts", {})
        scope = scoped_context.get("scope", "unknown")
        
        if self.config.include_json_in_prompt:
            # Format as JSON for clarity
            if self.config.pretty_print_json:
                facts_json = json.dumps(facts, indent=2)
            else:
                facts_json = json.dumps(facts)
            
            return (
                f"--- STATISTICAL FACTS (Scope: {scope}) ---\n"
                f"```json\n{facts_json}\n```\n"
                "--- END OF FACTS ---"
            )
        else:
            # Format as human-readable text
            return self._facts_to_text(facts, scope)
    
    def _facts_to_text(self, facts: Dict[str, Any], scope: str) -> str:
        """Convert facts dictionary to human-readable text format."""
        lines = [f"--- STATISTICAL FACTS (Scope: {scope}) ---"]
        
        for key, value in facts.items():
            if isinstance(value, dict):
                lines.append(f"\n{key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  - {sub_key.replace('_', ' ')}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"\n{key.replace('_', ' ').title()}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        
        lines.append("\n--- END OF FACTS ---")
        return "\n".join(lines)
    
    def _format_rag_context(self, rag_context: str) -> str:
        """Format RAG context for inclusion in prompt."""
        return (
            "--- REFERENCE KNOWLEDGE ---\n"
            f"{rag_context}\n"
            "--- END OF REFERENCE ---\n\n"
            "Use this reference knowledge to provide additional context and best practices, "
            "but always prioritize explaining the specific facts provided above."
        )
    
    def _get_output_expectations(self) -> str:
        """Define what output format is expected."""
        return (
            "--- OUTPUT REQUIREMENTS ---\n"
            "Please structure your explanation with these distinct sections:\n\n"
            
            "## Step 1: KEY FINDINGS\n"
            "Write a SHORT, executive-style summary using bullet points:\n"
            "- Dataset size and readiness for modeling or synthetic generation\n"
            "- Overall data quality status\n"
            "- Major risks (missingness, outliers, skewness)\n"
            "- One high-level recommendation\n"
            "Keep this section concise (3-5 bullet points maximum).\n\n"
            
            "## Step 2: DETAILED ANALYSIS\n"
            "Provide deeper analysis WITHOUT repeating dataset size or memory usage.\n"
            "Use bullet points (like Step 1) and cover:\n"
            "- Distribution issues (skewness, heavy tails, non-normality)\n"
            "- Missingness patterns and likely impact\n"
            "- Outlier presence and practical implications\n"
            "Keep bullets factual and grounded in the provided numbers.\n\n"
            
            "## Step 3: WHY THIS MATTERS\n"
            "Explain practical impact using bullet points (like Step 1):\n"
            "- How missing values or outliers could affect model training\n"
            "- What distribution issues mean for synthetic data generation\n"
            "- Which preprocessing steps would improve data quality\n"
            "Keep language neutral (use 'may', 'could', 'likely' instead of absolutes).\n"
            "Do NOT introduce new numbers or new findings—only interpret existing facts.\n\n"
            
            "## Step 4: RECOMMENDATIONS\n"
            "Provide 2–3 complete, actionable bullet points.\n"
            "Ensure each bullet is a full sentence and fully written (no incomplete items).\n"
            "This section MUST be present and use bullet points like Step 1.\n\n"

            "## Trust Note\n"
            "End with this exact sentence on its own line:\n"
            "This explanation is based on computed statistics from your data.\n\n"
            
            "CRITICAL RULES:\n"
            "- Reference specific values from the provided facts\n"
            "- Do NOT compute, calculate, or invent any new numbers\n"
            "- Do NOT introduce facts not present in the provided statistics\n"
            "- Use bullet points in KEY FINDINGS\n"
            "- Use short paragraphs in other sections\n"
            "- Avoid overconfident language (no 'guarantees', 'always', 'never')\n"
            "- Focus on actionable insights, not just descriptions"
        )


class PromptMetadataBuilder:
    """Builds metadata for prompt tracking and auditing."""
    
    @staticmethod
    def build(
        scoped_context: Dict[str, Any],
        tone: str,
        has_rag: bool,
        system_prompt_length: int,
        user_prompt_length: int
    ) -> Dict[str, Any]:
        """
        Build metadata about the generated prompt.
        
        Parameters
        ----------
        scoped_context : dict
            Scoped context from STEP 2
        tone : str
            Tone used for explanation
        has_rag : bool
            Whether RAG context was included
        system_prompt_length : int
            Character count of system prompt
        user_prompt_length : int
            Character count of user prompt
        
        Returns
        -------
        dict
            Prompt metadata
        """
        return {
            "scope": scoped_context.get("scope", "unknown"),
            "tone": tone,
            "has_rag": has_rag,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "prompt_lengths": {
                "system_prompt": system_prompt_length,
                "user_prompt": user_prompt_length,
                "total": system_prompt_length + user_prompt_length
            },
            "source_metadata": scoped_context.get("metadata", {}),
            "columns_analyzed": scoped_context.get("metadata", {}).get("columns_used", [])
        }


class ExplanationPromptBuilder:
    """
    Main orchestrator for building explanation prompts for LLM consumption.
    
    This class coordinates system prompt, user prompt, and metadata builders
    to produce safe, structured prompts that enforce the constraint that LLMs
    must only explain provided facts, never compute new ones.
    """
    
    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize the prompt builder with optional custom configuration.
        
        Parameters
        ----------
        config : PromptConfig, optional
            Custom configuration. If None, uses default configuration.
        """
        self.config = config or PromptConfig()
        
        # Initialize component builders
        self.system_prompt_builder = SystemPromptBuilder(self.config)
        self.user_prompt_builder = UserPromptBuilder(self.config)
        self.metadata_builder = PromptMetadataBuilder()
    
    def build(
        self,
        scoped_context: Dict[str, Any],
        rag_context: Optional[str] = None,
        tone: str = "clear"
    ) -> Dict[str, Any]:
        """
        Build a complete explanation prompt for LLM consumption.
        
        Parameters
        ----------
        scoped_context : dict
            Scoped context from STEP 2 (signal_selector.py).
            Must contain: scope, facts, metadata
        rag_context : str, optional
            Retrieved Augmented Generation context - background knowledge,
            best practices, or domain-specific explanations
        tone : str, default="clear"
            Explanation tone. Valid values:
            - "clear": Straightforward, accessible explanations
            - "concise": Brief, to-the-point insights
            - "technical": Precise statistical terminology
            - "beginner-friendly": Explained for newcomers
            - "detailed": Comprehensive, thorough analysis
        
        Returns
        -------
        dict
            Complete prompt package with structure:
            {
                "system_prompt": str,    # LLM role and constraints
                "user_prompt": str,      # Task and facts
                "metadata": dict         # Tracking and audit info
            }
        
        Notes
        -----
        This function does NOT call any LLM. It only prepares prompts.
        The prompts enforce that LLMs must:
        - NEVER compute or invent values
        - ONLY explain provided facts
        - Reference specific numbers explicitly
        - Focus on WHY patterns matter and WHAT to do
        
        Examples
        --------
        >>> from core.LLM.llm_explainability_engine import (
        ...     build_explainable_signals,
        ...     select_explainable_context,
        ...     build_explanation_prompt
        ... )
        >>> 
        >>> # Step 1: Extract signals
        >>> signals = build_explainable_signals(df)
        >>> 
        >>> # Step 2: Select scope
        >>> context = select_explainable_context(signals, "dataset_overview")
        >>> 
        >>> # Step 4: Build prompt (ready for LLM)
        >>> prompt = build_explanation_prompt(context, tone="clear")
        >>> print(prompt["system_prompt"][:100])
        'You are an expert data science assistant...'
        >>> 
        >>> # With RAG context
        >>> rag = "Best practice: Always check for outliers before modeling."
        >>> prompt = build_explanation_prompt(context, rag_context=rag, tone="technical")
        """
        # Validate inputs
        if not scoped_context:
            return self._error_response("Empty scoped_context provided")
        
        if "scope" not in scoped_context or "facts" not in scoped_context:
            return self._error_response(
                "Invalid scoped_context: missing 'scope' or 'facts' keys"
            )
        
        # Validate tone
        if tone not in self.config.valid_tones:
            tone = self.config.default_tone
        
        scope = scoped_context["scope"]
        
        # Build system prompt
        system_prompt = self.system_prompt_builder.build(tone, scope)
        
        # Build user prompt
        user_prompt = self.user_prompt_builder.build(scoped_context, rag_context)
        
        # Build metadata
        metadata = self.metadata_builder.build(
            scoped_context=scoped_context,
            tone=tone,
            has_rag=rag_context is not None,
            system_prompt_length=len(system_prompt),
            user_prompt_length=len(user_prompt)
        )
        
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "metadata": metadata
        }
    
    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            "system_prompt": "",
            "user_prompt": "",
            "metadata": {
                "error": True,
                "message": message,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }
        }


def build_explanation_prompt(
    scoped_context: Dict[str, Any],
    rag_context: Optional[str] = None,
    tone: str = "clear",
    config: Optional[PromptConfig] = None
) -> Dict[str, Any]:
    """
    Build a safe, structured prompt for LLM-based data explanations.
    
    This is the main convenience function (STEP 4 of the LLM pipeline) that
    prepares prompts for sending to an LLM. It acts as a SAFETY GATE, ensuring
    that the LLM will only explain provided facts and never invent statistics.
    
    Pipeline Context:
    - STEP 1: Extract signals from data (explainable_signals.py)
    - STEP 2: Select and scope signals (signal_selector.py)
    - STEP 3: (Optional) Retrieve RAG context
    - STEP 4: Build prompt (THIS MODULE) ← YOU ARE HERE
    - STEP 5: Call LLM with prompt (not implemented in this module)
    
    Parameters
    ----------
    scoped_context : dict
        Output from select_explainable_context() in STEP 2.
        Must contain:
        - scope: Analysis scope (e.g., "dataset_overview")
        - facts: Filtered statistical facts
        - metadata: Tracking information
    rag_context : str, optional
        Retrieved background knowledge, best practices, or domain expertise.
        Will be clearly separated in the prompt. If None, omitted.
    tone : str, default="clear"
        Explanation style. Options:
        - "clear": Accessible, straightforward (default)
        - "concise": Brief, focused insights
        - "technical": Precise statistical language
        - "beginner-friendly": Explained for non-experts
        - "detailed": Comprehensive, thorough
    config : PromptConfig, optional
        Custom configuration for prompt building
    
    Returns
    -------
    dict
        Complete prompt package:
        {
            "system_prompt": str,
                # Defines LLM role, constraints, tone, and scope guidance
            "user_prompt": str,
                # Contains task, formatted facts, optional RAG, and output format
            "metadata": dict
                # Tracking info: scope, tone, lengths, timestamps, etc.
        }
    
    Notes
    -----
    CRITICAL SAFETY CONSTRAINTS:
    - The generated prompts explicitly forbid the LLM from computing values
    - The LLM must only explain the specific facts provided
    - The LLM must reference actual numbers from the facts
    - This makes the explanations auditable and trustworthy
    
    This function does NOT:
    - Call any LLM (that's STEP 5)
    - Compute any statistics (done in STEP 1)
    - Filter or select data (done in STEP 2)
    - Print or display anything
    
    The output is designed to be:
    - Deterministic (same inputs = same prompts)
    - Auditable (prompts can be logged and reviewed)
    - Versionable (prompt templates can evolve)
    - Testable (easy to validate prompt structure)
    
    Examples
    --------
    >>> # Complete pipeline example
    >>> from core.LLM.llm_explainability_engine import (
    ...     build_explainable_signals,
    ...     select_explainable_context,
    ...     build_explanation_prompt
    ... )
    >>> 
    >>> # Step 1: Extract all signals from dataset
    >>> signals = build_explainable_signals(df)
    >>> 
    >>> # Step 2: Select dataset overview context
    >>> context = select_explainable_context(
    ...     signals,
    ...     scope="dataset_overview",
    ...     max_items=5
    ... )
    >>> 
    >>> # Step 4: Build prompt for LLM
    >>> prompt = build_explanation_prompt(context, tone="clear")
    >>> 
    >>> # Inspect the prompt
    >>> print("System Prompt Length:", len(prompt["system_prompt"]))
    >>> print("User Prompt Length:", len(prompt["user_prompt"]))
    >>> print("Scope:", prompt["metadata"]["scope"])
    >>> 
    >>> # Example with RAG context
    >>> best_practices = '''
    ... When analyzing outliers:
    ... 1. Check if they are data entry errors
    ... 2. Consider domain knowledge
    ... 3. Use robust statistical methods
    ... '''
    >>> 
    >>> prompt_with_rag = build_explanation_prompt(
    ...     context,
    ...     rag_context=best_practices,
    ...     tone="technical"
    ... )
    >>> 
    >>> # The prompt is now ready to send to LLM (STEP 5)
    >>> # response = llm.generate(
    >>> #     system=prompt["system_prompt"],
    >>> #     user=prompt["user_prompt"]
    >>> # )
    """
    builder = ExplanationPromptBuilder(config)
    return builder.build(scoped_context, rag_context, tone)


# Example usage and testing (for development only)
if __name__ == "__main__":
    # Sample scoped context from STEP 2
    sample_context = {
        "scope": "dataset_overview",
        "facts": {
            "basic_info": {
                "num_rows": 1000,
                "num_columns": 10,
                "memory_usage_mb": 1.2
            },
            "data_quality": {
                "total_missing_values": 50,
                "overall_missing_pct": 0.5
            },
            "column_types": {
                "num_numeric": 5,
                "num_categorical": 4,
                "num_datetime": 1
            }
        },
        "metadata": {
            "columns_used": [],
            "generated_at": "2026-02-03T10:00:00Z",
            "max_items": 5
        }
    }
    
    # Test basic prompt building
    print("=" * 60)
    print("Testing: Dataset Overview (Clear Tone)")
    print("=" * 60)
    prompt = build_explanation_prompt(sample_context, tone="clear")
    print("\nSystem Prompt:")
    print(prompt["system_prompt"][:300] + "...")
    print("\nUser Prompt:")
    print(prompt["user_prompt"][:300] + "...")
    print("\nMetadata:")
    print(json.dumps(prompt["metadata"], indent=2))
    
    # Test with RAG context
    print("\n" + "=" * 60)
    print("Testing: With RAG Context (Technical Tone)")
    print("=" * 60)
    rag = "Best practice: Always validate data quality before modeling."
    prompt_with_rag = build_explanation_prompt(
        sample_context,
        rag_context=rag,
        tone="technical"
    )
    print("\nPrompt lengths:")
    print(f"System: {prompt_with_rag['metadata']['prompt_lengths']['system_prompt']} chars")
    print(f"User: {prompt_with_rag['metadata']['prompt_lengths']['user_prompt']} chars")
    print(f"Has RAG: {prompt_with_rag['metadata']['has_rag']}")
    
    # Test error handling
    print("\n" + "=" * 60)
    print("Testing: Error Handling")
    print("=" * 60)
    error_prompt = build_explanation_prompt({})
    print("Error:", error_prompt["metadata"].get("message"))
