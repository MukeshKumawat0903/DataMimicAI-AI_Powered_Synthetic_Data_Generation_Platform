"""
LLaMA Inference Module for Data Explanations (STEP 5).

This module is a THIN execution layer that calls LLaMA via Groq to generate
explanations based on prompts prepared in STEP 4. It does NOT modify prompts,
compute statistics, or perform any intelligence. All logic happens before
(facts extraction, prompt building) and after (validation, UI presentation).

Key Principle: This module executes LLM inference. Nothing more, nothing less.

Why API Keys Are Used Here:
- STEP 5 is the ONLY step that requires external API access
- All previous steps (1-4) are deterministic and offline
- API keys are loaded from .env and never hardcoded

Author: DataMimicAI Team
Date: February 2026
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import warnings

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


@dataclass
class InferenceConfig:
    """Configuration for LLM inference."""
    
    # Model parameters
    default_model_name: str = "llama-3.3-70b-versatile"  # Groq LLaMA model
    default_temperature: float = 0.2  # Low for deterministic outputs
    default_max_tokens: int = 3000  # Increased for multi-section explanations
    
    # Timeouts
    request_timeout: int = 60  # seconds


class GroqAPIHandler:
    """Handles Groq API key retrieval and validation."""
    
    @staticmethod
    def get_api_key() -> str:
        """
        Safely retrieve GROQ_API_KEY from environment.
        
        Returns
        -------
        str
            The Groq API key
        
        Raises
        ------
        RuntimeError
            If GROQ_API_KEY is not found in environment
        """
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not found in environment variables.\n"
                "Please add it to your .env file:\n"
                "  GROQ_API_KEY=your_api_key_here\n"
                "Get your API key from: https://console.groq.com/keys"
            )
        
        return api_key


class GroqInferenceEngine:
    """
    Groq inference engine for LLaMA models.
    
    This class handles all interactions with the Groq API using ChatGroq.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize Groq inference engine.
        
        Parameters
        ----------
        config : InferenceConfig, optional
            Configuration for inference behavior
        """
        self.config = config or InferenceConfig()
        self.api_handler = GroqAPIHandler()
    
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate explanation using Groq's ChatGroq.
        
        Parameters
        ----------
        system_prompt : str
            System prompt defining role and constraints
        user_prompt : str
            User prompt with task and facts
        model_name : str, optional
            Groq model name (e.g., "llama-3.3-70b-versatile")
        temperature : float, optional
            Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens : int, optional
            Maximum tokens to generate
        
        Returns
        -------
        str
            Generated explanation text or error message
        """
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage
        except ImportError:
            return (
                "Error: Required packages not installed.\n"
                "Please install: pip install langchain-groq langchain-core"
            )
        
        try:
            # Get API key
            api_key = self.api_handler.get_api_key()
            
            # Use provided values or fall back to config defaults
            model_name = model_name or self.config.default_model_name
            temperature = temperature if temperature is not None else self.config.default_temperature
            max_tokens = max_tokens or self.config.default_max_tokens
            
            # Create ChatGroq client with specific parameters
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.config.request_timeout
            )
            
            # Build messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Generate response
            response = llm.invoke(messages)
            
            # Extract content
            generated_text = response.content
            
            if not generated_text:
                return "Error: Model returned empty response"
            
            return generated_text.strip()
        
        except RuntimeError as e:
            # API key or installation error - already has clear message
            return f"Error: {str(e)}"
        
        except Exception as e:
            # Catch all other errors (network, rate limit, etc.)
            return f"Error during Groq inference: {str(e)}"


class LLaMAInferenceEngine:
    """
    Main LLaMA inference engine orchestrating the inference process (STEP 5).
    
    This class coordinates the entire inference workflow from prompt to output.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize LLaMA inference engine.
        
        Parameters
        ----------
        config : InferenceConfig, optional
            Configuration for inference behavior
        """
        self.config = config or InferenceConfig()
        self.groq_engine = GroqInferenceEngine(self.config)
    
    def run_inference(
        self,
        prompt: Dict[str, Any],
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Run LLaMA inference via Groq to generate data explanations.
        
        This is the main method for LLM inference in the pipeline. It accepts
        a prepared prompt from STEP 4 and returns the raw LLM output using Groq's
        ChatGroq API.
        
        Pipeline Context:
        - STEP 1: Extract signals from data (explainable_signals.py)
        - STEP 2: Select and scope signals (signal_selector.py)
        - STEP 3: (Optional) Retrieve RAG context
        - STEP 4: Build safe prompt (prompt_builder.py)
        - STEP 5: Call LLM via Groq (THIS MODULE) ← YOU ARE HERE
        - STEP 6: (Future) Validate and post-process output
        
        Why API Keys Are Used Here:
        - STEP 5 is the ONLY step that requires external API access
        - All previous steps (1-4) are deterministic and offline
        - GROQ_API_KEY is loaded from .env file, never hardcoded
        - This keeps sensitive credentials isolated to inference only
        
        Parameters
        ----------
        prompt : dict
            Prompt dictionary from build_explanation_prompt() in STEP 4.
            Must contain: system_prompt, user_prompt, metadata
        model_name : str, optional
            Name of the Groq-hosted LLaMA model to use.
            If None, uses default: "llama-3.3-70b-versatile"
            Available models:
            - "llama-3.3-70b-versatile" (most capable)
            - "llama-3.2-11b-text-preview" (faster)
            - "llama-3.2-3b-preview" (smallest)
        temperature : float, optional
            Sampling temperature. Lower = more deterministic.
            Range: 0.0 (fully deterministic) to 1.0 (most creative)
            If None, uses default: 0.2 (good for factual explanations)
        max_tokens : int, optional
            Maximum number of tokens to generate.
            If None, uses default: 512 tokens
            Typical explanations need 256-1024 tokens.
        
        Returns
        -------
        str
            Raw explanation text generated by the LLM via Groq.
            If an error occurs, returns a controlled error message string.
            Never raises exceptions.
        
        Notes
        -----
        CRITICAL DESIGN PRINCIPLES:
        - This method does NOT modify prompts
        - This method does NOT compute statistics
        - This method does NOT add retries or fallbacks
        - This method does NOT validate outputs
        - This method is BORING by design
        
        All intelligence happens before (facts + prompt building) and after
        (validation + UI presentation). This is purely an execution layer.
        
        Requirements:
        - GROQ_API_KEY must be in .env file
        - langchain-groq must be installed: pip install langchain-groq
        - langchain-core must be installed: pip install langchain-core
        
        Error Handling:
        - Never raises exceptions (returns error strings instead)
        - API failures are caught and returned as messages
        - Missing API key returns clear instructions
        - Missing dependencies are detected and reported
        """
        # Validate prompt structure
        if not prompt:
            return "Error: Empty prompt provided"
        
        system_prompt = prompt.get("system_prompt", "")
        user_prompt = prompt.get("user_prompt", "")
        
        if not system_prompt or not user_prompt:
            return "Error: Invalid prompt structure. Missing system_prompt or user_prompt."
        
        # Suppress warnings during inference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            explanation = self.groq_engine.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        return explanation


# Convenience function for backward compatibility and ease of use
def run_llama_explanation(
    prompt: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    config: Optional[InferenceConfig] = None
) -> str:
    """
    Run LLaMA inference via Groq to generate data explanations (STEP 5).
    
    This is a convenience function that wraps the LLaMAInferenceEngine class.
    For more control, use LLaMAInferenceEngine directly.
    
    Parameters
    ----------
    prompt : dict
        Prompt dictionary from build_explanation_prompt() in STEP 4.
        Must contain: system_prompt, user_prompt, metadata
    model_name : str, optional
        Name of the Groq-hosted LLaMA model to use.
    temperature : float, optional
        Sampling temperature (0.0 = deterministic, 1.0 = creative)
    max_tokens : int, optional
        Maximum number of tokens to generate
    config : InferenceConfig, optional
        Custom configuration for inference behavior
    
    Returns
    -------
    str
        Raw explanation text generated by the LLM via Groq
    
    Examples
    --------
    >>> from core.LLM.llm_explainability_engine import (
    ...     build_explainable_signals,
    ...     select_explainable_context,
    ...     build_explanation_prompt,
    ...     run_llama_explanation
    ... )
    >>> 
    >>> # Complete pipeline
    >>> signals = build_explainable_signals(df)
    >>> context = select_explainable_context(signals, "dataset_overview")
    >>> prompt = build_explanation_prompt(context, tone="clear")
    >>> 
    >>> # Run inference via Groq (uses defaults)
    >>> explanation = run_llama_explanation(prompt)
    >>> print(explanation)
    'Based on the provided facts, the dataset contains...'
    >>> 
    >>> # Use specific model and parameters
    >>> explanation = run_llama_explanation(
    ...     prompt,
    ...     model_name="llama-3.2-11b-text-preview",
    ...     temperature=0.1,
    ...     max_tokens=256
    ... )
    >>> 
    >>> # Using class directly for more control
    >>> engine = LLaMAInferenceEngine(config=InferenceConfig())
    >>> explanation = engine.run_inference(prompt, temperature=0.0)
    """
    engine = LLaMAInferenceEngine(config=config)
    return engine.run_inference(
        prompt=prompt,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )


# Example usage and testing (for development only)
if __name__ == "__main__":
    # Sample prompt from STEP 4
    sample_prompt = {
        "system_prompt": (
            "You are an expert data science assistant. "
            "You must NEVER compute or invent values. "
            "You must ONLY explain the facts provided."
        ),
        "user_prompt": (
            "Explain the following dataset characteristics:\n"
            "Rows: 1000\n"
            "Columns: 10\n"
            "Missing values: 50 (0.5%)\n"
            "Please provide a clear explanation."
        ),
        "metadata": {
            "scope": "dataset_overview",
            "tone": "clear"
        }
    }
    
    print("=" * 60)
    print("Testing LLaMA Inference via Groq (Class-Based)")
    print("=" * 60)
    
    # Test 1: Using convenience function
    print("\n[Test 1] Using convenience function...")
    print("Note: This requires GROQ_API_KEY in .env file")
    
    explanation = run_llama_explanation(
        prompt=sample_prompt,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=256
    )
    
    print("\nGenerated Explanation:")
    print("-" * 60)
    print(explanation)
    print("-" * 60)
    
    # Test 2: Using class directly
    print("\n[Test 2] Using LLaMAInferenceEngine class...")
    engine = LLaMAInferenceEngine(config=InferenceConfig())
    explanation2 = engine.run_inference(
        prompt=sample_prompt,
        temperature=0.0,  # Fully deterministic
        max_tokens=128
    )
    print(explanation2[:200] + "...")
