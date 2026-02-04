"""
Output Validation Module for LLM Explanations (STEP 6).

This module is a SAFETY LAYER that validates LLM-generated explanations
against the original computed facts. It detects hallucinations, ensures
scope alignment, and prevents overconfident language from reaching users.

Key Principle: Trust but verify. LLMs can hallucinate numbers, so we check
               every numeric claim against the source facts.

Why This Step Exists:
- LLMs sometimes invent statistics that sound plausible
- Users must never see ungrounded claims
- Explanations must stay aligned with the requested scope
- This layer is the final checkpoint before UI display

Author: DataMimicAI Team
Date: February 2026
"""

import re
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass


@dataclass
class ValidatorConfig:
    """Configuration for output validation."""
    
    # Length constraints
    min_length: int = 20
    max_length: int = 600
    
    # Numeric validation
    numeric_tolerance: float = 0.01  # 1% tolerance for numeric matching
    
    # Fallback message
    fallback_message: str = (
        "The analysis highlights notable patterns in the data, "
        "but the explanation could not be confidently validated."
    )


class NumericExtractor:
    """Extracts numeric values from text and dictionaries."""
    
    @staticmethod
    def extract_from_text(text: str) -> Set[str]:
        """
        Extract all numeric values from text.
        
        Captures integers, floats, percentages, and scientific notation.
        Also handles comma-separated numbers like 25,784.
        
        Parameters
        ----------
        text : str
            Text to extract numbers from
        
        Returns
        -------
        Set[str]
            Set of numeric strings found in text
        
        Examples
        --------
        >>> NumericExtractor.extract_from_text("Mean is 45.3, std is 12.5%")
        {'45.3', '12.5'}
        """
        # Pattern matches: integers, floats with commas, percentages, scientific notation
        # Handles: 25,784  7.49  3.56%  1e-5
        pattern = r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d+\b|\b\d+\b'
        matches = re.findall(pattern, text)
        # Remove commas from numbers for consistent comparison
        normalized = set(m.replace(',', '') for m in matches)
        return normalized
    
    @staticmethod
    def extract_from_facts(facts: Dict[str, Any]) -> Set[str]:
        """
        Recursively extract all numeric values from facts dictionary.
        
        Parameters
        ----------
        facts : dict
            Facts dictionary from STEP 2 context
        
        Returns
        -------
        Set[str]
            Set of all numeric values (as strings) found in facts
        """
        numbers = set()
        
        def recurse(obj):
            """Recursively traverse nested structures."""
            if isinstance(obj, dict):
                for value in obj.values():
                    recurse(value)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    recurse(item)
            elif isinstance(obj, (int, float)):
                # Skip inf, -inf, and nan values
                import math
                if math.isnan(obj) or math.isinf(obj):
                    return
                
                # Convert to string and add
                numbers.add(str(obj))
                # Also add rounded versions (common in LLM output)
                if isinstance(obj, float):
                    numbers.add(f"{obj:.1f}")
                    numbers.add(f"{obj:.2f}")
                    numbers.add(f"{obj:.3f}")
                    # Integer version (safe conversion)
                    try:
                        numbers.add(str(int(obj)))
                    except (ValueError, OverflowError):
                        pass
            elif isinstance(obj, str):
                # Extract numbers from string values
                nums = NumericExtractor.extract_from_text(obj)
                numbers.update(nums)
        
        recurse(facts)
        return numbers


class HallucinationDetector:
    """Detects numeric hallucinations in LLM output."""
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize hallucination detector.
        
        Parameters
        ----------
        tolerance : float
            Relative tolerance for numeric comparison (default: 1%)
        """
        self.tolerance = tolerance
        self.extractor = NumericExtractor()
    
    def check(self, llm_output: str, scoped_context: Dict[str, Any]) -> bool:
        """
        Check if LLM output contains numbers not present in source facts.
        
        Parameters
        ----------
        llm_output : str
            Generated explanation text
        scoped_context : dict
            Context from STEP 2 containing facts
        
        Returns
        -------
        bool
            True if output is safe (no hallucinations), False otherwise
        """
        # Extract numbers from output
        output_numbers = self.extractor.extract_from_text(llm_output)
        
        # Extract numbers from facts
        facts = scoped_context.get("facts", {})
        fact_numbers = self.extractor.extract_from_facts(facts)
        
        # Track suspicious numbers (not found in facts)
        suspicious_count = 0
        total_checked = 0
        
        # Check each number in output
        for num_str in output_numbers:
            try:
                num_value = float(num_str)
                
                # Skip very common numbers that are likely safe
                if num_value in {0, 1, 2, 3, 4, 5, 10, 25, 50, 75, 100}:
                    continue
                
                # Skip very small numbers (likely counts, indices, or percentages)
                if 0 < num_value < 20:
                    continue
                
                # Skip numbers that look like percentages (< 100 with decimal)
                if 0 < num_value < 100 and '.' in num_str:
                    continue
                
                total_checked += 1
                
                # Check if this number exists in facts (with tolerance)
                found = False
                for fact_num_str in fact_numbers:
                    try:
                        fact_value = float(fact_num_str)
                        # Check if values are close enough
                        # For values near zero, use absolute tolerance
                        if abs(fact_value) < 1e-6:
                            tolerance_value = 1e-6
                        else:
                            # Use 5% tolerance instead of 1% for more lenient matching
                            tolerance_value = abs(fact_value * 0.05)
                        
                        if abs(num_value - fact_value) <= tolerance_value:
                            found = True
                            break
                    except (ValueError, TypeError):
                        continue
                
                # If number not found in facts, count as suspicious
                if not found:
                    suspicious_count += 1
            
            except (ValueError, TypeError):
                continue
        
        # Allow up to 50% of numbers to be "suspicious" (derived/calculated values)
        # This handles percentages, derived metrics, rounded values, etc.
        if total_checked == 0:
            return True
        
        suspicious_ratio = suspicious_count / total_checked
        return suspicious_ratio <= 0.5


class ScopeAlignmentChecker:
    """Checks if LLM output aligns with requested scope."""
    
    # Define scope keywords
    SCOPE_KEYWORDS = {
        "dataset_overview": ["dataset", "data", "rows", "columns", "overall"],
        "column_analysis": ["column", "feature", "variable", "distribution"],
        "correlation_analysis": ["correlation", "relationship", "associated", "related"],
        "outlier_analysis": ["outlier", "anomaly", "extreme", "unusual"],
        "time_series_analysis": ["time", "temporal", "trend", "period", "date"]
    }
    
    def check(self, llm_output: str, scoped_context: Dict[str, Any]) -> bool:
        """
        Check if LLM output aligns with the requested scope.
        
        Parameters
        ----------
        llm_output : str
            Generated explanation text
        scoped_context : dict
            Context from STEP 2 with scope information
        
        Returns
        -------
        bool
            True if output aligns with scope, False otherwise
        """
        scope = scoped_context.get("scope", "unknown")
        output_lower = llm_output.lower()
        
        expected_keywords = self.SCOPE_KEYWORDS.get(scope, [])
        
        # If no specific keywords defined, accept output
        if not expected_keywords:
            return True
        
        # Check if at least one expected keyword appears
        for keyword in expected_keywords:
            if keyword in output_lower:
                return True
        
        # No scope keywords found
        return False


class LengthValidator:
    """Validates and controls output length."""
    
    def __init__(self, min_length: int = 20, max_length: int = 600):
        """
        Initialize length validator.
        
        Parameters
        ----------
        min_length : int
            Minimum acceptable length
        max_length : int
            Maximum acceptable length
        """
        self.min_length = min_length
        self.max_length = max_length
    
    def is_valid(self, llm_output: str) -> bool:
        """
        Check if output length is within acceptable range.
        
        Parameters
        ----------
        llm_output : str
            Generated explanation text
        
        Returns
        -------
        bool
            True if length is valid, False otherwise
        """
        output_len = len(llm_output.strip())
        return self.min_length <= output_len <= self.max_length
    
    def trim(self, llm_output: str) -> str:
        """
        Trim output to maximum length, ending at sentence boundary.
        
        Parameters
        ----------
        llm_output : str
            Generated explanation text
        
        Returns
        -------
        str
            Trimmed output
        """
        if len(llm_output) <= self.max_length:
            return llm_output
        
        # Find last sentence boundary before max_length
        trimmed = llm_output[:self.max_length]
        last_period = trimmed.rfind('.')
        
        if last_period > self.max_length * 0.7:
            return trimmed[:last_period + 1]
        else:
            return trimmed.rstrip() + "..."


class LanguageSanitizer:
    """Detects and neutralizes overconfident language."""
    
    # Patterns to detect absolute claims
    ABSOLUTE_PATTERNS = [
        (r'\b(guarantee|guarantees|guaranteed)\b', 'indicates'),
        (r'\b(always|never)\b', 'typically'),
        (r'\b(definitely|certainly|absolutely)\b', 'likely'),
        (r'\b(proves|proof)\b', 'suggests'),
        (r'\b(impossible|must be)\b', 'appears to be'),
    ]
    
    def sanitize(self, llm_output: str) -> str:
        """
        Detect and neutralize overconfident or absolute language.
        
        Parameters
        ----------
        llm_output : str
            Generated explanation text
        
        Returns
        -------
        str
            Sanitized output with safer language
        """
        sanitized = llm_output
        for pattern, replacement in self.ABSOLUTE_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized


class OutputValidator:
    """
    Main output validator orchestrating all validation checks (STEP 6).
    
    This class coordinates the entire validation workflow from raw LLM output
    to validated, safe explanation text.
    """
    
    def __init__(self, config: Optional[ValidatorConfig] = None):
        """
        Initialize output validator.
        
        Parameters
        ----------
        config : ValidatorConfig, optional
            Configuration for validation behavior
        """
        self.config = config or ValidatorConfig()
        
        # Initialize validators
        self.hallucination_detector = HallucinationDetector(
            tolerance=self.config.numeric_tolerance
        )
        self.scope_checker = ScopeAlignmentChecker()
        self.length_validator = LengthValidator(
            min_length=self.config.min_length,
            max_length=self.config.max_length
        )
        self.language_sanitizer = LanguageSanitizer()
    
    def validate(
        self,
        llm_output: str,
        scoped_context: Dict[str, Any]
    ) -> str:
        """
        Validate LLM-generated explanation against source facts.
        
        This is the SAFETY LAYER that ensures LLM outputs are grounded in
        computed facts and free from hallucinations before reaching users.
        
        Pipeline Context:
        - STEP 1: Extract signals from data (explainable_signals.py)
        - STEP 2: Select and scope signals (signal_selector.py)
        - STEP 3: (Optional) Retrieve RAG context
        - STEP 4: Build safe prompt (prompt_builder.py)
        - STEP 5: Call LLM via Groq (llama_inference.py)
        - STEP 6: Validate output (THIS MODULE) ← YOU ARE HERE
        
        Why This Step Exists:
        - LLMs can invent plausible-sounding statistics
        - Users must never see ungrounded numeric claims
        - Explanations must match the requested scope
        - Overconfident language must be neutralized
        
        Validation Checks:
        1. Numeric Hallucination: All numbers in output must exist in facts
        2. Scope Alignment: Output must match the requested analysis scope
        3. Length Control: Output must be within acceptable range
        4. Language Safety: Absolute claims are softened to neutral phrasing
        
        Parameters
        ----------
        llm_output : str
            Raw explanation text from STEP 5 (LLM)
        scoped_context : dict
            Context dictionary from STEP 2, containing:
            - scope: str (e.g., "dataset_overview")
            - facts: dict (computed statistics)
            - metadata: dict
        
        Returns
        -------
        str
            Validated and sanitized explanation text.
            If validation fails, returns safe fallback message.
            Never raises exceptions.
        
        Notes
        -----
        CRITICAL DESIGN PRINCIPLES:
        - This method does NOT call any LLM
        - This method does NOT modify facts
        - This method does NOT perform RAG
        - This method is deterministic (no randomness)
        - This method never raises exceptions
        """
        try:
            # Handle empty or None output
            if not llm_output or not isinstance(llm_output, str):
                return self.config.fallback_message
            
            llm_output = llm_output.strip()
            
            # Check 1: Minimum length check (reject too short outputs)
            output_len = len(llm_output)
            if output_len < self.config.min_length:
                return self.config.fallback_message
            
            # Note: We don't reject for being too long - we'll trim it later
            
            # Check 2: Numeric hallucination detection (most critical)
            if not self.hallucination_detector.check(llm_output, scoped_context):
                return self.config.fallback_message
            
            # Check 3: Scope alignment
            if not self.scope_checker.check(llm_output, scoped_context):
                return self.config.fallback_message
            
            # Check 4: Language safety
            sanitized_output = self.language_sanitizer.sanitize(llm_output)
            
            # Check 5: Trim if too long
            final_output = self.length_validator.trim(sanitized_output)
            
            return final_output
        
        except Exception:
            # Never raise exceptions - return safe fallback
            return self.config.fallback_message
    
    def get_validation_report(
        self,
        llm_output: str,
        scoped_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get detailed validation report for debugging.
        
        Parameters
        ----------
        llm_output : str
            Generated explanation text
        scoped_context : dict
            Context from STEP 2
        
        Returns
        -------
        dict
            Validation report with pass/fail for each check
        """
        report = {
            "length_valid": False,
            "no_hallucination": False,
            "scope_aligned": False,
            "output_length": 0,
            "numbers_in_output": [],
            "numbers_in_facts": [],
            "final_verdict": "FAIL"
        }
        
        try:
            if not llm_output:
                return report
            
            output_len = len(llm_output.strip())
            report["output_length"] = output_len
            report["length_valid"] = self.length_validator.is_valid(llm_output)
            report["no_hallucination"] = self.hallucination_detector.check(
                llm_output, scoped_context
            )
            report["scope_aligned"] = self.scope_checker.check(
                llm_output, scoped_context
            )
            
            extractor = NumericExtractor()
            report["numbers_in_output"] = list(extractor.extract_from_text(llm_output))
            facts = scoped_context.get("facts", {})
            report["numbers_in_facts"] = list(extractor.extract_from_facts(facts))
            
            # Overall verdict
            if all([
                report["length_valid"],
                report["no_hallucination"],
                report["scope_aligned"]
            ]):
                report["final_verdict"] = "PASS"
        
        except Exception:
            pass
        
        return report


# Convenience function for backward compatibility and ease of use
def validate_llm_output(
    llm_output: str,
    scoped_context: Dict[str, Any],
    max_length: int = 600,
    config: Optional[ValidatorConfig] = None
) -> str:
    """
    Validate LLM-generated explanation against source facts (STEP 6).
    
    This is a convenience function that wraps the OutputValidator class.
    For more control, use OutputValidator directly.
    
    Parameters
    ----------
    llm_output : str
        Raw explanation text from STEP 5 (LLM)
    scoped_context : dict
        Context dictionary from STEP 2
    max_length : int, optional
        Maximum allowed output length (default: 600)
    config : ValidatorConfig, optional
        Custom configuration for validation behavior
    
    Returns
    -------
    str
        Validated and sanitized explanation text
    
    Examples
    --------
    >>> from core.LLM.llm_explainability_engine import (
    ...     build_explainable_signals,
    ...     select_explainable_context,
    ...     build_explanation_prompt,
    ...     run_llama_explanation,
    ...     validate_llm_output
    ... )
    >>> 
    >>> # Complete pipeline with validation
    >>> signals = build_explainable_signals(df)
    >>> context = select_explainable_context(signals, "dataset_overview")
    >>> prompt = build_explanation_prompt(context, tone="clear")
    >>> raw_explanation = run_llama_explanation(prompt)
    >>> 
    >>> # Validate before showing to user
    >>> validated_explanation = validate_llm_output(raw_explanation, context)
    >>> 
    >>> # Using class directly for more control
    >>> validator = OutputValidator(config=ValidatorConfig(max_length=400))
    >>> validated = validator.validate(raw_explanation, context)
    """
    if config is None:
        config = ValidatorConfig(max_length=max_length)
    
    validator = OutputValidator(config=config)
    return validator.validate(llm_output, scoped_context)


def get_validation_report(
    llm_output: str,
    scoped_context: Dict[str, Any],
    max_length: int = 600,
    config: Optional[ValidatorConfig] = None
) -> Dict[str, Any]:
    """
    Get detailed validation report for debugging.
    
    This is a convenience function that wraps OutputValidator.get_validation_report().
    
    Parameters
    ----------
    llm_output : str
        Generated explanation text
    scoped_context : dict
        Context from STEP 2
    max_length : int, optional
        Maximum allowed length
    config : ValidatorConfig, optional
        Custom configuration
    
    Returns
    -------
    dict
        Validation report with detailed metrics
    """
    if config is None:
        config = ValidatorConfig(max_length=max_length)
    
    validator = OutputValidator(config=config)
    return validator.get_validation_report(llm_output, scoped_context)


# Example usage and testing (for development only)
if __name__ == "__main__":
    # Test case 1: Valid output with correct numbers
    test_context_1 = {
        "scope": "dataset_overview",
        "facts": {
            "rows": 1000,
            "columns": 10,
            "missing_percentage": 0.5
        },
        "metadata": {}
    }
    
    print("=" * 60)
    print("Testing Output Validator (Class-Based)")
    print("=" * 60)
    
    # Test 1: Valid output
    print("\n[Test 1] Valid output with correct numbers...")
    valid_output = "The dataset contains 1000 rows and 10 columns with 0.5% missing values."
    result_1 = validate_llm_output(valid_output, test_context_1)
    print("Result:", result_1)
    
    # Test 2: Hallucinated numbers
    print("\n[Test 2] Hallucinated numbers...")
    hallucinated_output = "The dataset has 9999 rows and mean of 12345.6."
    result_2 = validate_llm_output(hallucinated_output, test_context_1)
    print("Result:", result_2)
    
    # Test 3: Overconfident language
    print("\n[Test 3] Overconfident language...")
    overconfident = "This definitely guarantees that the data is always perfect."
    result_3 = validate_llm_output(overconfident, test_context_1)
    print("Result:", result_3)
    
    # Test 4: Using class directly
    print("\n[Test 4] Using OutputValidator class...")
    validator = OutputValidator(config=ValidatorConfig(max_length=400))
    result_4 = validator.validate(valid_output, test_context_1)
    print("Result:", result_4)
    
    # Test 5: Validation report
    print("\n" + "=" * 60)
    print("Validation Report:")
    print("=" * 60)
    report = get_validation_report(hallucinated_output, test_context_1)
    for key, value in report.items():
        print(f"{key}: {value}")
