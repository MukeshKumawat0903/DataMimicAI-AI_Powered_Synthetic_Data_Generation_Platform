"""
RAG Knowledge Loader - Deterministic Rule Loading and Validation.

This module loads and validates YAML rule files from the rag_knowledge/ directory.
It provides a simple, deterministic, local-only knowledge loading mechanism.

STRICT RESPONSIBILITIES:
- Load YAML files from rag_knowledge/
- Validate rule schema
- Cache rules in memory
- No LLM calls, no embeddings, no external APIs

Author: DataMimicAI Team
Date: February 2026
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# SCHEMA DEFINITION
# ============================================================================

@dataclass
class RuleSchema:
    """
    Schema definition for a knowledge rule.
    
    All rules must conform to this structure:
    - rule_id: Unique identifier
    - issue_type: Type of issue this rule addresses
    - condition: Human-readable condition description
    - explanation: Neutral, factual explanation (NO recommendations)
    """
    rule_id: str
    issue_type: str
    condition: str
    explanation: str


# ============================================================================
# VALIDATION
# ============================================================================

def validate_rule(rule: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate a single rule against the schema.
    
    Parameters
    ----------
    rule : dict
        Rule dictionary to validate
    
    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
        - If valid: (True, None)
        - If invalid: (False, "error description")
    """
    required_fields = ["rule_id", "issue_type", "condition", "explanation"]
    
    # Check if rule is a dictionary
    if not isinstance(rule, dict):
        return False, f"Rule must be a dictionary, got {type(rule).__name__}"
    
    # Check required fields
    for field in required_fields:
        if field not in rule:
            return False, f"Missing required field: {field}"
        
        # Check field is non-empty string
        value = rule[field]
        if not isinstance(value, str):
            return False, f"Field '{field}' must be a string, got {type(value).__name__}"
        
        if not value.strip():
            return False, f"Field '{field}' cannot be empty"
    
    # Check for forbidden fields (recommendations, actions, etc.)
    forbidden_fields = ["recommendation", "action", "fix", "solution", "mitigation"]
    for field in forbidden_fields:
        if field in rule:
            logger.warning(f"Rule {rule.get('rule_id', 'unknown')} contains forbidden field: {field}")
            # Don't fail validation, just warn - allows future extensions
    
    return True, None


def validate_rule_file(rules: List[Dict[str, Any]], filename: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Validate all rules in a file.
    
    Parameters
    ----------
    rules : list
        List of rule dictionaries
    filename : str
        Name of the file being validated (for error messages)
    
    Returns
    -------
    Tuple[List[Dict[str, Any]], List[str]]
        (valid_rules, errors)
        - valid_rules: List of validated rules
        - errors: List of error messages for invalid rules
    """
    valid_rules = []
    errors = []
    
    if not isinstance(rules, list):
        errors.append(f"{filename}: 'rules' must be a list, got {type(rules).__name__}")
        return valid_rules, errors
    
    for idx, rule in enumerate(rules):
        is_valid, error_msg = validate_rule(rule)
        if is_valid:
            valid_rules.append(rule)
        else:
            errors.append(f"{filename}: Rule #{idx + 1}: {error_msg}")
    
    return valid_rules, errors


# ============================================================================
# KNOWLEDGE LOADER
# ============================================================================

class KnowledgeLoader:
    """
    Deterministic knowledge loader for RAG rules.
    
    This class loads YAML rule files from the rag_knowledge/ directory,
    validates them, and caches them in memory for fast retrieval.
    
    Characteristics:
    - Deterministic: Same files always produce same results
    - Local-only: No external API calls
    - Simple: No embeddings, no vector databases
    - Validated: All rules checked against schema
    """
    
    def __init__(self, knowledge_dir: Optional[str] = None):
        """
        Initialize the knowledge loader.
        
        Parameters
        ----------
        knowledge_dir : str, optional
            Path to the rag_knowledge/ directory.
            If None, uses default location relative to this file.
        """
        if knowledge_dir is None:
            # Default: rag_knowledge/ in the same parent directory as rag/
            current_dir = Path(__file__).parent
            knowledge_dir = current_dir.parent / "rag_knowledge"
        
        self.knowledge_dir = Path(knowledge_dir)
        self._rules_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._loaded = False
        self._load_errors: List[str] = []
        
        logger.debug(f"KnowledgeLoader initialized with directory: {self.knowledge_dir}")
    
    def load_all_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all YAML rule files from the knowledge directory.
        
        Returns
        -------
        dict
            Dictionary mapping category name to list of rules:
            {
                "eda": [...],
                "transformation": [...],
                "generator": [...],
                "privacy": [...]
            }
        
        Raises
        ------
        FileNotFoundError
            If knowledge directory doesn't exist
        ValueError
            If critical validation errors occur
        """
        if self._loaded:
            logger.debug("Returning cached rules")
            return self._rules_cache
        
        # Check directory exists
        if not self.knowledge_dir.exists():
            raise FileNotFoundError(f"Knowledge directory not found: {self.knowledge_dir}")
        
        if not self.knowledge_dir.is_dir():
            raise ValueError(f"Knowledge path is not a directory: {self.knowledge_dir}")
        
        logger.info(f"Loading knowledge rules from: {self.knowledge_dir}")
        
        # Expected YAML files
        expected_files = {
            "eda": "eda_rules.yaml",
            "transformation": "transformation_rules.yaml",
            "generator": "generator_rules.yaml",
            "privacy": "privacy_rules.yaml"
        }
        
        all_errors = []
        
        for category, filename in expected_files.items():
            file_path = self.knowledge_dir / filename
            
            if not file_path.exists():
                logger.warning(f"Rule file not found: {filename} (category: {category})")
                self._rules_cache[category] = []
                continue
            
            try:
                # Load YAML
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data is None:
                    logger.warning(f"Empty YAML file: {filename}")
                    self._rules_cache[category] = []
                    continue
                
                # Extract rules
                rules = data.get("rules", [])
                
                # Validate rules
                valid_rules, errors = validate_rule_file(rules, filename)
                
                if errors:
                    all_errors.extend(errors)
                
                self._rules_cache[category] = valid_rules
                logger.info(f"Loaded {len(valid_rules)} rules from {filename}")
                
            except yaml.YAMLError as e:
                error_msg = f"YAML parsing error in {filename}: {str(e)}"
                logger.error(error_msg)
                all_errors.append(error_msg)
                self._rules_cache[category] = []
            
            except Exception as e:
                error_msg = f"Unexpected error loading {filename}: {str(e)}"
                logger.error(error_msg)
                all_errors.append(error_msg)
                self._rules_cache[category] = []
        
        self._load_errors = all_errors
        self._loaded = True
        
        # Log summary
        total_rules = sum(len(rules) for rules in self._rules_cache.values())
        logger.info(f"Knowledge loading complete: {total_rules} total rules loaded")
        
        if all_errors:
            logger.warning(f"Encountered {len(all_errors)} errors during loading")
        
        return self._rules_cache
    
    def get_rules_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all rules for a specific category.
        
        Parameters
        ----------
        category : str
            Category name: 'eda', 'transformation', 'generator', or 'privacy'
        
        Returns
        -------
        list
            List of rules for the category (empty list if category not found)
        """
        if not self._loaded:
            self.load_all_rules()
        
        return self._rules_cache.get(category, [])
    
    def get_all_rules(self) -> List[Dict[str, Any]]:
        """
        Get all rules across all categories.
        
        Returns
        -------
        list
            Flat list of all rules
        """
        if not self._loaded:
            self.load_all_rules()
        
        all_rules = []
        for category_rules in self._rules_cache.values():
            all_rules.extend(category_rules)
        
        return all_rules
    
    def get_load_errors(self) -> List[str]:
        """
        Get list of errors encountered during loading.
        
        Returns
        -------
        list
            List of error messages (empty if no errors)
        """
        return self._load_errors
    
    def is_loaded(self) -> bool:
        """
        Check if rules have been loaded.
        
        Returns
        -------
        bool
            True if rules are loaded and cached
        """
        return self._loaded
    
    def reload(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Force reload of all rules from disk.
        
        Returns
        -------
        dict
            Reloaded rules by category
        """
        logger.info("Forcing reload of knowledge rules")
        self._loaded = False
        self._rules_cache.clear()
        self._load_errors.clear()
        return self.load_all_rules()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about loaded rules.
        
        Returns
        -------
        dict
            Summary with counts and status
        """
        if not self._loaded:
            self.load_all_rules()
        
        return {
            "loaded": self._loaded,
            "total_rules": sum(len(rules) for rules in self._rules_cache.values()),
            "rules_by_category": {
                category: len(rules)
                for category, rules in self._rules_cache.items()
            },
            "errors": len(self._load_errors),
            "knowledge_dir": str(self.knowledge_dir)
        }


# ============================================================================
# MODULE-LEVEL SINGLETON (Optional Convenience)
# ============================================================================

_default_loader: Optional[KnowledgeLoader] = None


def get_default_loader() -> KnowledgeLoader:
    """
    Get the default singleton knowledge loader instance.
    
    This is a convenience function for applications that want a single
    shared loader instance. For more control, create KnowledgeLoader directly.
    
    Returns
    -------
    KnowledgeLoader
        Default loader instance
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = KnowledgeLoader()
    return _default_loader


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create loader
    loader = KnowledgeLoader()
    
    # Load all rules
    rules_by_category = loader.load_all_rules()
    
    # Print summary
    summary = loader.get_summary()
    print("\n=== Knowledge Loader Summary ===")
    print(f"Total rules loaded: {summary['total_rules']}")
    print(f"Rules by category:")
    for category, count in summary['rules_by_category'].items():
        print(f"  {category}: {count}")
    
    if summary['errors'] > 0:
        print(f"\nErrors encountered: {summary['errors']}")
        for error in loader.get_load_errors():
            print(f"  - {error}")
    
    # Show sample EDA rule
    eda_rules = loader.get_rules_by_category("eda")
    if eda_rules:
        print(f"\n=== Sample EDA Rule ===")
        sample = eda_rules[0]
        print(f"Rule ID: {sample['rule_id']}")
        print(f"Issue Type: {sample['issue_type']}")
        print(f"Condition: {sample['condition']}")
        print(f"Explanation: {sample['explanation'][:100]}...")
