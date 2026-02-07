"""
RAG Retriever - Simple Rule-Based Knowledge Retrieval.

This module provides deterministic retrieval of rules from the knowledge base.
It uses simple filtering by issue_type and category - NO embeddings, NO vector DB, NO LLM.

STRICT RESPONSIBILITIES:
- Retrieve rules by issue_type
- Filter rules by category
- Match rules to diagnostics
- NO ranking, NO scoring, NO learning

Author: DataMimicAI Team
Date: February 2026
"""

import logging
from typing import Dict, List, Any, Optional
from .knowledge_loader import KnowledgeLoader, get_default_loader

logger = logging.getLogger(__name__)


# ============================================================================
# SIMPLE RETRIEVER
# ============================================================================

class SimpleRetriever:
    """
    Deterministic rule retriever with simple exact-match filtering.
    
    This retriever uses only exact string matching on issue_type and category.
    It is completely deterministic and requires no external services.
    
    Characteristics:
    - Deterministic: Same input always returns same rules
    - Local-only: No API calls or external services
    - Simple: Exact string matching only
    - Fast: In-memory dictionary lookups
    """
    
    def __init__(self, loader: Optional[KnowledgeLoader] = None):
        """
        Initialize the retriever.
        
        Parameters
        ----------
        loader : KnowledgeLoader, optional
            Knowledge loader instance. If None, uses default loader.
        """
        self.loader = loader if loader is not None else get_default_loader()
        self._index_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._indexed = False
        
        logger.debug("SimpleRetriever initialized")
    
    def _build_index(self):
        """
        Build an in-memory index mapping issue_type to rules.
        
        This creates a simple dictionary for O(1) lookups by issue_type.
        """
        if self._indexed:
            return
        
        logger.debug("Building issue_type index")
        
        # Get all rules
        all_rules = self.loader.get_all_rules()
        
        # Group by issue_type
        for rule in all_rules:
            issue_type = rule.get("issue_type", "unknown")
            if issue_type not in self._index_cache:
                self._index_cache[issue_type] = []
            self._index_cache[issue_type].append(rule)
        
        self._indexed = True
        logger.debug(f"Index built: {len(self._index_cache)} unique issue types")
    
    def retrieve_by_issue_type(self, issue_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve all rules matching a specific issue type.
        
        Parameters
        ----------
        issue_type : str
            Issue type to retrieve (e.g., 'high_skew', 'missing_values')
        
        Returns
        -------
        list
            List of matching rules (empty if no matches)
        """
        if not self._indexed:
            self._build_index()
        
        rules = self._index_cache.get(issue_type, [])
        logger.debug(f"Retrieved {len(rules)} rules for issue_type='{issue_type}'")
        return rules
    
    def retrieve_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Retrieve all rules from a specific category.
        
        Parameters
        ----------
        category : str
            Category name: 'eda', 'transformation', 'generator', or 'privacy'
        
        Returns
        -------
        list
            List of rules in the category
        """
        rules = self.loader.get_rules_by_category(category)
        logger.debug(f"Retrieved {len(rules)} rules for category='{category}'")
        return rules
    
    def retrieve_for_diagnostic(self, diagnostic: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant rules for a single diagnostic entry.
        
        This matches the diagnostic's issue_type with rule issue_types.
        
        Parameters
        ----------
        diagnostic : dict
            Diagnostic entry with 'issue_type' field
        
        Returns
        -------
        list
            List of matching rules
        """
        issue_type = diagnostic.get("issue_type")
        if not issue_type:
            logger.warning("Diagnostic missing 'issue_type' field")
            return []
        
        return self.retrieve_by_issue_type(issue_type)
    
    def retrieve_for_diagnostics(
        self,
        diagnostics: List[Dict[str, Any]],
        category_filter: Optional[str] = None
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Retrieve relevant rules for multiple diagnostics.
        
        This creates a mapping from diagnostic index to list of matching rules.
        
        Parameters
        ----------
        diagnostics : list
            List of diagnostic entries
        category_filter : str, optional
            If provided, only return rules from this category
        
        Returns
        -------
        dict
            Mapping from diagnostic index (int) to list of rules:
            {
                0: [rule1, rule2, ...],
                1: [rule3, ...],
                ...
            }
        """
        results = {}
        
        for idx, diagnostic in enumerate(diagnostics):
            rules = self.retrieve_for_diagnostic(diagnostic)
            
            # Apply category filter if specified
            if category_filter:
                # Get all rules from category
                category_rules = self.loader.get_rules_by_category(category_filter)
                category_rule_ids = {r["rule_id"] for r in category_rules}
                
                # Filter to only rules in the category
                rules = [r for r in rules if r["rule_id"] in category_rule_ids]
            
            if rules:  # Only include diagnostics that have matching rules
                results[idx] = rules
        
        logger.debug(f"Retrieved rules for {len(results)} out of {len(diagnostics)} diagnostics")
        return results
    
    def retrieve_for_diagnostics_dict(
        self,
        diagnostics_dict: Dict[str, Any],
        category_filter: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve rules for a diagnostics dictionary (from build_diagnostics).
        
        This is a convenience method that works with the standard diagnostics
        output format from diagnostics_builder.py.
        
        Parameters
        ----------
        diagnostics_dict : dict
            Output from build_diagnostics() with 'diagnostics' key
        category_filter : str, optional
            If provided, only return rules from this category
        
        Returns
        -------
        dict
            Mapping from diagnostic index to list of rules
        """
        diagnostics_list = diagnostics_dict.get("diagnostics", [])
        return self.retrieve_for_diagnostics(diagnostics_list, category_filter)
    
    def get_unique_issue_types(self) -> List[str]:
        """
        Get list of all unique issue types in the knowledge base.
        
        Returns
        -------
        list
            Sorted list of issue type strings
        """
        if not self._indexed:
            self._build_index()
        
        return sorted(self._index_cache.keys())
    
    def get_rule_count_by_issue_type(self) -> Dict[str, int]:
        """
        Get count of rules for each issue type.
        
        Returns
        -------
        dict
            Mapping from issue_type to count
        """
        if not self._indexed:
            self._build_index()
        
        return {
            issue_type: len(rules)
            for issue_type, rules in self._index_cache.items()
        }
    
    def reload_index(self):
        """
        Force reload of the index from knowledge loader.
        
        Call this if knowledge files are updated during runtime.
        """
        logger.info("Reloading retriever index")
        self._indexed = False
        self._index_cache.clear()
        self.loader.reload()
        self._build_index()


# ============================================================================
# AUGMENTATION HELPER
# ============================================================================

def augment_diagnostics_with_rules(
    diagnostics_dict: Dict[str, Any],
    retriever: Optional[SimpleRetriever] = None,
    category_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Augment diagnostics dictionary with retrieved RAG rules.
    
    This is a convenience function that adds a 'rag_context' field to
    the diagnostics dictionary containing relevant rules for each diagnostic.
    
    The original diagnostics structure is NOT modified - rules are added
    as supplementary context only.
    
    Parameters
    ----------
    diagnostics_dict : dict
        Output from build_diagnostics()
    retriever : SimpleRetriever, optional
        Retriever instance. If None, creates default retriever.
    category_filter : str, optional
        If provided, only include rules from this category
    
    Returns
    -------
    dict
        Augmented diagnostics with 'rag_context' field:
        {
            "diagnostics": [...],  # Original diagnostics
            "summary": {...},      # Original summary
            "metadata": {...},     # Original metadata
            "rag_context": {       # NEW: Retrieved rules
                "rules_by_diagnostic": {
                    0: [rule1, rule2, ...],
                    1: [rule3, ...],
                    ...
                },
                "retrieval_metadata": {
                    "total_rules_retrieved": int,
                    "category_filter": str or None,
                    "diagnostics_with_rules": int
                }
            }
        }
    """
    if retriever is None:
        retriever = SimpleRetriever()
    
    # Retrieve rules for all diagnostics
    rules_by_diagnostic = retriever.retrieve_for_diagnostics_dict(
        diagnostics_dict,
        category_filter=category_filter
    )
    
    # Count total rules
    total_rules = sum(len(rules) for rules in rules_by_diagnostic.values())
    
    # Create augmented result (shallow copy to avoid modifying original)
    augmented = dict(diagnostics_dict)
    augmented["rag_context"] = {
        "rules_by_diagnostic": rules_by_diagnostic,
        "retrieval_metadata": {
            "total_rules_retrieved": total_rules,
            "category_filter": category_filter,
            "diagnostics_with_rules": len(rules_by_diagnostic)
        }
    }
    
    logger.info(
        f"Augmented {len(diagnostics_dict.get('diagnostics', []))} diagnostics "
        f"with {total_rules} rules ({len(rules_by_diagnostic)} diagnostics have rules)"
    )
    
    return augmented


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create retriever
    retriever = SimpleRetriever()
    
    # Test retrieval by issue type
    print("\n=== Retrieve by Issue Type: 'high_skew' ===")
    skew_rules = retriever.retrieve_by_issue_type("high_skew")
    print(f"Found {len(skew_rules)} rules for high_skew")
    if skew_rules:
        print(f"Sample: {skew_rules[0]['rule_id']} - {skew_rules[0]['condition']}")
    
    # Test retrieval by category
    print("\n=== Retrieve by Category: 'privacy' ===")
    privacy_rules = retriever.retrieve_by_category("privacy")
    print(f"Found {len(privacy_rules)} privacy rules")
    
    # Show unique issue types
    print("\n=== Unique Issue Types ===")
    issue_types = retriever.get_unique_issue_types()
    print(f"Total unique issue types: {len(issue_types)}")
    print(f"Issue types: {', '.join(issue_types[:5])}...")
    
    # Show rule counts
    print("\n=== Rule Counts by Issue Type ===")
    counts = retriever.get_rule_count_by_issue_type()
    for issue_type, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {issue_type}: {count} rules")
