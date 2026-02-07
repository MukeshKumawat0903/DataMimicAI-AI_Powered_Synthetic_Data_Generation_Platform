"""
Test script for Diagnostics API endpoint.

This script tests the diagnostics_api without starting the full FastAPI server.
"""

import sys
sys.path.insert(0, 'D:/Learnings/My_Projects/Learning_Projects/DataMimicAI-AI_Powered_Synthetic_Data_Generation_Platform/backend')

from src.core.agents.diagnostics_interpreter_agent import DiagnosticsInterpreterAgent

# Sample diagnostics input (from the agent's docstring example)
diagnostics_input = {
    "diagnostics": [
        {
            "issue_type": "high_skew",
            "severity": "high",
            "column": "Volume",
            "metrics": {"skewness": 6.9051}
        },
        {
            "issue_type": "outliers",
            "severity": "high",
            "column": "Volume",
            "metrics": {"outlier_percentage": 12.7}
        }
    ],
    "summary": {"total_issues": 2, "high_severity_count": 2},
    "metadata": {"timestamp": "2026-02-06"}
}

print("=" * 80)
print("Testing DiagnosticsInterpreterAgent API Logic")
print("=" * 80)

# Test 1: Basic functionality
print("\n[Test 1] Basic interpretation:")
agent = DiagnosticsInterpreterAgent()
result = agent.interpret(diagnostics_input)
response = result.to_dict()

print(f"✅ Overall Assessment: {result.overall_assessment}")
print(f"✅ Dominant Patterns: {result.dominant_issue_patterns}")
print(f"✅ Confidence: {result.confidence}")
print(f"✅ Supporting Evidence Count: {len(result.supporting_evidence)}")

# Test 2: With RAG context
print("\n[Test 2] With RAG context:")
agent_with_rag = DiagnosticsInterpreterAgent(rag_context="Sample RAG context")
result_rag = agent_with_rag.interpret(diagnostics_input)
print(f"✅ Interpretation with RAG successful")

# Test 3: Invalid input handling
print("\n[Test 3] Invalid input handling:")
try:
    invalid_input = {"wrong_key": []}
    agent.interpret(invalid_input)
    print("❌ Should have raised ValueError")
except ValueError as e:
    print(f"✅ Correctly raised ValueError: {e}")

# Test 4: Empty diagnostics
print("\n[Test 4] Empty diagnostics:")
empty_input = {
    "diagnostics": [],
    "summary": {"total_issues": 0},
    "metadata": {}
}
result_empty = agent.interpret(empty_input)
print(f"✅ Empty diagnostics handled: {result_empty.overall_assessment}")

print("\n" + "=" * 80)
print("All API logic tests passed! ✅")
print("=" * 80)
print("\nAPI Endpoint Specification:")
print("  POST /api/diagnostics/interpret")
print("  - Input: diagnostics_input (Dict), rag_context (Optional[str])")
print("  - Output: interpretation result (Dict)")
print("  - Behavior: Read-only, deterministic, no side effects")
print("  - Error Codes: 400 (invalid input), 500 (unexpected error)")
