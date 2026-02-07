## 🤖 AI Assistance Module

### 1. Overview

The **AI Assistance Module** is DataMimicAI's read-only intelligence layer that provides diagnostic interpretation, transformation planning, and risk analysis capabilities. It operates as a **proposal-only system** that enhances human decision-making without executing actions or mutating data.

AI Assistance interprets complex diagnostic outputs, recognizes data quality patterns, and generates actionable recommendations. However, it **never executes transformations** or makes autonomous decisions. All proposed actions require explicit human approval before execution by the deterministic engine.

This architecture ensures safety, auditability, and governance while leveraging AI capabilities to accelerate data quality workflows and reduce cognitive load on data practitioners.

**Core Principle:** AI assists decisions; humans approve actions.

---

### 2. Responsibilities

The AI Assistance Module is responsible for:

- **Diagnostics Interpretation**: Translating technical diagnostic outputs into human-readable insights
- **Issue Pattern Recognition**: Identifying data quality patterns, anomalies, and structural issues across diagnostics
- **Transformation Proposal Generation**: Suggesting data transformations based on detected issues and best practices
- **Risk & Privacy Analysis**: Semantic analysis of data fields for privacy risks and compliance concerns (conceptual)
- **Explainability**: Providing transparent reasoning for all recommendations and interpretations
- **Context-Aware Suggestions**: Leveraging dataset metadata and domain knowledge to improve recommendation relevance

---

### 3. What AI Assistance DOES

The AI Assistance Module performs the following read-only operations:

✅ **Reads and Interprets Diagnostics**
   - Consumes structured diagnostic outputs from the Diagnostics Builder
   - Identifies severity levels, impact areas, and root causes
   - Generates natural language summaries of technical findings

✅ **Uses RAG-Based Knowledge**
   - Retrieves transformation rules from internal knowledge base
   - Applies domain-specific best practices and heuristics
   - References historical patterns and successful remediation strategies

✅ **Produces Structured Plans**
   - Generates transformation proposals with clear rationale
   - Provides step-by-step action plans with expected outcomes
   - Estimates impact, complexity, and risk for each proposed transformation

✅ **Explains Reasoning Transparently**
   - Documents why specific transformations are recommended
   - Shows which diagnostics triggered which recommendations
   - Provides confidence scores and alternative approaches when applicable

---

### 4. What AI Assistance DOES NOT DO

The AI Assistance Module has strict boundaries to ensure safety and control:

❌ **Execute Transformations**
   - AI never runs transformations against actual datasets
   - All execution is performed by the deterministic engine after human approval

❌ **Modify Datasets**
   - AI cannot write to databases, files, or data stores
   - All data mutations require explicit user action

❌ **Auto-Approve Plans**
   - No transformation can proceed without human confirmation
   - AI cannot bypass the approval gate under any circumstances

❌ **Tune Hyperparameters**
   - AI does not automatically adjust algorithm configurations
   - Parameter tuning remains a human-controlled activity

❌ **Make Irreversible Decisions**
   - AI cannot commit changes, delete data, or finalize outputs
   - All decisions remain reversible until execution phase

---

### 5. Subcomponents

| Component | Type | Purpose | Execution Authority |
|-----------|------|---------|---------------------|
| **Diagnostics Builder** | Computation | Runs all diagnostic checks on dataset; produces structured issue catalog | Read-only computation |
| **Diagnostics Interpreter Agent** | AI Agent | Reads diagnostics output; generates natural language insights and severity rankings | Interpretation only |
| **Transformation Planner Agent** | AI Agent | Analyzes interpreted diagnostics; proposes transformation sequences with reasoning | Proposal generation |
| **RAG Knowledge Base** | Data Store | Stores transformation rules, best practices, and domain knowledge for retrieval | Read-only retrieval |
| **Human Approval Gate** | Control Point | Enforces explicit user confirmation before any transformation execution | Approval enforcement |
| **Privacy & Risk Auditor** | AI Agent (Optional) | Semantic analysis of field names and values for PII, sensitive data, compliance risks | Analysis only |

---

### 6. Folder Structure

The AI Assistance Module is organized into functional subdirectories:

```
backend/src/core/ai_assistance/
├── explainability/
│   ├── diagnostics_interpreter.py      # Converts diagnostics to human-readable insights
│   ├── reasoning_engine.py             # Generates explanations for recommendations
│   └── confidence_scoring.py           # Assigns confidence levels to AI suggestions
├── agents/
│   ├── diagnostics_agent.py            # Primary agent for diagnostics interpretation
│   ├── planner_agent.py                # Transformation planning and sequencing
│   └── privacy_agent.py                # Privacy risk detection (optional)
├── rag/
│   ├── knowledge_base.py               # RAG-based rule retrieval system
│   ├── embeddings.py                   # Vector embeddings for semantic search
│   └── transformation_templates.json   # Predefined transformation patterns
└── approval/
    ├── approval_gate.py                # Human-in-the-loop enforcement
    ├── plan_serialization.py           # Converts AI proposals to executable plans
    └── audit_log.py                    # Logs all AI recommendations and approvals
```

---

### 7. Mapping to UI

The AI Assistance Module directly maps to three primary UI tabs:

| UI Tab | AI Assistance Role | User Interaction |
|--------|-------------------|------------------|
| **Diagnostics** | Displays interpreted diagnostics with natural language explanations; highlights critical issues | Read-only view; users review AI-generated insights |
| **Action Planner** | Presents transformation proposals with reasoning; allows users to approve, reject, or modify plans | Interactive approval; users gate execution |
| **Privacy & Risk Audit** | Performs semantic analysis on field names/values; flags potential PII, sensitive data, compliance risks | Risk review; users decide on data handling policies |

**Workflow:**
1. User uploads dataset → Diagnostics Builder runs checks
2. **Diagnostics Tab**: AI interprets results → User reviews insights
3. **Action Planner Tab**: AI proposes transformations → User approves selected actions
4. Execution Engine applies approved transformations (outside AI Assistance scope)
5. **Privacy & Risk Audit Tab**: AI flags sensitive fields → User configures privacy controls

---

### 8. Design Principles

The AI Assistance Module adheres to the following architectural principles:

**🛡️ Proposal-Only Agents**
- AI generates recommendations, not actions
- All outputs are JSON proposals, not executable code
- Clear separation between "suggest" and "execute" responsibilities

**🔒 Deterministic Outputs**
- Where possible, AI reasoning follows deterministic rules
- Non-deterministic LLM calls are used only for interpretation, not execution logic
- Results are reproducible and auditable

**👤 Human-in-the-Loop Governance**
- Every transformation requires explicit user approval
- No automated decision-making for data mutations
- Users can override, modify, or reject any AI suggestion

**📋 Auditable Reasoning**
- All AI recommendations are logged with justifications
- Users can trace why specific transformations were suggested
- Confidence scores and alternative approaches are documented

**🎯 Context-Aware Intelligence**
- Leverages dataset metadata, schema, and domain knowledge
- Adapts recommendations based on data characteristics
- Uses RAG to retrieve relevant transformation patterns

**⚡ Performance-Oriented**
- AI operations are asynchronous and non-blocking
- Diagnostics interpretation happens in the background
- Users can proceed with manual workflows if AI is slow or unavailable

---

### 9. Libraries & Tools

The AI Assistance Module leverages the following technologies:

#### Core AI & LLM Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| **OpenAI API** | Primary LLM provider for diagnostics interpretation and transformation planning | `openai>=1.0.0` |
| **LangChain** | Agent orchestration, prompt templating, and chain-of-thought reasoning | `langchain>=0.1.0` |
| **LangChain Community** | RAG integrations, vector store connectors, and retrieval chains | `langchain-community>=0.0.20` |
| **Pydantic** | Structured output validation and schema enforcement for AI responses | `pydantic>=2.0.0` |

#### RAG & Vector Search

| Library | Purpose | Version |
|---------|---------|---------|
| **ChromaDB** | Lightweight vector database for embedding storage and semantic search | `chromadb>=0.4.0` |
| **SentenceTransformers** | Generate embeddings for transformation rules and knowledge base entries | `sentence-transformers>=2.2.0` |
| **FAISS** (Optional) | High-performance similarity search for large-scale RAG deployments | `faiss-cpu>=1.7.0` |

#### Data Processing & Analysis

| Library | Purpose | Version |
|---------|---------|---------|
| **Pandas** | Diagnostic data manipulation and structured output processing | `pandas>=2.0.0` |
| **NumPy** | Numerical operations for confidence scoring and risk calculations | `numpy>=1.24.0` |
| **JSONSchema** | Validation of AI-generated JSON proposals against expected schemas | `jsonschema>=4.17.0` |

#### Logging & Auditing

| Library | Purpose | Version |
|---------|---------|---------|
| **Loguru** | Structured logging for AI decisions, reasoning traces, and approval events | `loguru>=0.7.0` |
| **Python JSON Logger** | JSON-formatted logs for integration with monitoring systems | `python-json-logger>=2.0.0` |

#### Async & Background Processing

| Library | Purpose | Version |
|---------|---------|---------|
| **asyncio** | Non-blocking AI operations and concurrent diagnostic processing | Built-in |
| **aiohttp** | Asynchronous HTTP calls to external LLM APIs | `aiohttp>=3.9.0` |
| **Celery** (Optional) | Distributed task queue for long-running AI analysis jobs | `celery>=5.3.0` |

#### Testing & Quality Assurance

| Library | Purpose | Version |
|---------|---------|---------|
| **pytest** | Unit testing for AI agents, RAG retrieval, and approval logic | `pytest>=7.4.0` |
| **pytest-asyncio** | Testing async AI operations | `pytest-asyncio>=0.21.0` |
| **Mock** | Mocking LLM responses for deterministic testing | `unittest.mock` (Built-in) |

#### Configuration Management

| Library | Purpose | Use Case |
|---------|---------|----------|
| **python-dotenv** | Environment variable management for API keys and model configurations | API credential security |
| **PyYAML** | YAML parsing for RAG knowledge base and transformation templates | Configuration files |

---

### Summary

The AI Assistance Module transforms DataMimicAI from a tool into an intelligent assistant. By keeping AI strictly in the **advisory role**, the platform maintains full control, safety, and transparency while dramatically improving user productivity and decision quality.

**Key Takeaway:** AI explains, recommends, and assists. Humans decide, approve, and execute.
