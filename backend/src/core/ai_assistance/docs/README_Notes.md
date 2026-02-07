# 🚀 DataMimicAI

**DataMimicAI** is an end-to-end **agentic, human-in-the-loop synthetic data generation and analysis platform**. It combines deterministic data diagnostics, controlled AI agents, human approvals, deterministic execution, and objective validation into a single auditable workflow.

The system is designed to be:

* 🔒 **Safe-by-design** (no uncontrolled AI execution)
* 🧠 **Agentic but governed** (AI proposes, humans decide)
* 📊 **Evidence-driven** (before/after validation, no opinions)
* 🏗️ **Production-grade architecture** (clear separation of concerns)

---

## 🎯 Core Philosophy

> **AI should assist decisions, not silently make them.**

DataMimicAI follows a strict contract:

```
DIAGNOSE → INTERPRET → PLAN → APPROVE → EXECUTE → VALIDATE → REPORT
```

* AI **never executes code directly**
* Humans **explicitly approve** all transformations
* Execution is **deterministic and auditable**
* Results are **measured, not judged**

---

## 🧭 High-Level Architecture

### End-to-End Flow

```
┌────────────┐
│ Diagnostics│  (Deterministic)
└─────┬──────┘
      ↓
┌────────────┐
│ Interpreter│  (Agent – Read-only)
└─────┬──────┘
      ↓
┌────────────┐
│ Planner    │  (Agent – Proposals only)
└─────┬──────┘
      ↓
┌────────────┐
│ Human Gate │  (Explicit approval)
└─────┬──────┘
      ↓
┌────────────┐
│ Execution  │  (Deterministic engine)
└─────┬──────┘
      ↓
┌────────────┐
│ Validation │  (Before/After metrics)
└─────┬──────┘
      ↓
┌────────────┐
│ Report UI  │  (Decision evidence)
└────────────┘
```

---

## 🧩 System Components

### 1️⃣ Diagnostics Builder (Deterministic)

**Purpose:** Convert raw EDA outputs into structured, machine-readable diagnostics.

**Key characteristics:**

* Rule-based
* Deterministic
* No AI / no LLM

**Output example:**

```json
{
  "issue_type": "high_skew",
  "affected_columns": ["Volume"],
  "metric": "skewness",
  "value": 6.9,
  "severity": "high"
}
```

---

### 2️⃣ Diagnostics Interpreter Agent (Read-only Agent)

**Purpose:** Interpret diagnostics to identify **cross-cutting issue patterns**.

**What it does:**

* Aggregates issues into dominant patterns (e.g. *Skew + Outliers*)
* Provides confidence level
* Cites supporting evidence

**What it does NOT do:**

* No recommendations
* No execution
* No parameter tuning

---

### 3️⃣ Transformation Planner Agent (Proposal-only Agent)

**Purpose:** Propose **conceptual transformation plans** based on interpretation.

**Strict constraints:**

* Proposal-only
* Uses fixed transformation vocabulary
* Deterministic output

**Allowed transformations:**

* `log_transform`
* `sqrt_transform`
* `winsorization`
* `scaling`
* `imputation`
* `encoding`
* `feature_deduplication`
* `dimensionality_reduction`

**Example plan:**

```json
{
  "plan_id": "TP-001",
  "proposed_transformations": [
    {
      "transformation": "log_transform",
      "target_columns": ["Volume"],
      "rationale": "Addresses heavy right-skew"
    }
  ]
}
```

---

### 4️⃣ Human-in-the-Loop Approval Gate

**Purpose:** Enforce explicit human governance.

**Capabilities:**

* Approve / reject plans
* Capture reviewer notes
* Prevent unapproved execution

**Key rule:**

> **No approved plan → no execution.**

---

### 5️⃣ Deterministic Execution Engine

**Purpose:** Execute **only approved plans**.

**Design principles:**

* No AI
* No reasoning
* Fixed transformation mapping
* Fail-fast on errors

**Execution output:**

```json
{
  "execution_status": "SUCCESS",
  "applied_transformations": ["log_transform", "winsorization"],
  "validation_available": true
}
```

---

### 6️⃣ Validation Feedback Loop

**Purpose:** Measure impact objectively.

**What it compares:**

* Skewness
* Missing values
* Outliers
* Correlations

**Output:** Before vs After metrics with deltas.

---

### 7️⃣ Decision Report UI

**Purpose:** Present factual evidence for decisions.

**Characteristics:**

* Read-only
* No recommendations
* No judgments

**Example:**

| Metric            | Before | After | Delta |
| ----------------- | ------ | ----- | ----- |
| Skewness (Volume) | 6.90   | 1.20  | -5.70 |

---

## 🖥️ User Interface Structure

### AI Assistance Tabs

* 🔍 **Diagnostics** – Read-only dataset health
* ⚙️ **Action Planner** – Interpret → Plan → Approve → Execute
* ⚠️ **Privacy & Risk Audit** – (v2 / optional)
* 📄 **Decision Report** – Validation results

---

## 🔐 Safety & Governance Guarantees

* ❌ No auto-execution
* ❌ No hidden AI actions
* ✅ Explicit approvals required
* ✅ Deterministic execution
* ✅ Full audit trail

---

## 🧪 Testing & Reliability

* Unit tests for diagnostics & validation
* Determinism checks
* Safe error handling
* Fail-fast execution model

---

## 🚦 Current Status

**Version:** v1.0

**State:**

* ✅ End-to-end functional
* ✅ Backend + Frontend fully wired
* ✅ Interview & demo ready

---

## 🛣️ Future Roadmap (Optional)

* Privacy Risk Auditor Agent
* Generator Strategy Agent
* Persistent storage (DB)
* Exportable audit reports

---

## 🏁 Final Note

DataMimicAI demonstrates how **agentic AI systems can be built responsibly**:

> AI reasons → Humans decide → Systems execute → Metrics validate.

This project intentionally prioritizes **safety, transparency, and correctness** over unchecked automation.

---

## Other

---

### UI

🔍 Diagnostics | ⚙️ Action Planner | ⚠️ Privacy & Risk Audit | 📄 Decision Report

--

### 🧭 Updated Roadmap (From Where You Are)

| Order       | Step                                  | Type          | Status   |
| ----------- | ------------------------------------- | ------------- | -------- |
| 1           | Diagnostics Builder                   | Deterministic | ✅ Done  |
| 2           | Explain Refactor                      | Presentation  | ✅ Done  |
| 3           | RAG Foundation                        | Knowledge     | ✅ Done  |
| 4           | Diagnostics Interpreter               | 🧠 Agent      | ✅ Done  |
| 5           | Transformation Planner                | 🧠 Agent      | ✅ Done  |
| **6** | **Plan Review & Approval Gate** | 🛡️ Control  | ✅ Done  |
| 7           | Generator Strategy Agent              | 🧠 Agent      | ⏳ Later |
| 8           | Privacy & Risk Auditor                | 🧠 Agent      | ⏳ Later |
| 9           | Execution Engine                      | ⚙️ Engine   | ✅ Done  |
| 10          | Validation Feedback Loop              | ⚙️ Engine   | ✅ Done  |
