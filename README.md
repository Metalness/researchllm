# Cross-Domain Mechanism Discovery

> **Finding engineering solutions by transferring mechanisms from unrelated domains.**

## TO RUN
pip install torch transformers faiss-cpu flask requests python-dotenv numpy
run python server.py


## Problem

LLMs are good at generating solutions, but they usually rely on patterns already associated with the problem's domain.

That makes them less likely to discover **non-obvious solutions inspired by completely different fields**.

This project explores whether retrieving problem-solving mechanisms from unrelated patents can expand the solutions an LLM produces.

---

## Solution

Instead of directly asking an LLM for a solution, we:

1. Extract the underlying mechanism required by the problem.
2. Search thousands of patent-derived mechanisms for structurally similar mechanisms.
3. Transfer useful mechanisms into the new domain.
4. Generate one concrete solution using those mechanisms.

The important distinction is:

> **We retrieve mechanisms, not answers.**

---

## Technology

* **LLM** — problem abstraction + solution synthesis
* **Qwen/Qwen3-Embedding-0.6B** — local embeddings
* **FAISS** — vector similarity search
* **Python** — pipeline
* **Flask + SSE** — web interface and live pipeline updates
* **Kilo / OpenRouter** — LLM API
* **~4,234 patent-derived mechanisms** — retrieval corpus

---

## How We Do It

### 1. Abstract

LLM Call #1 converts:

```text
Problem
→
Functional mechanism
+ constraints
+ domain-specific residual terms
```

It is instructed **not to design the solution**.

### 2. Retrieve

The mechanism is embedded locally and searched against the patent corpus using FAISS.

```text
4,234+ mechanisms
        ↓
semantic retrieval
        ↓
Top 20
        ↓
Top 10 → synthesis
```

### 3. Transfer

LLM Call #2 receives the original problem and retrieved mechanisms.

It determines:

* what mechanism is useful
* how it transfers to the new domain
* why the transfer is causal
* what additional engineering is required

Then it produces **one concrete solution**.

---

## Example

**Problem:** Detect early postoperative complications despite different patient baselines and noisy measurements.

**Retrieved mechanism:** An industrial system dynamically adjusts detection thresholds based on the frequency of threshold violations.

**Transferred idea:** Dynamically adjust patient-specific alert sensitivity based on repeated physiological deviations.

The source patent and hospital problem are unrelated, but the **underlying mechanism is transferable**.

---

## Research Question

> **Does cross-domain mechanism retrieval cause an LLM to produce solutions it would otherwise be unlikely to generate?**
