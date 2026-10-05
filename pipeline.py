import os
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from kilo_gateway import chat


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = (
    BASE_DIR
    / "phase2b_embeddings"
    / "mechanism_residual.faiss"
)

METADATA_PATH = (
    BASE_DIR
    / "phase2b_embeddings"
    / "metadata.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "end_to_end_solution.json"
)


# ============================================================
# EMBEDDING CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

MAX_LENGTH = 2048
EMBED_BATCH_SIZE = 16

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RETRIEVAL CONFIG
# ============================================================

RETRIEVAL_K = 20

# Number actually passed into Call 2.
# Retrieval still obtains RETRIEVAL_K results.
SYNTHESIS_K = 10


# ============================================================
# OPENROUTER CONFIG
# ============================================================

CALL1_MODEL = os.getenv(
    "OPENROUTER_MODEL_CALL1",
    os.getenv("OPENROUTER_MODEL", ""),
)

CALL1_REASONING = (
    os.getenv(
        "OPENROUTER_REASONING_CALL1",
        "false",
    ).lower()
    == "true"
)

CALL1_REASONING_EFFORT = os.getenv(
    "OPENROUTER_REASONING_EFFORT_CALL1",
    "low",
)


CALL2_MODEL = os.getenv(
    "OPENROUTER_MODEL_CALL2",
    os.getenv("OPENROUTER_MODEL", ""),
)

CALL2_REASONING = (
    os.getenv(
        "OPENROUTER_REASONING_CALL2",
        "true",
    ).lower()
    == "true"
)

CALL2_REASONING_EFFORT = os.getenv(
    "OPENROUTER_REASONING_EFFORT_CALL2",
    "medium",
)


# ============================================================
# PROMPT — CALL 1
# ============================================================

PASS1_SYSTEM = r"""
You are a mechanism abstraction engine for cross-domain innovation research.

Your task is to transform an external real-world problem into:

1. A concise problem abstraction.
2. The implementation-independent functional mechanism required to address it.
3. Domain-specific residual terms.

IMPORTANT:

Do NOT propose a solution architecture.

Do NOT recommend:
- technologies
- algorithms
- sensors
- models
- architectures
- materials
- products
- implementation details

Do NOT introduce mechanisms that are not implied by the original problem.

Describe WHAT must be accomplished and the causal difficulty,
not HOW a particular engineering system should implement it.

Preserve important causal relationships such as:

- temporal relationships
- spatial relationships
- conditional behavior
- feedback
- comparison
- accumulation
- thresholds
- selection
- switching
- repetition
- persistence
- adaptation
- failure and recovery

The abstraction should be useful for retrieving mechanisms from
completely different engineering domains.

The mechanism should normally be 1–3 sentences.

Residual terms are domain-specific context that helps preserve
the original problem context during retrieval.

Residual terms are NOT proposed solutions.

Return ONLY valid JSON with exactly this structure:

{
  "problem": "...",
  "constraints": "...",
  "required_capabilities": [
    "...",
    "..."
  ],
  "mechanism": "...",
  "residual_terms": [
    "...",
    "..."
  ]
}
"""


# ============================================================
# PROMPT — CALL 2
# ============================================================

PASS2_SYSTEM = r"""
You are a cross-domain engineering innovation synthesizer.

You are given:

- an external problem
- a problem abstraction
- a functional mechanism
- mechanisms retrieved from unrelated domains

Your task is to produce ONE concrete solution to the external problem.

The retrieved mechanisms are sources of causal inspiration.

A retrieved mechanism does NOT need to solve the entire target problem.
It may contribute only one useful causal building block.

IMPORTANT:

Do NOT force retrieval to matter.

Use a retrieved mechanism only if it provides a causal principle
that materially improves or changes the proposed solution.

If none of the retrieved mechanisms provide useful causal inspiration,
say so explicitly and solve the problem independently.

Avoid superficial transfer based only on shared nouns.

For every retrieved mechanism that is actually used, explain:

1. source rank
2. source id
3. original principle
4. transferred principle
5. adaptation to the target domain
6. why the transfer is causally useful

Clearly distinguish:

- retrieved inspiration
- domain adaptation
- independent engineering additions

Do not merely combine mechanisms because they appear related.

The final solution should be one coherent system rather than
a list of unrelated ideas.

Return ONLY valid JSON with this structure:

{
  "solution": "...",
  "retrieval_used": true,
  "retrieval_influence": [
    {
      "source_rank": 1,
      "source_id": "...",
      "original_principle": "...",
      "transferred_principle": "...",
      "domain_adaptation": "...",
      "causal_reason": "..."
    }
  ],
  "design_decisions_caused_by_retrieval": [
    "..."
  ],
  "independent_engineering_additions": [
    "..."
  ]
}

If retrieval is not useful:

{
  "solution": "...",
  "retrieval_used": false,
  "retrieval_influence": [],
  "design_decisions_caused_by_retrieval": [],
  "independent_engineering_additions": [
    "..."
  ]
}
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):
    if not text:
        raise ValueError(
            "LLM returned empty response."
        )

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    # Try complete response first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON object.
    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in LLM response:\n"
            + text[:2000]
        )

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True

        elif char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                candidate = text[start:i + 1]

                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        "Found JSON-like object but it could not "
                        "be parsed.\n\n"
                        + text[:3000]
                    ) from e

    raise ValueError(
        "Incomplete JSON object in LLM response:\n"
        + text[:3000]
    )


# ============================================================
# LLM HELPER
# ============================================================

def ask_json(
    user_prompt,
    system_prompt,
    max_tokens,
    model,
    reasoning,
    reasoning_effort,
):
    raw = chat(
        user_prompt,
        system=system_prompt,
        model=model,
        reasoning=reasoning,
        reasoning_effort=reasoning_effort,
        temperature=0.2,
        max_tokens=max_tokens,
    )

    return extract_json(raw)


# ============================================================
# EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    print("=" * 80)
    print("Loading embedding model")
    print("=" * 80)

    print(f"Model:  {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = model.to(DEVICE)
    model.eval()

    print("Embedding model loaded.")

    return tokenizer, model


# ============================================================
# QWEN LAST-TOKEN POOLING
# ============================================================

def last_token_pool(
    last_hidden_state,
    attention_mask,
):
    left_padding = (
        attention_mask[:, -1].sum()
        == attention_mask.shape[0]
    )

    if left_padding:
        return last_hidden_state[:, -1]

    sequence_lengths = (
        attention_mask.sum(dim=1) - 1
    )

    batch_indices = torch.arange(
        last_hidden_state.shape[0],
        device=DEVICE,
    )

    return last_hidden_state[
        batch_indices,
        sequence_lengths,
    ]


# ============================================================
# EMBEDDING
# ============================================================

@torch.no_grad()
def embed_texts(
    texts,
    tokenizer,
    model,
):
    embeddings = []

    for start in range(
        0,
        len(texts),
        EMBED_BATCH_SIZE,
    ):
        batch = texts[
            start:start + EMBED_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)

        pooled = last_token_pool(
            outputs.last_hidden_state,
            encoded["attention_mask"],
        )

        pooled = torch.nn.functional.normalize(
            pooled,
            p=2,
            dim=1,
        )

        embeddings.append(
            pooled.float().cpu().numpy()        )

    return np.vstack(embeddings).astype(
        "float32"
    )


# ============================================================
# METADATA
# ============================================================

def load_metadata():
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# EXACT PHASE 2B REPRESENTATION
# ============================================================

def make_mechanism_residual_representation(
    mechanism,
    residual_terms,
):
    """
    This MUST remain identical to the representation used
    when mechanism_residual.faiss was created.
    """

    return (
        f"Functional mechanism: {mechanism}\n"
        f"Domain-specific terms: {str(residual_terms)}"
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query_embedding,
    index,
    metadata,
    k,
):
    scores, indices = index.search(
        query_embedding.astype("float32"),
        k,
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        item = metadata[idx]

        # Metadata generated by the embedding pipeline
        # should contain the mechanism information.
        result = {
            "rank": rank,
            "score": float(score),

            "id": item.get(
                "patent_id",
                item.get("id", idx),
            ),

            "domain": item.get(
                "domain",
                item.get("field", ""),
            ),

            "title": item.get(
                "title",
                "",
            ),

            "problem": item.get(
                "problem",
                "",
            ),

            "constraint": item.get(
                "constraint",
                item.get("constraints", ""),
            ),

            "mechanism": item.get(
                "mechanism",
                "",
            ),

            "residual_terms": item.get(
                "residual_terms",
                [],
            ),
        }

        results.append(result)

    return results


# ============================================================
# CALL 1 — PROBLEM ABSTRACTION
# ============================================================

def call1_problem_abstraction(
    external_problem,
):
    prompt = f"""
External problem:

{external_problem}

Extract:

1. the problem abstraction
2. the constraints
3. the required capabilities
4. the implementation-independent functional mechanism
5. domain-specific residual terms

Remember:

- Do NOT propose a solution.
- Do NOT recommend technologies.
- Do NOT recommend algorithms.
- Do NOT recommend sensors.
- Do NOT recommend architectures.
- Do NOT introduce mechanisms not implied by the problem.
- Preserve causal relationships.
- Preserve temporal and conditional behavior.
- Keep the mechanism compact.
"""

    result = ask_json(
        prompt,
        PASS1_SYSTEM,
        max_tokens=1200,
        model=CALL1_MODEL,
        reasoning=CALL1_REASONING,
        reasoning_effort=CALL1_REASONING_EFFORT,
    )

    return result


# ============================================================
# CALL 2 — SOLUTION SYNTHESIS
# ============================================================

def call2_synthesize(
    external_problem,
    abstraction,
    retrieved,
):
    candidates = retrieved[
        :SYNTHESIS_K
    ]

    retrieval_blocks = []

    for item in candidates:
        retrieval_blocks.append(
            f"""
--- Retrieved mechanism #{item["rank"]} ---

Source ID:
{item["id"]}

Domain:
{item["domain"]}

Title:
{item["title"]}

Problem:
{item["problem"]}

Constraint:
{item["constraint"]}

Mechanism:
{item["mechanism"]}

Residual terms:
{item["residual_terms"]}

Similarity:
{item["score"]:.5f}
"""
        )

    retrieval_text = "\n".join(
        retrieval_blocks
    )

    prompt = f"""
EXTERNAL PROBLEM
================

{external_problem}


PROBLEM ABSTRACTION
===================

{json.dumps(
    abstraction,
    indent=2,
    ensure_ascii=False,
)}


RETRIEVED CROSS-DOMAIN MECHANISMS
=================================

{retrieval_text}


TASK
====

Design ONE concrete solution to the external problem.

The retrieved mechanisms are only sources of possible causal
inspiration.

A retrieved mechanism does not need to solve the whole problem.
It can provide one useful causal building block.

Do NOT force retrieval to matter.

Use a retrieved mechanism only when it provides a causal principle
that materially improves or changes the solution.

If a retrieved mechanism is not useful, ignore it.

If none of the retrieved mechanisms are useful, solve the problem
independently and explicitly state that retrieval was not used.

Avoid superficial transfer based only on shared nouns.

For every retrieved mechanism actually used, explain:

- original principle
- transferred principle
- domain adaptation
- causal reason for transfer

Clearly distinguish:

- retrieved inspiration
- domain adaptation
- independent engineering additions

Do not simply combine mechanisms because they are similar.

Produce one coherent solution.
"""

    result = ask_json(
        prompt,
        PASS2_SYSTEM,
        max_tokens=10000,
        model=CALL2_MODEL,
        reasoning=CALL2_REASONING,
        reasoning_effort=CALL2_REASONING_EFFORT,
    )

    return result


# ============================================================
# OPTIONAL STANDALONE TEST
# ============================================================

def main():
    """
    Allows pipeline.py to still be tested directly:

        python pipeline.py

    The Flask server does NOT use this.
    """

    import faiss

    problem = input(
        "Enter external problem:\n> "
    ).strip()

    if not problem:
        raise ValueError(
            "Problem cannot be empty."
        )

    print("\n[1/4] Loading embedding model...")
    tokenizer, model = load_embedding_model()

    print("\n[2/4] Call 1...")
    abstraction = call1_problem_abstraction(
        problem
    )

    print(
        json.dumps(
            abstraction,
            indent=2,
            ensure_ascii=False,
        )
    )

    rep = make_mechanism_residual_representation(
        abstraction["mechanism"],
        abstraction["residual_terms"],
    )

    print("\nEmbedding representation:")
    print(rep)

    print("\n[3/4] Embedding + retrieval...")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    metadata = load_metadata()

    q = embed_texts(
        [rep],
        tokenizer,
        model,
    )

    if q.shape[1] != index.d:
        raise ValueError(
            f"Dimension mismatch: "
            f"query={q.shape[1]}, "
            f"index={index.d}"
        )

    retrieved = retrieve(
        q,
        index,
        metadata,
        RETRIEVAL_K,
    )

    for item in retrieved:
        print(
            f'{item["rank"]:2d}. '
            f'{item["score"]:.5f} '
            f'{item["id"]} '
            f'{item["title"]}'
        )

    print("\n[4/4] Call 2...")
    solution = call2_synthesize(
        problem,
        abstraction,
        retrieved,
    )

    print(
        json.dumps(
            solution,
            indent=2,
            ensure_ascii=False,
        )
    )

    output = {
        "external_problem": problem,
        "problem_abstraction": abstraction,
        "embedding_representation": rep,
        "retrieval": retrieved[:SYNTHESIS_K],
        "solution": solution,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
