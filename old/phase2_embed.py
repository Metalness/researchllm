"""
Phase 2: Embed Phase 1 mechanisms with Qwen3-Embedding-0.6B.

Input:
    structured_mechanisms.json

Outputs:
    phase2_embeddings/
        mechanisms.json
        mechanism_embeddings.npy
        problem_mechanism_embeddings.npy
        problem_constraint_mechanism_embeddings.npy
        mechanism.faiss
        problem_mechanism.faiss
        problem_constraint_mechanism.faiss

Three representations are embedded for every patent:

1. mechanism
2. problem + mechanism
3. problem + constraint + mechanism

The embeddings are L2-normalized, so FAISS inner product
is equivalent to cosine similarity.

Designed for local CUDA inference.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import faiss

from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "structured_mechanisms.json"
OUTPUT_DIR = "phase2_embeddings"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Number of records processed by the model at once.
# RTX 5060 should comfortably handle small batches.
BATCH_SIZE = 16

# Maximum token length.
# Patent mechanisms should generally be much shorter than this.
MAX_LENGTH = 2048

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Only embed patents where Phase 1 found a mechanism.
ONLY_WITH_MECHANISM = True


# ============================================================
# FILE NAMES
# ============================================================

MECHANISMS_JSON = os.path.join(
    OUTPUT_DIR,
    "mechanisms.json"
)

MECHANISM_NPY = os.path.join(
    OUTPUT_DIR,
    "mechanism_embeddings.npy"
)

PROBLEM_MECHANISM_NPY = os.path.join(
    OUTPUT_DIR,
    "problem_mechanism_embeddings.npy"
)

PROBLEM_CONSTRAINT_MECHANISM_NPY = os.path.join(
    OUTPUT_DIR,
    "problem_constraint_mechanism_embeddings.npy"
)

MECHANISM_INDEX = os.path.join(
    OUTPUT_DIR,
    "mechanism.faiss"
)

PROBLEM_MECHANISM_INDEX = os.path.join(
    OUTPUT_DIR,
    "problem_mechanism.faiss"
)

PROBLEM_CONSTRAINT_MECHANISM_INDEX = os.path.join(
    OUTPUT_DIR,
    "problem_constraint_mechanism.faiss"
)


# ============================================================
# UTILITIES
# ============================================================

def clean_text(value):
    """
    Convert database values into clean strings.
    """
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(x) for x in value)

    return str(value).strip()


def has_mechanism(record):
    """
    Handle different possible representations of the Phase 1
    has_mechanism field.
    """

    value = record.get("has_mechanism")

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value == 1

    if isinstance(value, str):
        return value.lower().strip() in {
            "true",
            "1",
            "yes"
        }

    return False


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print(f"Loading {INPUT_FILE}...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "structured_mechanisms.json must contain a JSON list."
        )

    original_count = len(data)

    if ONLY_WITH_MECHANISM:

        data = [
            record
            for record in data
            if has_mechanism(record)
            and clean_text(record.get("mechanism"))
        ]

    print(
        f"Loaded {original_count} records."
    )

    if ONLY_WITH_MECHANISM:
        print(
            f"Using {len(data)} records with mechanisms."
        )

    if not data:
        raise ValueError(
            "No usable mechanism records found."
        )

    return data


# ============================================================
# BUILD REPRESENTATIONS
# ============================================================

def build_representations(records):

    mechanism_texts = []
    problem_mechanism_texts = []
    full_texts = []

    for record in records:

        problem = clean_text(
            record.get("problem")
        )

        constraint = clean_text(
            record.get("constraint")
        )

        mechanism = clean_text(
            record.get("mechanism")
        )

        # ----------------------------------------------------
        # Representation 1
        # ----------------------------------------------------

        representation_1 = mechanism

        # ----------------------------------------------------
        # Representation 2
        # ----------------------------------------------------

        representation_2 = (
            f"Problem: {problem}\n"
            f"Mechanism: {mechanism}"
        )

        # ----------------------------------------------------
        # Representation 3
        # ----------------------------------------------------

        representation_3 = (
            f"Problem: {problem}\n"
            f"Constraint: {constraint}\n"
            f"Mechanism: {mechanism}"
        )

        mechanism_texts.append(
            representation_1
        )

        problem_mechanism_texts.append(
            representation_2
        )

        full_texts.append(
            representation_3
        )

    return (
        mechanism_texts,
        problem_mechanism_texts,
        full_texts
    )


# ============================================================
# LOAD QWEN
# ============================================================

def load_model():

    print()
    print("=" * 60)
    print("Loading Qwen3-Embedding-0.6B")
    print("=" * 60)

    print(
        f"Model:  {MODEL_NAME}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU:    {torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"VRAM:   {total_memory:.2f} GB"
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

        print(
            "Embedding will run on CPU."
        )

    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.float16
            if DEVICE == "cuda"
            else torch.float32
        ),
        trust_remote_code=True
    )

    model.to(DEVICE)

    model.eval()

    print("Model loaded.")

    return tokenizer, model


# ============================================================
# EMBEDDING
# ============================================================

def last_token_pool(
    last_hidden_states,
    attention_mask
):
    """
    Qwen embedding models use last-token pooling.

    Because padding is on the left, the final token in
    each sequence is at position -1.
    """

    return last_hidden_states[:, -1]


@torch.inference_mode()
def embed_texts(
    texts,
    tokenizer,
    model,
    description
):

    all_embeddings = []

    total = len(texts)

    print()
    print(
        f"Embedding: {description}"
    )

    print(
        f"Documents: {total}"
    )

    start_time = time.time()

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = texts[start:end]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(
            **encoded
        )

        embeddings = last_token_pool(
            outputs.last_hidden_state,
            encoded["attention_mask"]
        )

        # Normalize here so cosine similarity can
        # be calculated using FAISS inner product.
        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1
        )

        embeddings = (
            embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        all_embeddings.append(
            embeddings
        )

        processed = end

        elapsed = time.time() - start_time

        rate = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = total - processed

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"\r  {processed}/{total} "
            f"({processed / total * 100:.1f}%) "
            f"| {rate:.2f} docs/s "
            f"| ETA {eta:.0f}s",
            end="",
            flush=True
        )

    print()

    result = np.vstack(
        all_embeddings
    )

    print(
        f"Embedding shape: {result.shape}"
    )

    return result


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(
    embeddings,
    output_path
):

    dimension = embeddings.shape[1]

    print(
        f"Building FAISS index "
        f"({dimension} dimensions)..."
    )

    # Because embeddings are normalized,
    # inner product == cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        output_path
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Vectors: {index.ntotal}"
    )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(records):

    metadata = []

    for i, record in enumerate(records):

        metadata.append({
            "vector_id": i,
            "patent_id": record.get("patent_id"),
            "domain": record.get("domain"),
            "title": record.get("title"),
            "problem": record.get("problem"),
            "constraint": record.get("constraint"),
            "mechanism": record.get("mechanism"),
            "mechanism_raw": record.get("mechanism_raw"),
            "residual_terms": record.get("residual_terms")
        })

    with open(
        MECHANISMS_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Saved metadata: {MECHANISMS_JSON}"
    )


# ============================================================
# TEST RETRIEVAL
# ============================================================

def test_retrieval(
    index,
    records,
    embeddings,
    representation_name,
    k=5
):

    if len(records) < 2:
        return

    print()
    print("=" * 60)
    print(
        f"Example retrieval: {representation_name}"
    )
    print("=" * 60)

    query_index = 0

    query_vector = (
        embeddings[query_index]
        .reshape(1, -1)
    )

    scores, indices = index.search(
        query_vector,
        min(k + 1, len(records))
    )

    query_record = records[query_index]

    print()
    print(
        "QUERY:"
    )

    print(
        f"Patent: {query_record.get('patent_id')}"
    )

    print(
        f"Domain: {query_record.get('domain')}"
    )

    print(
        f"Mechanism: "
        f"{query_record.get('mechanism')}"
    )

    print()
    print(
        "RESULTS:"
    )

    result_number = 0

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        # Skip the query itself.
        if idx == query_index:
            continue

        record = records[idx]

        result_number += 1

        print()
        print(
            f"{result_number}. "
            f"score={score:.4f}"
        )

        print(
            f"   Patent: "
            f"{record.get('patent_id')}"
        )

        print(
            f"   Domain: "
            f"{record.get('domain')}"
        )

        print(
            f"   Title: "
            f"{record.get('title')}"
        )

        print(
            f"   Mechanism: "
            f"{record.get('mechanism')}"
        )

        if result_number >= k:
            break


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print("=" * 60)
    print("PHASE 2 — QWEN3 EMBEDDING")
    print("=" * 60)

    print(
        f"Input:  {INPUT_FILE}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"Model:  {MODEL_NAME}"
    )

    print(
        f"Batch:  {BATCH_SIZE}"
    )

    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load Phase 1 data
    # --------------------------------------------------------

    records = load_data()

    # --------------------------------------------------------
    # Build text representations
    # --------------------------------------------------------

    (
        mechanism_texts,
        problem_mechanism_texts,
        full_texts
    ) = build_representations(
        records
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        records
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tokenizer, model = load_model()

    # --------------------------------------------------------
    # Representation 1
    # mechanism
    # --------------------------------------------------------

    mechanism_embeddings = embed_texts(
        mechanism_texts,
        tokenizer,
        model,
        "mechanism"
    )

    np.save(
        MECHANISM_NPY,
        mechanism_embeddings
    )

    print(
        f"Saved: {MECHANISM_NPY}"
    )

    build_faiss_index(
        mechanism_embeddings,
        MECHANISM_INDEX
    )

    # --------------------------------------------------------
    # Representation 2
    # problem + mechanism
    # --------------------------------------------------------

    problem_mechanism_embeddings = embed_texts(
        problem_mechanism_texts,
        tokenizer,
        model,
        "problem + mechanism"
    )

    np.save(
        PROBLEM_MECHANISM_NPY,
        problem_mechanism_embeddings
    )

    print(
        f"Saved: {PROBLEM_MECHANISM_NPY}"
    )

    build_faiss_index(
        problem_mechanism_embeddings,
        PROBLEM_MECHANISM_INDEX
    )

    # --------------------------------------------------------
    # Representation 3
    # problem + constraint + mechanism
    # --------------------------------------------------------

    full_embeddings = embed_texts(
        full_texts,
        tokenizer,
        model,
        "problem + constraint + mechanism"
    )

    np.save(
        PROBLEM_CONSTRAINT_MECHANISM_NPY,
        full_embeddings
    )

    print(
        f"Saved: "
        f"{PROBLEM_CONSTRAINT_MECHANISM_NPY}"
    )

    build_faiss_index(
        full_embeddings,
        PROBLEM_CONSTRAINT_MECHANISM_INDEX
    )

    # --------------------------------------------------------
    # Example retrieval
    # --------------------------------------------------------

    test_retrieval(
        faiss.read_index(
            MECHANISM_INDEX
        ),
        records,
        mechanism_embeddings,
        "mechanism"
    )

    test_retrieval(
        faiss.read_index(
            PROBLEM_MECHANISM_INDEX
        ),
        records,
        problem_mechanism_embeddings,
        "problem + mechanism"
    )

    test_retrieval(
        faiss.read_index(
            PROBLEM_CONSTRAINT_MECHANISM_INDEX
        ),
        records,
        full_embeddings,
        "problem + constraint + mechanism"
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        print()
        print(
            f"GPU memory allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        print(
            f"GPU memory reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    elapsed = (
        time.time() - total_start
    )

    print()
    print("=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)

    print(
        f"Patents embedded: {len(records)}"
    )

    print(
        f"Total time: {elapsed / 60:.2f} minutes"
    )

    print()
    print("Files:")

    print(
        f"  {MECHANISM_INDEX}"
    )

    print(
        f"  {PROBLEM_MECHANISM_INDEX}"
    )

    print(
        f"  {PROBLEM_CONSTRAINT_MECHANISM_INDEX}"
    )

    print()
    print(
        "These three indexes can now be compared "
        "for cross-domain retrieval quality."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()