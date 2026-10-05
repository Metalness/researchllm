import json
import os
import numpy as np
import torch
import faiss

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "structured_mechanisms.json"
OUTPUT_DIR = "phase2b_embeddings"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

BATCH_SIZE = 16
MAX_LENGTH = 2048

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("PHASE 2B — MULTI-REPRESENTATION EMBEDDINGS")
print("=" * 80)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Loaded {len(records)} records.")

# Only records that actually have a mechanism
records = [
    r for r in records
    if str(r.get("has_mechanism", "")).lower() in
       {"yes", "true", "1"}
    and r.get("mechanism")
]

print(f"Using {len(records)} records with mechanisms.")


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {total_vram:.2f} GB")
else:
    device = torch.device("cpu")
    print("WARNING: CUDA not available. Using CPU.")


# ============================================================
# LOAD MODEL
# ============================================================

print(f"\nLoading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
)

model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# TEXT REPRESENTATIONS
# ============================================================

def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def build_representations(r):
    problem = clean(r.get("problem"))
    constraint = clean(r.get("constraint"))
    mechanism = clean(r.get("mechanism"))
    mechanism_raw = clean(r.get("mechanism_raw"))
    residual = clean(r.get("residual_terms"))

    # --------------------------------------------------------
    # A — Current baseline
    # --------------------------------------------------------
    rep_mechanism = mechanism

    # --------------------------------------------------------
    # B — Problem + mechanism
    # --------------------------------------------------------
    rep_problem_mechanism = (
        f"Problem: {problem}\n"
        f"Mechanism: {mechanism}"
    )

    # --------------------------------------------------------
    # C — Problem + constraint + mechanism
    # --------------------------------------------------------
    rep_problem_constraint_mechanism = (
        f"Problem: {problem}\n"
        f"Constraint: {constraint}\n"
        f"Mechanism: {mechanism}"
    )

    # --------------------------------------------------------
    # D — Raw mechanism + abstracted mechanism
    #
    # Gives the model both the concrete implementation and
    # the LLM abstraction that Phase 1 already produced.
    # --------------------------------------------------------
    rep_raw_abstract = (
        f"Concrete mechanism: {mechanism_raw}\n"
        f"Functional mechanism: {mechanism}"
    )

    # --------------------------------------------------------
    # E — Problem + raw + abstract
    # --------------------------------------------------------
    rep_problem_raw_abstract = (
        f"Problem: {problem}\n"
        f"Concrete mechanism: {mechanism_raw}\n"
        f"Functional mechanism: {mechanism}"
    )

    # --------------------------------------------------------
    # F — Problem + constraint + raw + abstract
    # --------------------------------------------------------
    rep_full = (
        f"Problem: {problem}\n"
        f"Constraint: {constraint}\n"
        f"Concrete mechanism: {mechanism_raw}\n"
        f"Functional mechanism: {mechanism}"
    )

    # --------------------------------------------------------
    # G — Mechanism with residual/domain-specific terms
    #
    # We keep this separate rather than attempting dangerous
    # automatic deletion. This lets us compare later.
    # --------------------------------------------------------
    rep_mechanism_residual = (
        f"Functional mechanism: {mechanism}\n"
        f"Domain-specific terms: {residual}"
    )

    return {
        "mechanism": rep_mechanism,
        "problem_mechanism": rep_problem_mechanism,
        "problem_constraint_mechanism":
            rep_problem_constraint_mechanism,
        "raw_abstract": rep_raw_abstract,
        "problem_raw_abstract": rep_problem_raw_abstract,
        "full": rep_full,
        "mechanism_residual": rep_mechanism_residual,
    }


# ============================================================
# LAST-TOKEN POOLING
# ============================================================

def last_token_pool(last_hidden_state, attention_mask):
    """
    Qwen embedding models use the final valid token as the
    sequence representation.
    """

    left_padding = (
        attention_mask[:, -1].sum() ==
        attention_mask.shape[0]
    )

    if left_padding:
        return last_hidden_state[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1

    batch_indices = torch.arange(
        last_hidden_state.shape[0],
        device=last_hidden_state.device
    )

    return last_hidden_state[
        batch_indices,
        sequence_lengths
    ]


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

@torch.no_grad()
def embed_texts(texts):

    all_embeddings = []

    for start in tqdm(
        range(0, len(texts), BATCH_SIZE),
        desc="Embedding"
    ):

        batch = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }

        outputs = model(**encoded)

        embeddings = last_token_pool(
            outputs.last_hidden_state,
            encoded["attention_mask"]
        )

        # Normalize so inner product = cosine similarity
        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1
        )

        all_embeddings.append(
            embeddings.cpu().float().numpy()
        )

    return np.concatenate(all_embeddings, axis=0)


# ============================================================
# BUILD TEXT DATA
# ============================================================

print("\nBuilding representations...")

representations = {
    name: []
    for name in [
        "mechanism",
        "problem_mechanism",
        "problem_constraint_mechanism",
        "raw_abstract",
        "problem_raw_abstract",
        "full",
        "mechanism_residual",
    ]
}

for record in records:

    reps = build_representations(record)

    for name, text in reps.items():
        representations[name].append(text)


print("Representations built.")

for name, texts in representations.items():
    print(f"  {name}: {len(texts)}")


# ============================================================
# SAVE METADATA
# ============================================================

metadata = []

for r in records:

    metadata.append({
        "patent_id": r.get("patent_id"),
        "domain": r.get("domain"),
        "title": r.get("title"),
        "problem": r.get("problem"),
        "constraint": r.get("constraint"),
        "mechanism": r.get("mechanism"),
        "mechanism_raw": r.get("mechanism_raw"),
        "residual_terms": r.get("residual_terms"),
        "has_mechanism": r.get("has_mechanism"),
    })

with open(
    os.path.join(OUTPUT_DIR, "metadata.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nSaved metadata.json")


# ============================================================
# EMBED EACH REPRESENTATION
# ============================================================

embedding_files = {}

for name, texts in representations.items():

    print("\n" + "-" * 80)
    print(f"Embedding representation: {name}")
    print("-" * 80)

    embeddings = embed_texts(texts)

    print(f"Embedding shape: {embeddings.shape}")

    npy_path = os.path.join(
        OUTPUT_DIR,
        f"{name}.npy"
    )

    np.save(npy_path, embeddings)

    embedding_files[name] = npy_path

    print(f"Saved: {npy_path}")


# ============================================================
# BUILD FAISS INDEXES
# ============================================================

print("\n" + "=" * 80)
print("BUILDING FAISS INDEXES")
print("=" * 80)

for name, npy_path in embedding_files.items():

    embeddings = np.load(npy_path)

    dimension = embeddings.shape[1]

    # Normalized embeddings + inner product
    # = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    index_path = os.path.join(
        OUTPUT_DIR,
        f"{name}.faiss"
    )

    faiss.write_index(index, index_path)

    print(
        f"{name:35s} "
        f"{index.ntotal:5d} vectors "
        f"dim={dimension}"
    )


# ============================================================
# SAVE REPRESENTATION DESCRIPTIONS
# ============================================================

description = {
    "mechanism":
        "Phase 1 functional mechanism only",

    "problem_mechanism":
        "Problem + Phase 1 functional mechanism",

    "problem_constraint_mechanism":
        "Problem + constraint + functional mechanism",

    "raw_abstract":
        "Concrete mechanism + functional mechanism",

    "problem_raw_abstract":
        "Problem + concrete mechanism + functional mechanism",

    "full":
        "Problem + constraint + concrete mechanism + functional mechanism",

    "mechanism_residual":
        "Functional mechanism + Phase 1 residual/domain-specific terms"
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "representations.json"
    ),
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        description,
        f,
        indent=2
    )


# ============================================================
# SAMPLE RETRIEVAL
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE RETRIEVAL")
print("=" * 80)

query_index = 0

print("\nQUERY:")
print(metadata[query_index]["patent_id"])
print("Domain:", metadata[query_index]["domain"])
print("Mechanism:", metadata[query_index]["mechanism"])

for name in representations.keys():

    index_path = os.path.join(
        OUTPUT_DIR,
        f"{name}.faiss"
    )

    index = faiss.read_index(index_path)

    query_vector = np.load(
        os.path.join(
            OUTPUT_DIR,
            f"{name}.npy"
        )
    )[query_index:query_index + 1]

    scores, indices = index.search(
        query_vector,
        6
    )

    print("\n" + "-" * 60)
    print(name)

    shown = 0

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == query_index:
            continue

        print(
            f"{shown + 1}. "
            f"{metadata[idx]['patent_id']} | "
            f"{metadata[idx]['domain']} | "
            f"score={score:.4f}"
        )

        print(
            "   ",
            metadata[idx]["mechanism"][:300]
        )

        shown += 1

        if shown >= 5:
            break


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"\nOutput directory: {OUTPUT_DIR}")

print("\nFiles:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(" ", filename)

print("\nNo LLM calls were made.")
print("Phase 1 data was reused directly.")