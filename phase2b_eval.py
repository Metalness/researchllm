import json
import os
import csv
import numpy as np
import faiss

from collections import Counter


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "phase2b_embeddings"
OUTPUT_DIR = "phase2b_evaluation"

TOP_K = 50
NUM_QUERIES = 250
TOP_PAIRS_PER_REPRESENTATION = 20

SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


REPRESENTATIONS = [
    "mechanism",
    "problem_mechanism",
    "problem_constraint_mechanism",
    "raw_abstract",
    "problem_raw_abstract",
    "full",
    "mechanism_residual",
]


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("PHASE 2B — CROSS-DOMAIN ANALOGY CANDIDATE GENERATOR")
print("=" * 90)

with open(
    os.path.join(INPUT_DIR, "metadata.json"),
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

N = len(metadata)

print(f"Records: {N}")
print(f"Representations: {len(REPRESENTATIONS)}")


def get_domain(r):
    d = str(r.get("domain", "")).strip()
    return d if d else "UNKNOWN"


domains = [
    get_domain(r)
    for r in metadata
]

print(
    f"Unique domains: "
    f"{len(set(domains))}"
)


# ============================================================
# LOAD INDEXES + EMBEDDINGS
# ============================================================

indexes = {}
embeddings = {}

for rep in REPRESENTATIONS:

    index_path = os.path.join(
        INPUT_DIR,
        f"{rep}.faiss"
    )

    embedding_path = os.path.join(
        INPUT_DIR,
        f"{rep}.npy"
    )

    indexes[rep] = faiss.read_index(
        index_path
    )

    embeddings[rep] = np.load(
        embedding_path
    )

    print(
        f"{rep:35s} "
        f"{embeddings[rep].shape}"
    )


# ============================================================
# SELECT QUERIES
# ============================================================

rng = np.random.default_rng(SEED)

if NUM_QUERIES >= N:
    query_indices = np.arange(N)
else:
    query_indices = rng.choice(
        N,
        size=NUM_QUERIES,
        replace=False
    )

query_indices = sorted(
    query_indices.tolist()
)


# ============================================================
# RETRIEVE CROSS-DOMAIN CANDIDATES
# ============================================================

all_candidates = {
    rep: []
    for rep in REPRESENTATIONS
}


for rep in REPRESENTATIONS:

    print("\n" + "=" * 90)
    print(f"PROCESSING: {rep}")
    print("=" * 90)

    index = indexes[rep]

    candidates = []

    for q_idx in query_indices:

        query_vector = embeddings[
            rep
        ][q_idx:q_idx + 1]

        # Retrieve more than TOP_K because we remove:
        #   1. the query itself
        #   2. same-domain results
        search_k = min(
            TOP_K + 25,
            N
        )

        scores, indices = index.search(
            query_vector,
            search_k
        )

        query_domain = domains[q_idx]

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            idx = int(idx)
            score = float(score)

            # Don't match a patent with itself
            if idx == q_idx:
                continue

            candidate_domain = domains[idx]

            # Only cross-domain candidates
            if candidate_domain == query_domain:
                continue

            candidates.append({
                "query_index": q_idx,
                "candidate_index": idx,
                "query_id":
                    metadata[q_idx].get("patent_id"),
                "candidate_id":
                    metadata[idx].get("patent_id"),
                "query_domain":
                    query_domain,
                "candidate_domain":
                    candidate_domain,
                "rank": rank,
                "similarity": score,
            })

    # Sort globally by similarity
    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # --------------------------------------------------------
    # Avoid repeatedly showing the same pair.
    # --------------------------------------------------------

    seen_pairs = set()
    selected = []

    for c in candidates:

        pair = (
            c["query_index"],
            c["candidate_index"]
        )

        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)

        selected.append(c)

        if len(selected) >= TOP_PAIRS_PER_REPRESENTATION:
            break

    all_candidates[rep] = selected

    print(
        f"Selected {len(selected)} "
        f"top cross-domain candidates."
    )


# ============================================================
# CREATE HUMAN-READABLE ANALOGY TEST
# ============================================================

test_path = os.path.join(
    OUTPUT_DIR,
    "ANALOGY_TEST.txt"
)

with open(
    test_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 100 + "\n")
    f.write("CROSS-DOMAIN ANALOGY TEST\n")
    f.write("=" * 100 + "\n\n")

    f.write(
        "Judge whether the TWO MECHANISMS share a "
        "transferable underlying operation.\n\n"
    )

    f.write(
        "Do NOT mark Y merely because they:\n"
    )

    f.write(
        "  - use the same physical component\n"
    )

    f.write(
        "  - both involve hinges/opening/closing\n"
    )

    f.write(
        "  - both mention similar objects\n"
    )

    f.write(
        "  - solve generally similar problems\n\n"
    )

    f.write(
        "Mark Y when you can plausibly describe both using "
        "the same implementation-independent operation.\n\n"
    )

    f.write(
        "Y = genuine transferable analogy\n"
    )

    f.write(
        "N = merely surface similarity / unrelated mechanism\n"
    )

    f.write(
        "S = genuinely interesting but uncertain\n\n"
    )

    f.write("=" * 100 + "\n\n")


    number = 1

    for rep in REPRESENTATIONS:

        f.write("\n")
        f.write("#" * 100 + "\n")
        f.write(
            f"REPRESENTATION: {rep}\n"
        )
        f.write("#" * 100 + "\n\n")

        for item in all_candidates[rep]:

            q = metadata[
                item["query_index"]
            ]

            c = metadata[
                item["candidate_index"]
            ]

            f.write(
                "\n"
                + "-" * 100
                + "\n"
            )

            f.write(
                f"ANALOGY CANDIDATE #{number}\n"
            )

            f.write(
                f"Similarity: "
                f"{item['similarity']:.4f}\n"
            )

            f.write(
                f"Representation: {rep}\n"
            )

            f.write(
                f"Retrieval rank: {item['rank']}\n\n"
            )

            # ------------------------------------------------
            # QUERY
            # ------------------------------------------------

            f.write(
                "QUERY\n"
            )

            f.write(
                f"ID: {q.get('patent_id')}\n"
            )

            f.write(
                f"Domain: {q.get('domain')}\n\n"
            )

            f.write(
                "Problem:\n"
            )

            f.write(
                f"{q.get('problem', '')}\n\n"
            )

            f.write(
                "Constraint:\n"
            )

            f.write(
                f"{q.get('constraint', '')}\n\n"
            )

            f.write(
                "Concrete mechanism:\n"
            )

            f.write(
                f"{q.get('mechanism_raw', '')}\n\n"
            )

            f.write(
                "Functional mechanism:\n"
            )

            f.write(
                f"{q.get('mechanism', '')}\n\n"
            )

            # ------------------------------------------------
            # CANDIDATE
            # ------------------------------------------------

            f.write(
                "CANDIDATE\n"
            )

            f.write(
                f"ID: {c.get('patent_id')}\n"
            )

            f.write(
                f"Domain: {c.get('domain')}\n\n"
            )

            f.write(
                "Problem:\n"
            )

            f.write(
                f"{c.get('problem', '')}\n\n"
            )

            f.write(
                "Constraint:\n"
            )

            f.write(
                f"{c.get('constraint', '')}\n\n"
            )

            f.write(
                "Concrete mechanism:\n"
            )

            f.write(
                f"{c.get('mechanism_raw', '')}\n\n"
            )

            f.write(
                "Functional mechanism:\n"
            )

            f.write(
                f"{c.get('mechanism', '')}\n\n"
            )

            f.write(
                "YOUR LABEL:\n"
            )

            f.write(
                "[ ] Y — transferable analogy\n"
            )

            f.write(
                "[ ] N — not an analogy\n"
            )

            f.write(
                "[ ] S — unsure\n"
            )

            f.write(
                "\n"
            )

            number += 1


# ============================================================
# SAVE STRUCTURED JSON
# ============================================================

json_output = []

for rep in REPRESENTATIONS:

    for item in all_candidates[rep]:

        q = metadata[
            item["query_index"]
        ]

        c = metadata[
            item["candidate_index"]
        ]

        json_output.append({
            "representation": rep,

            "similarity":
                item["similarity"],

            "rank":
                item["rank"],

            "query": {
                "id":
                    q.get("patent_id"),

                "domain":
                    q.get("domain"),

                "problem":
                    q.get("problem"),

                "constraint":
                    q.get("constraint"),

                "mechanism_raw":
                    q.get("mechanism_raw"),

                "mechanism":
                    q.get("mechanism"),
            },

            "candidate": {
                "id":
                    c.get("patent_id"),

                "domain":
                    c.get("domain"),

                "problem":
                    c.get("problem"),

                "constraint":
                    c.get("constraint"),

                "mechanism_raw":
                    c.get("mechanism_raw"),

                "mechanism":
                    c.get("mechanism"),
            }
        })


json_path = os.path.join(
    OUTPUT_DIR,
    "analogy_candidates.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_output,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "analogy_summary.csv"
)

summary_rows = []

for rep in REPRESENTATIONS:

    items = all_candidates[rep]

    similarities = [
        x["similarity"]
        for x in items
    ]

    unique_domains = set()

    for x in items:
        unique_domains.add(
            x["query_domain"]
        )
        unique_domains.add(
            x["candidate_domain"]
        )

    summary_rows.append({
        "representation":
            rep,

        "candidates":
            len(items),

        "mean_similarity":
            float(np.mean(similarities))
            if similarities else 0,

        "highest_similarity":
            float(np.max(similarities))
            if similarities else 0,

        "unique_domains":
            len(unique_domains),
    })


with open(
    summary_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "representation",
            "candidates",
            "mean_similarity",
            "highest_similarity",
            "unique_domains",
        ]
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 90)
print("ANALOGY TEST GENERATED")
print("=" * 90)

print(
    f"\nHuman-readable test:"
)
print(
    f"  {test_path}"
)

print(
    f"\nStructured candidates:"
)
print(
    f"  {json_path}"
)

print(
    f"\nSummary:"
)
print(
    f"  {summary_path}"
)

print(
    "\nOpen ANALOGY_TEST.txt and send me the candidates "
    "with your Y/N/S judgments."
)

print(
    "\nNo LLM calls were made."
)