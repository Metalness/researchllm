"""
Phase 2.5: Evaluate cross-domain mechanism retrieval.

Reads:
    phase2_embeddings/mechanisms.json
    phase2_embeddings/*.faiss

Evaluates:
    1. mechanism
    2. problem + mechanism
    3. problem + constraint + mechanism

For each query:
    - Retrieves candidates from FAISS
    - Removes the query itself
    - Prioritizes cross-domain candidates
    - Shows top candidates
    - Allows manual functional-analogy labeling
    - Saves all judgments to JSON

Labels:
    y = genuine functional/mechanistic analogy
    n = not an analogy
    s = skip / unsure
    q = quit

The goal is NOT to judge whether patents are similar.
The goal is to judge whether their underlying mechanisms
could plausibly be transferred between domains.
"""

import json
import os
import random
import sys
from datetime import datetime

import faiss
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "phase2_embeddings"

METADATA_FILE = os.path.join(
    DATA_DIR,
    "mechanisms.json"
)

RESULTS_FILE = os.path.join(
    DATA_DIR,
    "evaluation_results.json"
)

NUM_QUERIES = 50

# Number of candidates retrieved from FAISS before
# cross-domain filtering.
RETRIEVE_K = 50

# Number shown to the human evaluator.
DISPLAY_K = 10

# Whether to randomly select queries.
# False = first NUM_QUERIES records.
RANDOM_QUERIES = True

RANDOM_SEED = 42

# Representations/indexes.
INDEXES = {
    "mechanism": os.path.join(
        DATA_DIR,
        "mechanism.faiss"
    ),

    "problem_mechanism": os.path.join(
        DATA_DIR,
        "problem_mechanism.faiss"
    ),

    "problem_constraint_mechanism": os.path.join(
        DATA_DIR,
        "problem_constraint_mechanism.faiss"
    )
}


# ============================================================
# LOAD DATA
# ============================================================

def load_metadata():

    print(
        f"Loading metadata: {METADATA_FILE}"
    )

    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Missing {METADATA_FILE}"
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        records = json.load(f)

    if not records:
        raise ValueError(
            "No records found."
        )

    print(
        f"Loaded {len(records)} records."
    )

    return records


# ============================================================
# LOAD INDEXES
# ============================================================

def load_indexes():

    indexes = {}

    for name, path in INDEXES.items():

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing FAISS index: {path}"
            )

        print(
            f"Loading {name} index..."
        )

        indexes[name] = faiss.read_index(
            path
        )

    return indexes


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def embedding_file(name):

    if name == "mechanism":
        return os.path.join(
            DATA_DIR,
            "mechanism_embeddings.npy"
        )

    if name == "problem_mechanism":
        return os.path.join(
            DATA_DIR,
            "problem_mechanism_embeddings.npy"
        )

    if name == "problem_constraint_mechanism":
        return os.path.join(
            DATA_DIR,
            "problem_constraint_mechanism_embeddings.npy"
        )

    raise ValueError(name)


def load_embeddings():

    embeddings = {}

    for name in INDEXES:

        path = embedding_file(name)

        print(
            f"Loading embeddings: {path}"
        )

        embeddings[name] = np.load(
            path,
            mmap_mode="r"
        )

    return embeddings


# ============================================================
# RESULTS
# ============================================================

def load_previous_results():

    if not os.path.exists(
        RESULTS_FILE
    ):
        return []

    try:

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            results = json.load(f)

        print(
            f"Loaded {len(results)} previous judgments."
        )

        return results

    except Exception:

        print(
            "WARNING: Could not read previous results."
        )

        return []


def save_results(results):

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# QUERY SELECTION
# ============================================================

def choose_queries(
    records,
    previous_results
):

    already_evaluated = {
        (
            item.get("query_id"),
            item.get("representation")
        )
        for item in previous_results
    }

    candidates = []

    for record in records:

        query_id = record.get(
            "patent_id"
        )

        for representation in INDEXES:

            if (
                query_id,
                representation
            ) not in already_evaluated:

                candidates.append(
                    (
                        query_id,
                        representation
                    )
                )

    # We want the same query to be evaluated under
    # all three representations, so choose query IDs
    # rather than individual representation entries.

    available_ids = []

    for record in records:

        query_id = record.get(
            "patent_id"
        )

        complete = all(
            (
                query_id,
                representation
            ) in already_evaluated
            for representation in INDEXES
        )

        if not complete:
            available_ids.append(
                query_id
            )

    if RANDOM_QUERIES:

        rng = random.Random(
            RANDOM_SEED
        )

        rng.shuffle(
            available_ids
        )

    return available_ids[
        :NUM_QUERIES
    ]


# ============================================================
# FIND RECORD
# ============================================================

def make_record_map(records):

    return {
        record["patent_id"]: record
        for record in records
    }


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_candidates(
    index,
    embeddings,
    records,
    query_idx,
    display_k=DISPLAY_K
):

    query_vector = np.asarray(
        embeddings[query_idx],
        dtype=np.float32
    ).reshape(1, -1)

    scores, indices = index.search(
        query_vector,
        min(
            RETRIEVE_K,
            len(records)
        )
    )

    query_record = records[
        query_idx
    ]

    query_domain = query_record.get(
        "domain",
        ""
    )

    candidates = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        # Don't retrieve the query itself.
        if idx == query_idx:
            continue

        candidate = records[idx]

        candidate_domain = candidate.get(
            "domain",
            ""
        )

        # We want cross-domain retrieval.
        if candidate_domain == query_domain:
            continue

        candidates.append({
            "index": int(idx),
            "score": float(score),
            "record": candidate
        })

        if len(candidates) >= display_k:
            break

    return candidates


# ============================================================
# PRINT QUERY
# ============================================================

def print_query(
    record,
    representation
):

    print()
    print("=" * 80)
    print(
        f"QUERY — {representation}"
    )
    print("=" * 80)

    print(
        f"Patent:  {record.get('patent_id')}"
    )

    print(
        f"Domain:  {record.get('domain')}"
    )

    print(
        f"Title:   {record.get('title') or '(none)'}"
    )

    print()

    print(
        "PROBLEM:"
    )

    print(
        record.get("problem")
        or "(none)"
    )

    print()

    print(
        "CONSTRAINT:"
    )

    print(
        record.get("constraint")
        or "(none)"
    )

    print()

    print(
        "MECHANISM:"
    )

    print(
        record.get("mechanism")
        or "(none)"
    )


# ============================================================
# PRINT CANDIDATE
# ============================================================

def print_candidate(
    number,
    candidate
):

    record = candidate["record"]

    print()
    print("-" * 80)

    print(
        f"[{number}] "
        f"similarity = {candidate['score']:.4f}"
    )

    print(
        f"Patent: {record.get('patent_id')}"
    )

    print(
        f"Domain: {record.get('domain')}"
    )

    print(
        f"Title:  {record.get('title') or '(none)'}"
    )

    print()

    print(
        "Problem:"
    )

    print(
        record.get("problem")
        or "(none)"
    )

    print()

    print(
        "Mechanism:"
    )

    print(
        record.get("mechanism")
        or "(none)"
    )


# ============================================================
# LABEL INPUT
# ============================================================

def get_label():

    while True:

        print()

        print(
            "Label this candidate:"
        )

        print(
            "  y = genuine functional/mechanistic analogy"
        )

        print(
            "  n = not an analogy"
        )

        print(
            "  s = unsure / skip"
        )

        print(
            "  q = quit"
        )

        value = input(
            "> "
        ).strip().lower()

        if value in {
            "y",
            "n",
            "s",
            "q"
        }:
            return value


# ============================================================
# EVALUATE ONE REPRESENTATION
# ============================================================

def evaluate_representation(
    query_record,
    query_idx,
    representation,
    index,
    embeddings,
    records,
    results
):

    query_id = query_record[
        "patent_id"
    ]

    # Don't redo this representation if already labeled.
    existing = [
        x
        for x in results
        if (
            x.get("query_id") == query_id
            and x.get("representation")
            == representation
        )
    ]

    if existing:
        print(
            f"\nAlready evaluated "
            f"{query_id} / {representation}."
        )

        return True

    print_query(
        query_record,
        representation
    )

    candidates = retrieve_candidates(
        index,
        embeddings,
        records,
        query_idx
    )

    if not candidates:

        print(
            "\nNo cross-domain candidates found."
        )

        return True

    print()
    print(
        f"Showing {len(candidates)} "
        f"cross-domain candidates."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Judge FUNCTIONAL/MЕCHANISTIC similarity, "
        "not whether the patents discuss similar technology."
    )

    print(
        "Ask: could the underlying operation be transplanted "
        "into the other domain?"
    )

    # --------------------------------------------------------
    # Label each candidate
    # --------------------------------------------------------

    for number, candidate in enumerate(
        candidates,
        start=1
    ):

        print_candidate(
            number,
            candidate
        )

        label = get_label()

        if label == "q":
            save_results(results)
            return False

        if label == "s":
            continue

        candidate_record = candidate[
            "record"
        ]

        results.append({

            "timestamp":
                datetime.now().isoformat(),

            "query_id":
                query_id,

            "query_domain":
                query_record.get("domain"),

            "candidate_id":
                candidate_record.get(
                    "patent_id"
                ),

            "candidate_domain":
                candidate_record.get(
                    "domain"
                ),

            "representation":
                representation,

            "rank":
                number,

            "similarity":
                candidate["score"],

            "label":
                1 if label == "y" else 0
        })

        save_results(
            results
        )

    return True


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    if not results:

        print(
            "No labeled results yet."
        )

        return

    representations = list(
        INDEXES.keys()
    )

    for representation in representations:

        subset = [
            x
            for x in results
            if x.get("representation")
            == representation
        ]

        if not subset:
            continue

        positives = sum(
            x["label"] == 1
            for x in subset
        )

        negatives = sum(
            x["label"] == 0
            for x in subset
        )

        total = positives + negatives

        precision = (
            positives / total
            if total
            else 0
        )

        print()
        print(
            representation
        )

        print(
            f"  Judgments: {total}"
        )

        print(
            f"  Analogies: {positives}"
        )

        print(
            f"  Non-analogies: {negatives}"
        )

        print(
            f"  Precision@displayed: "
            f"{precision:.3f}"
        )

    # --------------------------------------------------------
    # Cross-domain stats
    # --------------------------------------------------------

    domains = {}

    for item in results:

        key = (
            item.get("query_domain"),
            item.get("candidate_domain")
        )

        domains[key] = domains.get(
            key,
            0
        ) + 1

    print()
    print(
        f"Saved judgments: {len(results)}"
    )

    print(
        f"Results file: {RESULTS_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PHASE 2.5 — CROSS-DOMAIN RETRIEVAL EVALUATION")
    print("=" * 80)

    print()

    records = load_metadata()

    indexes = load_indexes()

    embeddings = load_embeddings()

    results = load_previous_results()

    record_map = make_record_map(
        records
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    for name in INDEXES:

        index = indexes[name]

        vectors = embeddings[name]

        if index.ntotal != len(records):

            raise ValueError(
                f"{name}: FAISS has "
                f"{index.ntotal} vectors but "
                f"metadata has {len(records)} records."
            )

        if vectors.shape[0] != len(records):

            raise ValueError(
                f"{name}: embedding count does "
                f"not match metadata count."
            )

        if vectors.shape[1] != index.d:

            raise ValueError(
                f"{name}: embedding dimension "
                f"does not match FAISS."
            )

    # --------------------------------------------------------
    # Select queries
    # --------------------------------------------------------

    query_ids = choose_queries(
        records,
        results
    )

    if not query_ids:

        print(
            "\nNo unevaluated queries remaining."
        )

        print_summary(
            results
        )

        return

    print()
    print(
        f"Selected {len(query_ids)} queries."
    )

    print(
        f"Each query will be tested against "
        f"{len(INDEXES)} representations."
    )

    print()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for query_number, query_id in enumerate(
        query_ids,
        start=1
    ):

        query_record = record_map[
            query_id
        ]

        query_idx = next(
            i
            for i, record
            in enumerate(records)
            if record["patent_id"]
            == query_id
        )

        print()
        print()
        print("#" * 80)

        print(
            f"QUERY {query_number}/{len(query_ids)}"
        )

        print(
            f"Patent: {query_id}"
        )

        print(
            f"Domain: "
            f"{query_record.get('domain')}"
        )

        print(
            "#" * 80
        )

        for representation in INDEXES:

            should_continue = (
                evaluate_representation(
                    query_record,
                    query_idx,
                    representation,
                    indexes[representation],
                    embeddings[representation],
                    records,
                    results
                )
            )

            if not should_continue:

                print()
                print(
                    "Evaluation stopped."
                )

                print_summary(
                    results
                )

                return

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_summary(
        results
    )

    print()
    print(
        "Evaluation complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()