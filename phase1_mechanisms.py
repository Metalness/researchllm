"""
Phase 1: Extract transferable mechanisms from patents.

Features:
- Multiple Kilo API keys
- Equal patent allocation across keys
- Multiple workers per key
- Sends BATCH_SIZE patents per LLM request
- Never permanently skips a patent because of gateway/API errors
- Retries failed batches indefinitely
- Only saves a batch after the entire batch receives valid results
- Two-pass extraction:
    Pass 1 -> identify/extract mechanism
    Pass 2 -> functional abstraction
- Existing completed patents are skipped
"""

import os
import sys
import json
import time
import sqlite3
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from kilo_gateway import chat


# ============================================================
# CONFIG
# ============================================================

DB_PATH = "patents.db"

# Number of patents sent in ONE LLM request.
BATCH_SIZE = 5

# Number of simultaneous batches per API key.
WORKERS_PER_KEY = 2

# Model prompt version.
PROMPT_VERSION = "patent-v7-batched-functional-abstraction"

# Maximum number of patents to take from DB.
# None = all.
LIMIT = None


# ============================================================
# API KEYS
# ============================================================

def get_api_keys():
    """
    Supports:

        KILO_API_KEYS=key1,key2,key3

    and falls back to:

        KILO_API_KEY=key1,key2,key3
    """

    raw = os.getenv("KILO_API_KEYS") or os.getenv("KILO_API_KEY")

    if not raw:
        sys.exit(
            "Set KILO_API_KEYS or KILO_API_KEY in .env."
        )

    keys = [k.strip() for k in raw.split(",") if k.strip()]

    if not keys:
        sys.exit("No API keys found.")

    return keys


# ============================================================
# DATABASE
# ============================================================

def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def ensure_schema():
    conn = connect_db()
    cur = conn.cursor()

    # Original fields
    cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            field TEXT,
            title TEXT,
            abstract TEXT,
            mechanism TEXT
        )
    """)

    # New fields
    columns = {
        "has_mechanism": "INTEGER",
        "problem": "TEXT",
        "constraint_text": "TEXT",
        "status": "TEXT",
        "reason": "TEXT",
        "mechanism_raw": "TEXT",
        "residual_terms": "TEXT",
        "prompt_version": "TEXT",
    }

    existing = {
        row[1]
        for row in cur.execute("PRAGMA table_info(items)").fetchall()
    }

    for name, typ in columns.items():
        if name not in existing:
            cur.execute(
                f"ALTER TABLE items ADD COLUMN {name} {typ}"
            )

    conn.commit()
    conn.close()


# ============================================================
# LOAD WORK
# ============================================================

def load_items():
    conn = connect_db()
    cur = conn.cursor()

    query = """
        SELECT
            id,
            field,
            title,
            abstract
        FROM items
        WHERE id LIKE 'patent:%'
          AND abstract IS NOT NULL
          AND length(abstract) >= 300
          AND (
              status IS NULL
              OR status NOT IN ('complete', 'none')
          )
        ORDER BY id
    """

    if LIMIT is not None:
        query += f" LIMIT {int(LIMIT)}"

    rows = cur.execute(query).fetchall()

    conn.close()
    return rows


# ============================================================
# BATCHING
# ============================================================

def make_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def split_equally(items, n):
    """
    Split items as evenly as possible.
    """

    n = min(n, len(items))

    result = []
    base = len(items) // n
    remainder = len(items) % n

    start = 0

    for i in range(n):
        size = base + (1 if i < remainder else 0)

        result.append(
            items[start:start + size]
        )

        start += size

    return result


# ============================================================
# PROMPTS
# ============================================================

PASS1_SYSTEM = """
You are extracting transferable engineering/scientific mechanisms
from patent abstracts.

The goal is NOT to summarize the patent.

The goal is to determine whether the patent contains a genuine
problem-solving mechanism that can potentially be transferred to
a completely different field.

A mechanism describes HOW something produces a useful effect.

Good examples:

- A movement that performs one action automatically causes another
  required state change.
- Repeated subdivision of limited space increases useful interaction
  while balancing structural support and throughput.
- A system maintains operation despite losing connectivity by allowing
  independent local operation followed by later reconciliation.
- A controlled weak boundary allows a desired region to separate while
  neighboring regions remain intact.

Bad examples:

- Lists of components.
- Lists of materials.
- Product features.
- Static architecture.
- "The device has X, Y and Z."
- Merely replacing nouns with generic nouns.
- Restating the abstract.
- Naming the technology instead of explaining the causal principle.

The mechanism must explain the functional relationship that creates
the desired effect.
"""


def build_pass1_prompt(rows):
    patents = []

    for idx, row in enumerate(rows):
        patent_id, field, title, abstract = row

        patents.append({
            "index": idx,
            "id": patent_id,
            "field": field,
            "title": title,
            "abstract": abstract,
        })

    return f"""
Analyze the following {len(patents)} patents independently.

For EACH patent return exactly one result.

Determine whether it contains a transferable problem-solving mechanism.

For a valid mechanism provide:

- has_mechanism: true
- reason_if_false: ""
- problem: the problem being solved
- mechanism: the causal functional mechanism
- constraint: the important constraint/tradeoff

For patents without a meaningful transferable mechanism:

- has_mechanism: false
- reason_if_false: short explanation
- problem: ""
- mechanism: ""
- constraint: ""

Do not reject a mechanism merely because it is technically specific.
Instead identify the underlying causal operation.

Return ONLY valid JSON in this exact structure:

{{
  "results": [
    {{
      "index": 0,
      "has_mechanism": true,
      "reason_if_false": "",
      "problem": "...",
      "mechanism": "...",
      "constraint": "..."
    }}
  ]
}}

There must be exactly {len(patents)} results.

PATENTS:

{json.dumps(patents, ensure_ascii=False, indent=2)}
"""


PASS2_SYSTEM = """
You are converting patent mechanisms into implementation-independent
functional abstractions.

The objective is to remove field-specific implementation while
preserving the actual causal structure.

DO NOT merely replace technical nouns with generic nouns.

Preserve important:

- causal relationships
- constraints
- sequence
- timing
- thresholds
- branching
- repetition
- accumulation
- selection
- comparison
- switching
- feedback
- spatial relationships when functionally important
- hierarchy
- redundancy
- failure behavior
- tradeoffs

Remove:

- brand names
- patent-specific component names
- unnecessary materials
- dimensions
- model numbers
- implementation-specific terminology

A good abstraction should be usable as a mechanism in a completely
different field.

Example:

Bad:
"Use a spring-loaded detent to hold the rotating shaft in position."

Better:
"Use a resilient element to create discrete stable states so a moving
part remains in a selected position without continuous actuation."

Another example:

Bad:
"A scraper removes toner from the OPC drum."

Better:
"Place a stationary element against a continuously moving surface so
accumulated material is removed as the surface passes the element."

Return only JSON.
"""


def build_pass2_prompt(rows):
    """
    rows contains dictionaries produced by Pass 1.
    """

    return f"""
Convert each mechanism below into a transferable functional abstraction.

Return exactly one result for every input item.

JSON structure:

{{
  "results": [
    {{
      "index": 0,
      "mechanism": "...",
      "residual_terms": []
    }}
  ]
}}

The mechanism should describe the functional principle, not the patent's
specific implementation.

residual_terms should contain technical terms that could not reasonably
be removed without damaging the mechanism.

There must be exactly {len(rows)} results.

INPUT:

{json.dumps(rows, ensure_ascii=False, indent=2)}
"""


# ============================================================
# ROBUST JSON REQUEST
# ============================================================

def call_json_forever(system_prompt, user_prompt, api_key, label):
    """
    Keep retrying until a valid JSON response is obtained.

    This is deliberately infinite.

    A gateway error must NEVER cause a batch to disappear.
    """

    attempt = 0

    while True:
        attempt += 1

        try:
            response = chat(
                user_prompt,
                system=system_prompt,
                api_key=api_key,
                temperature=0.2,
                max_tokens=5000,
            )

        except Exception as e:
            wait = min(60, 5 * attempt)

            print(
                f"  [{label}] gateway error on attempt {attempt}: "
                f"{str(e)[:300]}",
                flush=True,
            )

            print(
                f"  [{label}] retrying in {wait}s...",
                flush=True,
            )

            time.sleep(wait)
            continue

        # ----------------------------------------------------
        # Parse JSON ourselves so we can retry malformed output.
        # ----------------------------------------------------

        text = response.strip()

        # Remove markdown fences
        if text.startswith("```"):
            lines = text.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            print(
                f"  [{label}] invalid JSON on attempt {attempt}",
                flush=True,
            )
            time.sleep(min(60, 5 * attempt))
            continue

        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            print(
                f"  [{label}] JSON parse error on attempt {attempt}: "
                f"{e}",
                flush=True,
            )
            time.sleep(min(60, 5 * attempt))
            continue

        if not isinstance(data, dict):
            print(
                f"  [{label}] JSON is not an object; retrying",
                flush=True,
            )
            time.sleep(min(60, 5 * attempt))
            continue

        if not isinstance(data.get("results"), list):
            print(
                f"  [{label}] missing results array; retrying",
                flush=True,
            )
            time.sleep(min(60, 5 * attempt))
            continue

        return data


# ============================================================
# PROCESS ONE BATCH
# ============================================================

def process_batch(rows, api_key, batch_number):
    """
    Process a batch completely.

    Nothing is returned until BOTH passes succeed.
    """

    batch_label = f"BATCH {batch_number}"

    # --------------------------------------------------------
    # PASS 1
    # --------------------------------------------------------

    pass1_prompt = build_pass1_prompt(rows)

    pass1 = call_json_forever(
        PASS1_SYSTEM,
        pass1_prompt,
        api_key,
        f"{batch_label} PASS1",
    )

    pass1_results = pass1["results"]

    if len(pass1_results) != len(rows):
        print(
            f"  [{batch_label}] PASS1 returned "
            f"{len(pass1_results)} results for "
            f"{len(rows)} patents. Retrying...",
            flush=True,
        )

        return process_batch(rows, api_key, batch_number)

    # Check indices
    expected_indices = set(range(len(rows)))

    returned_indices = {
        x.get("index")
        for x in pass1_results
        if isinstance(x, dict)
    }

    if returned_indices != expected_indices:
        print(
            f"  [{batch_label}] PASS1 indices don't match. Retrying...",
            flush=True,
        )

        return process_batch(rows, api_key, batch_number)

    # --------------------------------------------------------
    # Build candidates for PASS 2
    # --------------------------------------------------------

    candidates = []

    for result in pass1_results:
        idx = result["index"]

        if not result.get("has_mechanism"):
            continue

        mechanism = result.get("mechanism", "").strip()

        if not mechanism:
            continue

        candidates.append({
            "index": idx,
            "problem": result.get("problem", ""),
            "constraint": result.get("constraint", ""),
            "mechanism": mechanism,
        })

    # --------------------------------------------------------
    # PASS 2
    # --------------------------------------------------------

    pass2_results = {}

    if candidates:
        pass2_prompt = build_pass2_prompt(candidates)

        pass2 = call_json_forever(
            PASS2_SYSTEM,
            pass2_prompt,
            api_key,
            f"{batch_label} PASS2",
        )

        returned = pass2["results"]

        if len(returned) != len(candidates):
            print(
                f"  [{batch_label}] PASS2 returned "
                f"{len(returned)} results for "
                f"{len(candidates)} mechanisms. Retrying...",
                flush=True,
            )

            return process_batch(rows, api_key, batch_number)

        for result in returned:
            idx = result.get("index")

            if idx not in {
                c["index"] for c in candidates
            }:
                print(
                    f"  [{batch_label}] PASS2 invalid index. Retrying...",
                    flush=True,
                )

                return process_batch(rows, api_key, batch_number)

            pass2_results[idx] = result

    # --------------------------------------------------------
    # Assemble final results
    # --------------------------------------------------------

    final = []

    for result in pass1_results:
        idx = result["index"]

        row = rows[idx]

        patent_id = row[0]
        field = row[1]
        title = row[2]

        has_mechanism = bool(result.get("has_mechanism"))

        if not has_mechanism:
            final.append({
                "id": patent_id,
                "field": field,
                "title": title,
                "has_mechanism": 0,
                "problem": "",
                "constraint_text": "",
                "mechanism": "",
                "mechanism_raw": "",
                "residual_terms": [],
                "status": "none",
                "reason": result.get(
                    "reason_if_false",
                    "No transferable mechanism",
                ),
                "prompt_version": PROMPT_VERSION,
            })

            continue

        raw_mechanism = result.get(
            "mechanism",
            ""
        ).strip()

        p2 = pass2_results.get(idx)

        if not p2:
            # This should never happen because PASS2 is required
            # for every candidate.
            raise RuntimeError(
                f"Missing PASS2 result for {patent_id}"
            )

        final_mechanism = (
            p2.get("mechanism") or raw_mechanism
        ).strip()

        residual_terms = p2.get(
            "residual_terms",
            []
        )

        if not isinstance(residual_terms, list):
            residual_terms = []

        final.append({
            "id": patent_id,
            "field": field,
            "title": title,
            "has_mechanism": 1,
            "problem": result.get("problem", ""),
            "constraint_text": result.get(
                "constraint",
                "",
            ),
            "mechanism": final_mechanism,
            "mechanism_raw": raw_mechanism,
            "residual_terms": residual_terms,
            "status": "complete",
            "reason": "",
            "prompt_version": PROMPT_VERSION,
        })

    return final


# ============================================================
# DATABASE SAVE
# ============================================================

def save_results(results):
    """
    Save an entire successfully processed batch.

    Uses a separate short-lived DB connection so workers never
    write to SQLite simultaneously.
    """

    if not results:
        return

    conn = connect_db()
    cur = conn.cursor()

    for r in results:
        cur.execute("""
            UPDATE items
            SET
                mechanism = ?,
                has_mechanism = ?,
                problem = ?,
                constraint_text = ?,
                status = ?,
                reason = ?,
                mechanism_raw = ?,
                residual_terms = ?,
                prompt_version = ?
            WHERE id = ?
        """, (
            r["mechanism"],
            r["has_mechanism"],
            r["problem"],
            r["constraint_text"],
            r["status"],
            r["reason"],
            r["mechanism_raw"],
            json.dumps(
                r["residual_terms"],
                ensure_ascii=False,
            ),
            r["prompt_version"],
            r["id"],
        ))

    conn.commit()
    conn.close()


# ============================================================
# WORKER
# ============================================================

def process_key_slice(key_index, api_key, rows):
    """
    Process one equal slice assigned to one API key.
    """

    batches = list(
        make_batches(
            rows,
            BATCH_SIZE,
        )
    )

    print(
        f"[KEY {key_index + 1}] "
        f"{len(rows)} patents -> "
        f"{len(batches)} batches "
        f"using {WORKERS_PER_KEY} workers",
        flush=True,
    )

    completed = 0

    # Each worker handles whole batches.
    with ThreadPoolExecutor(
        max_workers=WORKERS_PER_KEY
    ) as executor:

        futures = {}

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):
            future = executor.submit(
                process_batch,
                batch,
                api_key,
                f"K{key_index + 1}-{batch_number}",
            )

            futures[future] = (
                batch_number,
                batch,
            )

        for future in as_completed(futures):
            batch_number, batch = futures[future]

            # Because process_batch retries indefinitely,
            # this should only raise on an actual programming
            # error, not a temporary gateway error.
            results = future.result()

            save_results(results)

            completed += len(results)

            mechanisms = sum(
                1
                for r in results
                if r["has_mechanism"]
            )

            print(
                f"[KEY {key_index + 1}] "
                f"{completed}/{len(rows)} "
                f"saved | batch {batch_number} | "
                f"MECH {mechanisms}/{len(results)}",
                flush=True,
            )

    return completed


# ============================================================
# REPORT
# ============================================================

def report():
    conn = connect_db()
    cur = conn.cursor()

    total = cur.execute("""
        SELECT COUNT(*)
        FROM items
        WHERE id LIKE 'patent:%'
    """).fetchone()[0]

    complete = cur.execute("""
        SELECT COUNT(*)
        FROM items
        WHERE id LIKE 'patent:%'
          AND status = 'complete'
    """).fetchone()[0]

    none = cur.execute("""
        SELECT COUNT(*)
        FROM items
        WHERE id LIKE 'patent:%'
          AND status = 'none'
    """).fetchone()[0]

    pending = cur.execute("""
        SELECT COUNT(*)
        FROM items
        WHERE id LIKE 'patent:%'
          AND (
              status IS NULL
              OR status NOT IN ('complete', 'none')
          )
    """).fetchone()[0]

    print()
    print("=" * 60)
    print("PATENT PHASE 1 REPORT")
    print("=" * 60)
    print(f"Total patents : {total}")
    print(f"Mechanisms    : {complete}")
    print(f"No mechanism  : {none}")
    print(f"Pending       : {pending}")
    print("=" * 60)

    conn.close()


# ============================================================
# RESET
# ============================================================

def reset():
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
        UPDATE items
        SET
            mechanism = NULL,
            has_mechanism = NULL,
            problem = NULL,
            constraint_text = NULL,
            status = NULL,
            reason = NULL,
            mechanism_raw = NULL,
            residual_terms = NULL,
            prompt_version = NULL
        WHERE id LIKE 'patent:%'
    """)

    conn.commit()
    conn.close()

    print("Reset all patent mechanism results.")


# ============================================================
# MAIN
# ============================================================

def main():
    command = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "run"
    )

    ensure_schema()

    if command == "report":
        report()
        return

    if command == "reset":
        reset()
        return

    keys = get_api_keys()

    rows = load_items()

    if not rows:
        print("No pending patents.")
        report()
        return

    print(
        f"Processing {len(rows)} patents "
        f"with {len(keys)} API keys",
        flush=True,
    )

    print(
        f"BATCH_SIZE = {BATCH_SIZE}",
        flush=True,
    )

    print(
        f"WORKERS_PER_KEY = {WORKERS_PER_KEY}",
        flush=True,
    )

    print(
        f"Maximum concurrent requests = "
        f"{len(keys) * WORKERS_PER_KEY}",
        flush=True,
    )

    # --------------------------------------------------------
    # Equal allocation
    # --------------------------------------------------------

    slices = split_equally(
        rows,
        len(keys),
    )

    for i, slice_rows in enumerate(slices):
        print(
            f"  KEY {i + 1}: "
            f"{len(slice_rows)} patents",
            flush=True,
        )

    print()

    # --------------------------------------------------------
    # One executor per API key.
    #
    # Each key gets its own fixed slice.
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=len(keys)
    ) as executor:

        futures = {}

        for i, (key, slice_rows) in enumerate(
            zip(keys, slices)
        ):
            future = executor.submit(
                process_key_slice,
                i,
                key,
                slice_rows,
            )

            futures[future] = i

        for future in as_completed(futures):
            key_index = futures[future]

            try:
                count = future.result()

                print(
                    f"[KEY {key_index + 1}] "
                    f"FINISHED: {count} patents",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"[KEY {key_index + 1}] "
                    f"FATAL WORKER ERROR: {e}",
                    flush=True,
                )

    print()
    report()


if __name__ == "__main__":
    main()