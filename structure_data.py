import sqlite3
import json

DB_PATH = "patents.db"
OUTPUT_PATH = "structured_mechanisms.json"

# -----------------------------
# 1. Connect to Phase 1 database
# -----------------------------

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Only export successfully processed patents
cursor.execute("""
    SELECT
        id,
        field,
        title,
        abstract,
        problem,
        constraint_text,
        mechanism_raw,
        mechanism,
        residual_terms,
        has_mechanism,
        status,
        reason,
        prompt_version
    FROM items
    WHERE status IN ('complete', 'none')
    ORDER BY id
""")

rows = cursor.fetchall()

columns = [description[0] for description in cursor.description]

# -----------------------------
# 2. Convert into structured data
# -----------------------------

structured_data = []

for row in rows:

    data = dict(zip(columns, row))

    structured_mechanism = {
        "patent_id": data["id"],

        "domain": data["field"],

        "title": data["title"],

        "abstract": data["abstract"],

        "problem": data["problem"],

        "constraint": data["constraint_text"],

        "mechanism": data["mechanism"],

        "mechanism_raw": data["mechanism_raw"],

        "residual_terms": data["residual_terms"],

        "has_mechanism": data["has_mechanism"],

        "status": data["status"],

        "reason": data["reason"],

        "prompt_version": data["prompt_version"]
    }

    structured_data.append(structured_mechanism)


# -----------------------------
# 3. Save structured dataset
# -----------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        structured_data,
        f,
        indent=2,
        ensure_ascii=False
    )


# -----------------------------
# 4. Statistics
# -----------------------------

total = len(structured_data)
with_mechanism = sum(
    1 for x in structured_data
    if x["has_mechanism"] in (1, True, "true", "True")
)

without_mechanism = total - with_mechanism

print("Conversion complete.")
print("Total processed patents:", total)
print("Patents with mechanisms:", with_mechanism)
print("Patents without mechanisms:", without_mechanism)
print("Output file:", OUTPUT_PATH)

# -----------------------------
# 5. Show example
# -----------------------------

if structured_data:

    print("\nExample:")
    print(
        json.dumps(
            structured_data[0],
            indent=2,
            ensure_ascii=False
        )
    )

conn.close()