"""
Sample open-access papers equally from every OpenAlex field into SQLite,
saving the abstract AND a PDF link, so fetch_conclusions.py can pull the
conclusion section later.

Usage:
    export OPENALEX_API_KEY=your_key     # free at openalex.org/settings/api
    python load_openalex.py 20           # quick test: 20 per field
    python load_openalex.py              # 200 per field

Output: mechanisms.db, table "items". If you already have an old
mechanisms.db, rename it first (e.g. mechanisms_old.db) so this starts clean.

UNVERIFIED (written from memory, not in the docs I read): the /fields
endpoint, filters primary_topic.field.id / has_abstract / is_oa / language,
and the location field pdf_url. Marked CHECK; the script prints API errors.
"""
import math
import os
import sqlite3
import sys
import time

import requests
import dotenv
dotenv.load_dotenv()
BASE = "https://api.openalex.org"
KEY = os.getenv("OPENALEX_API_KEY")
QUOTA = int(sys.argv[1]) if len(sys.argv) > 1 else 200
MIN_CHARS = 300
SEED = 42


def get(path, params):
    params = dict(params)
    if KEY:
        params["api_key"] = KEY
    for attempt in range(5):
        r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500):
            time.sleep(2 ** attempt)
            continue
        print("API error:", r.status_code, r.text[:300])
        r.raise_for_status()
    raise RuntimeError("too many retries")


def rebuild_abstract(inv):
    if not inv:
        return ""
    words = {}
    for word, positions in inv.items():
        for p in positions:
            words[p] = word
    return " ".join(words[i] for i in sorted(words))


def list_fields():
    data = get("fields", {"per_page": 100})  # CHECK
    return [(f["id"], f["display_name"]) for f in data["results"]]


def sample_field(field_id, quota):
    """Over-fetch 3x: many works lack a usable abstract or a direct PDF link."""
    want = quota * 3
    rows = []
    for page in range(1, math.ceil(want / 100) + 1):
        data = get("works", {
            # CHECK all filter names
            "filter": (f"primary_topic.field.id:{field_id},has_abstract:true,"
                       "type:article,is_oa:true,language:en"),
            "sample": want, "seed": SEED, "per_page": 100, "page": page,
            "select": "id,title,abstract_inverted_index,best_oa_location",
        })
        if not data["results"]:
            break
        for w in data["results"]:
            text = rebuild_abstract(w.get("abstract_inverted_index"))
            loc = w.get("best_oa_location") or {}
            pdf = loc.get("pdf_url")  # CHECK
            if len(text) >= MIN_CHARS and w.get("title") and pdf:
                rows.append((w["id"], w["title"], text, pdf))
    return rows[:quota]


def main():
    db = sqlite3.connect("mechanisms.db")
    db.execute("""CREATE TABLE IF NOT EXISTS items (
        id TEXT PRIMARY KEY, field TEXT, title TEXT,
        abstract TEXT, mechanism TEXT)""")
    cols = {r[1] for r in db.execute("PRAGMA table_info(items)")}
    for col in ("pdf_url", "conclusion", "conclusion_status"):
        if col not in cols:
            db.execute(f"ALTER TABLE items ADD COLUMN {col} TEXT")

    fields = list_fields()
    print(f"{len(fields)} fields found, {QUOTA} papers each")
    for field_id, name in fields:
        rows = sample_field(field_id, QUOTA)
        db.executemany(
            "INSERT OR IGNORE INTO items (id, field, title, abstract, pdf_url) "
            "VALUES (?,?,?,?,?)",
            [(i, name, t, a, p) for i, t, a, p in rows])
        db.commit()
        print(f"  {name}: saved {len(rows)}")
    print("Done. Next: python fetch_conclusions.py 20")


if __name__ == "__main__":
    main()