"""
Load granted-patent abstracts into patents.db, sampled equally per CPC
technology class (so one busy area like electronics can't swamp the rest).

Download these PatentsView tables first (.zip or .tsv both work, no unzipping):
    g_patent_abstract   (required, ~1.5 GiB zipped)
    g_cpc_current       (required, ~0.5 GiB zipped)
    g_patent            (optional, ~0.2 GiB zipped; only used to add titles)

Usage:
    python load_patents.py g_patent_abstract.tsv.zip g_cpc_current.tsv.zip [g_patent.tsv.zip] [per_class]

    per_class = patents to keep per CPC class (default 40; about 130 classes
    exist, so ~5,000 patents total). Try a small number first, e.g. 3.

The files are huge, so everything is streamed row by row; memory use stays small.
Takes a few minutes per pass over the big files.

UNVERIFIED: column names. The table names and sizes were confirmed from the
PatentsView/USPTO pages, but the columns are from memory. The candidate lists
below are tried in order; if none match, the script prints the real header
so you can add the right name. Check the data dictionary (ODP page sidebar).
"""
import csv
import io
import random
import sqlite3
import sys
import zipfile
from collections import defaultdict

csv.field_size_limit(2 ** 30)

ID_NAMES = ["patent_id"]
ABSTRACT_NAMES = ["patent_abstract", "abstract"]
TITLE_NAMES = ["patent_title", "title"]
CLASS_NAMES = ["cpc_class"]
SECTION_NAMES = ["cpc_section"]
SEQ_NAMES = ["cpc_sequence"]

USE_SEQUENCE = True      # only each patent's first CPC row (its main class)
POOL_FACTOR = 4          # sample 4x per class, since some abstracts are too short
MIN_CHARS = 150
SEED = 42

SECTION_LABEL = {
    "A": "Human necessities", "B": "Operations and transport",
    "C": "Chemistry and metallurgy", "D": "Textiles and paper",
    "E": "Fixed constructions", "F": "Mechanical engineering, heating, weapons",
    "G": "Physics", "H": "Electricity", "Y": "Emerging cross-sectional tech",
}


def open_tsv(path):
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        name = z.namelist()[0]
        return io.TextIOWrapper(z.open(name), encoding="utf-8",
                                errors="replace", newline="")
    return open(path, encoding="utf-8", errors="replace", newline="")


def reader(path):
    r = csv.reader(open_tsv(path), delimiter="\t")
    return next(r), r


def col(header, names, what, path, required=True):
    lower = [h.strip().strip('"').lower() for h in header]
    for n in names:
        if n in lower:
            return lower.index(n)
    if required:
        sys.exit(f"\nNo {what} column found in {path}.\nColumns present: {header}\n"
                 f"Add the right name to the candidate list at the top of load_patents.py.")
    return None


def pass_cpc(path, per_class):
    """Reservoir-sample patent ids per CPC class without loading the file."""
    header, r = reader(path)
    i_id = col(header, ID_NAMES, "patent id", path)
    i_cls = col(header, CLASS_NAMES, "CPC class", path)
    i_sec = col(header, SECTION_NAMES, "CPC section", path, required=False)
    i_seq = col(header, SEQ_NAMES, "CPC sequence", path, required=False) if USE_SEQUENCE else None

    pool_size = per_class * POOL_FACTOR
    pool, seen = defaultdict(list), defaultdict(int)
    rng = random.Random(SEED)
    need = max(x for x in (i_id, i_cls, i_sec, i_seq) if x is not None)

    for n, row in enumerate(r, 1):
        if n % 5_000_000 == 0:
            print(f"    ...{n:,} CPC rows", flush=True)
        if len(row) <= need:
            continue
        if i_seq is not None and row[i_seq].strip() != "0":
            continue
        pid, cls = row[i_id].strip(), row[i_cls].strip()
        if not pid or not cls:
            continue
        sec = row[i_sec].strip() if i_sec is not None else cls[:1]
        key = cls if cls[:1] == sec else sec + cls
        seen[key] += 1
        lst = pool[key]
        if len(lst) < pool_size:
            lst.append(pid)
        else:
            j = rng.randrange(seen[key])
            if j < pool_size:
                lst[j] = pid
    if not pool:
        sys.exit("No CPC rows matched. If the sequence column starts at 1, set "
                 "USE_SEQUENCE = False at the top of the script and retry.")
    return pool


def pass_text(path, wanted, names, what, min_chars=0):
    """Stream a table and keep one text column for the wanted patent ids."""
    header, r = reader(path)
    i_id = col(header, ID_NAMES, "patent id", path)
    i_txt = col(header, names, what, path)
    out = {}
    for n, row in enumerate(r, 1):
        if n % 2_000_000 == 0:
            print(f"    ...{n:,} rows", flush=True)
        if len(row) <= max(i_id, i_txt):
            continue
        pid = row[i_id].strip()
        if pid in wanted:
            text = " ".join(row[i_txt].split())
            if len(text) >= min_chars:
                out[pid] = text
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.isdigit()]
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    if len(args) < 2:
        sys.exit(__doc__)
    abs_path, cpc_path = args[0], args[1]
    title_path = args[2] if len(args) > 2 else None
    per_class = int(nums[0]) if nums else 40

    print("Pass 1: sampling patent ids per CPC class...")
    pool = pass_cpc(cpc_path, per_class)
    id2cls = {}
    for key, ids in pool.items():
        for pid in ids:
            id2cls.setdefault(pid, key)
    print(f"  {len(pool)} classes, {len(id2cls):,} candidate patents")

    print("Pass 2: reading abstracts...")
    abstracts = pass_text(abs_path, id2cls, ABSTRACT_NAMES, "abstract", MIN_CHARS)
    print(f"  {len(abstracts):,} abstracts long enough")

    titles = {}
    if title_path:
        print("Pass 3: reading titles...")
        titles = pass_text(title_path, set(abstracts), TITLE_NAMES, "title")

    db = sqlite3.connect("patents.db")
    db.execute("""CREATE TABLE IF NOT EXISTS items (
        id TEXT PRIMARY KEY, field TEXT, title TEXT,
        abstract TEXT, mechanism TEXT)""")
    have = {r[1] for r in db.execute("PRAGMA table_info(items)")}
    for c in ("pdf_url", "conclusion", "conclusion_status"):
        if c not in have:
            db.execute(f"ALTER TABLE items ADD COLUMN {c} TEXT")

    saved = 0
    for key, ids in sorted(pool.items()):
        label = f"Patent: {SECTION_LABEL.get(key[:1], key[:1])} ({key})"
        kept = 0
        for pid in ids:
            if kept >= per_class:
                break
            if id2cls.get(pid) == key and pid in abstracts:
                db.execute(
                    "INSERT OR IGNORE INTO items (id, field, title, abstract) "
                    "VALUES (?,?,?,?)",
                    (f"patent:{pid}", label, titles.get(pid, ""), abstracts[pid]))
                kept += 1
        saved += kept
    db.commit()
    print(f"Done. Saved {saved:,} patents into patents.db")


if __name__ == "__main__":
    main()