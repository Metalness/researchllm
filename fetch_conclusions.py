"""
Download each paper's open-access PDF and pull out the conclusion section.

    pip install pypdf requests fonttools

    python fetch_conclusions.py 20       # test on 20 papers
    python fetch_conclusions.py          # everything not done yet
    python fetch_conclusions.py report   # stats only
    python fetch_conclusions.py reset    # reset all conclusions/statuses

Fills items.conclusion and items.conclusion_status:

    ok | no_section | download_fail | not_pdf | too_big | parse_fail

Safe to stop and re-run.
Publishers often block scripted downloads, so
expect a real share of download_fail; that is normal.
"""

import io
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

import requests
from pypdf import PdfReader


WORKERS = 4
MAX_BYTES = 15_000_000
MAX_PAGES = 40
MAX_CHARS = 2500
MIN_CHARS = 200

UA = {"User-Agent": "Mozilla/5.0 (research-prototype)"}

PRIMARY = re.compile(
    r"(?im)^[ \t]*(?:\d+(?:\.\d+)*[ \t]*|[IVX]+[ \t]*)?"
    r"(conclusions?(?:[ \t]+and[ \t]+(?:future[ \t]+work|outlook|discussion))?"
    r"|concluding[ \t]+remarks|summary[ \t]+and[ \t]+conclusions?)[ \t]*:?[ \t]*$"
)

FALLBACK = re.compile(
    r"(?im)^[ \t]*(?:\d+(?:\.\d+)*[ \t]*|[IVX]+[ \t]*)?"
    r"(discussion(?:[ \t]+and[ \t]+conclusions?)?)[ \t]*:?[ \t]*$"
)

END = re.compile(
    r"(?im)^[ \t]*(?:\d+(?:\.\d+)*[ \t]*)?"
    r"(references|bibliography|acknowledge?ments?|"
    r"appendix|appendices|funding|author contributions?|"
    r"conflicts? of interest|data availability|declaration)\b"
)


def extract_conclusion(text):
    """Return the conclusion (or discussion) section text, or None."""

    for pattern in (PRIMARY, FALLBACK):
        matches = list(pattern.finditer(text))

        # Last match wins: skips a table-of-contents hit near the top.
        for m in reversed(matches):
            start = m.end()

            end_m = END.search(text, start)
            end = end_m.start() if end_m else len(text)

            section = re.sub(r"\s+", " ", text[start:end]).strip()[:MAX_CHARS]

            if len(section) >= MIN_CHARS:
                return section

    return None


def process(row):
    item_id, url = row

    try:
        r = requests.get(
            url,
            headers=UA,
            timeout=45,
            stream=True
        )

        if r.status_code != 200:
            return item_id, None, "download_fail"

        data = b""

        for chunk in r.iter_content(65536):
            data += chunk

            if len(data) > MAX_BYTES:
                return item_id, None, "too_big"

    except requests.RequestException:
        return item_id, None, "download_fail"

    if not data.startswith(b"%PDF"):
        return item_id, None, "not_pdf"

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = reader.pages[:MAX_PAGES]

        text = "\n".join(
            (p.extract_text() or "")
            for p in pages
        )

    except Exception:
        return item_id, None, "parse_fail"

    section = extract_conclusion(text)

    return item_id, section, ("ok" if section else "no_section")


def report(db):
    rows = db.execute(
        "SELECT conclusion_status, COUNT(*) "
        "FROM items "
        "GROUP BY conclusion_status"
    ).fetchall()

    print("conclusion_status:", dict(rows))


def reset(db):
    print("Resetting all conclusion data...")

    db.execute(
        "UPDATE items "
        "SET conclusion = NULL, conclusion_status = NULL"
    )

    db.commit()

    print("Done. All conclusion and conclusion_status values cleared.")


def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else None

    db = sqlite3.connect("mechanisms.db")

    # Reset everything.
    if arg == "reset":
        reset(db)
        db.close()
        return

    # Report only.
    if arg == "report":
        report(db)
        db.close()
        return

    # Fetch unfinished items.
    todo = db.execute(
        "SELECT id, pdf_url "
        "FROM items "
        "WHERE pdf_url IS NOT NULL "
        "AND conclusion_status IS NULL"
    ).fetchall()

    # Optional test limit.
    if arg and arg.isdigit():
        todo = todo[:int(arg)]

    print(f"Fetching {len(todo)} PDFs with {WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for n, (item_id, section, status) in enumerate(
            pool.map(process, todo),
            1
        ):
            db.execute(
                "UPDATE items "
                "SET conclusion=?, conclusion_status=? "
                "WHERE id=?",
                (section, status, item_id)
            )

            db.commit()

            print(
                f"  {n}/{len(todo)} {status}",
                flush=True
            )

    report(db)
    db.close()


if __name__ == "__main__":
    main()
