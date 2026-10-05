"""
LLM test on 40 papers from mechanisms.db, using your Kilo key from .env

.env (same folder, one per line):
    KILO_API_KEY=your_key
    KILO_MODEL=exact-model-id

    python compare_40.py            # 20 papers per group, seed 1
    python compare_40.py 20 1       # n per group, seed

It re-creates the same random draw as compare_db.py (same seed, same query) so
you get the SAME 40 papers as in compare_sample.txt, provided mechanisms.db has
not changed since. Runs:
    WITH group    (20): abstract only (A)  AND  abstract + conclusion (B)
    WITHOUT group (20): abstract only (A)
= 60 API calls. Nothing is written to the database.
Results go to compare_40_results.txt.
"""
import os
import random
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


load_env()   # must happen BEFORE importing kilo_gateway (it reads the key on import)
if not os.environ.get("KILO_API_KEY"):
    sys.exit("No KILO_API_KEY found in .env or the environment.")
if not os.environ.get("KILO_MODEL"):
    sys.exit("No KILO_MODEL found in .env (find it with: python kilo_gateway.py models nemotron).")
from kilo_gateway import chat_json

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
WORKERS = 2          # the provider has been overloaded before; raise to 4 if stable
MAX_CHARS = 1500

SYSTEM = """You convert research abstracts into domain-free mechanism descriptions.
The goal is a stored "how it works" idea that someone in a completely
different field could read and recognize.

Decide first: does the text describe HOW something works or HOW something
is done? Answer true if it presents or explains ANY of: a process, a
structure, a feedback loop, a causal chain, a design principle, or a new
method, material, device, algorithm, or technique together with the reason
it works. Answer false ONLY if it purely measures, surveys, counts,
describes a population, or reports statistics with no method or principle
behind them, or if it is junk (web-page boilerplate, no real abstract).
Do not invent details the text does not state, but do extract the
underlying principle when the text gives enough to infer it.

If it does, write:
- "problem": one sentence, what difficulty or goal this addresses.
- "mechanism": 2-3 sentences, present tense. Say what acts on what, what
  gets amplified, suppressed, stored or selected, and why it works.
- "constraint": one sentence, what limits it or when it fails
  (write "unknown" if the text does not say).

Rules for the wording:
- No field-specific nouns. Replace them with roles: "a medium", "a signal",
  "a stock", "agents", "a layered structure", "a waste by-product",
  "a population".
- Keep the shape words: periodic, threshold, delay, feedback, nested,
  proportional, irreversible, minimum, and so on. These carry the idea.
- Do not name chemicals, organisms, places, instruments or methods.
- Ignore patent phrasing (wherein, said, comprising); describe only how it works.
- The text may be in any language. Always answer in English.
- Test before answering: could a person in an unrelated field read this and
  recognize it in their own work? If not, rewrite it more abstractly.

The input may include a CONCLUSION section. Prefer explanations of why
something works found there, but still describe only the idea, not the field.

Return ONLY JSON:
{"has_mechanism": true/false, "reason_if_false": "...",
 "problem": "...", "mechanism": "...", "constraint": "..."}

Example 1 (input: UV reflections in a penguin beak with crystal-like periodic structure)
{"has_mechanism": true, "reason_if_false": "",
 "problem": "A signal must be reflected strongly at one chosen wavelength without using any dye.",
 "mechanism": "A stack of thin layers with a regular repeating spacing makes the reflections from each boundary reinforce each other, but only for waves whose length matches that spacing. Changing the spacing changes which wavelength is selected, so the structure itself acts as a tunable filter.",
 "constraint": "Only works while the spacing stays regular; irregular spacing blurs the selection."}

Example 2 (input: logistic regression study of accident causes among surveyed mine workers)
{"has_mechanism": false, "reason_if_false": "statistical survey, no mechanism described",
 "problem": "", "mechanism": "", "constraint": ""}"""


def run(task):
    key, title, abstract, conclusion = task
    user = f"TITLE: {title}\nABSTRACT: {abstract[:MAX_CHARS]}"
    if conclusion:
        user += f"\nCONCLUSION: {conclusion[:1500]}"
    try:
        data = chat_json(SYSTEM, user, temperature=0.1, max_tokens=1500)
    except Exception as e:
        print(f"  FAIL {key}: {str(e)[:100]}", flush=True)
        return key, None
    tag = "?" if data is None else ("MECH" if data.get("has_mechanism") else "none")
    print(f"  {key[1]:2s} {tag:5s} {title[:60]!r}", flush=True)
    return key, data


def yes(d):
    return bool(d and d.get("has_mechanism"))


def main():
    db = sqlite3.connect("mechanisms.db")
    db.row_factory = sqlite3.Row
    rng = random.Random(SEED)
    with_rows = db.execute("SELECT * FROM items WHERE pdf_url IS NOT NULL "
                           "AND conclusion_status = 'ok'").fetchall()
    out_rows = db.execute("SELECT * FROM items WHERE pdf_url IS NOT NULL "
                          "AND conclusion_status IS NOT NULL "
                          "AND conclusion_status != 'ok'").fetchall()
    with_s = rng.sample(with_rows, min(N, len(with_rows)))
    out_s = rng.sample(out_rows, min(N, len(out_rows)))

    tasks = []
    for r in with_s:
        tasks.append(((r["id"], "A"), r["title"], r["abstract"], None))
        tasks.append(((r["id"], "B"), r["title"], r["abstract"], r["conclusion"]))
    for r in out_s:
        tasks.append(((r["id"], "A"), r["title"], r["abstract"], None))
    print(f"{len(with_s)} WITH + {len(out_s)} WITHOUT papers, {len(tasks)} calls, "
          f"{WORKERS} workers\n")

    with ThreadPoolExecutor(WORKERS) as pool:
        res = dict(pool.map(run, tasks))

    # ---------- numbers ----------
    ok_pairs = [(r, res[(r["id"], "A")], res[(r["id"], "B")]) for r in with_s
                if res.get((r["id"], "A")) is not None and res.get((r["id"], "B")) is not None]
    ok_out = [(r, res[(r["id"], "A")]) for r in out_s if res.get((r["id"], "A")) is not None]
    bad = len(tasks) - sum(v is not None for v in res.values())

    n = len(ok_pairs)
    a = sum(yes(x) for _, x, _ in ok_pairs)
    b = sum(yes(y) for _, _, y in ok_pairs)
    gained = [(r, x, y) for r, x, y in ok_pairs if yes(y) and not yes(x)]
    lost = [(r, x, y) for r, x, y in ok_pairs if yes(x) and not yes(y)]
    o = sum(yes(x) for _, x in ok_out)

    lines = []
    P = lines.append
    P("=" * 70)
    P(f"Failed or unparseable calls: {bad} of {len(tasks)}")
    P(f"\nSAME {n} PAPERS, with a conclusion available:")
    P(f"  abstract only        : {a}/{n} have a mechanism ({100*a/max(n,1):.0f}%)")
    P(f"  abstract + conclusion: {b}/{n} have a mechanism ({100*b/max(n,1):.0f}%)")
    P(f"  gained with conclusion: {len(gained)}   lost: {len(lost)}")
    P(f"\nPAPERS WITHOUT a conclusion ({len(ok_out)}), abstract only:")
    P(f"  {o}/{len(ok_out)} have a mechanism ({100*o/max(len(ok_out),1):.0f}%)")
    P("\nNOTE: with ~20 papers per group, a difference of 1-3 papers is noise.")
    P("Read the examples below; they matter more than the percentages.")

    def show(r, d, label):
        P(f"\n[{r['field']}] {r['title'][:100]}")
        if d and d.get("has_mechanism"):
            P(f"  {label} mechanism: {d.get('mechanism', '')}")
        elif d:
            P(f"  {label} none: {str(d.get('reason_if_false', ''))[:120]}")

    P("\n" + "#" * 70 + "\nGAINED with conclusion (A said no, B found a mechanism)\n" + "#" * 70)
    for r, x, y in gained:
        show(r, x, "A"); show(r, y, "B")
    P("\n" + "#" * 70 + "\nLOST with conclusion (A found one, B did not)\n" + "#" * 70)
    for r, x, y in lost:
        show(r, x, "A"); show(r, y, "B")
    P("\n" + "#" * 70 + "\nBOTH found one: does the conclusion change the wording?\n" + "#" * 70)
    for r, x, y in [t for t in ok_pairs if yes(t[1]) and yes(t[2])][:6]:
        show(r, x, "A"); show(r, y, "B")
    P("\n" + "#" * 70 + "\nWITHOUT-conclusion papers that had a mechanism\n" + "#" * 70)
    for r, x in [t for t in ok_out if yes(t[1])]:
        show(r, x, "A")

    text = "\n".join(lines)
    open("compare_40_results.txt", "w", encoding="utf-8").write(text)
    print("\n" + "\n".join(lines[:12]))
    print("\nFull results with all mechanisms: compare_40_results.txt")


if __name__ == "__main__":
    main()