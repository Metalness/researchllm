"""
Cross-domain solver: top-20 mechanisms -> ONE request -> a solution.

Default cost per problem: 2 LLM requests
    1. analyze the problem (problem, constraints, 4-6 required capabilities)
    2. solve: the 20 retrieved mechanisms go in; judgments + the design come out in ONE response
Retrieval itself uses no LLM (local embeddings + FAISS).

Usage (same folder as kilo_gateway.py, .env and phase2b_embeddings/):
    python cross_domain.py --problem "your problem text"
    python cross_domain.py --problems problems.txt
    python cross_domain.py --problems problems.txt --ablate    # +2 requests per problem
    python cross_domain.py --problems problems.txt --judge     # +1 request per problem (weak)

--ablate answers "does retrieval change the designs?" by also solving each problem with
    A = no mechanisms, B = 20 RANDOM mechanisms   (C = the retrieved 20 always runs)
and writes ablation_runs/blind_packet.md (designs shuffled as X/Y/Z, no provenance) plus
blind_key.json. Rank the designs yourself BEFORE opening the key.

Rate limits: requests are sequential with a pause (--delay, default 3s). If the provider
returns 429 the gateway waits 20s, 40s, ...; the run stops after 2 failed problems in a row
and keeps everything finished so far.

Retrieval design: one query per required capability plus one for the whole problem, results
interleaved; max 2 per CPC class and 6 per section (loosened step by step only if the corpus
is too narrow); near-duplicate mechanisms dropped. The model sees opaque shuffled IDs
(M01...) and never the score, domain, title or raw text.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

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


load_env()   # before importing kilo_gateway (it reads the key on import)
if not os.environ.get("KILO_API_KEY") or not os.environ.get("KILO_MODEL"):
    sys.exit("Set KILO_API_KEY and KILO_MODEL in .env first.")
from kilo_gateway import chat_json  # noqa: E402

# ------------------------------------------------------------------ config
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "phase2b_embeddings" / "mechanism_residual.faiss"
METADATA_PATH = BASE_DIR / "phase2b_embeddings" / "metadata.json"
OUT_DIR = BASE_DIR / "ablation_runs"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MAX_LENGTH = 2048

TOP_K = 20             # mechanisms shown to the model
SEARCH_K = 150         # candidates fetched per query before diversification
PER_CLASS_CAP = 2      # max results from one CPC class (e.g. "G05")
PER_SECTION_CAP = 6    # max results from one CPC section letter (e.g. "G")
DUP_JACCARD = 0.40     # mechanism word overlap above this = near-duplicate
TEMPERATURE = 0.2
DELAY = 3.0            # seconds before every request (overridden by --delay)
SOLVE_MAX_TOKENS = 12000   # overridden by --max-tokens

STOP = set("that this with from into when then than they them their have been being "
           "while which would could should about using used based each other more most "
           "also only over under after before between within without such these those "
           "where what will can may not and the for are but its it".split())


# ------------------------------------------------------------------ helpers
def clean(v):
    return "" if v is None else str(v).strip()


def call(system, user, max_tokens, what):
    """One request. The gateway already retries transport errors, so exceptions propagate.
    Only invalid JSON is retried (once)."""
    for attempt in range(2):
        time.sleep(DELAY)
        data = chat_json(system, user, temperature=TEMPERATURE, max_tokens=max_tokens)
        if data is not None:
            return data
        print(f"  [{what}] model did not return valid JSON; "
              f"{'retrying once' if attempt == 0 else 'giving up'}", flush=True)
    raise RuntimeError(f"{what}: no valid JSON")


def tokens(text):
    return {w for w in re.findall(r"[a-z]{4,}", clean(text).lower()) if w not in STOP}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        md = json.load(f)
    if isinstance(md, list):
        return md
    if isinstance(md, dict):
        for key in ("records", "metadata"):
            if isinstance(md.get(key), list):
                return md[key]
        if all(isinstance(v, dict) for v in md.values()):
            return list(md.values())
    raise RuntimeError("Could not determine metadata format.")


def rec_get(rec, *names, default=""):
    for n in names:
        if rec.get(n) not in (None, ""):
            return rec[n]
    return default


CLASS_RE = re.compile(r"\(([A-HY]\d{2})\)")


def class_of(rec):
    dom = clean(rec_get(rec, "domain", "field"))
    m = CLASS_RE.search(dom)
    cls = m.group(1) if m else (dom or "unknown")
    return cls, cls[:1]


# ------------------------------------------------------------------ retriever
class FaissRetriever:
    """Real retriever: Qwen3-Embedding-0.6B + your FAISS index. Heavy imports are lazy."""

    def __init__(self):
        import faiss
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
        for p in (INDEX_PATH, METADATA_PATH):
            if not p.exists():
                raise FileNotFoundError(p)
        self.torch, self.F = torch, F
        self.index = faiss.read_index(str(INDEX_PATH))
        self.records = load_metadata()
        if len(self.records) < self.index.ntotal:
            raise RuntimeError("Metadata has fewer records than the FAISS index.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Index vectors: {self.index.ntotal}, dim {self.index.d}, device {self.device}")
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model.to(self.device).eval()

    def _pool(self, hidden, mask):
        if mask[:, -1].sum() == mask.shape[0]:
            return hidden[:, -1]
        lengths = mask.sum(dim=1) - 1
        return hidden[self.torch.arange(hidden.shape[0], device=hidden.device), lengths]

    def embed(self, text):
        enc = self.tok([text], padding=True, truncation=True,
                       max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model(**enc)
            emb = self.F.normalize(self._pool(out.last_hidden_state, enc["attention_mask"]),
                                   p=2, dim=1)
        vec = emb.float().cpu().numpy().astype(np.float32)   # bfloat16 -> float32 first
        if vec.shape[1] != self.index.d:
            raise RuntimeError(f"Dimension mismatch: query {vec.shape[1]}, index {self.index.d}")
        return vec

    def search(self, vec, k):
        return self.index.search(vec, min(k, self.index.ntotal))


def load_retriever():
    return FaissRetriever()


# ------------------------------------------------------------------ prompts
PASS1_SYSTEM = r"""
You are the first stage of a cross-domain inventive problem-solving system.
The user provides a NEW problem. Your job is NOT to solve it.

Do NOT propose sensors, algorithms, hardware, software, architectures, AI, machine
learning, SLAM, sensor fusion, cameras, LiDAR, radar, GPS, IMUs, or any conventional
solution. Decompose the problem into its fundamental functional challenge.

Extract:
1. problem: what must ultimately be achieved, stated functionally (one or two sentences).
2. constraints: what makes it difficult? Preserve interacting environmental, physical,
   operational, reliability, timing, resource and failure constraints.
3. required_capabilities: what must the system be able to do, expressed functionally,
   without naming implementation technologies. Make each capability SELF-CONTAINED (it
   will be searched on its own) and keep them to 4-6 items.

Instead of "use sensor fusion" write "combine partially independent observations whose
reliability can change over time". Instead of "use SLAM" write "maintain a consistent
estimate of location relative to previously encountered structure".

Return ONLY valid JSON:
{"problem": "...", "constraints": "...", "required_capabilities": ["...", "..."]}
"""

SOLVE_SYSTEM = r"""
You are a systems engineer. You get a target problem and up to 20 candidate mechanisms
retrieved from unrelated fields, each with an ID (M01, M02, ...). In ONE response do two things.

1. JUDGE every candidate, briefly. For each give "v": "s" (strong), "m" (moderate),
   "w" (weak) or "n" (none), and "std": true if a competent engineer in the target field
   would already use this idea by default. Do not reward shared nouns, the same industry,
   similar wording, or generic words like feedback, sensing, AI or fusion. Reward a causal
   principle that would still work with the original application removed.
   One entry per candidate, same IDs, same order, no re-ranking, no prose.

2. DESIGN a solution. Candidates are inspiration, not instructions. Use a candidate only if
   it is rated s or m and it materially changes the design; use at most 4 and prefer
   std=false ones. Rules:
   - every part carries an "origin": a candidate ID, or "standard" for ordinary engineering
   - a novelty claim must state exactly which feature differs from a standard design
   - if no candidate materially changes the design, leave borrowed_mechanisms empty
   - no confidence statement
   - concrete enough to prototype: parts, information flow, degraded operation, recovery
   - tests_that_would_show_this_is_wrong: 3-5 concrete experiments or measurements
   Keep every field short and concrete.

Return ONLY valid JSON:
{"judgments": [{"id": "M01", "v": "m", "std": false}],
 "solution_title": "...", "core_idea": "...",
 "parts": [{"component": "...", "role": "...", "origin": "M07 or standard"}],
 "operating_sequence": ["..."],
 "borrowed_mechanisms": [{"id": "M07", "principle": "...", "adaptation": "...",
                          "changes_design_how": "..."}],
 "failure_modes": [{"failure": "...", "response": "..."}],
 "novelty_claims": [{"claim": "...", "differs_from_standard_design_by": "..."}],
 "tests_that_would_show_this_is_wrong": ["..."]}
"""

NO_INSPIRATION = ('\n\nNO CANDIDATE MECHANISMS ARE AVAILABLE FOR THIS RUN. Skip step 1 (judgments '
                  'must be an empty list). Design from ordinary engineering knowledge. Every '
                  'origin must be "standard" and borrowed_mechanisms must be an empty list.')

JUDGE_SYSTEM = r"""
You compare anonymous candidate designs for the same problem. Rank them best to worst by:
which design contains the most USEFUL idea that a competent engineer would NOT list by
default, while still being physically plausible. Ignore length, polish, and confident tone.
A design made only of standard parts ranks low even if it is well written.

Return ONLY valid JSON: {"ranking": ["X", "Z", "Y"], "reason": "one or two sentences"}
"""


# ------------------------------------------------------------------ pipeline steps
def pass1(problem):
    d = call(PASS1_SYSTEM, problem, 4000, "analyze")
    caps = [clean(c) for c in (d.get("required_capabilities") or []) if clean(c)]
    if not caps:
        raise RuntimeError("analysis returned no capabilities")
    return {"problem": clean(d.get("problem")), "constraints": clean(d.get("constraints")),
            "required_capabilities": caps}


def query_text(text):
    # same layout the index was built with; no residual terms (the old overall query had 20)
    return f"Functional mechanism: {text}\nDomain-specific terms: []"


def build_queries(struct):
    queries = [("overall", query_text(struct["problem"]))]
    for i, cap in enumerate(struct["required_capabilities"], 1):
        queries.append((f"capability {i}: {cap[:70]}", query_text(cap)))
    return queries


def retrieve(ret, queries, rng):
    """Interleave results from every query, then apply diversity rules."""
    per_query = []
    for name, text in queries:
        scores, idxs = ret.search(ret.embed(text), SEARCH_K)
        per_query.append((name, list(zip(scores[0], idxs[0]))))

    def run(class_cap, section_cap):
        chosen, seen, cls_n, sec_n, bags = [], set(), Counter(), Counter(), []
        pos = [0] * len(per_query)
        while len(chosen) < TOP_K and any(pos[i] < len(per_query[i][1]) for i in range(len(per_query))):
            for qi, (name, hits) in enumerate(per_query):
                if len(chosen) >= TOP_K:
                    break
                while pos[qi] < len(hits):
                    score, idx = hits[pos[qi]]
                    pos[qi] += 1
                    if idx < 0 or idx >= len(ret.records):
                        continue
                    rec = ret.records[idx]
                    pid = rec_get(rec, "patent_id", "id", default=idx)
                    if pid in seen:
                        continue
                    cls, sec = class_of(rec)
                    if cls_n[cls] >= class_cap or sec_n[sec] >= section_cap:
                        continue
                    bag = tokens(rec_get(rec, "mechanism"))
                    if any(jaccard(bag, b) >= DUP_JACCARD for b in bags):
                        continue
                    seen.add(pid); cls_n[cls] += 1; sec_n[sec] += 1; bags.append(bag)
                    chosen.append({"patent_id": pid, "domain": clean(rec_get(rec, "domain", "field")),
                                   "score": float(score), "matched": name, "rec": rec})
                    break
        return chosen

    # Start strict; if the corpus cannot fill TOP_K under the caps, loosen them one step
    # at a time (never all at once, or one busy class takes over the list).
    for step in range(0, 8):
        class_cap, section_cap = PER_CLASS_CAP + step, PER_SECTION_CAP + 2 * step
        chosen = run(class_cap, section_cap)
        if len(chosen) >= TOP_K:
            break
    if step:
        print(f"  (corpus too narrow for the strict caps; used max {class_cap} per class, "
              f"{section_cap} per section)")
    return label(chosen, rng)


def random_pool(records, rng):
    pool = [r for r in records if clean(rec_get(r, "mechanism"))]
    rows = []
    for rec in rng.sample(pool, min(TOP_K, len(pool))):
        rows.append({"patent_id": rec_get(rec, "patent_id", "id"),
                     "domain": clean(rec_get(rec, "domain", "field")),
                     "score": None, "matched": "random", "rec": rec})
    return label(rows, rng)


def label(rows, rng):
    """Shuffle, then give opaque IDs so position and score cannot leak."""
    rows = list(rows)
    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["id"] = f"M{i:02d}"
        rec = r["rec"]
        r["view"] = {"problem": clean(rec_get(rec, "problem")),
                     "constraint": clean(rec_get(rec, "constraint", "constraint_text")),
                     "mechanism": clean(rec_get(rec, "mechanism"))}
    return rows


def block(item):
    v = item["view"]
    return (f"ID: {item['id']}\nProblem: {v['problem']}\nConstraint: {v['constraint']}\n"
            f"Mechanism: {v['mechanism']}\nRetrieved for: {item['matched']}\n")


def target_text(struct):
    return (f"TARGET PROBLEM:\n{struct['problem']}\n\nTARGET CONSTRAINTS:\n{struct['constraints']}\n\n"
            f"TARGET CAPABILITIES:\n{json.dumps(struct['required_capabilities'], indent=2)}")


def solve(struct, items):
    """ONE request: judge all candidates and design the solution."""
    if items:
        user = target_text(struct) + "\n\nCANDIDATES:\n\n" + "\n".join(block(i) for i in items)
        system = SOLVE_SYSTEM
    else:
        user, system = target_text(struct), SOLVE_SYSTEM + NO_INSPIRATION
    sol = call(system, user, SOLVE_MAX_TOKENS, "solve")
    # provenance sanity checks: the model may cite IDs that were never shown
    valid = {i["id"] for i in items}
    used = {clean(p.get("origin")) for p in sol.get("parts") or []} | \
           {clean(b.get("id")) for b in sol.get("borrowed_mechanisms") or []}
    sol["_invalid_ids"] = sorted(u for u in used if u and u.lower() != "standard" and u not in valid)
    judged = {clean(j.get("id")) for j in sol.get("judgments") or []}
    sol["_unjudged_ids"] = sorted(valid - judged)
    return sol


# ------------------------------------------------------------------ comparison helpers
def design_text(sol, with_origin=False):
    lines = [f"Title: {clean(sol.get('solution_title'))}",
             f"Core idea: {clean(sol.get('core_idea'))}", "Parts:"]
    for p in sol.get("parts") or []:
        tag = f" [{clean(p.get('origin'))}]" if with_origin else ""
        lines.append(f"- {clean(p.get('component'))}: {clean(p.get('role'))}{tag}")
    lines.append("Operating sequence:")
    for i, s in enumerate(sol.get("operating_sequence") or [], 1):
        lines.append(f"{i}. {clean(s)}")
    lines.append("Failure handling:")
    for f in sol.get("failure_modes") or []:
        lines.append(f"- {clean(f.get('failure'))} -> {clean(f.get('response'))}")
    return "\n".join(lines)


def design_bag(sol):
    return tokens(design_text(sol))


def retrieved_share(sol):
    parts = sol.get("parts") or []
    if not parts:
        return 0.0
    return sum(1 for p in parts if clean(p.get("origin")).lower() != "standard") / len(parts)


def slug(text, n=40):
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:n].strip("_") or "problem"


# ------------------------------------------------------------------ main flow
def solve_problem(num, problem, arms, ret, records, seed):
    print("\n" + "=" * 90 + f"\nPROBLEM {num}: {problem[:100]}\n" + "=" * 90)
    rng = random.Random(seed * 1000 + num)
    struct = pass1(problem)
    print(f"  [request 1] analyzed: {len(struct['required_capabilities'])} capabilities")
    run = {"problem": problem, "struct": struct, "arms": {}}

    for arm in ("C", "A", "B"):
        if arm not in arms:
            continue
        queries = []
        if arm == "C":
            queries = build_queries(struct)
            items = retrieve(ret, queries, rng)
            cls = Counter(class_of(i["rec"])[0] for i in items)
            sec = Counter(class_of(i["rec"])[1] for i in items)
            hit = Counter(i["matched"].split(":")[0] for i in items)
            sc = [i["score"] for i in items]
            print(f"  [retrieval, no LLM] {len(items)} mechanisms, {len(cls)} classes, sections "
                  f"{dict(sec)}, scores {min(sc):.2f}-{max(sc):.2f}")
            print(f"                      contributed by query: {dict(hit)}")
        elif arm == "B":
            items = random_pool(records, rng)
        else:
            items = []
        sol = solve(struct, items)
        j = Counter(clean(x.get("v")).lower() for x in sol.get("judgments") or [])
        extra = ""
        if sol["_invalid_ids"]:
            extra += f"  INVALID IDs cited: {sol['_invalid_ids']}"
        if items and sol["_unjudged_ids"]:
            extra += f"  unjudged: {len(sol['_unjudged_ids'])}"
        print(f"  [solve, arm {arm}] judged {dict(j)}; borrowed "
              f"{[b.get('id') for b in sol.get('borrowed_mechanisms') or []]}{extra}")
        run["arms"][arm] = {
            "queries": [q[0] for q in queries],
            "retrieved": [{"id": i["id"], "patent_id": i["patent_id"], "domain": i["domain"],
                           "score": i["score"], "matched": i["matched"], **i["view"]} for i in items],
            "solution": sol}

    bags = {a: design_bag(r["solution"]) for a, r in run["arms"].items()}
    metrics = {"retrieved_share": {a: round(retrieved_share(r["solution"]), 2)
                                   for a, r in run["arms"].items()}}
    for x, y in (("C", "A"), ("B", "A"), ("C", "B")):
        if x in bags and y in bags:
            metrics[f"word_overlap_{x}_vs_{y}"] = round(jaccard(bags[x], bags[y]), 2)
    run["metrics"] = metrics
    if len(bags) > 1:
        print(f"  metrics: {metrics}")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{num:02d}_{slug(problem)}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved {path.name}")
    return run


def write_blind(runs, seed):
    rng = random.Random(seed)
    key, lines = {}, ["# Blind comparison\n",
                      "Rank the designs for each problem BEFORE opening blind_key.json.\n",
                      "Question: which contains the most useful idea you would NOT have listed by default?\n"]
    for n, run in enumerate(runs, 1):
        arms = list(run["arms"])
        if len(arms) < 2:
            continue
        rng.shuffle(arms)
        labels = dict(zip("XYZ", arms))
        key[n] = labels
        lines.append(f"\n---\n## Problem {n}\n{run['problem']}\n")
        for lab, arm in labels.items():
            lines.append(f"### Design {lab}\n```\n{design_text(run['arms'][arm]['solution'])}\n```\n")
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "blind_packet.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "blind_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return key


def llm_judge(runs, key):
    """Weak signal: the same model family judges. Use only to flag, not to conclude."""
    ranks = {}
    for n, run in enumerate(runs, 1):
        if n not in key:
            continue
        labels = key[n]
        user = f"PROBLEM:\n{run['problem']}\n\n" + "\n\n".join(
            f"DESIGN {lab}:\n{design_text(run['arms'][arm]['solution'])}" for lab, arm in labels.items())
        try:
            d = call(JUDGE_SYSTEM, user, 1500, "judge")
        except Exception:
            continue
        order = [clean(x) for x in d.get("ranking", []) if clean(x) in labels]
        for pos, lab in enumerate(order, 1):
            ranks.setdefault(labels[lab], []).append(pos)
    return ranks


def summarize(runs, ranks=None):
    print("\n" + "=" * 90 + "\nSUMMARY ACROSS PROBLEMS\n" + "=" * 90)
    for k in ("word_overlap_C_vs_A", "word_overlap_B_vs_A", "word_overlap_C_vs_B"):
        vals = [r["metrics"][k] for r in runs if k in r["metrics"]]
        if vals:
            print(f"  mean {k}: {np.mean(vals):.2f}  (higher = designs more alike)")
    shares = [r["metrics"]["retrieved_share"].get("C") for r in runs
              if "C" in r["metrics"]["retrieved_share"]]
    if shares:
        print(f"  mean share of arm-C parts that came from retrieved mechanisms: {np.mean(shares):.2f}")
    if ranks:
        print("  LLM judge, mean rank (1 = best):  [weak signal, same model family]")
        for arm, rs in sorted(ranks.items()):
            print(f"    arm {arm}: {np.mean(rs):.2f} over {len(rs)} problems")
    print("\nNow read ablation_runs/blind_packet.md, rank designs yourself, then open blind_key.json.")


def main():
    global DELAY, SOLVE_MAX_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem")
    ap.add_argument("--problems")
    ap.add_argument("--ablate", action="store_true", help="also run arms A (none) and B (random)")
    ap.add_argument("--judge", action="store_true", help="LLM ranks the blind designs (weak signal)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--delay", type=float, default=DELAY, help="seconds before each request")
    ap.add_argument("--max-tokens", type=int, default=SOLVE_MAX_TOKENS)
    args = ap.parse_args()
    DELAY, SOLVE_MAX_TOKENS = args.delay, args.max_tokens
    arms = ["A", "B", "C"] if args.ablate else ["C"]

    if args.problems:
        problems = [p.strip() for p in Path(args.problems).read_text(encoding="utf-8").splitlines()
                    if p.strip() and not p.startswith("#")]
    elif args.problem:
        problems = [args.problem]
    else:
        problems = [input("Problem (no solution included)\n> ").strip()]
    problems = [p for p in problems if p]
    if not problems:
        sys.exit("No problem given.")
    per = 2 + (2 if args.ablate else 0) + (1 if args.judge and args.ablate else 0)
    print(f"{len(problems)} problem(s), about {per} LLM request(s) each = ~{per * len(problems)} total")

    ret = records = None
    if "C" in arms:
        ret = load_retriever()
        records = ret.records
    if "B" in arms and records is None:
        records = load_metadata()

    runs, fails = [], 0
    for n, p in enumerate(problems, 1):
        try:
            runs.append(solve_problem(n, p, arms, ret, records, args.seed))
            fails = 0
        except Exception as e:   # keep finished work; stop if the provider keeps failing
            fails += 1
            print(f"  PROBLEM {n} FAILED: {str(e)[:200]}")
            if fails >= 2:
                print("  Two failures in a row: stopping. Finished problems are saved in ablation_runs/.")
                break
    if not runs:
        sys.exit("All problems failed.")
    if len(arms) > 1:
        key = write_blind(runs, args.seed)
        ranks = llm_judge(runs, key) if args.judge else None
        summarize(runs, ranks)
    else:
        print("\nDone. Designs are in ablation_runs/*.json (key 'solution'; every part has an origin).")


if __name__ == "__main__":
    main()