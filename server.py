"""
Backend for the cross-domain innovation UI.

Setup:
    1. Save your original script in this folder as  pipeline.py
    2. pip install flask
    3. python server.py   ->  http://localhost:5000

Streams each pipeline stage to the browser as Server-Sent Events.
"""

import json
import traceback

import faiss
from flask import Flask, Response, request, send_from_directory, stream_with_context

import pipeline as p  # your original script

app = Flask(__name__, static_folder=".")

_cache = {}


def resources():
    """Load the embedding model, FAISS index and metadata once."""
    if not _cache:
        if not p.INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found: {p.INDEX_PATH}")
        if not p.METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata not found: {p.METADATA_PATH}")
        _cache["tok"], _cache["model"] = p.load_embedding_model()
        _cache["index"] = faiss.read_index(str(p.INDEX_PATH))
        _cache["meta"] = p.load_metadata()
    return _cache


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def home():
    return send_from_directory(".", "index.html")


@app.post("/api/run")
def run():
    problem = (request.get_json(silent=True) or {}).get("problem", "").strip()

    def stream():
        if not problem:
            yield sse("error", {"message": "Describe a problem first."})
            return
        try:
            # Step 1: abstraction (LLM call 1)
            yield sse("step", {"id": "abstract", "status": "running"})
            abstraction = p.call1_problem_abstraction(problem)
            missing = [k for k in ("problem", "constraints", "required_capabilities",
                                   "mechanism", "residual_terms") if k not in abstraction]
            if missing:
                raise ValueError("Abstraction is missing fields: " + ", ".join(missing))
            yield sse("step", {"id": "abstract", "status": "done", "data": abstraction})

            # Step 2: embedding
            yield sse("step", {"id": "embed", "status": "running"})
            res = resources()
            rep = p.make_mechanism_residual_representation(
                abstraction["mechanism"], abstraction["residual_terms"])
            q = p.embed_texts([rep], res["tok"], res["model"])
            if q.shape[1] != res["index"].d:
                raise ValueError(f"Dimension mismatch: query={q.shape[1]}, index={res['index'].d}")
            yield sse("step", {"id": "embed", "status": "done", "data": {
                "representation": rep, "dimension": int(q.shape[1]), "model": p.MODEL_NAME}})

            # Step 3: retrieval
            yield sse("step", {"id": "retrieve", "status": "running"})
            retrieved = p.retrieve(q, res["index"], res["meta"], p.RETRIEVAL_K)
            yield sse("step", {"id": "retrieve", "status": "done", "data": retrieved})

            # Step 4: synthesis (LLM call 2)
            yield sse("step", {"id": "synthesize", "status": "running"})
            solution = p.call2_synthesize(problem, abstraction, retrieved)
            yield sse("step", {"id": "synthesize", "status": "done", "data": solution})

            with open(p.OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"external_problem": problem, "problem_abstraction": abstraction,
                           "embedding_representation": rep,
                           "retrieval": retrieved[:p.SYNTHESIS_K], "solution": solution},
                          f, indent=2, ensure_ascii=False)
            yield sse("done", {})
        except Exception as e:
            traceback.print_exc()
            yield sse("error", {"message": str(e)})

    return Response(stream_with_context(stream()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)