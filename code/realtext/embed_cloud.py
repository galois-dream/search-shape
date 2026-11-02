# embed_cloud.py -- embed every C250/C750 chunk with the API models: OpenAI text-embedding-3-small/-large and
# text-embedding-ada-002; Google Vertex AI text-embedding-005 twice (task_type RETRIEVAL_DOCUMENT for passages,
# RETRIEVAL_QUERY for queries) and gemini-embedding-001 (task_type SEMANTIC_SIMILARITY, one vector per chunk).
# Small batches, each written to disk as a float32 .npy part (resumable), merged at the end; L2-normalisation happens
# downstream (features.py). Prints a cost estimate first and aborts if it exceeds $2. Never prints credentials.
# Credentials come only from the environment: OPENAI_API_KEY; VERTEX_PROJECT (your Google Cloud project id) and
# VERTEX_ACCESS_TOKEN (if unset, `gcloud auth print-access-token` is called).
# Usage: [CORPUS=wiki] python embed_cloud.py estimate | run <model>   (CORPUS default nobel)
import json, os, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
BASE = ROOT / os.environ["CORPUS"] if os.environ.get("CORPUS", "nobel") != "nobel" else ROOT  # wiki -> $RT_WORK/wiki/
DATA, EMB = BASE / "data", BASE / "emb"
# Published list prices at the time of the experiment (not verified against a billing export):
PRICE = {"openai-3-small": ("tok", 0.02 / 1e6), "openai-3-large": ("tok", 0.13 / 1e6),
         "vertex-005": ("char", 0.000025 / 1e3), "gemini-001": ("tok", 0.15 / 1e6),  # gemini: unverified list price
         "openai-ada-002": ("tok", 0.10 / 1e6)}


def load(g):
    return [json.loads(l) for l in open(DATA / f"chunks_c{g}.jsonl")]


def estimate():
    tok = sum(len(load(g)) * g for g in (250, 750))
    ch = sum(len(r["text"]) for g in (250, 750) for r in load(g))
    tot = 0
    for m, (u, p) in PRICE.items():
        n = tok if u == "tok" else 2 * ch  # vertex: two task types
        c = n * p; tot += c
        print(f"[estimate] {m}: {n} {u}s x ${p:.3g} = ${c:.4f}")
    print(f"[estimate] TOTAL ${tot:.4f} (cap $2.00)")
    return tot


def post(url, body, headers, tries=12):  # backoff up to 60 s (per-minute quota 429s)
    for t in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:300]
            if e.code in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524) and t < tries - 1:
                time.sleep(min(60, 2 ** t) + 1); continue
            raise RuntimeError(f"HTTP {e.code}: {msg!r}")
        except urllib.error.URLError:
            if t < tries - 1:
                time.sleep(min(60, 2 ** t) + 1); continue
            raise


def openai_batch(model, texts):
    h = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"}
    r = post("https://api.openai.com/v1/embeddings", {"model": model, "input": texts, "encoding_format": "float"}, h)
    d = sorted(r["data"], key=lambda x: x["index"])
    return np.array([x["embedding"] for x in d], np.float32), {"usage_tokens": r["usage"]["total_tokens"],
                                                               "model_returned": r.get("model")}


_tok = {}


def gtoken():
    if not _tok or time.time() - _tok["t"] > 1800:
        tok = os.environ.get("VERTEX_ACCESS_TOKEN")
        if not tok:
            tok = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                                 text=True, check=True).stdout.strip()
        _tok["v"], _tok["p"], _tok["t"] = tok, os.environ["VERTEX_PROJECT"], time.time()
    return _tok["v"], _tok["p"]


def vertex_batch(task, texts, model="text-embedding-005"):
    tok, proj = gtoken()
    url = (f"https://us-central1-aiplatform.googleapis.com/v1/projects/{proj}/locations/us-central1/"
           f"publishers/google/models/{model}:predict")
    body = {"instances": [{"content": t, "task_type": task} for t in texts], "parameters": {"autoTruncate": True}}
    r = post(url, body, {"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    P = r["predictions"]
    stats = [p["embeddings"]["statistics"] for p in P]
    return np.array([p["embeddings"]["values"] for p in P], np.float32), {
        "token_counts": [s.get("token_count") for s in stats], "truncated": [bool(s.get("truncated")) for s in stats],
        "deployedModelId": r.get("deployedModelId"), "model_version": r.get("modelVersionId")}


JOBS = {  # name -> list of (out-subdir, batch fn, batch size per granularity)
    "openai-3-small": [("passage", lambda t: openai_batch("text-embedding-3-small", t), {250: 64, 750: 32})],
    "openai-3-large": [("passage", lambda t: openai_batch("text-embedding-3-large", t), {250: 64, 750: 32})],
    "vertex-005": [("passage", lambda t: vertex_batch("RETRIEVAL_DOCUMENT", t), {250: 50, 750: 16}),
                   ("query", lambda t: vertex_batch("RETRIEVAL_QUERY", t), {250: 50, 750: 16})],
    # symmetric task type (same embedding for query and passage); batches of 16 (250) and 8 (750) texts
    "gemini-001": [("passage", lambda t: vertex_batch("SEMANTIC_SIMILARITY", t, "gemini-embedding-001"), {250: 16, 750: 8})],
    # an older closed generation (8,191-token input, 1,536 dimensions)
    "openai-ada-002": [("passage", lambda t: openai_batch("text-embedding-ada-002", t), {250: 64, 750: 32})],
}


def run(model):
    for role, fn, bs in JOBS[model]:
        for g in (250, 750):
            rows = load(g)
            d = EMB / model / f"{role}_c{g}"; d.mkdir(parents=True, exist_ok=True)
            b = bs[g]
            for k in range(0, len(rows), b):
                part = d / f"part{k:05d}.npy"
                if part.exists():
                    continue
                X, meta = fn([r["text"] for r in rows[k:k + b]])
                assert X.shape[0] == len(rows[k:k + b])
                meta["ids"] = [r["id"] for r in rows[k:k + b]]
                np.save(part, X)
                json.dump(meta, open(d / f"part{k:05d}.json", "w"))
            parts = sorted(d.glob("part*.npy"))
            X = np.concatenate([np.load(p) for p in parts]).astype(np.float32)
            ids = sum((json.load(open(p.with_suffix(".json")))["ids"] for p in parts), [])
            assert ids == [r["id"] for r in rows], "id order mismatch"
            np.save(EMB / model / f"{role}_c{g}.npy", X)
            metas = [json.load(open(p.with_suffix(".json"))) for p in parts]
            summ = {"n": len(ids), "dim": int(X.shape[1]), "time": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
            if "usage_tokens" in metas[0]:
                summ["usage_tokens"] = sum(m["usage_tokens"] for m in metas)
                summ["model_returned"] = metas[0]["model_returned"]
            if "truncated" in metas[0]:
                tc = sum((m["token_counts"] for m in metas), []); tr = sum((m["truncated"] for m in metas), [])
                summ.update(n_truncated=int(sum(tr)), token_count_max=max(tc), token_count_sum=int(sum(tc)),
                            token_count_median=float(np.median(tc)), deployedModelId=metas[0].get("deployedModelId"),
                            model_version=metas[0].get("model_version"))
            json.dump(summ, open(EMB / model / f"{role}_c{g}.summary.json", "w"), indent=1)
            print(f"[{time.strftime('%H:%M:%S')}] {model} {role} C{g}: {summ}", flush=True)


if __name__ == "__main__":
    if estimate() > 2.0:
        sys.exit("estimate exceeds $2 -- stopping")
    if sys.argv[1] == "run":
        run(sys.argv[2])
