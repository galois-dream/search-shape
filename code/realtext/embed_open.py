# embed_open.py -- embed every C250/C750 chunk with an open model via sentence-transformers (2 torch threads; device
# EMB_DEVICE, default cpu; weights always in float32); same text as passage and query, no instruction prefix; records
# the model's own tokenizer lengths and how many chunks exceed max_seq_length (i.e. are truncated), and the device and
# dtype. Writes emb/<tag>/passage_c{250,750}.npy (float32) + summary json.
# Usage: [CORPUS=wiki] [EMB_DEVICE=mps] python embed_open.py <hf-model-id> <tag>
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(2)
ROOT = Path(os.environ.get("RT_WORK", str(Path(__file__).resolve().parent / "work")))  # see README
BASE = ROOT if os.environ.get("CORPUS", "nobel") == "nobel" else ROOT / os.environ["CORPUS"]
mid, tag = sys.argv[1], sys.argv[2]
DEV = os.environ.get("EMB_DEVICE", "cpu")
m = SentenceTransformer(mid, device=DEV, model_kwargs={"dtype": torch.float32})  # float32 always (the Qwen3 config defaults to bfloat16)
E = BASE / "emb" / tag; E.mkdir(parents=True, exist_ok=True)
for g in (250, 750):
    rows = [json.loads(l) for l in open(BASE / "data" / f"chunks_c{g}.jsonl")]
    texts = [r["text"] for r in rows]
    t0 = time.time()
    X = m.encode(texts, batch_size=8, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
    np.save(E / f"passage_c{g}.npy", X.astype(np.float32))
    lens = [len(m.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    s = dict(model=mid, n=len(rows), dim=int(X.shape[1]), max_seq_length=m.max_seq_length,
             n_truncated=int(sum(l > m.max_seq_length for l in lens)), tok_len_median=float(np.median(lens)),
             tok_len_max=int(max(lens)), device=DEV, dtype=str(next(m.parameters()).dtype), seconds=round(time.time() - t0, 1),
             time=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    json.dump(s, open(E / f"passage_c{g}.summary.json", "w"), indent=1)
    print(s, flush=True)
