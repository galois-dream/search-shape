# ambigqa.py -- minimal helper for p2.py: CHUNKS_PER_PAGE, CORPUS_CAP and chunks(), copied unchanged from the
# authors' evidence/ambigqa.py (written 2026-10-15 19:35 CEST), the module that built the registered AmbigQA corpus
# (docs/analysis_plan.pdf, Data). The rest of that module (gold-page extraction, embedding, topic clustering) is not
# needed here. chunks() cuts every page of OUT/pages.parquet with the Blowfish library's TokenSlidingWindowChunker
# (open source, Apache-2.0; 250 o200k tokens per chunk) and marks the first CHUNKS_PER_PAGE chunks as in_corpus.
from pathlib import Path

import pandas as pd

from blowfish.datasets.chunkers import TokenSlidingWindowChunker
from blowfish.datasets.types import Document

CHUNKS_PER_PAGE = 5
CORPUS_CAP = 50_000


def chunks(data, out):
    P = pd.read_parquet(Path(out) / "pages.parquet")
    chunker = TokenSlidingWindowChunker()
    rows = [{"title": t, "idx": i, "text": c.text}
            for t, text in zip(P.title, P.text)
            for i, c in enumerate(chunker.chunk(Document(doc_id=t, title=t, text=text)))]
    C = pd.DataFrame(rows)
    C["in_corpus"] = C.idx < CHUNKS_PER_PAGE
    C.to_parquet(Path(out) / "chunks.parquet")
    per_page = C.groupby("title").size()
    print(f"chunks: {len(C)} total over {len(per_page)} pages; corpus {int(C.in_corpus.sum())} (cap {CORPUS_CAP})")
    print("chunks per page quantiles (0, .1, .25, .5, .75, .9, .99, 1):",
          per_page.quantile([0, .1, .25, .5, .75, .9, .99, 1]).astype(int).tolist())
    print(f"pages with < 10 chunks: {int((per_page < 10).sum())}")
