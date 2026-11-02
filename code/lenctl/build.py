# build.py -- written on day 4 at 10:07. Builds the length-controlled topic-count test of docs/lenctl_plan.md: selects 240
# Wikipedia pages (seed 0), cuts 18 atoms of 83 cl100k tokens per page, forms Q1/C1 (one page per text) and, for
# grouping seeds 0-2, Q3/C3 (three pages per text, same atoms), and embeds every text with two open models.
# Needs tiktoken and sentence-transformers; input: the AmbigQA Wikipedia dump (docs.db):
#   python build.py DOCS_DB OUT_DIR
# Outputs: OUT_DIR/texts.json (pages, atoms, groupings) and OUT_DIR/emb_<model>.npz (unit vectors per text set).
import html, json, sqlite3, sys
from pathlib import Path
import numpy as np
import tiktoken

NPAGE, NATOM, ATOM = 240, 18, 83
MODELS = {"bge-base": "BAAI/bge-base-en-v1.5", "bge-m3": "BAAI/bge-m3"}
SKIP = ("(disambiguation)", "List of", "Index of")


def pick_pages(db_path, enc):
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ids = [r[0] for r in db.execute("SELECT id FROM documents")]
    out = []
    for i in np.random.default_rng(0).permutation(len(ids)):
        text = db.execute("SELECT text FROM documents WHERE id = ?", (ids[i],)).fetchone()[0]
        title, _, body = html.unescape(text).partition("\n")
        if any(s in title for s in SKIP):
            continue
        tok = enc.encode(body.strip())
        if len(tok) >= NATOM * ATOM:
            out.append((title, tok[:NATOM * ATOM]))
            if len(out) == NPAGE:
                return out
    raise SystemExit("not enough pages")


def triples(items, page_of, rng):
    """Group items into threes from three different pages: order items by a random page order (items of one page
    adjacent, shuffled within the page), then group positions j, j+m, j+2m (m = n/3). A page holds at most 15 items
    and m >= 240, so the three items of a group always come from different pages. Group order is then shuffled.
    (10:11: replaces rejection sampling, which practically never succeeded.)"""
    items = np.asarray(items); pg = np.asarray([page_of[a] for a in items])
    rank = rng.permutation(pg.max() + 1)
    L = items[np.lexsort((rng.random(len(items)), rank[pg]))]
    m = len(L) // 3
    g = [[int(L[j]), int(L[j + m]), int(L[j + 2 * m])] for j in range(m)]
    assert all(len({page_of[a] for a in t}) == 3 for t in g)
    return [g[k] for k in rng.permutation(m)]


def main(db_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("cl100k_base")
    pages = pick_pages(db_path, enc)
    atoms, page_of = [], []
    for p, (_, tok) in enumerate(pages):
        for a in range(NATOM):
            atoms.append(enc.decode(tok[a * ATOM:(a + 1) * ATOM])); page_of.append(p)
    qa = [p * NATOM + a for p in range(NPAGE) for a in range(3)]            # block 0 of each page
    ca = [p * NATOM + a for p in range(NPAGE) for a in range(3, NATOM)]     # blocks 1-5
    q1 = [qa[k:k + 3] for k in range(0, len(qa), 3)]
    c1 = [ca[k:k + 3] for k in range(0, len(ca), 3)]
    sets = {"Q1": q1, "C1": c1}
    for g in range(3):
        rng = np.random.default_rng((g, 7))
        sets[f"Q3_g{g}"] = triples(qa, page_of, rng)
        sets[f"C3_g{g}"] = triples(ca, page_of, rng)
    text = {k: [" ".join(atoms[a] for a in grp) for grp in v] for k, v in sets.items()}
    ntok = {k: sorted({sum(len(enc.encode(atoms[a])) for a in grp) for grp in v})[:3] for k, v in sets.items()}
    json.dump({"pages": [t for t, _ in pages], "page_of": page_of, "sets": sets, "atom_tokens": ATOM,
               "decoded_token_counts_sample": ntok}, open(out / "texts.json", "w"))
    print("pages", len(pages), "atoms", len(atoms), {k: len(v) for k, v in sets.items()}, flush=True)
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for name, hf in MODELS.items():
        m = SentenceTransformer(hf, device=dev)
        m.max_seq_length = 512
        E = {k: m.encode(v, batch_size=16, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
             for k, v in text.items()}
        np.savez_compressed(out / f"emb_{name}.npz", **E)
        print(name, dev, {k: v.shape for k, v in E.items()}, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
