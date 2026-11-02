# ring_examples.py -- written on day 1 at 23:48. Illustrative examples for the paper's appendix, NOT evidence:
# public Wikipedia pages (AmbigQA 2020-01-20 dump, CC BY-SA 3.0) for several senses of "ring", "mercury", "jaguar" and
# "python", searched together inside a whole-page background of ~50k chunks, embedded with MiniLM (same chunker,
# embedder and background rule as the module hook_data.py of a demonstration package, which is imported, not
# copied, and not included here because it would identify the authors; the README, section 4, gives the setup). For each query: the top-50
# neighbours, the longest edge of the minimum spanning tree of their unit offset directions (= L_max), and the two
# groups that edge separates, with their page titles.
# Added on day 2: a 2-D classical-MDS layout of the 50 offsets (display only), the MST edges, and HDBSCAN clusters of
# the offsets (min_cluster_size=5; robust single linkage, as the library's topic clustering) named by majority page.
# Usage:  HOOK_DIR=<dir of hook_data.py> python ring_examples.py DATA_DIR REAL_DIR OUT_JSON
import collections, html, json, os, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN

sys.path.insert(0, os.environ["HOOK_DIR"])
import hook_data as H  # noqa: E402  (chunker, embedder name, background size, K)

SENSES = {  # sense label -> candidate page titles (first match in the dump is used)
    "ring: algebra": ["Ring (mathematics)"], "ring: jewellery": ["Ring (jewellery)", "Ring (jewelry)"],
    "ring: Tolkien": ["One Ring"], "ring: boxing": ["Boxing ring"],
    "mercury: planet": ["Mercury (planet)"], "mercury: element": ["Mercury (element)"], "mercury: singer": ["Freddie Mercury"],
    "jaguar: animal": ["Jaguar"], "jaguar: cars": ["Jaguar Cars"],
    "python: language": ["Python (programming language)"], "python: snakes": ["Pythonidae"]}
QUERIES = ["ring", "what is a ring", "ring ideal and quotient", "wedding ring", "who forged the one ring", "boxing ring size",
           "mercury", "how long is a year on mercury", "jaguar", "jaguar top speed", "python", "how big do pythons get"]
# Added on day 4 at 09:50: EXAMPLE_SET=crane runs the earlier preprint's crane sentences (its introduction used them as an
# illustration only; never searched). "Neck" stands in for the verb sense (craning one's neck), which has no page.
if os.environ.get("EXAMPLE_SET") == "crane":
    SENSES = {"crane: machine": ["Crane (machine)"], "crane: bird": ["Crane (bird)"], "neck (craning)": ["Neck"]}
    QUERIES = ["I saw a crane.", "He craned his neck to watch the crane lift near the crane.",
               "He craned his neck to see the construction site.", "The crane lifted the heavy beams effortlessly.",
               "A crane flew over the lake at sunset.", "crane"]


def find_pages(db_path, titles):
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    want, found = {f"{t}\n" for t in titles}, {}
    for pid, head in db.execute("SELECT id, substr(text, 1, 80) FROM documents"):
        line = html.unescape(head.split("\n", 1)[0]) + "\n"
        if line in want and line not in found:
            found[line] = db.execute("SELECT text FROM documents WHERE id = ?", (pid,)).fetchone()[0]
            if len(found) == len(want):
                break
    return {k[:-1]: v for k, v in found.items()}


def main(data, real, out):
    from sentence_transformers import SentenceTransformer
    texts = find_pages(Path(data) / "docs.db", [t for c in SENSES.values() for t in c])
    page_sense = {}
    for sense, cands in SENSES.items():
        hit = next((t for t in cands if t in texts), None)
        print(f"{sense:18s} -> {hit}", flush=True)
        if hit:
            page_sense[hit] = sense
    chunker = H.TokenSlidingWindowChunker()
    rows = [{"page": p, "text": c.text} for p in page_sense for c in chunker.chunk(H.Document(doc_id=p, title=p, text=texts[p]))]
    C = pd.DataFrame(rows)
    m = SentenceTransformer(H.EMBEDDER, device="cpu")
    E = m.encode(C.text.tolist(), batch_size=32, normalize_embeddings=True).astype(np.float32)
    Q = m.encode(QUERIES, normalize_embeddings=True).astype(np.float32)
    B = pd.read_parquet(Path(real) / "chunks.parquet")
    sizes = B[~B.title.isin(list(page_sense))].groupby("title").size().sample(frac=1, random_state=0)
    picked = sizes.index[(sizes.cumsum() <= H.MAX_SEARCH - len(E)).to_numpy()]
    keep = B.title.isin(picked).to_numpy()
    E = np.r_[E, np.load(Path(real) / "chunk_emb.npy")[keep]]
    title = np.r_[C.page.to_numpy(), B.title[keep].to_numpy()]
    print(f"search space {len(E)} chunks ({len(C)} from the {len(page_sense)} sense pages)", flush=True)
    res, dump = [], {}
    for text, q in zip(QUERIES, Q):
        top = np.argsort(-(E @ q), kind="stable")[:H.K]
        U = E[top] - q
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        D = squareform(pdist(U))
        T = minimum_spanning_tree(D).toarray()
        mst = [(int(a), int(b), float(T[a, b])) for a, b in zip(*np.nonzero(T))]
        J = np.eye(len(D)) - 1.0 / len(D)
        w, V = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
        xy = V[:, -2:][:, ::-1] * np.sqrt(np.maximum(w[-2:][::-1], 0))   # classical MDS, display only
        hl = HDBSCAN(min_cluster_size=5).fit_predict(U)
        senses = [page_sense.get(t, "other pages") for t in title[top]]
        clusters = []
        for c in sorted(set(hl) - {-1}):
            cs = collections.Counter(s_ for s_, h in zip(senses, hl) if h == c).most_common(3)
            clusters.append({"label": int(c), "size": int((hl == c).sum()), "senses": cs})
        edges = np.sort(T[T > 0])
        i, j = np.unravel_index(np.argmax(T), T.shape)
        T[i, j] = 0
        _, lab = connected_components(T + T.T, directed=False)
        groups = []
        for g in sorted(set(lab), key=lambda g: -(lab == g).sum()):
            idx = top[lab == g]
            labels = [page_sense.get(t, "other pages") for t in title[idx]]
            groups.append({"size": int(len(idx)), "senses": collections.Counter(labels).most_common(4)})
        sense_counts = collections.Counter(page_sense.get(t, "other pages") for t in title[top]).most_common(6)
        res.append({"query": text, "top1_cos": float((E[top] @ q).max()), "lmax": float(edges[-1]), "second": float(edges[-2]),
                    "senses_in_top50": sense_counts, "split": groups, "longest_edge": [int(i), int(j)],
                    "xy": np.round(xy, 4).tolist(), "sense": senses, "hdbscan": hl.tolist(), "clusters": clusters, "mst": mst})
        if os.environ.get("DUMP_NEIGHBOURS"):   # added on day 4 at 10:13: vectors for persistence features and controls
            dump[f"q{len(res) - 1}"] = q; dump[f"n{len(res) - 1}"] = E[top]
            dump[f"pool{len(res) - 1}"] = E[np.random.default_rng(len(res)).choice(np.setdiff1d(np.arange(len(E)), top), 250, replace=False)]
        print(f"\n{text!r}: Lmax {edges[-1]:.3f} (next {edges[-2]:.3f}); top-50 {sense_counts}", flush=True)
        for g in groups:
            print(f"   group of {g['size']:2d}: {g['senses']}", flush=True)
        print(f"   HDBSCAN: {[(c['size'], c['senses']) for c in clusters]}; noise {int((hl == -1).sum())}", flush=True)
    if dump:
        np.savez_compressed(Path(out).with_suffix(".npz"), **dump)
    Path(out).write_text(json.dumps({"embedder": H.EMBEDDER, "k": H.K, "search_space": int(len(E)),
                                     "source": "English Wikipedia dump 2020-01-20 (AmbigQA docs.db), CC BY-SA 3.0",
                                     "pages": page_sense, "queries": res}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
