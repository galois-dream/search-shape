# Shipped copy of paper/trec_p2/explore.py; only this line added for the repo layout (EXPLORATORY.md = docs/trec_p2_exploratory.md; results/ = data/reference_outputs/trec_p2/).
# explore.py -- P2 exploratory analyses E1 and E2, specified in EXPLORATORY.md (written 2026-11-14 23:01 CET).
# Not registered; nothing here changes the registered verdict. Usage: python explore.py ARM_DIR TOPICS_TSV
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import chi2
import p2

TOPO = ["longbar", "w1", "ltmax", "eig1"]


def _logit_fit(X, y, off):  # unpenalised logistic regression with an offset (Newton); returns beta, log-lik, Wald SE
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-(off + X @ b))); H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-(off + X @ b)))
    return b, np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), np.sqrt(np.diag(np.linalg.inv(H)))


def _z(x):
    return (x - x.mean()) / x.std()


def _job(args):
    F, y, cols, lin, r = args
    return p2._oof(F, y, cols, lin, r)


def main(arm, tsv):
    arm = Path(arm); F = pd.read_csv(arm / "feats.csv"); T = p2._topics(tsv, with_type=True)
    lab = dict(zip(T.number, T.type)); F = F[F.number.map(lab).isin(["ambiguous", "faceted"])].reset_index(drop=True)
    y = (F.number.map(lab) == "ambiguous").to_numpy().astype(int)
    cos = [c for c in F.columns if c.startswith("cos")]; m0 = ["nwords", "pages", "coverage"] + cos
    with ProcessPoolExecutor(4) as ex:
        P0 = np.stack(list(ex.map(_job, [(F, y, m0, cos, r) for r in range(20)])))
        P1 = np.stack(list(ex.map(_job, [(F, y, m0 + TOPO, cos, r) for r in range(20)])))
    L = [f"# explore.py -- {arm}; n1 = {y.sum()}, n2 = {len(y) - y.sum()} (exploratory, not registered)"]

    # E2 first: reproduce the registered bootstrap exactly (same code path as p2.fit's run()), then read it as probabilities
    ll = lambda P, ix: np.mean([np.mean(y[ix] * np.log(p[ix]) + (1 - y[ix]) * np.log(1 - p[ix])) for p in P])
    au = lambda P, ix: np.mean([p2._auc(p[ix], y[ix]) for p in P])
    ix0 = np.arange(len(y)); rng = np.random.default_rng(0); bl, ba = [], []
    for _ in range(2000):
        ix = rng.integers(0, len(y), len(y))
        if 0 < y[ix].sum() < len(ix):
            bl.append(ll(P1, ix) - ll(P0, ix)); ba.append(au(P1, ix) - au(P0, ix))
    bl, ba = np.array(bl), np.array(ba); qa, ql = np.percentile(ba, [2.5, 97.5]), np.percentile(bl, [2.5, 97.5])
    L.append(f"E2 reproduction: AUROC gain {au(P1, ix0) - au(P0, ix0):+.3f} [{qa[0]:+.3f}, {qa[1]:+.3f}]; log-likelihood gain "
             f"{ll(P1, ix0) - ll(P0, ix0):+.4f} [{ql[0]:+.4f}, {ql[1]:+.4f}] (must match result.txt 'real')")
    L.append(f"E2: share of {len(ba)} bootstrap draws with AUROC gain >= +0.03: {np.mean(ba >= 0.03):.3f}; with AUROC gain > 0: "
             f"{np.mean(ba > 0):.3f}; with log-likelihood gain > 0: {np.mean(bl > 0):.3f}")

    # E1: offset = mean out-of-fold logit of M0 over the 20 repetitions
    off = np.mean(np.log(P0 / (1 - P0)), axis=0); one = np.ones((len(y), 1))
    _, l0, _ = _logit_fit(one, y, off)
    for tag, suf in (("real", ""), ("control corp", "_corp"), ("control swap", "_swap")):
        X1 = np.column_stack([one, _z(F["longbar" + suf].to_numpy(float))]); b, l1, se = _logit_fit(X1, y, off)
        st = 2 * (l1 - l0)
        X4 = np.column_stack([one] + [_z(F[t + suf].to_numpy(float)) for t in TOPO]); _, l4, _ = _logit_fit(X4, y, off)
        st4 = 2 * (l4 - l0)
        L.append(f"E1 {tag}: longest bar given M0, beta per SD {b[1]:+.3f} [{b[1] - 1.96 * se[1]:+.3f}, {b[1] + 1.96 * se[1]:+.3f}], "
                 f"LR chi2(1) = {st:.2f}, p = {chi2.sf(st, 1):.3g}; E1b four features: LR chi2(4) = {st4:.2f}, p = {chi2.sf(st4, 4):.3g}")
    (arm / "explore.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))


def e3(arm, tsv):  # confounders, EXPLORATORY.md section E3
    from scipy.stats import mannwhitneyu, spearmanr
    arm = Path(arm); F = pd.read_csv(arm / "feats.csv"); T = p2._topics(tsv, with_type=True)
    lab = dict(zip(T.number, T.type)); F = F[F.number.map(lab).isin(["ambiguous", "faceted"])].reset_index(drop=True)
    y = (F.number.map(lab) == "ambiguous").to_numpy().astype(int); n = F.nwords.to_numpy()
    cos = [c for c in F.columns if c.startswith("cos")]; L = [f"# explore.py e3 -- {arm} (exploratory, not registered)"]
    mw = lambda x, m: mannwhitneyu(x[m & (y == 1)], x[m & (y == 0)], alternative="two-sided")
    def auc_line(name, x, m):
        a, b = (m & (y == 1)).sum(), (m & (y == 0)).sum(); u = mw(x, m)
        return f"{name}: AUROC {u.statistic / (a * b):.3f}, U = {u.statistic:.0f}, n1 = {a}, n2 = {b}, p = {u.pvalue:.3g}"
    for k, m in (("1 word", n == 1), ("2 words", n == 2), ("3+ words", n >= 3)):
        L.append(f"E3a {k}: ambiguous {(m & (y == 1)).sum()}, faceted {(m & (y == 0)).sum()}")
    L.append("E3a " + auc_line("query length (words), ambiguous vs faceted", n.astype(float), np.ones(len(y), bool)))
    yr = 2009 + (F.number.to_numpy() - 1) // 50
    L.append("E3a ambiguous share by year: " + ", ".join(f"{t} {y[yr == t].sum()}/{(yr == t).sum()}" for t in range(2009, 2013)))
    sets = {"(i) length alone": ["nwords"], "(ii) baseline without length": ["pages", "coverage"] + cos,
            "(iii) full baseline": ["nwords", "pages", "coverage"] + cos, "(iv) length + topology": ["nwords"] + TOPO}
    P = {}
    with ProcessPoolExecutor(3) as ex:
        for k, cols in sets.items():
            P[k] = np.stack(list(ex.map(_job, [(F, y, cols, cos, r) for r in range(20)])))
    au = lambda Q, ix: np.mean([p2._auc(p[ix], y[ix]) for p in Q])
    rng = np.random.default_rng(0); B = {k: [] for k in P}
    for _ in range(2000):
        ix = rng.integers(0, len(y), len(y))
        if 0 < y[ix].sum() < len(ix):
            for k in P:
                B[k].append(au(P[k], ix))
    for k in P:
        q = np.percentile(B[k], [2.5, 97.5]); L.append(f"E3b {k}: CV AUROC {au(P[k], np.arange(len(y))):.3f} [{q[0]:.3f}, {q[1]:.3f}]")
    for k, m in (("1 word", n == 1), ("2+ words", n >= 2)):
        L.append("E3c " + auc_line(f"{k}, longest bar", F.longbar.to_numpy(float), m))
        L.append("E3c " + auc_line(f"{k}, top cosine", F.cos00.to_numpy(float), m))
    r1, r2 = spearmanr(F.longbar, n), spearmanr(F.longbar, F.cos00)
    L.append(f"E3d Spearman: longest bar vs length rho = {r1.statistic:+.3f} (p = {r1.pvalue:.3g}); vs top cosine rho = "
             f"{r2.statistic:+.3f} (p = {r2.pvalue:.3g}); n = {len(y)}")
    (arm / "explore_e3.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))


def e4(tsv, bm25_csv, index_db, out):  # names, Wikipedia coverage, year: EXPLORATORY.md section E4
    import html, sqlite3
    from scipy.stats import fisher_exact, mannwhitneyu
    T = p2._topics(tsv, with_type=True).drop_duplicates("number"); T = T[T.type.isin(["ambiguous", "faceted"])].reset_index(drop=True)
    y = (T.type == "ambiguous").to_numpy().astype(int); n = np.array([len(q.split()) for q in T["query"]])
    q = [s.lower().strip() for s in T["query"]]; want = set(q) | {s + " (disambiguation)" for s in q}
    art, dis = set(), set()
    for t, d in sqlite3.connect(f"file:{index_db}?mode=ro", uri=True).execute("SELECT title, disamb FROM meta"):
        t = html.unescape(t).lower()
        if t in want:
            (dis if d or t.endswith(" (disambiguation)") else art).add(t.removesuffix(" (disambiguation)"))
    a = np.array([s in art for s in q], int); b = np.array([s in dis for s in q], int)
    B = pd.read_csv(bm25_csv); B = B[B.arm == "withdis"]; c = T.number.map(B.groupby("number").disamb.sum()).fillna(0).to_numpy()
    L = [f"# explore.py e4 -- names, Wikipedia coverage, year (exploratory, not registered); n1 = {y.sum()}, n2 = {len(y) - y.sum()}"]
    ll = lambda X: _logit_fit(np.column_stack([np.ones(len(y))] + X), y, np.zeros(len(y)))[1]
    z = lambda x: (x - x.mean()) / (x.std() or 1)
    l_len = ll([z(n.astype(float))])
    for name, x in (("E4a own article", a), ("E4b disambiguation page", b)):
        for k, m in (("all", n > 0), ("1 word", n == 1), ("2+ words", n >= 2)):
            t = [[(x[m & (y == 1)] == 1).sum(), (x[m & (y == 1)] == 0).sum()], [(x[m & (y == 0)] == 1).sum(), (x[m & (y == 0)] == 0).sum()]]
            f = fisher_exact(t)
            L.append(f"{name}, {k}: ambiguous {t[0][0]}/{sum(t[0])}, faceted {t[1][0]}/{sum(t[1])}; Fisher OR = {f.statistic:.2f}, p = {f.pvalue:.3g}")
        st = 2 * (ll([z(n.astype(float)), z(x.astype(float))]) - l_len)
        L.append(f"{name}: AUROC {p2._auc(x.astype(float), y):.3f}; given length LR chi2(1) = {st:.2f}, p = {chi2.sf(st, 1):.3g}")
    for k, m in (("all", n > 0), ("1 word", n == 1), ("2+ words", n >= 2)):
        u = mannwhitneyu(c[m & (y == 1)], c[m & (y == 0)], alternative="two-sided"); n1, n2 = (m & (y == 1)).sum(), (m & (y == 0)).sum()
        L.append(f"E4c disambiguation pages in top 20, {k}: median ambiguous {np.median(c[m & (y == 1)]):.0f}, faceted "
                 f"{np.median(c[m & (y == 0)]):.0f}; AUROC {u.statistic / (n1 * n2):.3f}, U = {u.statistic:.0f}, n1 = {n1}, n2 = {n2}, p = {u.pvalue:.3g}")
    st = 2 * (ll([z(n.astype(float)), z(c)]) - l_len); L.append(f"E4c given length: LR chi2(1) = {st:.2f}, p = {chi2.sf(st, 1):.3g}")
    yr = 2009 + (T.number.to_numpy() - 1) // 50; Y = [(yr == t).astype(float) for t in (2010, 2011, 2012)]
    l_yr = ll(Y); l_both = ll([z(n.astype(float))] + Y)
    s1, s2 = 2 * (l_both - l_len), 2 * (l_both - l_yr)
    L.append(f"E4d year given length: LR chi2(3) = {s1:.2f}, p = {chi2.sf(s1, 3):.3g}; length given year: LR chi2(1) = {s2:.2f}, p = {chi2.sf(s2, 1):.3g}")
    for k, m in (("1 word", n == 1), ("2+ words", n >= 2)):
        L.append(f"E4d ambiguous share by year, {k}: " + ", ".join(f"{t} {y[m & (yr == t)].sum()}/{(m & (yr == t)).sum()}" for t in range(2009, 2013)))
    Path(out).write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"e3": e3, "e4": e4}.get(cmd, main)(*sys.argv[2 if cmd in ("e3", "e4") else 1:])
