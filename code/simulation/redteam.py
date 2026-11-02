"""Red-team tables appended to stats.md by stats.py (all-50-neighbour setting):
  R1 score/coherence baselines AUROC [CI]; R2 grouped 5-fold logistic incremental test (folds by seed x parent group;
  R2b: folds by seed), A0 = cosine stats, A = A0 + pairwise coherence + components, B = A + W1, LT_max, longest H0 bar;
  R3 W1 AUROC vs real/null by child-size bin (S1 vs S2); DT1-DT6: distractor variant (distractor.py).
Cluster bootstrap identical to stats.py (B = 2000, fixed seed).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stats import wauc, ci, B, BOOT_SEED, EPS

COS = ["cos_top1", "cos_mean", "cos_std", "cos_slope"]
COH = ["pair_coherence", "n_components"]
TOPO = ["all_w1_h0", "all_ltmax_h1", "all_h0_maxbar"]
BASE = COS + ["off_R", "mean_sin_half"] + COH
NAMES = {"cos_top1": "top-1 cosine", "cos_mean": "mean cosine", "cos_std": "std cosine", "cos_slope": "slope (cos on rank)",
         "off_R": "R = |mean unit offset|", "mean_sin_half": "mean sin(θ/2)", "pair_coherence": "pairwise coherence",
         "n_components": "components @ median pair-cos", "all_w1_h0": "W1(H0)", "all_ltmax_h1": "LT_max(H1)", "all_h0_maxbar": "max H0 bar"}
CMPS = [(2, "S1 vs S2"), (4, "S1 vs mixed")]
BINS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 28), (29, 31)]


def load(feat, cosd):
    cos = pd.concat([pd.read_csv(f) for f in sorted(cosd.glob("cosine_*_seed*.csv*"))], ignore_index=True)
    frames = []
    for c in cos.config.unique():
        d = pd.concat([pd.read_csv(f) for f in sorted(feat.glob(f"{c}_seed*.csv*"))], ignore_index=True)
        frames.append(d[np.isclose(d.depsilon, EPS)])
    d = pd.concat(frames, ignore_index=True)
    m = d.merge(cos.drop(columns=["parent_group", "corpus_nfeat", "query_nfeat"]), on=["config", "sim_seed", "scenario", "query_idx"],
                how="inner", validate="one_to_one")
    assert len(m) == len(cos), "row alignment failed"
    chk = m.check_w1.notna()
    assert np.allclose(m.loc[chk, "check_w1"], m.loc[chk, "all_w1_h0"]), "W1 spot-check failed"
    for c in ("all_ltmax_h1", "null_all_ltmax_h1"):
        m[c] = m[c].fillna(0.0)
    m["cluster"] = m.sim_seed.astype(str) + "_" + m.parent_group.astype(str)
    m["n_child"] = np.where(m.scenario == 2, m.query_nfeat, m.corpus_nfeat)
    return m, int(chk.sum())


def boot_weights(cl):
    codes, K = pd.factorize(cl)[0], cl.nunique()
    W = np.random.default_rng(BOOT_SEED).multinomial(K, np.full(K, 1.0 / K), size=B).astype(float)
    return codes, W


def auc_ci(x, y, cx, cy, W):
    a = wauc(x, y, np.ones_like(x), np.ones_like(y))[0]
    bt = wauc(x, y, W[:, cx], W[:, cy])
    return a, bt


def oof(X, y, groups, seed=0):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        mdl.fit(X[tr], y[tr]); p[te] = mdl.decision_function(X[te])
    return p


def incremental(m, group_col, title, note):
    """Grouped 5-fold logistic incremental test (A0, A, B); group_col = 'cluster' (seed x parent) or 'sim_seed'."""
    L = [title, "", note, "",
         "| config | comparison | n S1 / other | AUROC A0 | AUROC A | AUROC B | A − A0 [95% CI] | B − A [95% CI] | B − A0 [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|"]
    for cfg in m.config.unique():
        g0 = m[m.config == cfg]
        for sb, cname in CMPS:
            g = g0[g0.scenario.isin([1, sb])].reset_index(drop=True)
            if (g.scenario == sb).sum() == 0:
                continue
            y = (g.scenario == 1).to_numpy().astype(int)
            codes, W = boot_weights(g.cluster)
            res = {}
            for name, cols in (("A0", COS), ("A", COS + COH), ("B", COS + COH + TOPO)):
                sc = oof(g[cols].to_numpy(float), y, g[group_col].astype(str).to_numpy())
                res[name] = auc_ci(sc[y == 1], sc[y == 0], codes[y == 1], codes[y == 0], W)
            def d(u, v):
                lo, hi = ci(res[u][1] - res[v][1]); return f"{res[u][0] - res[v][0]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            L.append(f"| {cfg} | {cname} | {int(y.sum())} / {int((1 - y).sum())} | {res['A0'][0]:.3f} | {res['A'][0]:.3f} | "
                     f"{res['B'][0]:.3f} | {d('A', 'A0')} | {d('B', 'A')} | {d('B', 'A0')} |")
    return L + [""]


SEED_NOTE = ("Folds grouped by SEED (5 seeds -> 5 folds), so seed-constant S2 corpus statistics cannot leak across folds "
             ". Δ CIs: cluster bootstrap (seed x parent) of out-of-fold scores.")


def bands(m, feats, title):
    """AUROC (S1 vs S2) per child-size band for the given (label, column) pairs."""
    L = [title, "", "| config | n_child | n S1 / S2 | " + " | ".join(l for l, _ in feats) + " |", "|---|---|---|" + "---|" * len(feats)]
    for cfg in m.config.unique():
        g = m[m.config == cfg]
        codes, W = boot_weights(g.cluster)
        g = g.assign(code=codes)
        top = int(g.n_child.max())
        for lo_, hi_ in BINS:
            if lo_ > top:
                continue
            gb = g[(g.n_child >= lo_) & (g.n_child <= hi_)]
            A, Bd = gb[gb.scenario == 1], gb[gb.scenario == 2]
            cells = []
            for _, f in feats:
                a, bt = auc_ci(A[f].to_numpy(float), Bd[f].to_numpy(float), A.code.to_numpy(), Bd.code.to_numpy(), W)
                l, h = ci(bt); cells.append(f"{a:.3f} [{l:.3f}, {h:.3f}]")
            L.append(f"| {cfg} | {lo_}–{min(hi_, top)} | {len(A)} / {len(Bd)} | " + " | ".join(cells) + " |")
    return L + [""]


def tables(feat, cosd):
    m, nchk = load(feat, cosd)
    L = ["# Red-team tables (all 50 neighbours)", "",
         f"Rows joined on (config, sim_seed, scenario, query_idx); alignment proven by a W1 spot-check on {nchk} queries (exact match).",
         "AUROC orientation as above (P(S1 > other)); CIs: cluster bootstrap over seed x parent group, B=2000, fixed seed.",
         "Note: in S2 every query of a seed retrieves the same 50 parents, so corpus-only statistics (pairwise coherence,",
         "components) are constant within a seed for S2. AUROC near 0 = perfect separation in the opposite direction (S1 lower).",
         "Configs: superpos = D=256, N_top=1024 random unit topics, n_parent=32, sigma=0.1 (S1, S2); superpos_s02 = same at",
         "sigma=0.2 with the mixed arm. Their eps=0.4 / all-50 rows appear in the first two tables under the same names.", ""]
    # R1
    L += ["## R1. Score and coherence baselines: AUROC [95% CI]", "",
          "| config | comparison | " + " | ".join(NAMES[f] for f in BASE) + " |", "|---|---|" + "---|" * len(BASE)]
    for cfg in m.config.unique():
        g = m[m.config == cfg]
        codes, W = boot_weights(g.cluster)
        g = g.assign(code=codes)
        for sb, cname in CMPS:
            A, Bd = g[g.scenario == 1], g[g.scenario == sb]
            if Bd.empty:
                continue
            cells = []
            for f in BASE:
                a, bt = auc_ci(A[f].to_numpy(float), Bd[f].to_numpy(float), A.code.to_numpy(), Bd.code.to_numpy(), W)
                lo, hi = ci(bt); cells.append(f"{a:.3f} [{lo:.3f}, {hi:.3f}]")
            L.append(f"| {cfg} | {cname} | " + " | ".join(cells) + " |")
    L.append("")
    # R2
    L += ["## R2. Incremental value: grouped 5-fold logistic regression (out-of-fold AUROC)", "",
          "A0 = top-1, mean, std, slope of the 50 cosines; A = A0 + pairwise coherence + components; B = A + W1(H0), LT_max(H1),",
          "longest H0 bar. Standardised, L2 logistic (C=1), GroupKFold(5) on seed x parent group. Δ CIs: cluster bootstrap of the",
          "fixed out-of-fold scores (fold refitting not resampled).", "",
          "| config | comparison | n S1 / other | AUROC A0 | AUROC A | AUROC B | A − A0 [95% CI] | B − A [95% CI] | B − A0 [95% CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cfg in m.config.unique():
        g0 = m[m.config == cfg]
        for sb, cname in CMPS:
            g = g0[g0.scenario.isin([1, sb])].reset_index(drop=True)
            if (g.scenario == sb).sum() == 0:
                continue
            y = (g.scenario == 1).to_numpy().astype(int)
            codes, W = boot_weights(g.cluster)
            res = {}
            for name, cols in (("A0", COS), ("A", COS + COH), ("B", COS + COH + TOPO)):
                s = oof(g[cols].to_numpy(float), y, g.cluster.to_numpy())
                res[name] = auc_ci(s[y == 1], s[y == 0], codes[y == 1], codes[y == 0], W)
            def d(u, v):
                lo, hi = ci(res[u][1] - res[v][1]); return f"{res[u][0] - res[v][0]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            L.append(f"| {cfg} | {cname} | {int(y.sum())} / {int((1 - y).sum())} | {res['A0'][0]:.3f} | {res['A'][0]:.3f} | "
                     f"{res['B'][0]:.3f} | {d('A', 'A0')} | {d('B', 'A')} | {d('B', 'A0')} |")
    L.append("")
    L += incremental(m, "sim_seed", "## R2b. Incremental value, folds grouped by seed", SEED_NOTE)
    # R3
    L += ["## R3. W1(H0) AUROC by child size n_child (S1 vs S2, all 50 neighbours)", "",
          "n_child = topics in the S1 corpus children / in the S2 query child.", "",
          "| config | n_child | n S1 / S2 | W1 AUROC [95% CI] | null W1 AUROC [95% CI] | mean-cosine AUROC [95% CI] |", "|---|---|---|---|---|---|"]
    for cfg in m.config.unique():
        g = m[m.config == cfg]
        codes, W = boot_weights(g.cluster)
        g = g.assign(code=codes)
        top = int(g.n_child.max())
        for lo_, hi_ in BINS:
            if lo_ > top:
                continue
            gb = g[(g.n_child >= lo_) & (g.n_child <= hi_)]
            A, Bd = gb[gb.scenario == 1], gb[gb.scenario == 2]
            cells = []
            for f in ("all_w1_h0", "null_all_w1_h0", "cos_mean"):
                a, bt = auc_ci(A[f].to_numpy(float), Bd[f].to_numpy(float), A.code.to_numpy(), Bd.code.to_numpy(), W)
                l, h = ci(bt); cells.append(f"{a:.3f} [{l:.3f}, {h:.3f}]")
            L.append(f"| {cfg} | {lo_}–{min(hi_, top)} | {len(A)} / {len(Bd)} | " + " | ".join(cells) + " |")
    L.append("")
    return L


# --- distractor variant (distractor.py): top-50 retrieved from n = 2,000 -------------
def distractor_tables(disd):
    files = sorted(disd.glob("distractor_*_seed*.csv*"))
    if not files:
        return []
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("all_ltmax_h1", "null_all_ltmax_h1"):
        m[c] = m[c].fillna(0.0)
    m["cluster"] = m.sim_seed.astype(str) + "_" + m.parent_group.astype(str)
    L = ["# Distractor variant: top-50 retrieved from n = 2,000 (K/n = 0.025; all 50 retrieved neighbours)", "",
         "S1: designed 50 own children of size n_child + 1,950 children of other parents (uniform sizes); S2: the 50 parents +",
         "1,950 freshly generated parents (n_parent topics, same sigma); S3 mixed: its designed 50 + 1,950 children of other",
         "parents. Designed fraction = share of the top-50 that are the designed corpus. Features/null/cosine stats/coherence",
         "computed on the retrieved 50; LT_max = 0 when no H1 point. Seeds 0-4; bootstrap as above.", "",
         "## DT1. Retrieval composition", "", "| config | arm | n queries | K/n | designed fraction median [IQR] | own parent in top-50 (S2) |",
         "|---|---|---|---|---|---|"]
    lab = {1: "S1 multi-factual", 2: "S2 polysemic", 4: "S3 mixed"}
    for cfg in m.config.unique():
        for sc in sorted(m[m.config == cfg].scenario.unique()):
            g = m[(m.config == cfg) & (m.scenario == sc)]
            q = g.frac_designed.quantile([.25, .5, .75]).to_numpy()
            op = f"{g.own_parent_retrieved.mean():.3f}" if sc == 2 else "–"
            L.append(f"| {cfg} | {lab[sc]} | {len(g)} | {K_N:.3f} | {q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}] | {op} |")
    L += ["", "## DT2. Persistence features vs matched-cosine null (retrieved 50)", "",
          "| config | comparison | feature | Δmedian | AUROC [95% CI] | null AUROC [95% CI] | AUROC − null [95% CI] | per-seed AUROC |",
          "|---|---|---|---|---|---|---|---|"]
    rowsB = []
    for cfg in m.config.unique():
        g0 = m[m.config == cfg]
        codes, W = boot_weights(g0.cluster)
        g0 = g0.assign(code=codes)
        for sb, cname in CMPS:
            A, Bd = g0[g0.scenario == 1], g0[g0.scenario == sb]
            if Bd.empty:
                continue
            for f in TOPO:
                ra, rb = auc_ci(A[f].to_numpy(float), Bd[f].to_numpy(float), A.code.to_numpy(), Bd.code.to_numpy(), W)
                na, nb = auc_ci(A["null_" + f].to_numpy(float), Bd["null_" + f].to_numpy(float), A.code.to_numpy(), Bd.code.to_numpy(), W)
                ps = [wauc(A[A.sim_seed == s][f].to_numpy(float), Bd[Bd.sim_seed == s][f].to_numpy(float),
                           np.ones((A.sim_seed == s).sum()), np.ones((Bd.sim_seed == s).sum()))[0] for s in sorted(A.sim_seed.unique())]
                lo, hi = ci(rb); nlo, nhi = ci(nb); dlo, dhi = ci(rb - nb)
                L.append(f"| {cfg} | {cname} | {NAMES[f]} | {A[f].median() - Bd[f].median():.3f} | {ra:.3f} [{lo:.3f}, {hi:.3f}] | "
                         f"{na:.3f} [{nlo:.3f}, {nhi:.3f}] | {ra - na:+.3f} [{dlo:+.3f}, {dhi:+.3f}] | {min(ps):.3f}–{max(ps):.3f} |")
            cells = []
            for f in COS + ["pair_coherence", "n_components"]:
                a, bt = auc_ci(A[f].to_numpy(float), Bd[f].to_numpy(float), A.code.to_numpy(), Bd.code.to_numpy(), W)
                lo, hi = ci(bt); cells.append(f"{a:.3f} [{lo:.3f}, {hi:.3f}]")
            rowsB.append(f"| {cfg} | {cname} | " + " | ".join(cells) + " |")
    L += ["", "## DT3. Cosine-statistic and coherence AUROCs (retrieved 50)", "",
          "| config | comparison | " + " | ".join(NAMES[f] for f in COS + ["pair_coherence", "n_components"]) + " |",
          "|---|---|" + "---|" * 6] + rowsB
    L += [""] + incremental(m, "sim_seed", "## DT4. Incremental value (grouped logistic, folds by seed; models as in R2)", SEED_NOTE)
    m["n_child"] = np.where(m.scenario == 2, m.query_nfeat, m.corpus_nfeat)
    L += bands(m, [("W1", "all_w1_h0"), ("null W1", "null_all_w1_h0"), ("LT_max", "all_ltmax_h1"), ("null LT_max", "null_all_ltmax_h1"),
                   ("max H0 bar", "all_h0_maxbar"), ("null max bar", "null_all_h0_maxbar"), ("mean cosine", "cos_mean")],
               "## DT5. AUROC [95% CI] by child size (S1 vs S2, distractor variant, retrieved 50)")
    L += ["## DT6. Designed fraction of the top-50 by child size (median)", "", "| config | n_child | S1 multi-factual | S2 polysemic | S3 mixed |", "|---|---|---|---|---|"]
    for cfg in m.config.unique():
        g = m[m.config == cfg]
        for lo_, hi_ in BINS:
            gb = g[(g.n_child >= lo_) & (g.n_child <= hi_)]
            if gb.empty:
                continue
            med = [f"{gb[gb.scenario == sc].frac_designed.median():.2f}" if (gb.scenario == sc).any() else "–" for sc in (1, 2, 4)]
            L.append(f"| {cfg} | {lo_}–{min(hi_, int(g.n_child.max()))} | " + " | ".join(med) + " |")
    L.append("")
    return L


K_N = 50 / 2000
