# vmf_stats.py -- analyses (a) and (b) of the vMF plan on the vmf_<model>.csv.gz tables from vmf.py, plus the
# longest-edge audit and a reproducibility check against features_<model>.csv.{gz,xz}.
#   (a) main contrast, full arm: AUROC P(Q750|C250 > Q250|C750) for vmf2_r5 (and r3, r8, eig2).
#   (b) fixed query length: Q250 searched against C750 vs the size-matched C250, full arm (1 control draw) and
#       leave-document-out arm (20 draws, one AUROC per draw, averaged); real minus control; paired document-bootstrap
#       interval (Bayesian bootstrap over documents, 2,000 draws, seed 0), as in bayes.py.
#   audit: share of real neighbourhoods whose longest MST edge cuts off 1, 2-3 or >= 4 chunks (all cells pooled).
#   check: le_len (longest MST edge of the real offsets) must equal features' longbar for the same query and cell.
# Orientation: in (b) AUROC = P(size-matched C250 > C750), as bayes.py and the paper's fixed-length table;
# d = AUROC_real - mean AUROC_control in that orientation. auc_real / auc_ctrl are reported so the side of 0.5 is visible.
# Usage: [OUTROOT=out_clean] [VMF_K=100] python vmf_stats.py  -> $OUTROOT/vmf_stats{_k100}.csv and .md
import os
import numpy as np
import pandas as pd
from bayes import CORPORA, MODELS, wauc

K = int(os.environ.get("VMF_K", "50")); TAG = "" if K == 50 else f"_k{K}"
B = 2000
FEATS = ["vmf2_r5", "vmf2_r3", "vmf2_r8", "eig2"]


def boot_d(a_real, b_real, a_ctrl, b_ctrl, da, db, rng):
    """Paired Bayesian bootstrap over documents of AUROC(real) - mean_k AUROC(control_k). a_ctrl: n_a x k."""
    docs = np.unique(np.r_[da, db]); ix = {d: i for i, d in enumerate(docs)}
    ia, ib = np.array([ix[d] for d in da]), np.array([ix[d] for d in db])
    W = rng.dirichlet(np.ones(len(docs)), size=B)
    Wa, Wb = W[:, ia], W[:, ib]
    r = wauc(a_real, b_real, Wa, Wb)
    c = np.mean([wauc(a_ctrl[:, k], b_ctrl[:, k], Wa, Wb) for k in range(a_ctrl.shape[1])], axis=0)
    one = lambda x, y: wauc(x, y, np.ones((1, len(x))), np.ones((1, len(y))))[0]
    est = one(a_real, b_real) - np.mean([one(a_ctrl[:, k], b_ctrl[:, k]) for k in range(a_ctrl.shape[1])])
    return est, r - c


def main():
    rows, audit, check = [], [], []
    for corp, p in CORPORA.items():
        for m in MODELS:
            f = p / f"vmf{TAG}_{m}.csv.gz"
            if not f.exists():
                continue
            V = pd.read_csv(f)
            sel = lambda v, c: V[(V.variant == v) & (V.cell == c)]
            one = lambda x, y: wauc(x, y, np.ones((1, len(x))), np.ones((1, len(y))))[0]
            for feat in FEATS:  # (a)
                rows.append(dict(corpus=corp, model=m, analysis="main", feature=feat,
                                 est=one(sel("primary", "Q750|C250")[feat].to_numpy(float), sel("primary", "Q250|C750")[feat].to_numpy(float))))
            for arm, (va, vb) in {"all": ("primary", "dens"), "lodo": ("lodo", "lodo_dens")}.items():  # (b)
                a, b = sel(va, "Q250|C750"), sel(vb, "Q250|C250")
                for feat in ("vmf2_r5", "eig2"):
                    base = {"vmf2_r5": "vmf2_enull", "eig2": "eig2_enull"}[feat]
                    cn = [base] if arm == "all" else [f"{base}{k:02d}" for k in range(20)]
                    est, d = boot_d(b[feat].to_numpy(float), a[feat].to_numpy(float), b[cn].to_numpy(float), a[cn].to_numpy(float),
                                    b.lecture.to_numpy(), a.lecture.to_numpy(), np.random.default_rng(0))
                    ar = one(b[feat].to_numpy(float), a[feat].to_numpy(float))
                    ac = float(np.mean([one(b[c].to_numpy(float), a[c].to_numpy(float)) for c in cn]))
                    rows.append(dict(corpus=corp, model=m, analysis=f"fixed_{arm}", feature=feat, auc_real=ar, auc_ctrl=ac, est=est,
                                     lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), n_draws=len(cn)))
            n = V.le_small_n
            audit.append(dict(corpus=corp, model=m, n=len(V), one=(n == 1).mean(), two_three=n.between(2, 3).mean(), four_plus=(n >= 4).mean()))
            F = p / f"features_{m}.csv.gz"
            F = F if F.exists() else F.with_suffix(".xz")   # shipped tables are .csv.xz
            if F.exists() and K == 50:
                Fe = pd.read_csv(F, usecols=["variant", "cell", "query_id", "longbar"])
                j = V[["variant", "cell", "query_id", "le_len"]].merge(Fe, on=["variant", "cell", "query_id"])
                check.append(dict(corpus=corp, model=m, n_matched=len(j), max_abs_diff=float((j.le_len - j.longbar).abs().max())))
            print(corp, m, "done", flush=True)
    R = pd.DataFrame(rows); A = pd.DataFrame(audit); C = pd.DataFrame(check)
    out = CORPORA["nobel"]
    R.to_csv(out / f"vmf_stats{TAG}.csv", index=False); A.to_csv(out / f"vmf_audit{TAG}.csv", index=False)
    if len(C):
        C.to_csv(out / "vmf_repro_check.csv", index=False)
    md = [f"# vMF statistic (docs/vmf_plan.md), K = {K}", "", R.round(4).to_string(index=False), "", "## Longest-edge audit (share of neighbourhoods)", "",
          A.round(3).to_string(index=False)] + (["", "## Reproducibility: le_len vs features longbar", "", C.to_string(index=False)] if len(C) else [])
    (out / f"vmf_stats{TAG}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:3]))


if __name__ == "__main__":
    main()
