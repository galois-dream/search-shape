# How far can W1 and Lmax depart from nearest-neighbour statistics when the cosines differ?

Written 2026-11-02 07:59 CET. Numbers from `python3 -I paper/theory/check_bound.py 300` run 2026-11-02 07:58 CET
(synthetic only; seed 20261102). Notation follows `paper/v4/main.tex` (sec:geometry, eq:dist, sec:nnbars, eq:ratio,
app:whylittle). Nothing here has been checked on real embeddings.

## Notation

Unit query `q`, unit passages `c_1..c_N` (N = 50), angle `θ_i`, `c_i = cos θ_i q + sin θ_i t_i`, bearing `t_i ⊥ q`.
`a_i = ‖c_i − q‖ = 2 s_i` with `s_i = sin(θ_i/2)`, `κ_i = cos(θ_i/2)`, `Δ_ij = θ_i − θ_j`.
Offset directions `u_i = (c_i − q)/a_i`; `d_ij = ‖u_i − u_j‖`.
**Rescaled passage metric** `ρ_ij = ‖c_i − c_j‖ / √(a_i a_j)` (symmetric; equals eq:ratio when `a_i = a_j`).
For any metric `m` on the N points with distinct distances: `T_m` its MST, `e_1(m) ≤ … ≤ e_{N−1}(m)` its sorted edge
lengths, `W1(m) = Σ_k e_k(m) / (2(N−1))`, `Lmax(m) = e_{N−1}(m)`, `δ_i(m) = min_{j≠i} m_ij`.
The paper's statistic: `r_i = min_j ‖c_i − c_j‖ / a_i`, `S_r = (1/2N) Σ_i r_i`. Write `S_m = (1/2N) Σ_i δ_i(m)`.

## 1. Pairwise: the cosines distort only the radial term, exactly

**Lemma 1 (exact identity).** For all i, j,

    ‖c_i − c_j‖² = 4 sin²(Δ_ij/2) + sin θ_i sin θ_j ‖t_i − t_j‖²,                       (1)
    d_ij² = ρ_ij² − E_ij,   E_ij = 4 sin²(Δ_ij/4) · [cos²(Δ_ij/4) / (s_i s_j) − 1].       (2)

*Proof.* (1): expand `‖(cos θ_i − cos θ_j) q + sin θ_i t_i − sin θ_j t_j‖²`, use `t_i·t_j = 1 − ½‖t_i − t_j‖²` and
`(cos θ_i − cos θ_j)² + (sin θ_i − sin θ_j)² = 4 sin²(Δ/2)`. Divide (1) by `a_i a_j = 4 s_i s_j`; since
`sin θ_i = 2 s_i κ_i`, the bearing term becomes `κ_i κ_j ‖t_i − t_j‖²`, which is exactly the bearing term of eq:dist.
Subtract eq:dist; with `sin(Δ/2) = 2 sin(Δ/4) cos(Δ/4)` the radial terms differ by `E_ij`. ∎

So the bearings enter `d` and `ρ` identically; only the radial term differs (`2 sin(Δ/4)` in `d`, about
`2 sin(Δ/2)/√(a_i a_j)` in `ρ`, an inflation of roughly `1/√(s_i s_j)`, i.e. 2–3× for cosines in [0.5, 0.8]).
Equal angles give `E = 0`, recovering eq:ratio.

**Corollary 1.** If all cosines are ≥ 0 (θ ≤ π/2), then `E_ij ≥ 0` (since `s_i s_j ≤ ½ < cos²(π/8) ≤ cos²(Δ/4)`), and

    0 ≤ ρ_ij − d_ij = E_ij / (ρ_ij + d_ij) ≤ √E_ij,      E_ij ≤ (Δ_ij²/4)(1/(s_i s_j) − 1).          (3)

(`(ρ − d)² ≤ (ρ − d)(ρ + d) = E`; `4 sin²(x/4) ≤ x²/4`.) For θ_i ∈ [θ_lo, θ_hi] put `E_max = sup E` over the box
(for [0.5, 0.8] attained, numerically, at the two ends; `|Δ| ≤ |cos θ_i − cos θ_j| / sin θ_lo` converts to cosines).

**Multiplicatively the two metrics are NOT uniformly close** (a clean negative): when `t_i = t_j`,
`d_ij/ρ_ij = √(s_i s_j)/cos(Δ/4)`, which stays near `s_i ≈ 0.32–0.5` even as Δ → 0. So `ρ ≥ d ≥ ρ √(s_i s_j)/cos(Δ/4)`
is the best bi-Lipschitz statement and it is tight (radially aligned pairs). The useful closeness is additive, (3),
and it vanishes with the spread of the angles.

## 2. Sorted MST edges, W1 and Lmax under the distortion

**Lemma 2 (monotone/stable MST).** Let `m ≤ m'` pointwise. Then (a) `e_k(m) ≤ e_k(m')` for every k;
(b) `Σ_k e_k(m') ≤ Σ_{e∈T_m} m'_e`; (c) `Lmax(m') ≤ max_{e∈T_m} m'_e`; (d) if also `m' ≤ m + η` then `e_k(m') ≤ e_k(m) + η`.

*Proof.* (a) `e_k(m)` is the least t at which the graph `{m ≤ t}` has ≤ N − k components; `{m' ≤ t} ⊆ {m ≤ t}`.
(b) `T_m` is a spanning tree, so its m'-weight is at least the minimum. (c) `Lmax` is the minimax spanning-tree edge
(the MST is a bottleneck spanning tree). (d) apply (a) to `m'` and `m + η`. ∎

(a)+(d) is the H0 case of bottleneck stability of Rips persistence under a sup-norm perturbation of the metric
[Chazal, Cohen-Steiner, Guibas, Mémoli, Oudot 2009; Chazal, de Silva, Oudot 2014: `d_B ≤ 2 d_GH ≤ ‖m − m'‖_∞`].
Part (b)/(c) — only the N − 1 tree edges matter — is folklore for MSTs but is what makes the bound usable.

**Theorem 1 (offset barcode vs rescaled passage barcode).** With cosines ≥ 0,

    0 ≤ W1(ρ) − W1(d) ≤ (1/(2(N−1))) Σ_{(i,j)∈T_d} E_ij / (ρ_ij + d_ij) ≤ √E_max / 2,
    0 ≤ Lmax(ρ) − Lmax(d) ≤ max_{(i,j)∈T_d} E_ij / (ρ_ij + d_ij) ≤ √E_max,
    e_k(d) ≤ e_k(ρ) ≤ e_k(d) + √E_max  for every k.

*Proof.* Corollary 1 gives `d ≤ ρ ≤ d + √E_max`; apply Lemma 2 with `m = d`, `m' = ρ`, and (3) on the tree edges. ∎

## 3. W1 against nearest-neighbour statistics: an exact decomposition

**Proposition 2 (NN decomposition; any metric with distinct distances).** Let `p` be the number of mutual
nearest-neighbour pairs, `M` the sum of their lengths, and `G` the sum of the MST edges that are not NN edges. Then
the NN graph is a forest with exactly `p` trees, it has `N − p` edges, all in the MST, and

    2(N−1) W1(m) = Σ_i δ_i(m) − M + G,      G consists of p − 1 edges, each ≤ Lmax(m) and ≥ max(δ_i, δ_j).   (4)

*Proof.* NN edges are in the MST (cut property; app:whylittle item 2). The map `i ↦ NN(i)` is a functional graph:
every component has exactly one cycle, and along a pointer path the distances strictly decrease, so every cycle is a
mutual pair. Hence components = p, distinct NN edges = N − p, and `Σ_i δ_i` counts each mutual edge twice and each
other NN edge once, i.e. NN-forest weight = `Σ δ_i − M`. The remaining p − 1 MST edges join the p trees. ∎

So `W1(m) = (N/(N−1)) S_m + (G − M)/(2(N−1))`, with crude bounds `Σδ/2 ≤ 2(N−1)W1 ≤ Σδ − M + (p−1)Lmax`.

**Theorem 2 (W1 vs the paper's statistic, three named error pieces).** Exactly,

    W1(d) − (N/(N−1)) S_r  =  (G_d − M_d)/(2(N−1))  +  (N/(N−1))(S_d − S_ρ)  +  (N/(N−1))(S_ρ − S_r),
                               gap − mutual            radial distortion        asymmetric scaling

with, for cosines ≥ 0 and `j(i)` the d-nearest neighbour of i,

    −(1/2N) Σ_i E_{i j(i)}/(ρ + d)_{i j(i)} ≤ S_d − S_ρ ≤ 0,     √(a_i/a_max) r_i ≤ δ_i(ρ) ≤ √(a_i/a_min) r_i.

*Proof.* (4) for `m = d`, then add and subtract. `d ≤ ρ` gives `δ_i(d) ≤ δ_i(ρ) ≤ ρ_{i j(i)}`; apply (3).
`ρ_ij = (‖c_i − c_j‖/a_i) √(a_i/a_j)` and `a_min ≤ a_j ≤ a_max`. ∎

**Proposition 3 (Lmax: no upper bound by NN statistics, even at equal cosines).** `Lmax ≥ max_i δ_i` (app:whylittle
item 5), but two tight groups of 25 bearings at bearing distance g have `max δ → 0` while `Lmax → cos(θ/2)·g`.
So no function of the NN distances bounds Lmax from above, whatever the cosine spread. ∎ This is consistent with
the measured median Spearman 0.65 for Lmax vs max r_i: the cosines are not what breaks it.

## 4. Constants for cosines in [0.5, 0.8]

θ ∈ [36.9°, 60°], Δ_max = 0.404 rad, s ∈ [0.316, 0.5], a ∈ [0.632, 1.0].
- `E_max = 0.214`, `√E_max = 0.462` (pair at cos 0.5 and 0.8). Pairs with |Δcos| ≤ 0.05 / 0.1 / 0.2: `√E ≤ 0.11 / 0.20 / 0.35`.
- Uniform bound: `0 ≤ W1(ρ) − W1(d) ≤ 0.231`; `Lmax(ρ) − Lmax(d) ≤ 0.462`. **Useless**: W1 itself is 0.05–0.6 here.
- Asymmetric scaling: `δ_i(ρ)/r_i ∈ [0.795, 1.258]`. Also loose.
- Bi-Lipschitz: `d/ρ ∈ [≈0.32, 1]`, tight. Useless as a uniform statement.

The worst-case constants are therefore too loose to explain anything. **The tree-edge form of Theorem 1 and the
exact pieces of Theorem 2 are tight**, because MST/NN edges join passages that are close, hence usually at similar
cosines, and E_ij is quadratic in Δ_ij.

## 5. Numerical check (check_bound.py, 300 synthetic queries per family, D = 256, N = 50)

All identities hold to ≤ 2·10⁻¹⁴ in every query (Lemma 1, Prop. 2, monotonicity, Theorem 1, Theorem 2 bounds).

| family (cos in [0.5,0.8] unless noted) | (W1ρ−W1d)/W1d median / max | actual ÷ tree bound, median (min) | actual ÷ uniform bound, max | Spearman W1 vs S_r / vs S_ρ | pairs certified concordant / observed |
|---|---|---|---|---|---|
| random bearings | 0.008 / 0.013 | 0.79 (0.57) | 0.034 | 0.77 / 0.94 | 0.00 / 0.79 |
| clustered bearings | 0.045 / 0.29 | 0.76 (0.34) | 0.115 | 0.998 / 0.998 | 0.72 / 0.98 |
| clusters at own cosine level | 0.004 / 0.066 | 0.99 (0.52) | 0.056 | 0.998 / 0.998 | 0.85 / 0.98 |
| radial (tight bearings, spread cosines; adversarial) | 0.070 / 0.43 | 0.53 (0.32) | 0.071 | 0.997 / 0.997 | 0.68 / 0.98 |
| retrieval top-50 (cosines from geometry, median range [0.62, 0.67]) | 0.001 / 0.012 | 1.00 (0.80) | 0.030 | 0.999 / 0.999 | 0.93 / 0.99 |

"Certified concordant": a pair of queries whose `(N/(N−1))S_r` differ by more than the sum of their Theorem 2 error
bounds (radial piece bounded, the other two exact) must be ranked the same by W1 (if `|X−Y| ≤ η` per query, pairs with
`|Y_a − Y_b| > η_a + η_b` are concordant). Observed concordance = Kendall-type pair agreement.

Error pieces, median |piece| ÷ across-query SD of W1: clustered 0.07 (gap−mutual) / 0.09 (radial) / 0.01 (asym);
retrieval 0.05 / 0.002 / 0.004; random bearings 3.4 / 1.7 / 8.6 (SD of W1 only 0.003 there).
Spread sweep (clustered, cosines in 0.65 ± w/2): median relative W1 distortion 0 / 0.8% / 2.5% / 3.6% / 9.8% for
w = 0 / 0.1 / 0.2 / 0.3 / 0.5; Spearman W1 vs S_r stays 0.996–0.998 throughout.

Readings (synthetic; hedge accordingly):
1. Unequal cosines move W1 by a median ≤ 5% of W1 for spreads up to [0.5, 0.8]; the tree bound captures 53–100% of
   the actual shift; the uniform bound is roughly 10–100× too loose (actual ÷ uniform: max 0.03–0.12, median 0.007–0.05).
2. Rank agreement is governed by **error ÷ across-query spread of W1**, not by error alone. Where queries differ
   (SD ≈ 0.1) every piece is < 0.1 SD and Spearman ≈ 0.998; where they don't (random bearings, SD 0.003), tiny errors
   scramble ranks (0.77). The certificate explains 68–93% of pairs; the rest agree for reasons the bound does not see.
3. In realistic families the **largest piece is gap − mutual (arrangement), not the cosines**.
4. The symmetric `S_ρ` (divide by `√(a_i a_j)`) ranked W1 better than the paper's `S_r` in the one family where they
   differed (0.94 vs 0.77). Untested on real data.

## 6. Does this explain the measured 0.93–0.997?

Partly, and only in data-dependent form. No statement in terms of `[θ_lo, θ_hi]` alone can, because (i) the uniform
constants are of the order of W1 itself, and (ii) rank agreement depends on the across-query spread of W1, which no
per-query bound controls. What the theorems do give is an exact accounting: the departure of W1 from `(N/(N−1))S_r` is
gap − mutual + radial + asymmetric, each computable per query, the radial one bounded by Theorem 1/2. Whether the
measured 0.93 cells (Wikipedia) are low because of the gap piece or the cosine pieces is answerable from data (below).

**Conjecture (unproved).** For neighbours sampled i.i.d. from a smooth density on a set of intrinsic dimension k,
MST length and Σ NN distance are both Euclidean functionals with the same `n^{(k−1)/k} ∫ f^{(k−1)/k}` growth
[Steele 1988; Yukich 1998, I believe — citations unverified], so `W1/S` → a constant depending on k only. That would
explain high rank agreement across queries of similar intrinsic dimension, and predicts lower agreement where k varies
across queries. N = 50 is far from asymptotic; treat as heuristic.

## 7. Novelty (honest)

- Lemma 1: elementary trigonometry on the tangent–normal decomposition already in sec:geometry. The observation that
  `ρ = ‖c_i − c_j‖/√(a_i a_j)` matches the bearing term **exactly**, leaving a radial-only residual, is, as far as I know,
  not in the literature, but it is a two-line computation; present it as a remark, not a theorem.
- Lemma 2 / Theorem 1: direct consequences of bottleneck stability for Rips filtrations [Cohen-Steiner, Edelsbrunner,
  Harer 2007; Chazal et al. 2009; Chazal, de Silva, Oudot 2014] and MST folklore; the tree-edge refinement is folklore.
- Prop. 2: NN graph ⊆ MST and "components = mutual pairs" are standard (functional-graph argument; e.g. Eppstein,
  Paterson, Yao 1997 on nearest-neighbour graphs, I believe). The exact identity (4) is, I believe, folklore too.
- Prop. 3: trivial counterexample.
- New, if anything: the assembly — the three-piece error identity (Theorem 2) with a computable certificate, and the
  finding that the uniform cosine-spread bound is useless while the tree-edge bound is tight. No new mathematics.

## 8. Suggested real-embedding check (not run; heavy: run on a separate machine)

Extend `paper/nobel/src/mstcheck.py` inside its per-query loop (it already has `U, X, a, DU, DC, T`):
`A = np.sqrt(2 − 2a)`; `RHO = DC / np.sqrt(np.outer(A, A))`; `θ = arccos(a)`; E from (2); assert `DU² ≈ RHO² − E`.
Record per query: `W1(RHO) − W1(DU)`, the tree bound, the three pieces of Theorem 2, `S_ρ`, and the certificate
fraction per cell. Print per corpus × model: median |piece| ÷ within-cell SD(W1), Spearman W1 vs `S_ρ` next to vs `S_r`,
and the per-query cosine range. Same 45 combinations, same `out_clean` cells.
