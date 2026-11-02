# Two-meaning von Mises–Fisher statistic: plan written before any real-text value was computed

Written on day 2 at 09:00, after the synthetic check (`code/realtext/vmf_highdim.py`: an
ambient-dimension vMF mixture never detects two meanings 30° apart at n = 50, D = 64–3,072; after projecting onto the
top 5 principal directions it does, as does the second Gram eigenvalue). No vMF quantity has been computed on any
real corpus at the time of writing. The authors decided to include it in the submission.

## Statistic (fixed now)

For each query, take the same 50 unit offset directions `u_i` that the persistence features use
(`features.py: offsets`, after the same retrieval exclusions).

1. **Project.** Let `V_r` hold the top `r = 5` right singular vectors of the centred 50 × D matrix. Set `y_i = u_i V_rᵀ`
   and renormalise each `y_i` to unit length. With m = 50 points, structure is estimable only in about r·log D
   dimensions (effective rank; Vershynin, *High-Dimensional Probability*, 2nd ed., Remark 5.6.3), so the ambient fit is
   not used.
2. **Fit.** On S^{r−1}, fit:
   - k = 1: a vMF with the mean direction in closed form and κ from the Banerjee et al. (2005) approximation;
   - k = 2: a vMF mixture with shared κ, by EM (50 iterations, 5 restarts, seeded by the query index).

   `log I_ν` uses the uniform (Debye) expansion.
3. **Score.** `vmf2 = BIC(k=1) − BIC(k=2)`, with `k(r−1) + 1 + (k−1)` parameters and n = 50. Positive values favour
   two meanings.
4. **Secondary statistics** (reported, not used for the conclusion):
   - `vmf2` with r = 3 and r = 8;
   - `eig2`, the second eigenvalue's share of the uncentred 50 × 50 Gram matrix of the `u_i`.

## Analyses (identical to the longest H0 bar's)

- **(a) Main contrast.** AUROC of Q750|C250 against Q250|C750 (full arm).
- **(b) Fixed query length, leave-document-out arm.** Q250 against the size-matched C250 and against C750.
  - Report the real AUROC minus the corpus-bearing control's AUROC.
  - The control uses the same 20 draws as `features.py` (`default_rng((k, 1, query index))`), with one AUROC per draw,
    averaged.
  - Interval: a paired document bootstrap, as in `bayes.py`.
  - Count, per corpus, the models whose paired interval lies above 0.
  - The same comparison is made in the full arm with one control draw.
- **(c) Flexible classifier.** Out-of-fold AUROC of A′ against A′ + `vmf2` (same folds and 200 refits as
  `controls.py hgb`), for the five models if time allows; otherwise text-embedding-3-small and ModernBERT first.

## Predictions (written now; they may fail)

- **P1.** `vmf2` separates the main contrast with AUROC above 0.7 in at least 20 of the 25 corpus × model
  combinations. Query length drives it, as it drives every feature.
- **P2.** At fixed query length (leave-document-out), `vmf2` beats its corpus-bearing control where the longest bar
  does: at least 3 of 5 models on Nobel and at least 3 on the essays; at most 1 of 5 on Wikipedia.
- **P3.** Adding `vmf2` to the flexible classifier changes its out-of-fold AUROC by less than 0.01 in every
  combination fitted. The classifier already holds the top five Gram eigenvalues.

The AmbigQA labels remain untouched.

## Amendment 1 (on day 2 at 09:44, before any full real-text result)

**What changed.** `log I_ν` is now computed exactly with scipy's exponentially scaled `ive` for ν < 50. That covers
every order used here: r = 3, 5, 8 gives ν = 0.5, 1.5, 3. The uniform (Debye) expansion is kept for ν ≥ 50.

**Why.** At these small orders the Debye expansion's leading term is off by up to 0.05 in `log I` (κ from 1 to 100,
checked against the power series). Summed over 50 directions, that shifts the BIC gap by a few units.

**What had been seen when this was decided.**
- One 30-query smoke test (Nobel, text-embedding-3-small, leave-document-out, Q250|C750 only), run with the Debye
  version. It printed the medians of `vmf2_r5` (real −1.8; first control draw 35.4) and the sizes of the
  longest-edge sides.
- The partly finished full runs were stopped and their outputs deleted unread.

Nothing else changes: the statistic, analyses and predictions stay as written above.

## Amendment 2 (on day 2 at 09:54, before any full real-text result)

**What changed.** We add a sensitivity run with K = 100 neighbours. Everything else is unchanged: the statistic,
r = 5 (with 3 and 8), the corpus-bearing control drawing K chunks, and the analyses (a)–(c). Outputs are
`vmf_k100_<model>.csv.gz`.

**Why.** With K = 50, the fit has 50 points. The neighbourhood's effective rank (trace over largest eigenvalue of
the covariance of the 50 centred offsets) was measured on 200 Nobel queries before this run. Its median is
9.3 [10th–90th percentile 7.1, 12.9] for text-embedding-3-small and 8.9 [7.1, 11.5] for bge-base. Doubling K doubles
the points per fit.

**Caveat, stated now.** The 750-token chunkings hold 367–794 chunks, so K = 100 reaches 13–27% of the corpus and
mixes in neighbouring topics. K = 100 is a sensitivity check, not the main analysis. The human-labelled plan
already registers K ∈ {25, 50, 100}.
