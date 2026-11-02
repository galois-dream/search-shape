# Matched-cosine simulation: can topology see what the cosines cannot?

Written on day 2 at 14:44, before any run, for the authors to approve; changes after approval become dated
amendments.

*Order: this plan (day 2), amendment 1 (day 2) and amendment 2 (day 3) were all written before any result was read.
Amendment 2 was written as the run started (day 3 at 22:51; the run took 22:51 to 23:07) and approved at 22:53, during
the run, before any result was read. The results are in `docs/matched_results.md`. Part 2 of the same plan, the transfer
test, is `docs/transfer_plan.md`.*

**Question.** By eq. (6) of the paper, every neighbourhood statistic is a function of the cosines to the query and the
bearings. If two classes of neighbourhood have *identical cosines*, only the arrangement of the bearings differs. Can
persistent homology then tell **one meaning** from **two meanings**, and how does that depend on how many directions the
spread occupies (a co-author's "too noisy" hypothesis)?

**Construction (per sample).**
- Query q = e₁ in D = 768 dimensions. A check at D = 1,536 is run for the main cell only.
- Cosines: 50 values drawn once per sample from uniform [0.55, 0.75]. The **same list** is used for its one-meaning and
  its two-meaning neighbourhood, so the cosine profiles match exactly.
- Bearings: unit vectors ⊥ q, built as `t_i ∝ m_g(i) + s·ξ_i`:
  - one meaning: a single centre m;
  - two meanings: two centres at bearing angle φ = 60°, 25 neighbours each;
  - spread `ξ_i`: Gaussian inside a random p-dimensional subspace ⊥ q, with p ∈ {2, 5, 20, 100, 767} (767 = all of q⊥).
- Neighbour i is then `cos θ_i q + sin θ_i t_i`.
- **Level A (cosines matched):** the same spread s for both classes, set so a bearing has cosine 0.6 to its centre.
- **Level B (cosines and mean pairwise similarity matched):** s is tuned per class so that both classes have the same
  mean pairwise cosine among the 50 neighbours (coherence). This is the paper's "two tight groups vs one wide group"
  case.
- Sizes: 200 samples per class per cell, seeds 0–4. This gives 5 values of p × 2 levels, plus 2 cells for φ = 30° and
  90° at p = 5, Level B.

**Statistics compared** (AUROC, one vs two meanings; cluster bootstrap over samples):
- cosine statistics (top-1, mean, std, slope);
- coherence (mean pairwise cosine);
- the share of the second eigenvalue of the 50 × 50 Gram matrix of the offsets;
- vmf2 (r = 5, as in `docs/vmf_plan.md`);
- W1(H0), L_max(H0), LT_max(H1), and the maximum nearest-neighbour distance among the offsets.

**What counts as topology "winning".** A persistence feature has AUROC ≥ 0.75 with its 95% interval clear of 0.5, where
the cosine statistics are at 0.5 (by construction) **and** coherence is within 0.05 of 0.5 (Level B).
- **Topology "adds" beyond simple arrangement statistics** if its AUROC also exceeds both the Gram eigenvalue share and
  the maximum nearest-neighbour distance by ≥ 0.05.

**Predictions (written now).**
- **M1.** Cosine statistics: AUROC within 0.5 ± 0.05 in every cell (a check of the construction).
- **M2.** Level A, p ≤ 20: coherence, the Gram eigenvalue share and L_max all reach AUROC ≥ 0.9.
- **M3.** Level B, p ≤ 20: coherence ≈ 0.5, while L_max and the Gram eigenvalue share stay ≥ 0.8. Topology "wins" here
  as defined above, but does not "add" beyond the Gram eigenvalue share or the maximum nearest-neighbour distance.
- **M4.** As p grows to 767, every arrangement statistic falls towards 0.5. The p at which L_max drops below 0.75 is
  reported, along with the simulated neighbourhoods' effective rank at that p. The real neighbourhoods' effective rank
  is about 9 (`data/reference_outputs/analyses/effective_rank.txt`), so the comparison shows whether real text lies on the "visible" or the
  "drowned" side.
- **M5.** LT_max is weaker than L_max everywhere, and informative only for p ≤ 5.

**What each outcome means for the paper.**
- **M3 holds:** homology is demonstrably *not useless*. It sees arrangement the cosines cannot, though simpler
  arrangement statistics see it too.
- **M3 fails** (topology at 0.5 even here): homology adds nothing even in its best case.
- **M4** places the real data. If real neighbourhoods sit where topology would still see two meanings, the real-text
  null is evidence about real ambiguity proxies, not about noise. If they sit where it would not, the "too noisy"
  explanation stands, and the paper says so.

**Cost.** About 30k persistence computations, roughly 30 minutes on one machine. Code: `code/simulation/matched/matched.py` (to write,
reusing `simcore.persistence_features` and `vmf.py`'s statistic).


## Amendment 1 to part 1 (on day 2 at 15:34, before any run): a co-author's comments on the visual guide

1. **Noise on top of the structure.** Noise should be taken into account: it sets the amplitude of the spread over all
   directions, and as a Gaussian sample it has zero mean spread for each neighbour.
   - Each bearing becomes `t_i ∝ m_g(i) + s·ξ_i + τ·η_i`: the low-dimensional spread `ξ_i` (in p directions) plus
     zero-mean isotropic Gaussian noise `η_i` over all of q⊥.
   - τ is set so that the noise-to-signal ratio matches the main simulation at σ = 0.1 and σ = 0.2: there, a passage's
     noise has norm σ√D against topic norm √n.
   - τ = 0 is kept as the noise-free reference.
   - This adds a second knob, so the p sweep runs at τ ∈ {0, τ(0.1), τ(0.2)}.
2. **More meanings, where loops rather than gaps may carry the difference.**
   - Add a three-meaning class (centres 120° apart in a plane of q⊥) and a ring class (bearings spread along a circle
     in a plane of q⊥, no gap).
   - Comparisons: one meaning vs three; ring vs one meaning; ring vs two meanings.
   - Prediction **M6**: LT_max beats L_max for ring vs one meaning at p ≤ 5. For three vs one, L_max and the Gram
     eigenvalue statistics stay ahead.
3. **Placing real text** (how is the placement inferred, and can LT_max be used?).
   - Real neighbourhoods are placed on the p axis by matching **several** yardsticks between simulated and real
     neighbourhoods, and whether they agree is reported:
     - effective rank (trace over largest eigenvalue);
     - participation ratio (trace² over the trace of the square);
     - the variance share of the top 5 directions;
     - the distribution of LT_max.
   - Real yardsticks are recomputed on the **clean** corpora, all models, with the paper's own retrieval (overlapping
     chunks excluded).
   - The earlier single number (effective rank about 9, all-text Nobel, two models, no exclusion) is only a first
     estimate.

## Amendment 2 to part 1 (on day 3 at 22:51, before any run): answers to Q1, Q4 and Q6

Written at the authors' request; approved by a co-author at 22:53, while the run was in progress, before any result
was read. Another co-author could still add to or override them before the results were read. Informed by the real-text results of day 3 (one similarity statistic matched persistence in 42 of
45 cells; the projected vMF test fires on elongation). No AUROC of any statistic in this simulation has been computed.

1. **Q1, matching.** Levels A and B stay as built. In addition, every arrangement statistic is tested *within bands*
   (deciles, pooled over the two classes) of each of: coherence; the **median nearest-neighbour cosine**; and the
   **effective rank** of the 50 offsets. A statistic sees arrangement beyond these only if its stratified AUROC stays
   clear of 0.5. The median nearest-neighbour cosine is added to the statistics computed.
2. **Q4, criteria.** "Topology wins" unchanged. **Adds**: the paired 95% bootstrap interval (over samples, 2,000 draws)
   of [persistence feature − best rival] lies above 0 and the point difference is >= 0.05; rivals = coherence, median
   nearest-neighbour cosine, Gram second-eigenvalue share, largest nearest-neighbour distance. **Redundant**: that
   interval lies within +-0.02. Otherwise **undecided**. A cell whose best rival has AUROC >= 0.98 is reported as **at
   the ceiling** (untestable) instead.
3. **Q6, predictions** (in addition to M1–M6):
   - **M7.** In the one-meaning class, vmf2 > 0 in more than 50% of samples at p <= 2, and the share falls as p grows.
   - **M8.** The Gram second-eigenvalue share separates one meaning from two with AUROC >= 0.9 for p <= 5 at tau = 0,
     and falls below 0.75 at tau = 0.8 (noise closes the eigengap; Davis–Kahan).
   - **M9.** On the yardsticks, clean real neighbourhoods (effective rank about 8–9) sit where L_max separates the
     Level B classes only partly (AUROC 0.6–0.8).
