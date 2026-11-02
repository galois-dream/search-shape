# Matched-cosine simulation: results against the registered predictions

Written on day 3 at 23:10 from `data/reference_outputs/matched/matched_auroc.csv` and `matched_summary.txt` (run on day 3, 22:51–23:07, on two
machines, 14 shards, 63 cells, 200 neighbourhoods per class per cell; analysis `matched_auroc.py` exactly as registered in
`docs/matched_plan.md` part 1, amendment 1 (day 2, 15:34) and amendment 2 (day 3, 22:51)). Written for the authors,
before the paper reported the simulation.

## Construction checks (read first)
- **Cosines equal by construction: holds.** Every cosine statistic has AUROC within 0.5 +- 0.05 in all 63 cells (M1).
- **One vs two, Level B: coherence matched** (AUROC 0.500 in every cell). This is the valid main comparison.
- **One vs three, Level B: coherence NOT matched** (AUROC 0.24–0.30). The one-meaning cloud cannot be widened enough
  to reach the three-way split's coherence; the pre-run check covered one vs two only. These cells cannot count as wins.
- **Ring vs one: partly matched** (0.57–0.67). **Ring vs two: not matched** (by design; the plan named no matching).
- **Within-band tests (Q1):** where a similarity statistic separates the classes perfectly there is no overlap to test
  (reported as missing); the band AUROCs at 1.00 rest on very few pairs and are not evidence.

## Verdicts (one vs two meanings)
- **Topology "wins"** (AUROC >= 0.75, interval clear of 0.5, cosines at 0.5, coherence at 0.5 at Level B): W1 in 18 of
  18 Level B cells (15 of 15 at Level A), L_max in 17 (15), LT_max in 12 (3). **Homology is not useless:** with the
  cosines and the mean similarity held equal, it sees one meaning against two.
- **Topology "adds"** (beyond the best of coherence, median nearest-neighbour cosine, Gram second-eigenvalue share,
  largest nearest-neighbour distance): **untestable in 32 of 33 cells** because a similarity statistic is already at
  AUROC >= 0.98 (the Gram second-eigenvalue share in 25 of the 33 cells, coherence in 15, the median nearest-neighbour
  cosine in 12, the largest nearest-neighbour distance in 8). **The one testable cell is the hardest** (two meanings
  30 deg apart, p = 5, tau = 0): W1 0.861 [0.833, 0.889] against the best rival 0.802, difference **+0.06
  [+0.02, +0.10] -> "adds"**; W1 also keeps separating within bands of the median nearest-neighbour cosine (0.91),
  effective rank (0.83) and coherence (0.94). L_max (0.63) and LT_max (0.54) are undecided there. One cell of 33:
  read as a lead, not a result.

## Registered predictions
| | prediction | outcome |
|---|---|---|
| M1 | cosine statistics at 0.5 +- 0.05 | holds (63/63) |
| M2 | Level A, p <= 20: coherence, Gram share, L_max >= 0.9 | fails as stated: coherence ~1.00, but the Gram share is 0.52–0.65 at p <= 5 and L_max 0.73–0.94 |
| M3 | Level B, p <= 20: coherence ~0.5, L_max and Gram share >= 0.8; wins but does not add | holds where testable: coherence 0.50, L_max 0.81–1.00 (0.63 at 30 deg), Gram share >= 0.96; "adds" untestable (ceiling) |
| M4 | every arrangement statistic falls towards 0.5 as p -> 767 | **fails**: at p = 767 L_max is 0.99–1.00 and the similarity statistics 1.00. Spreading the one-meaning cloud over all dimensions does not hide the difference; it makes it easier (larger nearest-neighbour distances) |
| M5 | LT_max weaker than L_max everywhere; informative only for p <= 5 | partly fails: weaker in most cells, but LT_max is 0.90–0.99 at p >= 20 |
| M6 | ring vs one, p <= 5: LT_max beats L_max | holds (0.99–1.00 against 0.81–0.92), with coherence only partly matched |
| M7 | vmf2 > 0 in > 50% of one-meaning clouds at p <= 2, falling with p | holds (100% at p = 2; <= 0.5% at p >= 5) |
| M8 | Gram share >= 0.9 for p <= 5 at tau = 0; < 0.75 at tau = 0.8 | partly fails: Level B 0.98–1.00 at tau = 0, but it stays 0.96–1.00 at tau = 0.8; Level A only 0.52–0.65 |
| M9 | real text (effective rank 8–9) sits where L_max separates Level B only partly (0.6–0.8) | fails: effective rank 8–9 matches the one-meaning clouds at p = 20, where L_max separates almost perfectly. Real chunkings do not behave like one-vs-two-meaning neighbourhoods of this construction |

## What it means (proposed reading)
1. When only the arrangement differs, persistence does detect two meanings; the "homology is useless" reading is wrong.
2. But simple similarity statistics (most often the Gram second-eigenvalue share) detect them just as well almost everywhere;
   only in the hardest case (30 deg apart, few spread dimensions) does W1 add a little. That is the paper's spine,
   "the similarity structure suffices", now with a known boundary.
3. "Too noisy to see" (M4) is not what this construction shows; the real-text weakness is not explained by noise of
   this kind (M9).
4. Design limits: one vs three and the ring comparisons are not properly matched; within-band tests lack overlap; one
   testable cell. A follow-up would match the median nearest-neighbour cosine by construction, with two spread knobs.

*Corrected after the results were written: two parentheticals named the median nearest-neighbour cosine as the
statistic at the ceiling; they now give the counts read from `matched_auroc.csv`. No number or verdict changed.*
