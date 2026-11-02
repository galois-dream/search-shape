# Fields medalists' ICM lectures: predictions written before any feature was computed
Written on day 1 at 20:54. Corpus: ICM lectures by Fields medalists (open IMU proceedings), selection rule in README once built; same pipeline as Nobel / essays / Turing / Wikipedia.
Context: on Turing, prediction 3 failed (longest bar beat the corpus null in 2 of 5 models, predicted >= 3). Mathematical text is formula-heavy; extraction breaks equations into symbol soup, which may make chunks less coherent.
If the paper's conclusions are right, on this corpus (all 5 models unless noted):
1. W1(H0), main contrast: multi-factual-like > polysemic-like, AUROC > 0.7 in every model.
2. W1, leave-document-out, fixed 250-token queries: |real - corpus-bearing null| <= 0.08 AUROC in at least 4 of 5 models.
3. Longest H0 bar, leave-document-out, fixed 250-token queries: real - corpus-bearing null > 0 with the real AUROC's 95% interval above the null, in at least 2 of 5 models (lowered from 3 after Turing; stated before seeing data).
4. Ratio (matched-cell) test under leave-document-out: W1 away from 0.5 in at least 4 of 5 models.
5. Flexible non-topological rival (3-small, ModernBERT): persistence adds |B'-A'| <= 0.01.
What would count against us: (2) or (5) failing reopens "topology adds nothing beyond cosines and corpus geometry"; (3) failing in all 5 models would mean the longest-bar effect does not reach mathematical text.

## Addendum: model-based prediction, written on day 1 at 21:02, still before any Fields feature exists
The hierarchical model fitted to the four earlier corpora (`code/realtext/bayes.py`, run at 21:01; corpus and model random effects) gives,
for the longest-bar advantage over the corpus-bearing null in a NEW corpus: posterior predictive P(theta_new > 0) = 0.67, 95% interval [-0.25, +0.36].
For W1: P(theta_new > 0) = 0.49 (no expected effect).
The Fields result will be scored against this predictive (log score and whether its corpus effect falls inside the interval) as well as against predictions 1-5 above.
