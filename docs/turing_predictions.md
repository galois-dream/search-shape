# Turing lectures: predictions written before any Turing feature was computed
Written on day 1 at 18:58, before any Turing feature was computed. Corpus: ACM Turing Award lectures with >= 3,000 extractable words (32 of 38), same pipeline as Nobel/essays/Wikipedia.
If today's claims are right, on Turing (all 5 models unless noted):
1. W1(H0), main contrast: multi-factual-like > polysemic-like, AUROC > 0.7 in every model.
2. W1, LODO, fixed 250-token queries: |real - corpus-bearing null| <= 0.08 AUROC in at least 4 of 5 models (cosines + corpus geometry account for W1).
3. Longest H0 bar, LODO, fixed 250-token queries: real - corpus-bearing null > 0 with the real AUROC's CI clear of the null, in at least 3 of 5 models (arrangement information in long-form lecture prose, as on Nobel and the essays). Of the five claims, this is the one most at risk.
4. Ratio (diagonal) test under LODO: W1 and longest bar both away from 0.5 in most models (the definition's ratio prediction fails).
5. Flexible non-topological rival (3-small, ModernBERT): persistence adds |B'-A'| <= 0.01.
What would count against us: (3) failing in most models would make the longest-bar result Nobel/essay-specific rather than a property of long-form prose; (2) or (5) failing would reopen the "topology adds nothing" conclusion.
