# Length-controlled topic-count test: plan and predictions

Written on day 4 at 10:05, before any text was selected, embedded or measured. The authors approved running it just before.
Committed to git before the first feature is computed; changes after that are dated amendments below.

## Why
On real text the paper uses chunk length as a stand-in for topic count, so length and topic count move together. The
ratio prediction failed there, and the paper cannot tell whether the definition, the stand-in or a plain length effect
is responsible. This test varies topic count while holding the text and its length fixed.

## Construction (same text, same length; only the grouping changes)
- **Pages.** 240 English Wikipedia pages from the 2020-01-20 dump distributed with AmbigQA (CC BY-SA 3.0), drawn
  uniformly at random (seed 0) among pages with at least 1,494 `cl100k_base` tokens; titles containing
  "(disambiguation)", "List of" or "Index of" are excluded. A page stands for one topic.
- **Atoms.** The first 1,494 tokens of each page, cut into 18 atoms of 83 tokens (decoded back to text).
  Block b = atoms 3b..3b+2 (b = 0..5). Block 0 of every page goes to the query pool, blocks 1-5 to the corpus pool,
  so query text never appears in the corpus.
- **C1** (one-topic passages): every corpus block as one passage (3 consecutive atoms of one page): 1,200 passages.
- **C3** (three-topic passages): the same 3,600 corpus atoms regrouped at random into 1,200 passages of 3 atoms from 3
  different pages (grouping seed g).
- **Q1** (one-topic queries): every query block (240 queries).
- **Q3** (three-topic queries): the same 720 query atoms regrouped into 240 queries of 3 atoms from 3 different pages.
- Every query and passage is 249 tokens of the same text pool; C1 and C3 hold exactly the same text, as do Q1 and Q3.
- Cells: Q3|C1 multi-factual (3 topics, more than any passage); Q1|C3 polysemic (1, fewer than any passage);
  Q1|C1 and Q3|C3 matched (neither).
- **Models:** BAAI/bge-base-en-v1.5 and BAAI/bge-m3 (open weights, normalised), on two desktop machines. K = 50, exact cosine.
- **Arms:** primary "all" (the corpus as built; query atoms are never in it). Secondary "own pages out": passages that
  contain an atom of any of the query's pages are excluded. The primary arm keeps them because, unlike the chunking
  test, the passages that share the query's topics are the point here.
- **Replication:** grouping seeds g = 0, 1, 2 (C3 and Q3 regrouped; C1 and Q1 fixed). Primary estimate: g = 0; the
  range over g is reported.

## Measurements (per query)
W1(H0), L_max(H0), LT_max(H1) of the unit offsets (as in the paper); cosine statistics (top-1, mean, s.d., rank
slope); similarity statistics (median nearest-neighbour cosine, mean pairwise cosine, second-eigenvalue share);
random-bearing and corpus-bearing controls (5 draws each, averaged). AUROC with a 95% bootstrap over queries
(B = 2,000, seed 0).

## Predictions, registered now
- **L1 (construction).** Token counts are identical across Q1/Q3 and across C1/C3 (checked, not tested).
- **L2 (the definition's ratio prediction).** The matched cells Q1|C1 and Q3|C3 are indistinguishable for W1 and
  L_max: AUROC within 0.5 +- 0.05, in both models, primary arm.
- **L3 (regime direction, from the simulation).** W1 is larger for multi-factual queries (Q3|C1) than for polysemic
  ones (Q1|C3): AUROC > 0.5 with its 95% interval above 0.5, in both models, primary arm.
- **L4 (cosines, from the simulation).** The best single cosine statistic also separates Q3|C1 from Q1|C3 (AUROC >= 0.75).
- **L5 (sufficiency, from the real-text results).** On the main contrast, the median nearest-neighbour cosine comes
  within 0.02 of the best persistence feature (oriented AUROC), in both models.
- No prediction for the gain over the corpus-bearing control; it is reported.

## Reading
- L2 and L3 hold: the definition's regimes are visible when length is fixed, and the chunking result was a length effect.
- L2 fails with length fixed: the ratio-only prediction fails on topic count itself (for this construction), not
  only on the length stand-in.
- L3 fails: the features do not show the simulated direction on real text even when topic count is the only change.
Whatever the outcome, it is reported in the paper as an added, exploratory test with this plan in the supplement.
Limits stated in advance: a page is a coarse topic; random triples of unrelated pages are an extreme form of a
multi-topic passage; two models; one corpus.

## Amendment 1 (on day 4 at 10:17, before any feature was computed; bge-m3 still embedding)
The crane illustration (Appendix G) suggested that the definition's difference lives in how the query relates to its
neighbours, not in how the neighbours sit relative to each other: a polysemic query is covered by one passage, a
multi-factual one only by several. We add one statistic and one prediction:
- **Coverage gain.** The best cosine between the query and a normalised sum of two of its 50 neighbours that share no
  page, minus the best cosine to a single neighbour. Also reported for any two neighbours.
- **L6.** The coverage gain is larger for multi-factual queries (Q3|C1) than for polysemic ones (Q1|C3): AUROC > 0.5 with
  its 95% interval above 0.5, in both models, primary arm.
L1-L5 are unchanged. L6 was added after the crane example was seen, on different data, and is reported as such.
