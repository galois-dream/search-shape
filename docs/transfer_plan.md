# Transfer test: do topological summaries transfer across embedding models better than cosine statistics?

Written on day 2 at 15:13, before any run. **Approved by the authors**, together with the
framing it tests: for flagging the regimes, the cosine profile may be *sufficient*, and then the persistence summaries
are *redundant given it*, not unnecessary. Two sufficient statistics can still differ in their invariances. This test
asks whether a threshold on a topological summary carries over from one embedding model to another better than a
threshold on a cosine statistic, which the paper notes do not transfer.

**Data.** The clean real-text features (`data/realtext/` and `data/reference_outputs/` in this archive), every corpus and model available, on the fixed-query-length
comparison, leave-document-out arm. The label is 1 for the size-matched 250-token chunking and 0 for the 750-token one.
The queries are the same chunks in every model, so models can be paired query by query.

**Statistics** (one number per query; orientation and threshold learned on the source model):
- cosine statistics: top-1, mean, standard deviation, rank slope, NQC (standard deviation over mean), coherence;
- persistence summaries: W1(H0), L_max(H0), LT_max(H1);
- arrangement statistics: the largest nearest-neighbour distance, the mean resultant length R.

**Procedure.** For each corpus, each ordered pair of models (source s, target t; s ≠ t) and each statistic:
1. Fix the orientation and the threshold that maximise balanced accuracy on s.
2. Apply them to t, and record the **transfer loss**: the best balanced accuracy achievable on t minus the balanced
   accuracy obtained with s's threshold.

Report the median loss over pairs, per statistic and corpus, with a bootstrap over documents (B = 1,000). AUROC is
reported alongside, since it needs no threshold.

**Predictions (written now).**
- **T1.** Level-based cosine statistics (top-1, mean, coherence) transfer worst: median loss ≥ 0.10 in at least 3 of the
  5 corpora.
- **T2.** W1 and L_max lose less than the mean cosine in at least 4 of 5 corpora. Their offsets are unit directions,
  which removes the overall cosine level.
- **T3.** Scale-free cosine statistics (NQC, rank slope) transfer about as well as the persistence summaries: within
  0.03.

**What counts as "topology transfers better".** The median loss of W1 or L_max is lower than that of *every* cosine
statistic by ≥ 0.03 in at least 3 of 5 corpora. If so, the paper names invariance as a concrete reason to prefer the
topological description of the same information. If T3 holds instead, the advantage belongs to scale-free statistics in
general, not to topology.

**Amendment to part 2 (on day 2 at 15:14, before any run).** The largest nearest-neighbour distance is not stored in
the feature tables. Computing it would require recomputing every neighbourhood, so it is dropped from part 2. It
remains in part 1. Nothing else changes.

(Part 2 of a two-part plan; part 1, the matched-cosine simulation, is `docs/matched_plan.md`.)
