# Human-labelled check on AmbigNQ *train*: plan and predictions

Written on day 4 at 10:19, before any question was embedded, retrieved or measured. Committed before the first
feature is computed; later changes are dated amendments below.

## Why, and what it must not touch
The paper has no human-labelled ambiguous queries except three crane sentences. The registered test (`docs/analysis_plan.pdf`,
Stage 1 locked 14 days before day 1) evaluates AmbigNQ **dev** exactly once, after the Stage-2 code freeze. This check uses the
**train** split only, which the registered plan reserves for fitting, and reads nothing from dev. It is exploratory,
is not the registered test, and does not change it.

## Data and retrieval
- **Questions:** all 10,036 AmbigNQ train questions (`sewon/ambig_qa`, CC BY-SA 3.0). Label: 1 if the annotation is
  `multipleQAs` (ambiguous), else 0 (the registered primary label; train has a single annotation, so it is noisier).
- **Corpus:** the registered retrieval corpus (`chunks.parquet`, rows with
  `in_corpus`: 37,528 chunks, at most 5 per gold page, over the pages named in `nq_doc_title` for train and dev), with
  its stored all-MiniLM-L6-v2 embeddings. Known bias (as registered): every gold page is in the corpus.
- **Retrieval:** questions embedded with all-MiniLM-L6-v2 (normalised); top K = 50 by exact cosine.

## Measurements (per question; as in the length-controlled test)
W1(H0), L_max(H0), LT_max(H1) of the unit offsets; random- and corpus-bearing controls (5 draws each, averaged;
corpus bearings from random non-neighbour corpus chunks); cosine statistics (top-1, mean, s.d., rank slope);
similarity statistics (median nearest-neighbour cosine, coherence, second-eigenvalue share); coverage gain (best
cosine of the question with a normalised sum of two neighbours, minus the best single cosine; and the same for two
neighbours from different pages); question length in words.
AUROC = P(statistic larger for an ambiguous question than for a clear one), with a 95% bootstrap over questions
(B = 2,000, seed 0); "oriented" = max(AUROC, 1 - AUROC).

## Predictions, registered now
- **A1 (sufficiency).** The best persistence feature's oriented AUROC is no more than 0.02 above the best similarity
  statistic's.
- **A2 (arrangement beyond cosines).** For W1 and L_max, |AUROC(real) - AUROC(corpus-bearing control)| <= 0.05.
- **A3 (coverage; the definition's polysemic side).** Ambiguous questions are covered by one passage at least as well
  as clear ones: the coverage gain (different pages) is smaller for ambiguous questions, AUROC < 0.5 with its 95%
  interval below 0.5.
- **A4 (length rival).** Question length's oriented AUROC is at least that of every persistence feature.
- No threshold for "detects ambiguity" is set; every AUROC is reported.

## Reading
A1 and A4 holding would say that, on human labels too, persistence carries nothing that simpler statistics lack.
A3 tests the definition's prediction directly on real ambiguous questions; it can fail because AmbigNQ's "ambiguous"
is answer-ambiguity, not a topic count. Limits stated now: one model, train labels with one annotator, a corpus built
from the gold pages, no multi-factual questions (AmbigNQ has none).

## Amendment 1 (on day 4 at 11:13, after the A1-A4 results were seen; a co-author's idea): flipping questions into topics
AmbigNQ's ambiguous train questions carry human-written disambiguated questions, one per reading. For each ambiguous
train question with at least two disambiguations: **polysemic** = the original question; **multi-factual** = all its
disambiguated questions joined into one query (one topic per reading); **single** = its first disambiguated question.
Same retrieval and statistics as above. Contrasts: multi-factual vs polysemic, and multi-factual vs single.
- **F1 (definition, from the simulation).** W1 is larger for the multi-factual query than for the polysemic one:
  AUROC > 0.5 with its 95% interval above 0.5.
- **F2 (coverage).** The coverage gain (different pages) is larger for the multi-factual query than for the polysemic
  one: AUROC > 0.5 with its interval above 0.5.
- **F3 (length rival).** The joined query is longer by construction; question length is reported, and any statistic
  that does not beat length is read as tracking length.
Exploratory, added after the main results were known; train only, dev untouched.

## Note on the analysis (on day 4 at 11:54)
The flipped-query intervals (amendment 1) use 1,000 bootstrap draws over source questions, not the 2,000 stated for the
main check; the main check's intervals use 2,000 as planned.
