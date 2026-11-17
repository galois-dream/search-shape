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

*Added to this copy on 2026-11-17 from the authors' plan file, with its calendar dates (the part above gives study days). Paths in the amendments: `PREREG.md` = the registered plan, `docs/analysis_plan.pdf`; `../tois/related_work_survey.md` = the authors' literature notes (not included); `out/`, `out_full/` = the output directories of `ambig.py` without and with `FULL=1`; results in `data/reference_outputs/ambig_train/qpp_*.txt` and `tests_*.txt`.*

## Amendment 2 (2026-11-14 09:32 CET, before any of these predictors is computed): query-performance baselines for TOIS
Why: the paper goes to ACM TOIS, whose readers will ask how persistence compares with the query-performance predictors
(QPP) of information retrieval. PREREG.md already registers a QPP family as P1's comparison set on dev (Methods,
"Retrieval-only predictors"); this amendment computes the same family on **train** only, with the registered
definitions, plus the classical originals reviewers expect. Exploratory, train only; dev untouched (no dev question is
embedded, retrieved or scored). Same retrieval and corpus as the main check (`out/`, registered corpus); repeated on
`out_full/` (whole gold pages) as in the main check.

**Predictors** (k = 50 unless stated; s_i = cosine of neighbour i; s_C = the query's cosine to the mean corpus embedding,
the dense stand-in for the corpus score; formulas checked against the original papers in `../tois/related_work_survey.md`
before computing, and any change to them is dated here first):
- Registered (PREREG.md, Methods), definitions unchanged: NQC (dense adaptation: std of s_i / their mean); coherence
  (mean pairwise cosine of the neighbours; already in feats.csv); Clarity (simplified, unweighted unigram KL of the
  top-k text against the corpus, lowercase `\w+`, add-one smoothing); Dense-QPP (30 shared N(0, I) perturbations, seed 0;
  noise std s in {0.01..0.09}; depth K in {100..1000}; average overlap; the (s, K) pair chosen on a random half of
  train, seed 0, by |AUROC - 0.5|, and reported on the other half only); direction spread R (norm of the mean unit
  offset); score gap (top-1 minus top-2).
- Classical originals, as published: NQC with the original normaliser (std of s_i / |s_C|); WIG ((mean s_i - s_C) /
  sqrt(question length in words)); SMV ((1/k) sum s_i |ln(s_i / mean s)| / s_C); sigma_max (the largest std of the top-k'
  scores over k' = 2..50); pre-retrieval avgIDF, maxIDF and SCQ over the corpus chunks' term statistics (same
  tokenisation as Clarity).

**Reported for each:** AUROC with its 95% bootstrap interval (2,000 draws over questions, seed 0, as in the main check)
and the Mann-Whitney U, z and two-sided p (`ambig.py tests`); every predictor is reported, none is dropped.

**Predictions, written now.** (Q1) No QPP predictor's oriented AUROC exceeds 0.56 on the registered corpus: every
statistic measured so far lies between 0.46 and 0.55, and these predictors are functions of the same scores, texts or
neighbours. (Q2) No QPP predictor beats question length's oriented AUROC (0.537) by more than 0.02. (Q3) Dense-QPP's
held-out AUROC lies within 0.45-0.55. If Q1 fails, the paper says which predictor and by how much, and does not
call persistence's near-chance result "shared by every retrieval-only rival".

### Amendment 2a (2026-11-14 09:50 CET, after the literature survey and before any predictor is computed): formulas by provenance
The survey (`../tois/related_work_survey.md`, section (i)) could not read every original (ACM and Springer refuse automated
access), and showed that two of amendment 2's "classical originals" were not as published. Each predictor is now labelled
by provenance; labels go into the paper. Replaces amendment 2's list of classical originals; the registered family is
unchanged.
- **Published dense forms (read in Holdcroft et al. 2026):** NQC = sd(s)/mean|s| (identical to the registered NQC);
  LSMV = s_1 * sd(s), which replaces SMV (formula not verified); sigma_max = max over k' = 2..50 of sd(top-k' scores);
  gap = s_1 - s_2.
- **Read in Vlachou & Macdonald 2024 (dense versions of Diaz 2007 and Arabzadeh et al. 2021):** AC-embs, the Pearson
  correlation of the 50 scores with W s, where W = the neighbours' cosine matrix; on the graph that keeps pairs whose
  cosine exceeds the mean pairwise cosine: WAND (mean over nodes of the average degree of each node's neighbours; 0 for an
  isolated node) and density 2|E|/(k(k-1)).
- **Ours, illustrative (labelled so in the paper):** NQC with the corpus normaliser, sd(s)/|s_C|; WIG_dense = mean(s) - s_C
  (no length term: one embedding's score does not grow with query length, and length is reported on its own); the number
  of connected components of the same mean-threshold graph, which are exactly the H0 classes alive at that one threshold.
- **Restated from secondary sources:** avgIDF, maxIDF, avgSCQ, maxSCQ (He & Ounis 2004; Zhao et al. 2008).
- **Registered, unchanged:** Clarity (simplified: unweighted, add-one smoothed; not the original's query-likelihood
  weighting with lambda = 0.6, which the paper states), coherence, Dense-QPP, direction spread R.
- **Not computed, with reasons:** pairRatio and n(sigma_x%) (each needs a cut-off the sources do not fix); average node
  connectivity (about 1.5 h of maxflow for a statistic of the same graph); supervised text classifiers (a different cost
  class; discussed in the paper with the published ClariQ numbers).
Predictions Q1-Q3 apply to every predictor above.

### Result of amendment 2/2a (recorded 2026-11-14 10:18 CET; registered corpus, `qpp_train.txt`)
- **Q1 fails for one predictor:** avgIDF, AUROC 0.434 [0.422, 0.444] (oriented 0.566; Mann-Whitney z = -11.5): ambiguous
  questions use commoner words. Every other predictor stays at or below 0.56 (maxIDF 0.558 oriented).
- **Q2 fails for the same predictor:** avgIDF beats question length (0.537) by 0.029. No other predictor beats it by
  more than 0.02.
- **Q3 holds:** Dense-QPP, (s, K) = (0.09, 100) chosen on the tuning half, held-out AUROC 0.536 [0.520, 0.551]; all 90
  variants on the held-out half 0.526 to 0.540.
- Every predictor built from the retrieved neighbourhood (scores, similarity graph, Clarity, Dense-QPP) lies between 0.48 and 0.54.
- Consequence, as amendment 2 required: the paper names avgIDF as the strongest retrieval-only signal (a property of the
  query's words, not of the neighbourhood) and does not say persistence's near-chance result is shared by every rival.
- **Whole gold pages** (`qpp_full.txt`, recorded 2026-11-14 11:21 CET): passage-based predictors 0.47 to 0.54 (WIG_dense 0.540,
  Dense-QPP held-out 0.538 at (0.09, 400)); avgIDF 0.429 [0.418, 0.440] (oriented 0.571) and maxIDF 0.439 (oriented
  0.561), so with whole pages Q1 fails for two predictors, both read the query's words against corpus statistics.
