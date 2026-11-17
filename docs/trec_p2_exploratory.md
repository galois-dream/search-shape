*Shipped copy of the authors' `paper/trec_p2/EXPLORATORY.md`, unchanged below this note. Paths: `explore.py`, `p2.py` = `code/trec_p2/`; `results/` = `data/reference_outputs/trec_p2/`; `index.db` and `bm25.csv` = the outputs of `build_index.py` and `p2.py bm25` (`bm25.csv` ships as `data/trec_p2/bm25.csv.xz`). The registration is `docs/analysis_plan.pdf` with `docs/analysis_plan_deviation7.md`.*

# P2 (TREC): exploratory analyses, not registered

Written 2026-11-14 23:01 CET by Claude, at Alex's request, before either analysis below was run.
**Already seen when this was written:** the registered results of three arms (primary/minilm, primary/mpnet,
withdis/minilm). **Not yet seen:** the two corpus A arms (still running). Nothing here changes the registered verdict
("not demonstrated" in the three arms seen); both analyses are reported in the paper, labelled exploratory, whatever
they show.

## Why
The registered test enters four topology features as natural splines (12 columns) that share one L2 penalty with 53
baseline columns, and its bootstrap intervals hold the fitted models fixed. E1 asks the narrowest version of the question
with one degree of freedom; E2 states the registered bootstrap as probabilities instead of a verdict.

## E1. Does the longest bar add anything given the baseline's prediction? (1 df)
- Offset o_i: the mean over the 20 repetitions (seeds 0-19, exactly as in `p2.py fit`) of the out-of-fold logit of M0
  for topic i.
- Unpenalised logistic regression y ~ 1 + offset(o) + z, z = the standardised longest bar; null y ~ 1 + offset(o).
  Likelihood-ratio test, chi-squared with 1 df, two-sided p. Coefficient per SD of z with 95% Wald interval. The theory
  predicts a positive coefficient.
- Secondary (E1b): all four topology features, linear and standardised: chi-squared with 4 df.
- Calibration: the same with the corpus-bearing and bearing-swap control features in place of the real ones. The LRT
  ignores that o was fitted on the other topics' labels; the controls show how far that matters.

## E2. Probability statements from the registered bootstrap
- Re-run the registered real-feature fit with the same seeds; it must reproduce `result.txt` (point estimates and
  intervals to 3 decimals) or E2 is not reported.
- Report the share of the 2,000 bootstrap draws with AUROC gain >= +0.03, and with log-likelihood gain > 0. Read as
  approximate posterior probabilities (Bayesian bootstrap, flat prior over topics), conditional on the fitted models:
  refitting variability is ignored, so they are overconfident.

## Arms
primary/minilm (registered primary), primary/mpnet, withdis/minilm, and A/minilm, A/mpnet once they finish.
Code: `explore.py` in this folder; output `results/explore_<arm>.txt`.

## E3. Confounders: is the separation query length, year, or retrieval? (added 2026-11-14 23:13 CET, before running)
Prompted by Alex's question (Occam): the ambiguous topics look like one-word names, the faceted ones like phrases. Not
yet looked at: any statistic below. Seen: the topic list itself (query text and type), the registered results, E1, E2.
- E3a. Query length (words) by type: counts for 1, 2, 3+ words; Mann-Whitney U. Ambiguous share by TREC year (topics
  1-50 = 2009, 51-100 = 2010, 101-150 = 2011, 151-200 = 2012).
- E3b. Cross-validated AUROC, same machinery and bootstrap as the registered fit: (i) query length alone; (ii) the
  baseline without query length; (iii) the full baseline (reproduces the registered M0); (iv) length plus the four
  topology features. If (i) is close to (iii), the "retrieval separates the types" sentence is mostly about length.
- E3c. Within one-word topics, and within topics of two or more words: Mann-Whitney AUROC of the longest bar and of
  the top cosine, with U, n1, n2 and p. If the longest bar's signal vanishes within length strata, it was length.
- E3d. Spearman correlation of the longest bar with query length, and with the top cosine.
Arms: primary/minilm, primary/mpnet. Output: results/explore_e3_<arm>.txt.

## E4. Names, Wikipedia coverage and year (added 2026-11-14 23:23 CET, before running)
Alex asked for hypotheses 3 and 4 of the E3 discussion to be checked. Not yet looked at: any statistic below.
- E4a. "Has its own Wikipedia article": a non-disambiguation page in the dump (`index.db` meta) whose title equals the
  query, ignoring case. E4b. "Has a disambiguation page": a page titled "<query> (disambiguation)", or the exact-title
  page is itself a disambiguation page. E4c. Number of disambiguation pages among the top 20 BM25 pages (withdis arm of
  `bm25.csv`). These are lookups, not retrieval geometry; they show what a dictionary check alone achieves.
- For each: rate (or median) by type, overall and within 1-word and 2+-word topics; Fisher exact test (odds ratio, p)
  for the binary ones, Mann-Whitney for E4c; AUROC overall. Unpenalised logistic LR test of each given query length,
  chi-squared(1).
- E4d. Year: unpenalised logistic y ~ length + year (three dummies), LR test for year, chi-squared(3); and for length
  given year, chi-squared(1). Ambiguous share by year within 1-word and 2+-word topics.
Output: results/explore_e4.txt (no embedding involved, so one file).
