# Pre-registered analysis plan: deviation 7 (test P2, TREC Web track topics)

*Deviation 7 of the registered analysis plan (`docs/analysis_plan.pdf`, which ends with deviation 6), filed on OSF as an update of the registration on 2026-11-14 (see "Filed" below). Copied from the authors' plan file `PREREG.md` on 2026-11-17; "this file" below means that plan file. Verbatim. Code: `code/trec_p2/`; results: `data/reference_outputs/trec_p2/`; exploratory follow-ups: `docs/trec_p2_exploratory.md`.*

7. **2026-11-06 21:10 CET: a second registered test, P2, on queries people typed as topic-ambiguous (agreed by both
authors, 2026-11-06).** P1 and P1b use AmbigQA's labels, which mark mostly underspecification (one topic, a detail left
open), not the polysemic regime the paper defines. P2 tests that regime directly. **Timing (revised 2026-11-14 09:25
CET, before filing; text approved by Alex 2026-11-14 19:27 CET; Thomas agreed, as relayed by Alex, 2026-11-14 19:29 CET):** the paper was declined by SIAM J. Mathematics of
Data Science on 2026-11-14 without review and goes next to ACM Transactions on Information Systems, whose readers will
ask for exactly this test. P2 is therefore run once, as soon as this deviation is filed on OSF, and before the paper is
submitted. Its result is reported whatever it is, including "not demonstrated". No TREC topic, label or document has
been downloaded, and no P1/P1b Stage-2 freeze, dev feature or dev label has been computed; the first TREC download
happens after filing, and its date is recorded here.
- **Queries and label.** The TREC Web track diversity topics of 2009–2012 (about 200; the exact count is taken at
  download) [Clarke, Craswell & Soboroff 2009; Clarke, Craswell, Soboroff & Cormack 2010; Clarke, Craswell, Soboroff &
  Voorhees 2011; Clarke, Craswell & Voorhees 2012, TREC proceedings, NIST]. Label: the NIST topic type, ambiguous = 1,
  faceted = 0. Input is the query text only; subtopic descriptions and judged documents are never used.
- **Embedders.** all-MiniLM-L6-v2 (primary), all-mpnet-base-v2 (robustness). K = 50.
- **Models.** M0, retrieval-only rivals: query length, the 50 sorted cosines, the number of distinct pages among the 50
  neighbours, and the corpus's coverage of the topic. M1: M0 plus the longest H0 bar, `w1_h0`, `lt_max_h1` and the
  top-eigenvalue share. Logistic regression with smooth (spline) terms for continuous covariates, never cuts.
- **Endpoint.** M1's gain over M0 in held-out log-likelihood and in AUROC, 10-fold cross-validation repeated 20 times,
  with a bootstrap 95% CI. Predicted direction: M1 beats M0, and within M1 the longest H0 bar is larger for ambiguous
  topics.
  - "Useful gain" if the log-likelihood gain's 95% CI lies above 0 and the AUROC rises by ≥ 0.03.
  - "No useful gain" if the AUROC gain's whole 95% CI lies within ±0.03.
  - Otherwise "not demonstrated". With ~200 topics (AUROC SE ≈ 0.04–0.05) this is the likely outcome unless the effect
    is large; stated now.
  - **Also reported, never used to decide (added 2026-11-14, before any TREC data):** the journal asks for a named test
    with its p-value. For the log-likelihood gain: a one-sided permutation test (topic labels permuted 1,000 times,
    seed 0; for cost, each permutation refits one repetition of the 10-fold cross-validation, and the observed gain is
    recomputed the same way, from repetition 1; p = (1 + #permuted gains ≥ observed)/1,001). For the
    longest bar's direction: the Mann–Whitney U with n1, n2 and its two-sided p; U/(n1·n2) is the bar's AUROC, so this
    is the same number as the interval, not a second analysis. The decision rule above is unchanged.
- **Corpora.** Primary, B: for each topic's query text, the top 20 pages by BM25 over titles and first paragraphs of the
  AmbigQA Wikipedia dump (2020-01-20), pooled into one corpus for all topics. Leak guards: the rule sees the query text
  only; disambiguation pages are excluded (the result with them included is also reported); coverage by topic type is
  reported before any feature is compared. Check, A: the registered AmbigQA corpus plus deviation 6(c)'s 50,000 chunks
  from random dump pages that are not gold pages, fixed before any TREC query is seen; a null there is weak, since a
  missing sense cannot show.
- **Specification details (added 2026-11-14 09:34 CET, before filing and before any TREC data; each closes a choice the
  text above left open, which would otherwise be made after seeing data).**
  - Search: BM25 as implemented by SQLite FTS5 (k1 = 1.2, b = 0.75) over one field per page, its title and first
    paragraph (the dump's text up to the first blank line after the title). Query: the topic's query text, lowercased,
    split into word tokens, joined by OR (disjunctive, as in standard BM25). A page is a disambiguation page if its title
    ends in "(disambiguation)" or its first paragraph contains "may refer to"; flagged when the index is built.
  - Chunking and retrieval: pages are cut as the registered corpus (Data: ~250-token chunks, first 5 per page); the 50
    neighbours are the top 50 chunks by exact cosine.
  - Coverage: the share of the topic's 50 neighbours that come from its own 20 BM25 pages (corpus B); in corpus A,
    from the 20 pages its BM25 search returns, though most of them are not in A.
  - The 6(c) chunks (also the construction P1's 6(c) uses): pages drawn uniformly without replacement, seed 0, from the
    dump's pages that are neither gold pages nor disambiguation pages, cut as the registered corpus (first 5 chunks per
    page), taken in draw order until 50,000 chunks; the last page may contribute fewer than 5.
  - Bearing swap: the other topic's bearings are carried to this query by the Householder reflection that maps the other
    query to this one, rank by rank, as in the paper's bearing-swap control.
  - Topics: those NIST typed ambiguous or faceted; any other type, and exact duplicate query texts after the first, are
    excluded and counted.
  - Fitting: L2-penalised logistic regression. Scalar continuous covariates enter as natural cubic splines with 3
    degrees of freedom (4 knots at the training fold's 5th, 35th, 65th and 95th percentiles; a covariate with fewer than
    4 distinct knot values, such as a short integer count, enters linearly); the 50 sorted cosines enter linearly
    (with ~200 topics, splines on all 50 would leave more parameters than topics). Covariates are standardised within each training fold, and the penalty is chosen by inner
    5-fold cross-validation on log-loss within each training fold, from 13 log-spaced values in [1e-3, 1e3].
  - Statistics: each repetition yields out-of-fold predictions for every topic from M0 and from M1. The log-likelihood
    gain is the mean per-topic out-of-fold log-likelihood of M1 minus M0; the AUROC gain is AUROC(M1) minus AUROC(M0);
    both are averaged over the 20 repetitions. Their 95% intervals resample topics (2,000 draws, seed 0) and recompute
    the averaged statistic from the same out-of-fold predictions, without refitting.
- **Report-only controls.** The corpus-bearing control of deviation 6(a) (20 draws) and the paper's bearing swap (each
  topic keeps its 50 cosines and takes another topic's real bearings; one draw per topic, seed 0): M1 refitted on each,
  reported beside the real M1.
- **Unchanged:** P1, P1b and every earlier deviation. The SIMODS manuscript (M197184, declined 2026-11-14) named this
  test in its section "Ambiguity as people label it"; the TOIS version reports its result.
- **Embargo.** The OSF registration's embargo end is set to 2027-12-31, matching this file (OSF showed 2026-12-31).
- **Filed:** as an update of OSF registration `hrakm` (revision 6af8aa663e69996f7b30c8a5): submitted 2026-11-14 18:40:17 UTC,
  approved by the admin 2026-11-14 18:41:20 UTC (19:41:20 CET), with this file as of commit 3f63271
  (`PREREG_deviation7_2026-11-14_1929.pdf`, SHA-256 prefix 5ad9e8bf757da721). On that date OSF still showed the embargo
  ending 2026-12-31. No TREC topic, label or document had been downloaded.
- **First TREC download:** the four NIST topic files (Web track 2009-2012) were saved by Alex from a browser on 2026-11-14
  between 19:43:48 and 19:44:19 CET (NIST refused scripted downloads), after the filing above. Check corpus A had been
  built at 19:41 CET, before any TREC file existed on any machine (50,000 random chunks from 24,508 pages).
