*Shipped copy of the authors' `paper/tois/EXPLORATORY.md`, unchanged below this note except the host name. Paths: `explore_e*.py` = `code/review_checks/`; `explore_out/` = `data/reference_outputs/review_checks/` (`e5_cells.csv` ships as `data/review_checks/e5_cells.csv.xz`); `sim/matched.py`, `sim/matched_auroc.py` = `code/simulation/matched/`; `sim/out/matched_shard*of14.csv` = `data/simulation/matched/` (xz); `sim/out/matched_auroc.csv` = `data/reference_outputs/matched/matched_auroc.csv`; `nobel/src/` = `code/realtext/`; `nobel/out_clean/swap_corpus.txt` = `results/swap_corpus.txt`; `paper/nobel/.../emb` = the embeddings (not shipped; README section 2.4). Thomas's annotated draft and e-mail are not included.*

# Exploratory checks prompted by Thomas's review of the TOIS draft

Written 2026-11-16 21:13 CET (Claude, for Alex), before either check is run. Source: Thomas's annotated PDF of the 2026-11-15 08:46
draft (pp. 16 and 21). Neither check is registered; both are reported as exploratory, whatever they show. Option (b)
of the reply (bearings from another corpus) waits for Alex.

What has already been seen: the 33 matched-score settings (sim/out/matched_auroc.csv: AUROCs by setting, read
for the reply to Thomas, before this plan was written), Fig. 6 (real bearings fill 21-25 effective dimensions by participation
ratio, corpus-drawn 23-29, random 46-48). Not yet seen: the participation ratio of any simulated setting, and any
statistic of random chunks per chunking.

## E5. Persistence against similarity statistics at a realistic dimension (Thomas, p. 16)
Question: where simulated neighbourhoods fill as many dimensions as real ones, does persistence tell one sense from
two better than the similarity statistics?
1. Existing data: median participation ratio of the one-sense and two-sense neighbourhoods in each of the 33
   one-vs-two settings (sim/out/matched_shard*of14.csv, column part_ratio). The setting family nearest the real range
   is the spread p in {2, 5, 20, 100, 767} whose two-sense median participation ratio is closest to 23 at Level B,
   noise 0. Report the AUROCs of the settings in that family as they stand.
2. New runs (sim/matched.py functions, unchanged): one sense vs two, Level B (coherence matched), D = 768, that p,
   angle between senses phi in {10, 20, 30, 45} degrees, noise tau in {0, 0.4}: 8 settings, 5 seeds x 40 pairs each,
   the same seeds and construction as matched.py. Statistics as matched.py. Output to paper/tois/explore_out/, never
   over sim/out/.
3. Analysis as sim/matched_auroc.py: oriented AUROC of W1(H0) and L_max(H0), the best rival among coherence, median
   nearest-neighbour similarity, second-eigenvalue share and largest nearest-neighbour distance, and the paired
   bootstrap interval (2,000 draws over pairs, seed 0) of feature minus best rival. Verdicts by the same rules:
   ceiling if the best rival reaches 0.98; adds if the interval is above 0 and the difference at least 0.05.
Prediction, written now: the similarity statistics match or beat persistence in most settings; persistence adds
(>= 0.05) only at small angles without noise, as in the one testable setting so far (+0.06 at 30 degrees, p = 5).
Use: one sentence in Section 6 (matched settings) and the reply to Thomas. If persistence adds in most realistic
settings, the conclusion's "similarity statistics ... as well as persistence" gets a qualifier for constructed
neighbourhoods; it would not change the real-text result.

## E6. Do longer chunks crowd together? (Thomas, p. 21)
Question: is there a corpus-wide reason why bearings drawn from the 750-token chunking give smaller bars than those
drawn from the 250-token chunking, in every corpus?
Hypothesis: a 750-token chunk averages more text, so its embedding lies closer to the corpus's mean direction and the
chunks crowd together.
Data: the 45 corpus x model embeddings already computed (paper/nobel/{emb,turing/emb,fields/emb,gutenberg/emb,wiki/emb},
passage_c250.npy and passage_c750.npy). For each, 500 chunks drawn at random (seed 0) from each chunking
(all if fewer), rows normalised to unit length:
 (i) mean pairwise cosine (coherence of random chunks);
 (ii) participation ratio of the centred embeddings;
 (iii) mean cosine to the chunking's own mean direction.
Report per combination the difference C750 minus C250 and the count of the 45 in which C750 is more crowded (higher
(i), lower (ii)). Descriptive only; the 45 are not independent (5 corpora). No test.
Prediction, written now: C750 is more crowded in at least 40 of 45 on (i). If it is, Section 8 says so in one sentence
as a likely reason ("a property of the corpus searched"); if not, the paper keeps saying the reason is unexplained.

### E6 amendment 1 (2026-11-16 21:14 CET), after the first run
The first run (explore_out/e6_run1.txt) drew min(500, n) chunks per chunking, so where the 750-token chunking has fewer
than 500 chunks the two samples differ in size, and (ii) depends on sample size. Seen in run 1: C750 lower
participation ratio in 45 of 45, higher coherence in 39 (all five gemini-001 combinations and Wikipedia bge-base
reversed), closer to the mean in 39. Amendment: both chunkings get the same number of chunks, m = min(500, n250, n750),
seed 0. Run 1 is kept and reported as superseded. The prediction is unchanged.

### E5 step 1 result and amendment 1 (2026-11-16 21:15 CET), before any new run
Median participation ratio (one sense / two senses) of the 33 existing settings, from sim/out/matched_shard*of14.csv:
at Level B without noise no spread p comes near the real 21-25 (two-sense medians 2.9 to 6.6), so the rule of step 1
("p nearest 23 at Level B, noise 0") would pick a setting far from real text. Settings whose two classes both lie in
15-30: Level A p = 20, noise 0.8 (21.5 / 19.4) and Level B p = 20, noise 0.8 (24.4 / 15.8). Their AUROCs, already
seen: both at the ceiling (best rival 1.000; W1 0.997 and 0.995). So at a realistic dimension the existing settings
cannot tell persistence from the similarity statistics.
Amendment: the new runs use p = 20, noise tau = 0.8 (the main simulation's sigma = 0.2), Levels A and B, phi in {10,
20, 30, 45} degrees: 8 settings, everything else as in step 2. The participation ratio of each new setting is reported
beside its AUROCs, so a reader can see whether it stays in the real range. Prediction unchanged.

### E6 result (2026-11-16 21:15 CET; explore_out/e6.txt, equal samples of m = 367 to 500 chunks)
- (ii) Participation ratio: lower for the 750-token chunks in 45 of 45 combinations (median difference -14.8).
- (i) Coherence: higher for the 750-token chunks in 39 of 45 (median +0.020); lower in all five gemini-001
  combinations and Wikipedia bge-base (-0.002 to -0.041). The prediction (at least 40 of 45 on (i)) fails narrowly.
- (iii) Cosine to the chunking's mean: higher in 39 of 45, the same exceptions.
Reading: longer chunks fill fewer effective dimensions everywhere; "closer to the corpus mean" does not hold for
gemini-001. Crowded bearings have closer nearest neighbours and so shorter bars, which makes fewer dimensions a likely
reason the corpus-bearing control reproduces the difference in every corpus. Descriptive, 45 dependent combinations,
not a demonstration. Paper: one sentence in Section 8, labelled exploratory.

### E5 result (2026-11-16 21:17 CET; explore_out/e5_auroc.txt, run on a desktop machine 21:15-21:17 CET)
The prediction fails. At a realistic dimension and noise, persistence does add in a window of angles:
- Level A (participation ratio 21.4-21.7, inside the real 21-25): W1(H0) beats the best similarity statistic by +0.106
  [+0.064, +0.148] at 20 degrees and +0.106 [+0.071, +0.138] at 30 degrees ("adds"); at 10 degrees nothing separates
  the classes (best 0.57); at 45 degrees coherence reaches the ceiling (0.98).
- Level B (coherence matched; participation ratio 23.5-33.9, above the real range): W1 adds +0.092 [+0.050, +0.136] at
  20 degrees; at 30 and 45 degrees the second-eigenvalue share reaches the ceiling (0.994, 1.000) and W1 is far behind
  (-0.32, -0.21).
- The longest bar adds nowhere. Every cosine statistic sits at 0.50, as constructed.
Reading: in constructed neighbourhoods that fill as many dimensions as real ones, with noise, W1(H0) is the better
detector of two close senses (20-30 degrees apart) by about 0.1 AUROC; when the senses are further apart a similarity
statistic is as good or better. This qualifies the paper's statement that simple similarity statistics do better in
constructed neighbourhoods; it does not touch the real-text results, where no gap between senses was found.

## E7. Bearings from another corpus (Thomas, p. 17) -- plan written 2026-11-16 21:19 CET, before any run
Question: does the corpus-bearing control reproduce the difference between the mismatched cells because of something
specific to each corpus, or because of a property of the chunk length that every corpus shares?
Already seen: nobel/out_clean/swap_corpus.txt (4 Nov, exploratory): bearings from the OTHER chunking of the same corpus
reverse the gap in all five corpora (text-embedding-3-small). Not yet seen: any bearing drawn from another corpus.
Design (extends nobel/src/swap_corpus.py; CLEAN=1, span-overlap exclusion, K = 50, one draw per query, rng seed 1):
for each corpus X and mismatched cell (Q250|C750, Q750|C250), each query keeps its 50 real scores, and its bearings
come from (a) its own corpus outside its top 50 (the corpus-bearing control), (b) the next corpus in the cycle
Nobel -> Turing -> Fields -> Essays -> Wikipedia -> Nobel, same chunking, outside the query's top 50 there, (c) uniform
random. Statistic: the mean bar (2 W1(H0)). Report per corpus the median per cell, the gap (multi-factual-like minus
polysemic-like) for real, (a), (b), (c), and the AUROC of the mean bar between the cells (oriented: polysemic-like
smaller) for real, (a) and (b). Models: text-embedding-3-small (as the 4 Nov probe) and bge-base-en-v1.5 (open).
Prediction, written now (from E6: longer chunks fill fewer dimensions in every corpus): the cross-corpus gap lies
within 0.03 of the own-corpus gap, and its AUROC within 0.05 of the own-corpus AUROC, in at least 4 of 5 corpora for
each model. If so, Section 8 says the difference is a property of the chunk length shared across corpora; if not,
that it depends on each corpus's own arrangement.

### E7 result (2026-11-16 21:21 CET; explore_out/e7_openai-3-small.txt, e7_bge-base.txt; run on a desktop machine 21:19-21:20 CET)
Sanity: the Nobel real gap (+0.170, text-embedding-3-small) equals the 4 Nov probe's.
- text-embedding-3-small: bearings from the next corpus give a gap within 0.03 of the own-corpus gap in 4 of 5
  corpora (Turing: +0.121 against +0.161), and an AUROC within 0.05 in 5 of 5 (0.999 to 1.000 against 0.995 to 1.000).
- bge-base: gap within 0.03 in 5 of 5; AUROC within 0.05 in 3 of 5. In the other two the cross-corpus bearings
  separate the cells BETTER than the corpus's own (Nobel 0.952 against 0.840; Fields 0.989 against 0.839).
- Cross-corpus AUROC is at least the own-corpus AUROC in all 10 corpus x model rows.
Against the prediction: holds for text-embedding-3-small; fails as worded for bge-base, in the direction of a stronger
effect. Reading: bearings from any corpus of the same chunk length reproduce the difference, so it is a property of
the chunk length shared across corpora, not of each corpus's own arrangement. Consistent with E6. Two models, one draw
per query; descriptive.

## E8. Thomas's two follow-ups to E7 (email reply of 2026-11-17, 18:41 UTC) -- plan written 2026-11-17 20:02 CET, before any run
Thomas: "The bearings from another corpus should be from the neighbour with k > 50 from another corpus not neighbours
from another corpus but 0<k<50. Another experiment to do is to try a control where the query from 1 corpus is tested
against another corpus entirely (score + bearings)."
Already true of E7 (explore_e7.py:42-43): the cross-corpus bearings were drawn uniformly from the other corpus's chunks
OUTSIDE the query's top 50 there, never from its top 50. Not yet run: bearings from the other corpus's NEAR shell, and a
search of the other corpus entirely.
Design (extends explore_e7.py; same corpora, cycle Nobel -> Turing -> Fields -> Essays -> Wikipedia -> Nobel, same
chunking, CLEAN=1, K = 50, one draw per query, rng seed 1; models text-embedding-3-small and bge-base-en-v1.5; mismatched
cells Q250|C750 (polysemic-like) and Q750|C250 (multi-factual-like); statistic the mean bar 2 W1(H0)):
(a) cross-shell: each query keeps its 50 real scores; bearings from 50 chunks drawn at random from ranks 51-500 of the
    query in the next corpus (the shell-bearing control of features.py:178, moved to the other corpus). Reported beside
    E7's own-corpus and cross-corpus (uniform outside top 50) columns, recomputed in the same run.
(b) elsewhere: each query searches the next corpus entirely; its neighbourhood is that corpus's top 50, with their own
    scores and bearings (no exclusion needed: no text is shared between corpora). Reported: the mean bar and the mean
    score per cell, their gaps (multi-factual-like minus polysemic-like), and the AUROC of the mean bar (oriented:
    polysemic-like smaller), beside the real own-corpus values.
Predictions, written now (from E6 and E7: the difference is a property of chunk length shared across corpora):
P8a: the cross-shell gap lies within 0.03 of the own-corpus (uniform outside top 50) gap in at least 4 of 5 corpora for
     each model, and its AUROC is at least 0.80 in all 10 corpus x model rows.
P8b: searching another corpus entirely, polysemic-like queries still have the smaller mean bar: gap > 0 in all 10 rows,
     and AUROC at least 0.80 in at least 4 of 5 corpora for each model.
If P8b fails, the difference needs the query's own topic in its neighbours, and Section 8's "property of the chunk length
shared across corpora" must be narrowed to bearings. Descriptive; no tests; two models, one draw per query.

### E8 result (2026-11-17 20:05 CET; explore_out/e8_openai-3-small.txt, e8_bge-base.txt; run on a desktop machine, finished 11:04 PST)
Sanity: the real gaps and AUROCs equal E7's in all 10 rows. The own and cross columns differ from E7's in the third
decimal, because the extra draws change the random sequence.
- P8a FAILS. Cross-shell gap within 0.03 of the own-corpus gap in 2 of 5 corpora for each model (text-embedding-3-small:
  Nobel, Wikipedia; bge-base: Turing, Wikipedia); elsewhere it is smaller, e.g. Fields +0.059 against +0.136 and Essays
  +0.063 against +0.129 (text-embedding-3-small). AUROC at least 0.80 in 7 of 10 rows (bge-base: Nobel 0.799, Fields
  0.750, Essays 0.789). The gap is positive in all 10 rows; AUROC 0.750 to 1.000.
- P8b holds for text-embedding-3-small (gap +0.078 to +0.135, AUROC 0.983 to 1.000) and FAILS as worded for bge-base
  (gap +0.027 to +0.048, positive in all 5; AUROC at least 0.80 in 3 of 5: Nobel 0.907, Fields 0.846, Wikipedia 0.802;
  Turing 0.799, Essays 0.752). Mean scores in the other corpus are lower, and still lower for polysemic-like queries in
  all 10 rows.
Reading: the direction survives everywhere: with bearings from the other corpus's shell, and with a search of another
corpus entirely, polysemic-like queries have the smaller mean bar in all 20 rows (AUROC 0.75 to 1.00). The size does
not: bearings drawn near the query (ranks 51-500), in another corpus, reproduce a smaller gap than bearings drawn from
anywhere outside the top 50. So the chunk length carries the direction across corpora, and what lies near the query
sets part of the size. Section 8's sentence is narrowed accordingly. Two models, one draw per query; descriptive.
