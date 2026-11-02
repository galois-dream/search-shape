# What Persistent Homology Sees Around a Search Query: code and derived data

Code for every number, table and figure of the paper by Thomas Barillot and Alex Castro (submitted to the SIAM Journal on
Mathematics of Data Science), and of the earlier preprint it revisits, Barillot and De Castro, "Topological
quantification of ambiguity in semantic search", arXiv:2406.07990. Licence: Apache-2.0 (LICENSE). The authors
contributed equally; Alex Castro is the corresponding author.

**Added for the SIMODS paper**

```
code/geometry/     mstcheck.py (H0 bars = spanning tree; nearest-neighbour share), mstpieces.py (the exact decomposition of
                   the total persistence into nearest-neighbour ratios, arrangement and two cosine terms; variance shares;
                   nulls), timing.py (cost per query)
code/theory/       nn_bound.md and check_bound.py (bounds for unequal cosines), random_nn_share.py (random-cloud yardsticks),
                   stability/ (floating-point, tie, perturbation and Monte Carlo probes)
code/figures/      the four-point figure and the generator of the real-barcode figure
results/           the outputs these scripts produced for the paper (each file says how it was run)
```

**Data.** No source text is included, and no embeddings except the crane example's vectors: several corpora are
copyrighted (see "Data and licences" below, which gives every selection rule and identifier needed to rebuild them).
Derived data that are not here (embeddings and per-query features for all nine models, about 2 GB) are available from
the authors on request, for research use where the source texts' terms permit.

---

## Full description (originally the paper's supplementary material)

Code and derived data for the synthetic simulation, the real-text study (Nobel lectures, ACM Turing Award lectures,
Fields medallists' ICM lectures, essays, Wikipedia; nine embedding models) and the didactic figures, plus the
pre-registered analysis plan, the dated plans of the two registered real-text tests (vMF statistic, transfer test) and
of the registered matched-cosine simulation, the written Turing and Fields predictions, and the plans of the two checks
"closer to the definition" (section 5). Everything runs on a laptop CPU (Python 3.14 was used; 3.11+ should work).
No source text is included, and no embeddings except the crane example's vectors (section 4; see "Data and licences").

```
README.md
code/simulation/   simulation: generator, persistence features, controls ("nulls" in code), baselines, distractor variant, statistics, figure data;
                   matched/: the matched-cosine simulation (section 1.1)
code/realtext/     real-text pipeline: corpus selection, chunking, embedding clients, artefact filter, retrieval + features,
                   controls, statistics, Bayesian layer, the later analyses (section 2.3), figures
code/didactic/     didactic figures: Python scripts that compute their data, TikZ/pgfplots sources, data/
code/paper_tables/ make_real_tables.py: the paper's real-text tables, the appendix dot plot's data and the results ledger
code/fields_corpus/ selection rule, download and text extraction for the Fields corpus (README.md there)
code/examples/     the ring/mercury/jaguar/python and crane examples (section 4)
code/lenctl/       length-controlled topic-count test (section 5.1)
code/ambig_train/  AmbigNQ train check, its flipped-query amendment, and the longest-bar split count (sections 5.2-5.3)
data/simulation/   per-query tables, seed 0 of each configuration (see section 1): features/, cosine/, distractor/;
                   matched/: every neighbourhood of the matched-cosine run (section 1.1)
data/realtext/     per corpus: clean-text per-query feature tables for five models; summary files for all nine
                   (flexible-classifier replicates, subsample seeds, document-matched control); corpus manifests
data/examples/     the examples' JSON outputs and the crane neighbourhoods' vectors (section 4)
data/lenctl/, data/ambig_train/  per-query feature tables of the two checks of section 5 (and lenctl's page groupings)
data/reference_outputs/  every number behind the paper (section 2.4): simulation statistics; real-text stats_long per
                   corpus and the Table S0-S5 summary (clean text, nine models); Bayesian layer (bayes/); later analyses
                   (analyses/); the all-text comparison (alltext/); the paper's table sources (paper_tables/); loops_check;
                   matched-cosine verdicts (matched/); child-size band check (simulation_band_controls.md);
                   section 5's results (lenctl/, ambig_train/); the crane topology table (crane_topo.dat)

docs/analysis_plan.pdf     pre-registered analysis plan for the AmbigQA test (anonymised copy of the registered file)
docs/vmf_plan.md           plan of the two-meaning vMF statistic, written before any real-text value, with its amendments
docs/transfer_plan.md      plan of the transfer test, written before it was run, with its amendment
docs/matched_plan.md       plan of the matched-cosine simulation, written before it was run, with its two amendments
docs/matched_results.md    its results against the registered predictions
docs/turing_predictions.md predictions for the Turing corpus, written before any Turing feature was computed
docs/fields_predictions.md the same for the Fields corpus, with the model-based addendum
docs/lenctl_plan.md        plan of the length-controlled test, written before any text was selected, with its amendment
docs/ambig_train_plan.md   plan of the AmbigNQ train check, written before any question was embedded, with its amendment
```

**Dated notes.** The plans and prediction notes give the study day (day 1 = the first day of the embedding runs,
whose calendar dates the paper gives; days 2 to 4 follow it) and the local clock time, so their order relative to the runs can be
checked. Names are replaced by "the authors" or "a co-author", quoted remarks are paraphrased, and internal file
paths are replaced by this archive's. The transfer plan and the matched-cosine plan are parts 2 and 1 of one file.
In the matched-cosine results, two parentheticals naming the statistic at the ceiling were corrected against
`matched_auroc.csv` (marked there). Nothing else is edited. The registered analysis plan
(`docs/analysis_plan.pdf`) keeps its calendar dates, as registered.

**Storage rule for all tables:** CSV, floats written with 10 significant digits, xz-compressed (`*.csv.xz`; pandas
reads them directly). Real-text features on the ε = 0.4 rule and all null-feature columns, which enter only through
AUROCs and 4-decimal medians, use 7 significant digits; the inputs of the gradient-boosting rival (sorted cosines,
Gram summaries, W1, LT_max, longest bar) are kept at full precision, because its tree splits are sensitive to
rounding. The matched-cosine shards and the per-query tables of section 5 are also kept at full precision.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r code/simulation/requirements.txt     # numpy, scipy, scikit-learn, pandas, matplotlib, ripser
pip install -r code/realtext/requirements.txt       # adds tiktoken (corpus building / chunking only)
```

## Definitions used everywhere

For a query q and its K = 50 nearest passages c_1..c_K (cosine on unit vectors):

- offsets u_i = (c_i − q + machine-eps)/‖·‖; Vietoris–Rips persistence of {u_i} with ripser (Z/2, max dimension 1);
- W1(H0) = mean of d/2 over the finite H0 deaths d; LT_max(H1) = max (d − b)/2 over H1 points, 0 if none;
  longest H0 bar (L_max) = largest finite H0 death;
- ε rule (secondary): keep neighbour i if ‖c_i − q‖/min_j ‖c_j − q‖ − 1 ≤ 0.4; at least 3 needed;
- the controls of the paper (called "nulls" in code, column names and these tables: `null` = random-bearing,
  `enull` / "emp" = corpus-bearing, `lnull` / "local" = shell-bearing). Each keeps a_i = cos(c_i, q) and redraws the
  bearing: c_i' = a_i q + √(1 − a_i²) v_i with
  - random-bearing control: v_i fresh uniform unit ⟂ q;
  - corpus-bearing control: v_i = the unit component ⟂ q of a chunk drawn without replacement from the searched
    chunks outside the query's top 50 (its cap) and outside the chunks excluded from its retrieval (chunks overlapping
    the query; in the leave-document-out arm, its whole document);
  - shell-bearing control: the same from the query's own ranks 51–500 (its shell), with replacement only if fewer
    than 50 remain;
  - document-matched control (`docmatch.py`): the bearing of a random other chunk of the neighbour's own document;
  - one draw per query in the full arm; 20 independent draws, averaged, in the leave-document-out arms;
- baselines: top-1, mean, std, OLS slope (on rank) of the 50 cosines; coherence = mean pairwise cosine of the 50;
  connectivity = connected components at the query's median pairwise cosine; R = ‖mean_i u_i‖ vs mean_i sin(θ_i/2).

Implementation: `code/simulation/simcore.py` (`persistence_features`, `matched_cosine_null`). The real-text code
uses the same definitions and checks itself against `simcore` (`features.py --refcheck`).

## 1. Simulation (`code/simulation/`)

| file | role |
|---|---|
| `simcore.py` | topic banks (orthonormal QR columns; random unit vectors for the superposition regime), passage generator, features, matched-cosine null |
| `run_sim.py` | configurations, scenarios S1/S2/S3 (code ids 1/2/4), random projections, per-query tables |
| `cosine_baselines.py` | cosine, coherence and connectivity baselines for the same queries |
| `distractor.py` | the designed 50 passages hidden among 1,950 distractors; top 50 retrieved from n = 2,000 |
| `stats.py`, `redteam.py` | AUROC, median differences, cluster bootstrap over parent groups (B = 2000), nulls, grouped 5-fold logistic test (R2: folds by parent group; R2b: by seed), distractor tables DT1–DT6 |
| `make_figs.py` | data tables (.dat) for the four simulation figures, drawn by `code/didactic/fig_sim.tex` |
| `loops_check.py` | the experiment behind "loops need not count topics" |
| `band_controls.py` | AUROC of W1, LT_max and the longest bar against their random-bearing controls within child-size bands |
| `matched/` | the matched-cosine simulation and its analysis (section 1.1) |
| `verify_reimplementation.py` | regenerates small runs and compares with `data/simulation` |

Configurations (`run_sim.py:CONFIGS`): `fig1` (D = 256, 64 topics, 32 per parent, σ = 0.1; main), `fig2` (σ = 0.2,
adds S3), `appD64`, `appD128`, `proj` (σ = 0.01; random projections to 224/192/128/64), `superpos`, `superpos_s02`
(1,024 random topics in D = 256). Seeds 0–4; 50 parent groups per seed; one query per parent group per child size.

```bash
cd code/simulation
python -W ignore stats.py          # -> results/stats.md, stats.csv, stats_all50.csv (+ R1-R3, R2b, DT1-DT6)
python -W ignore make_figs.py      # -> results/figs/sim_{main,dims,proj,mixed}_{w1,lt}.dat, sim_main_eps.dat
python -W ignore loops_check.py 100   # loops vs spread directions (100 samples per cell; data/reference_outputs/loops_check.txt)
```

`stats.py` for one configuration takes about 10–20 s and under 1 GB; all configurations at once were not timed on a
laptop (budget several minutes and several GB). After regenerating all five seeds (below), `results/stats*.csv` and
`stats.md` should match `data/reference_outputs/simulation_stats*`.

**Only seed 0 of each configuration is shipped** in `data/simulation/` (all five seeds would put the archive over its
100 MB limit). Run on the shipped tables, `stats.py` and `make_figs.py` report seed-0 statistics; the paper's numbers
use seeds 0–4, which the commands below regenerate.

| paper item | source |
|---|---|
| Table "Simulation, S1 against S2" (`tab:sim`) | `results/stats.md`: config `fig1` (ε = 0.4 table, all-50 table, R1, R2/R2b) |
| Appendix compact table (`tab:simcompact`) | `code/didactic/gen_sim_compact.py` (from `simulation_stats_all50.csv` and R2b of `simulation_stats.md`) |
| Full table of all configurations (`paper_tables/tab_sim_all.tex`) | `results/stats.csv`, `results/stats_all50.csv` |
| Paragraph "With retrieval" | `results/stats.md`, DT1–DT6 |
| Figures `fig:sim`, `fig:sim_mixed`, `fig:simall`, `fig:proj` | `results/figs/sim_*.dat`, drawn by `code/didactic/fig_sim.tex` (the paper's copies, from all 5 seeds, are in `code/didactic/data/sim/`) |
| Loops paragraph and appendix "Loops and spread" | `loops_check.py 100` (didactic figure `fig_loops` reads its output) |
| Paragraph "Within one child size, the random-bearing control matches W1" | `band_controls.py` (seeds 0–4: `data/reference_outputs/simulation_band_controls.md`) |

Regenerate the tables from scratch (about 25 CPU-minutes, plus ~2 for the baselines and ~5 for the distractors):

```bash
python -W ignore run_sim.py --configs fig1 fig2 appD64 appD128 proj superpos superpos_s02 --seeds 0 1 2 3 4 --workers 4 --outdir out
python -W ignore cosine_baselines.py --configs fig1 fig2 superpos superpos_s02 --seeds 0 1 2 3 4 --workers 4 --outdir out
python -W ignore distractor.py --configs fig1 fig2 --seeds 0 1 2 3 4 --workers 4 --outdir out
```

then compress them into `<dir>/{features,cosine,distractor}/` and set `SIM_DATA=<dir>` (`SIM_RESULTS` = output folder).

**Reproduction check (run on this code and data).** `python verify_reimplementation.py` (~50 s) regenerates and
compares: appD64 and superpos seed 0 (all rows and columns incl. the nulls), fig2 and proj seed 0 (first 2 parent
groups; covers S3 and the projections), cosine baselines and distractor variant for fig1 seed 0. All six PASS; largest
difference 5e-10, i.e. the 10-digit storage (against the original full-precision tables: 5.6e-16). On the five-seed tables (before seeds 1–4 were
left out of this archive), `stats.py` on the appD64 tables matched the reference statistics to 4e-10, and on fig1 every table line of the run behind the
paper (main tables, R1–R3, R2b, DT1–DT6; 41 lines) is identical.

### 1.1 Matched-cosine simulation (`code/simulation/matched/`)

Registered before it was run (`docs/matched_plan.md`: plan, amendment 1, amendment 2; results against the
predictions: `docs/matched_results.md`). Each sample draws one list of 50 cosines to the query and builds two
neighbourhoods with exactly those cosines, differing only in how the neighbours' bearings are arranged: one meaning
against two (also three, and a ring). Level A matches the cosines; Level B also matches the mean pairwise cosine
(coherence). 63 cells (spread dimension p, noise tau, angle phi, D), 200 samples per class per cell. AUROCs of the
cosine statistics, the similarity rivals (coherence, median nearest-neighbour cosine, Gram second-eigenvalue share,
largest nearest-neighbour distance) and W1, L_max, LT_max, with the registered "wins", "adds" and "ceiling" rules.

```bash
cd code/simulation/matched
for k in $(seq 0 13); do python -W ignore matched.py --shard $k/14 & done; wait   # -> results/matched_shard*of14.csv
python -W ignore matched_auroc.py results/matched_shard*of14.csv                  # -> results/matched_auroc.csv, matched_summary.txt
python -W ignore matched_auroc.py ../../../data/simulation/matched/matched_shard*of14.csv.xz   # from the shipped run (~75 s)
```

The run behind the paper took 16 minutes as 14 shards on two machines; `python matched.py --check` prints the construction
diagnostics only. `vmf2` imports `code/realtext/vmf.py` (`VMF_DIR` overrides the path). The shipped shards
(`data/simulation/matched/`, full precision) hold every neighbourhood's statistics.

**The paper's sentence** in Limitations ("with the cosines and the mean similarity held equal, W1 separates one meaning
from two in all 18 settings ... (W1, +0.06 [+0.02, +0.10])") rests on `data/reference_outputs/matched/matched_auroc.csv`,
rows `comparison = one_vs_two`: `w1_wins` is true in all 18 Level B rows; `best_rival_auc` >= 0.98 (verdict `ceiling`)
in 32 of the 33 one-vs-two rows; the remaining row (Level B, p = 5, tau = 0, phi = 30) has `w1_minus_best` = +0.060,
interval [`w1_diff_lo`, `w1_diff_hi`] = [+0.019, +0.098], verdict `adds`. Which rival reaches 0.98 varies: the Gram
second-eigenvalue share in 25 of the 33 rows, coherence in 15, the median nearest-neighbour cosine in 12.

**Reproduction check.** `matched_auroc.py` on the shipped shards reproduces `matched_summary.txt` exactly, and every
verdict, interval and band AUROC in `matched_auroc.csv`. Two columns differ only through floating-point ties:
`auc_coherence` by 1e-5 in one cell, and `auc_top5_share` (a yardstick, in no verdict) by up to 0.11 in the p = 2,
tau = 0 cells, where the top-5 variance share is 1 to within 1e-15 and its AUROC orders rounding noise.

## 2. Real text (`code/realtext/`)

### 2.1 Corpora, chunking and the artefact filter

Corpora: **Nobel**, 40 Nobel physics lectures as published PDFs (289,650 tokens;
1,139 C250 / 367 C750 chunks); **Essays**, 40 public-domain science and philosophy books from Project Gutenberg (one
window each, 596,250 tokens; 2,384 / 794); **Wikipedia**, 40 pages from the AmbigQA 2020-01-20 dump (317,272 tokens;
1,249 / 402); **Turing**, the 32 ACM Turing Award lectures with at least 3,000 words of extractable text (of 38;
311,257 tokens after a 15,750-token cap; 1,232 / 401), cleaned with the Nobel header/footer rule; **Fields**, one English-language
ICM lecture by each of the 40 most recent qualifying Fields medallists (1974–2022; 475,030 tokens after the cap;
1,886 / 620; selection and extraction in `code/fields_corpus/README.md`). Turing and Fields were analysed last, with
predictions written beforehand (`docs/turing_predictions.md`, `docs/fields_predictions.md`). Text is tokenised with tiktoken `cl100k_base` and cut into contiguous, non-overlapping 250- and
750-token chunks (C250, C750; each C750 = three consecutive C250). Cells Q250|C750, Q750|C250, Q250|C250, Q750|C750.
Arms: **primary** (corpus chunks overlapping the query's text are never candidates) and **LODO** (the query's whole
document is excluded); density control = C250 subsampled to |C750| per document (seed 0).

**Clean text (every reported real-text result).** Some PDFs set letters with extra spacing, and their text comes out
letter by letter. `features.py` with `CLEAN=1` drops every chunk in which more than half of the alphabetic tokens have
one or two letters, from both chunkings (a 750-token chunk goes if any of its three 250-token parts goes, and the
250-token parts of a dropped 750-token chunk go too, so the chunkings stay aligned), then retrieves and computes
everything on what is left; outputs go to `$OUTROOT=out_clean`. Chunks removed per corpus:
`data/reference_outputs/analyses/clean_filter_counts.csv` (Nobel 261 of 1,139 / 87 of 367; Turing 282 / 94; Fields
456 / 152; essays 12 / 4; Wikipedia none). The all-text run (no filter, `OUTROOT=out`) is kept only as a comparison:
`data/reference_outputs/alltext/` (five models; the paper's all-text Nobel effect and the Turing/Fields predictions,
which were scored on all text) and the dose-response below.

### 2.2 Embedding models and settings

Nine models, none fine-tuned. All vectors are L2-normalised before retrieval.

| tag | model (public name) | dim | settings |
|---|---|---|---|
| openai-3-small | OpenAI `text-embedding-3-small` | 1536 | `/v1/embeddings`, `encoding_format=float`, batches of 64 (C250) / 32 (C750); same vector for query and passage |
| openai-3-large | OpenAI `text-embedding-3-large` | 3072 | as above |
| openai-ada-002 | OpenAI `text-embedding-ada-002` | 1536 | as above |
| vertex-005 | Google Vertex AI `text-embedding-005` | 768 | us-central1, `autoTruncate: true`, batches of 50 / 16; task_type `RETRIEVAL_DOCUMENT` (passages), `RETRIEVAL_QUERY` (queries); no chunk truncated |
| gemini-001 | Google Vertex AI `gemini-embedding-001` | 3072 | us-central1, `autoTruncate: true`, batches of 16 / 8; task_type `SEMANTIC_SIMILARITY`, same vector for query and passage; no chunk truncated |
| modernbert-embed | `tasksource/ModernBERT-base-embed` (revision 9ba3d26d) | 768 | sentence-transformers 6.2.0, CPU, batch 8, no prefix |
| bge-base | `BAAI/bge-base-en-v1.5` (revision a5beb1e3) | 768 | as above; max_seq_length 512, so every C750 chunk is truncated |
| bge-m3 | `BAAI/bge-m3` | 1024 | sentence-transformers, integrated GPU, float32, batch 8, no prefix; no chunk truncated |
| qwen3-0.6b | `Qwen/Qwen3-Embedding-0.6B` | 1024 | as bge-m3 (weights loaded in float32; no instruction prefix); no chunk truncated |

The revisions of bge-m3 and Qwen3-Embedding-0.6B were not recorded (the Hugging Face default at the time of the run).

### 2.3 Code

| file | role |
|---|---|
| `prep.py` | Nobel header/footer rule, tokenisation, chunking (all corpora) |
| `wiki_select_remote.py`, `wiki_select_local.py` | uniform random (seed 0) selection of 40 eligible Wikipedia pages |
| `build_gutenberg.py` | Essays: selection rule (header of the file), download, PG header/footer stripping, windowing |
| `embed_cloud.py`, `embed_open.py` | embedding clients (API models; open models via sentence-transformers) |
| `features.py` | artefact filter (`CLEAN=1`), retrieval + all per-query features, controls (incl. `--nullseeds 20`), own-document share, Gram summaries |
| `controls.py` | `hgb`: flexible classifier A' (sorted cosines + similarity-matrix summaries) vs B' = A' + persistence (refit cluster bootstrap); `hgbvmf`: the same with the vMF statistic; `dens`: 20 density subsamples |
| `stats.py`, `combine.py` | document-level cluster bootstrap (B = 2000, seed 0) per corpus; corpus × model × arm summary (Tables S0–S5) |
| `bayes.py` | Bayesian bootstrap of real − corpus-bearing control per cell, and hierarchical models across cells; `--subvar` adds the subsample variance, `--prior-sweep` varies the scale priors |
| `figs.py` | KDE and interaction figures (ε-sweep figure only if prefix tables exist) |
| `run_features.sh` | features → stats → figures for one corpus |

Later analyses, one line each:

| file | what it does | output (`data/reference_outputs/...` unless stated) |
|---|---|---|
| `sufficiency.py` | does one similarity statistic do as well as persistence? AUROC of single cosine, neighbour-similarity and persistence statistics, and of each persistence feature within strata of coherence | `analyses/sufficiency.csv` |
| `docmatch.py` | document-matched control: each neighbour keeps its cosine and its document, and borrows the bearing of another chunk of that document | `data/realtext/<corpus>/docmatch_<model>.csv` (text-embedding-3-small, ModernBERT) |
| `dose.py` | artefact dose-response: the fixed-length real − control difference at cleaning thresholds 1.0 (all text), 0.7, 0.5 (the paper), 0.4, and a same-size random deletion | `analyses/dose.csv` |
| `subseed.py` | repeats the size-matched 250-token subsample with ten seeds, so the intervals can include that randomness | `data/realtext/<corpus>/subseed_<model>.csv`; folded in by `bayes.py --subvar` (`bayes/bb_cells_subvar.csv`, `bayes/hier_summary_subvar.csv`) |
| `transfer.py` | registered transfer test: how much balanced accuracy a threshold loses when it is set on one embedding model and used on another | `analyses/transfer.csv` (plan: `docs/transfer_plan.md`) |
| `vmf.py`, `vmf_stats.py` | registered two-meaning von Mises–Fisher statistic, its corpus-bearing control and the longest-edge audit | `analyses/vmf_stats*.{csv,md}`, `vmf_audit*.csv`, `vmf_repro_check.csv`; with `controls.py hgbvmf`: `data/realtext/<corpus>/hgbvmf_<model>.csv` (plan: `docs/vmf_plan.md`) |
| `vmf_elongation.py` | on real neighbourhoods, the vMF statistic tracks how elongated the cloud is | `analyses/vmf_elongation.txt` |
| `vmf_aniso_check.py`, `vmf_highdim.py` | synthetic checks: when a projected mixture "finds" two groups in one elongated group, and why the unprojected fit never finds two | `analyses/vmf_aniso_check.txt`, `analyses/vmf_highdim.txt` |
| `hdp_checks.py` | three high-dimensional-probability checks: concentration of the random-bearing control, random projection of the neighbourhoods, second-eigenvalue split against its control | `analyses/hdp_conc_openai-3-small.csv.xz`, `hdp_jl_openai-3-large.csv.xz`, `hdp_dk.csv` |
| `effective_rank.py` | effective rank of neighbourhood offsets and of the corpus (all text) | `analyses/effective_rank.txt` |
| `tasktype.py` | text-embedding-005: the same query texts embedded as queries (as in the paper) and as passages; medians, query-to-neighbour gap, main and fixed-length AUROCs | `analyses/tasktype_vertex-005.txt` (concatenated over the five corpora) |
| `mstcheck.py` | the $H_0$ barcode as nearest-neighbour distances: share of bars that are nearest-neighbour edges, overlap of the offset and passage spanning trees, rank correlation of $W_1$ with the mean nearest-neighbour ratio, main-contrast AUROCs | `analyses/mst_nn.txt` (one line per corpus and model) |
| `../paper_tables/make_real_tables.py` | the paper's real-text tables, the appendix dot plot's data and the ledger of every number in the real-text results section | `paper_tables/tab_real.tex`, `tab_real_full.tex`, `tab_ratio.tex`, `tab_controls.tex`, `ledger_sec6.txt`; `code/didactic/data/realctl.dat`, `realctl_meta.tex` |

### 2.4 What is shipped, and how the rest regenerates

- **Per-query feature tables (clean text) for five models**: text-embedding-3-small, text-embedding-3-large,
  text-embedding-005, ModernBERT-base-embed and bge-base-en-v1.5, in `data/realtext/<corpus>/features_<model>.csv.xz`
  (with the `ltmax_enull_<model>.csv.xz` sidecars). The tables of the other four models (gemini-embedding-001,
  text-embedding-ada-002, bge-m3, Qwen3-Embedding-0.6B) would put the archive over its 100 MB limit.
- **Every summary output for all nine models**: each corpus's `stats_long.csv` (every estimate, unrounded) and the
  Table S0–S5 summary; the Bayesian cells and hierarchical summaries; the flexible-classifier replicates (`hgb_*`,
  `hgbvmf_*`), subsample-seed and document-matched results; and the outputs of every later analysis above.
- **Regenerating the four other models' tables**: the commercial embeddings cannot be redistributed; they are
  recomputed by `embed_cloud.py` with your own API keys. The open models (bge-m3, Qwen3-Embedding-0.6B, and the shipped
  ModernBERT and bge-base) are recomputed by `embed_open.py`. Then `features.py` (with `CLEAN=1 OUTROOT=out_clean`)
  regenerates the per-query tables. Analyses that need the embeddings themselves (`docmatch.py`, `subseed.py`,
  `vmf.py`, `vmf_elongation.py`, `hdp_checks.py conc|jl`, `effective_rank.py`, `tasktype.py`, `mstcheck.py`) run only after this step.

**Reproduce from the shipped tables** (`$RT_WORK`, default `code/realtext/work`):

```bash
cd code/realtext
export RT_WORK=work OUTROOT=out_clean
mkdir -p work/out_clean/wiki work/out_clean/gutenberg work/out_clean/turing work/out_clean/fields
cp ../../data/realtext/nobel/*.csv* work/out_clean/
cp ../../data/realtext/wikipedia/*.csv* work/out_clean/wiki/
for c in gutenberg turing fields; do cp ../../data/realtext/$c/*.csv* work/out_clean/$c/; done
for c in nobel wiki gutenberg turing fields; do CORPUS=$c python -W ignore stats.py; done   # ~1.5-4 min per corpus
python combine.py                    # -> work/out_clean/stats_summary.md (Tables S0-S5, the five shipped models)
python bayes.py                      # -> work/out_clean/bayes/ (Bayesian layer, five shipped models; ~4 CPU-minutes)
python sufficiency.py                # -> work/out_clean/sufficiency.csv (five shipped models)
python transfer.py                   # -> work/out_clean/transfer.csv (pairs among the models present; the paper's uses nine)
python vmf_highdim.py; python vmf_aniso_check.py                                   # synthetic checks
OMP_NUM_THREADS=1 CORPUS=gutenberg python controls.py hgb openai-3-small 200       # flexible classifier, any corpus
# the paper's tables, dot-plot data and ledger, from the nine-model reference outputs:
mkdir -p work/tables && cp ../../data/reference_outputs/paper_tables/tab_{real,real_full,ratio}.tex work/tables/
python ../paper_tables/make_real_tables.py ../../data/reference_outputs work/tables
# the subsample-adjusted Bayesian cells and hierarchical summary for all nine models:
cp ../../data/reference_outputs/bayes/bb_cells.csv work/out_clean/bayes/ && python bayes.py --subvar
```

| paper item | source (`data/reference_outputs/...`; "gutenberg" = Essays) |
|---|---|
| `tab:real`, `tab:real_full`, `tab:ratio`, `tab:controls`, `fig:realctl` as typeset | `make_real_tables.py` (rewrites the table bodies from the unrounded cells in `realtext_<corpus>_stats_long.csv`, `bayes/bb_cells_subvar.csv` and `data/realtext/<corpus>/{hgb,docmatch}_*.csv`) |
| every number of the real-text results section | `paper_tables/ledger_sec6.txt` (written by `make_real_tables.py`) |
| real AUROC intervals, shell-bearing control, own-document share, 20-draw details, coherence and logistic tests | `realtext_stats_summary.md` (Tables S0–S5) and each corpus's `stats_long.csv` |
| fixed-length real − corpus-bearing control with subsample variance; hierarchical model; prior sensitivity | `bayes/bb_cells_subvar.csv`, `bayes/hier_summary_subvar.csv`, `bayes/prior_sweep.csv` (`bayes/bb_cells.csv`, `hier_summary.csv`: one subsample) |
| flexible classifier, every corpus × model | `data/realtext/<corpus>/hgb_<model>.csv` (point estimate = rep 0, 200 refit replicates) |
| one similarity statistic against persistence | `analyses/sufficiency.csv` |
| document-matched control | `data/realtext/<corpus>/docmatch_<model>.csv`, `paper_tables/tab_controls.tex` |
| extraction artefacts: filter and dose-response | `analyses/clean_filter_counts.csv`, `analyses/dose.csv`; all-text comparison in `alltext/` |
| transfer across embedding models | `analyses/transfer.csv` |
| vMF appendix | `analyses/vmf_*`, `analyses/effective_rank.txt`, `data/realtext/<corpus>/hgbvmf_<model>.csv` |
| high-dimensional checks | `analyses/hdp_*` |
| why text-embedding-005 differs (query and passage task types) | `analyses/tasktype_vertex-005.txt` |
| why persistence adds little to nearest-neighbour statistics (appendix) | `analyses/mst_nn.txt` |
| appendix on predictions and outcomes | `docs/turing_predictions.md`, `docs/fields_predictions.md`; `alltext/` (scored on all text) and the clean results above |

**Reproduction check (run on this code and data).** On the shipped tables, `make_real_tables.py` regenerates
`tab_real.tex`, `tab_real_full.tex`, `tab_ratio.tex`, `tab_controls.tex`, `ledger_sec6.txt`, `realctl.dat` and
`realctl_meta.tex` byte for byte. `stats.py` on the shipped Nobel tables reproduces every estimate the paper's tables
use (660 rows of `stats_long.csv`, five models) to within 8e-5, the effect of storing 10 significant digits; the
grouped-CV logistic AUROCs (Tables S1–S2) also depend on the scikit-learn version (up to 0.02 with scikit-learn 1.3
instead of the 1.9 used). `bayes.py` on the shipped tables reproduces the five models' rows of `bayes/bb_cells.csv`
to 3e-6 (the 7-digit control columns); `bayes.py --subvar` on the shipped nine-model `bb_cells.csv` reproduces
`bb_cells_subvar.csv` exactly and `hier_summary_subvar.csv` to 1e-16. `sufficiency.py` reproduces the five models' rows
of `analyses/sufficiency.csv` exactly. `combine.py` then reproduces the five models' Table S0 rows character for
character, except that it fills the flexible-classifier column for two cells (essays / bge-base, Turing /
ModernBERT) whose replicate files were finished after the shipped summary was written. On the original embeddings, `features.py --refcheck` agrees with
`simcore.persistence_features` to 0, and the `docmatch.py`, `subseed.py` and `vmf_stats.py` runs check their real
features and first control draw against the feature tables (`analyses/vmf_repro_check.csv`: longest MST edge equal to
the longest bar to 6e-8).

**Rebuild from raw text** (API keys needed for the API models; `features.py` takes 4–15 CPU-minutes per model and
corpus with `--nullseeds 20`; embedding with the open models 10–20 minutes per corpus):

```bash
# Nobel: fetch the 40 transcripts in data/realtext/nobel/lecture_manifest.csv (Nobel Foundation website, PDF),
#   pdftotext -q into work/text/<lecture_id>.txt (OCR the scanned richardson-lecture-1: pdftoppm -r 200 -gray + tesseract)
python prep.py
# Wikipedia (needs AmbigQA's docs.db):
mkdir -p work/wiki/raw && python wiki_select_remote.py docs.db 250 12000 > work/wiki/raw/wiki_cand.jsonl
python wiki_select_local.py && CORPUS=wiki CAP_TOKENS=15750 python prep.py
# Essays (Gutenberg): reproduces data/realtext/gutenberg/selection_manifest.csv (check text_sha256)
python build_gutenberg.py && CORPUS=gutenberg python prep.py
# Turing: obtain the PDFs for the DOIs in data/realtext/turing/turing_manifest.csv from the ACM Digital Library,
#   pdftotext -q into work/turing/text/ (no OCR), keep the 32 with kept = yes (>= 3,000 words), then
CORPUS=turing CAP_TOKENS=15750 python prep.py
# Fields: build the 40 texts with ../fields_corpus (fetch_volume.sh, build_corpus.py; README there), then
TXT_DIR=<fields texts> CORPUS=fields CAP_TOKENS=15750 python prep.py
# Embeddings, per corpus (CORPUS=nobel|wiki|gutenberg|turing|fields):
export OPENAI_API_KEY=... VERTEX_PROJECT=...          # VERTEX_ACCESS_TOKEN optional (else gcloud is called)
python embed_cloud.py estimate
for m in openai-3-small openai-3-large openai-ada-002 vertex-005 gemini-001; do python embed_cloud.py run $m; done
python embed_open.py tasksource/ModernBERT-base-embed modernbert-embed; python embed_open.py BAAI/bge-base-en-v1.5 bge-base
EMB_DEVICE=mps python embed_open.py BAAI/bge-m3 bge-m3; EMB_DEVICE=mps python embed_open.py Qwen/Qwen3-Embedding-0.6B qwen3-0.6b   # or cpu
# Features (per corpus and model; clean text), controls, statistics, later analyses:
export CLEAN=1 OUTROOT=out_clean
python features.py <model> --refcheck --nullseeds 20
OMP_NUM_THREADS=1 python controls.py hgb <model> 200
python vmf.py <model>; OMP_NUM_THREADS=1 python controls.py hgbvmf <model> 200   # vMF statistic (VMF_K=100: sensitivity run)
python subseed.py <model>; python docmatch.py <model>
python stats.py; python combine.py; python figs.py; python bayes.py; python bayes.py --subvar; python bayes.py --prior-sweep
python sufficiency.py; python transfer.py; python vmf_stats.py; python hdp_checks.py dk
python hdp_checks.py conc openai-3-small; python hdp_checks.py jl openai-3-large; python vmf_elongation.py
# Dose-response arms (features.py per model in each arm), then python dose.py:
#   OUTROOT=out (CLEAN=0), CLEAN_T=0.7 OUTROOT=out_t07, CLEAN_T=0.4 OUTROOT=out_t04,
#   CLEAN_T=0.5 CLEAN_RAND=<number of 750-token chunks 0.4 drops beyond 0.5> OUTROOT=out_t04rand
```

## 3. Didactic figures (`code/didactic/`)

| figure | data | source |
|---|---|---|
| `fig_overview` (page 1) | `data/overview_coords.tex` | `python overview.py` |
| `fig_tilt` | none (panel a analytic; panel b a schematic with hand-set angles) | `fig_tilt.tex` |
| `fig_toy_barcode` | `data/toy_*.dat` | `python toy_barcode.py` |
| `fig_loops` (not in the final paper; its numbers are in the loops paragraph) | `data/loops_T*.dat` | `python real_data.py` (from `data/reference_outputs/loops_check.txt`) |
| `fig_predictions` (schematic of the ratio test; the values are in `tab:ratio`) | none (`data/ratio_forest*` is kept for the forest version, not in the paper) | `python real_data.py` |
| `fig_nulls` (essays and Wikipedia) | `data/nulls_{gutenberg,wiki}_*.dat`, AUROCs in `data/nulls_summary.txt` | `python real_data.py` (from the shipped clean text-embedding-3-small tables, LODO arm) |
| `fig_effects` (query-length and corpus effects, every model) | `data/effects*.{dat,tex}` | `python real_data.py` (from `reference_outputs/realtext_*_stats_long.csv`) |
| `fig_realctl` (real − corpus-bearing control, every model and persistence feature) | `data/realctl.dat`, `data/realctl_meta.tex` | `make_real_tables.py` (code/paper_tables) |
| `fig_sim` (the four simulation figures: `\simMain`, `\simDims`, `\simProj`, `\simMixed`) | `data/sim/sim_*.dat` | `code/simulation/make_figs.py` |
| `fig_examples` (appendix "From flag to clarifying question") | `data/examples/` | section 4 |
| `fig_crane` (appendix `app:examples`, the crane sentences) | `data/examples_crane/ex*_{pts,mst,long}.dat`; `topo_all.dat` (= `data/reference_outputs/crane_topo.dat`) and its rows for the three drawn queries, `topo_{W1,Lmax,LTmax}.dat` | section 4 (`to_dat.py`, `crane_topo.py`) |
| `fig_vmf` (appendix `app:vmf`) | `data/vmf_fig/` | `python gen_vmf_fig.py data/vmf_fig` (imports `code/realtext/vmf.py`) |
| `tab_sim_compact` | none | `python gen_sim_compact.py` (from `reference_outputs/simulation_stats_all50.csv`, `simulation_stats.md`) |

The `.tex` files are TikZ/pgfplots figures; they read data via the macro `\didpath` (define it as the path to this
folder, e.g. `\newcommand{\didpath}{code/didactic/}`).

## 4. From flag to clarifying question (appendix `app:examples`)

`data/examples/ring_examples.json`: for 12 queries about *ring*, *mercury*, *jaguar* and *python*: the meanings (page titles)
among the 50 nearest chunks; the longest and second-longest MST edges of their unit offset directions and the two groups
the longest edge separates; HDBSCAN clusters of the offsets (scikit-learn 1.3.2, `min_cluster_size=5`); a 2-D classical-MDS
layout (display only); and every MST edge. Setup: `sentence-transformers/all-MiniLM-L6-v2`; 11 meaning pages from the
AmbigQA 2020-01-20 Wikipedia dump cut into 250-token chunks (o200k_base), searched together with whole background pages
from the same dump up to 49,999 chunks (seed 0). Two runs gave identical numbers. The script is
`code/examples/ring_examples.py`; it imports its chunker, embedder name and background rule from a module of a
demonstration package that is not included here (it would identify the authors), and reads the background chunks and
their embeddings (`REAL_DIR/chunks.parquet`, `chunk_emb.npy`), which that package builds. The JSON holds every number in
the appendix table and figure; `code/examples/to_dat.py` turns it into the figure's tables (`code/didactic/data/examples/`).
Illustration only, not evidence.

**The crane sentences** (`fig_crane`, paragraph "The earlier preprint's crane, searched"). The same script with
`EXAMPLE_SET=crane` searches six crane sentences with the pages Crane (machine), Crane (bird) and Neck (49,964 chunks);
`DUMP_NEIGHBOURS=1` also writes each query's vector, its 50 neighbours and 250 random non-neighbours to `crane.npz`
(float32, keys `q<k>`, `n<k>`, `pool<k>` for sentence k = 0..5 in the order of `crane.json`). It ships as text,
`data/examples/crane_vectors.csv.xz` (columns `key`, `row`, `v0..v383`; 9 significant digits, which restore the float32
values exactly); the first command below rebuilds the npz.

```bash
cd code/examples
python -c "import numpy as np, pandas as pd; d = pd.read_csv('../../data/examples/crane_vectors.csv.xz'); np.savez('crane.npz', **{k: (lambda a: a[0] if k[0] == 'q' else a)(g.iloc[:, 2:].to_numpy(np.float32)) for k, g in d.groupby('key', sort=False)})"
python to_dat.py ../../data/examples/crane.json out "crane" "I saw a crane." \
  "He craned his neck to watch the crane lift near the crane."     # -> out/ex{0,1,2}_*.dat = code/didactic/data/examples_crane/
python crane_topo.py crane.npz ../../data/examples/crane.json out/crane_topo.dat   # needs ripser
```

Both reproduce the shipped files byte for byte (`crane_topo.dat` = `data/reference_outputs/crane_topo.dat` =
`code/didactic/data/examples_crane/topo_all.dat`; `topo_{W1,Lmax,LTmax}.dat` are its rows for sentences 0, 1 and 5, the
three drawn). Per sentence and feature, `crane_topo.dat` gives the real value, the mean and s.d. of 50 corpus-bearing and
50 random-bearing control draws, and the median nearest-neighbour cosine; the paper's "only the loop summary differs
(0.035 against 0.019), within its spread over the six crane sentences (0.019 to 0.045)" is its `LT_max` rows. The split
sizes behind "the crane example shows the same" (section 5.3) are the `split` entries of `crane.json`.

## 5. Closer to the definition (section "Closer to the definition"; Discussion, "What the longest bar measures")

Two exploratory checks added at the end of the study. Each was written down with its predictions before any statistic
was computed: `docs/lenctl_plan.md` and `docs/ambig_train_plan.md`, each the plan, then amendment 1, then (in
`data/reference_outputs/`) the results. The length-controlled test's amendment (coverage gain, prediction L6) was
written before any of its features was computed. The AmbigNQ check's amendment (flipped queries, F1-F3) was written
after its main results (A1-A4) had been seen, and says so. Both use `ripser` and pandas; embeddings are not shipped.

### 5.1 Length-controlled topic count (`code/lenctl/`)

Topic count varies while text and length stay fixed: 240 random Wikipedia pages (AmbigQA 2020-01-20 dump, seed 0) are
cut into 83-token atoms and grouped into 249-token texts of one page (Q1, C1) or three pages (Q3, C3; grouping seeds
0-2). Cells: Q3|C1 multi-factual, Q1|C3 polysemic, Q1|C1 and Q3|C3 matched. Predictions L1-L6 are in the plan.

```bash
cd code/lenctl
python build.py <AmbigQA docs.db> work   # -> work/texts.json, work/emb_{bge-base,bge-m3}.npz (sentence-transformers, tiktoken)
python feats.py work                     # -> work/feats.csv
python analyze.py work/feats.csv > results.txt
python analyze.py ../../data/lenctl/feats.csv.xz   # from the shipped table
```

On the shipped table, `analyze.py` reproduces `data/reference_outputs/lenctl/results.txt` byte for byte.
`data/lenctl/texts.json` holds the 240 page titles, each atom's page and every text set's atom groupings (indices
only, no text). In `results.txt`, M = Q3|C1 against Q1|C3 (AUROC > 0.5: larger for Q3|C1), R = Q3|C3 against Q1|C1;
the paper uses the blocks `arm all` (primary), grouping seed g0, with bracketed 95% intervals.

| sentence of the paper | `results.txt` |
|---|---|
| W1 larger for multi-factual queries: 0.91 [0.88, 0.93] (bge-base) and 0.96 [0.94, 0.97] (bge-m3); the longest bar 0.94 and 0.99 | rows `w1`, `longbar`, column M |
| matched cells: W1 0.49 [0.44, 0.54] and 0.51 [0.45, 0.56]; the longest bar 0.26 in both models | rows `w1`, `longbar`, column R |
| no single cosine statistic reaches 0.75 (best 0.62 and 0.74) | line `L4 cosines` |
| median nearest-neighbour cosine 0.79 and 0.69 against 0.94 and 0.99 | line `L5 sufficiency` |
| coherence alone separates the regimes, 0.99 and 1.00 | row `coherence`, M (0.012 and 0.003, i.e. oriented 0.988 and 0.997) |
| the corpus-bearing control separates them better (0.97 to 1.00; every gain negative) | rows `w1_corp`, `longbar_corp`, M; lines `gain over corpus-bearing control` |
| three regroupings: W1 0.90 to 0.96 | lines `g0`-`g2`, `w1 M` |

The secondary arm (`ownout`: passages sharing a page with the query excluded) and L6 (line `L6 coverage`) are in the
same file.

### 5.2 AmbigNQ train and the flipped queries (`code/ambig_train/`)

All 10,036 AmbigNQ train questions (label: annotation `multipleQAs`) retrieve their 50 neighbours from the registered
test's corpus (37,528 chunks with all-MiniLM-L6-v2 embeddings; built as `docs/analysis_plan.pdf` describes, by code not
included here); the development split is not read. Amendment 1 searches, for the 4,749 ambiguous questions with at
least two disambiguated questions, the original (type 0), all disambiguations joined (type 1) and the first one (type 2).

```bash
cd code/ambig_train
python ambig.py retrieve DATA_DIR REAL_DIR train   # DATA_DIR/ambignq/train.json; REAL_DIR/chunks.parquet (title, in_corpus), chunk_emb.npy
python ambig.py feats train && python ambig.py auroc train                 # -> train/feats.csv, train/auroc.txt
python ambig.py flip_retrieve DATA_DIR REAL_DIR flip && python ambig.py feats flip && python ambig.py flip_auroc flip
python isolate.py ambig train/retrieved.npz train/corpus_emb.npy           # section 5.3
# from the shipped tables (each reproduces the reference file byte for byte):
mkdir -p t f && xz -dc ../../data/ambig_train/feats_train.csv.xz > t/feats.csv && python ambig.py auroc t        # = auroc_train.txt
xz -dc ../../data/ambig_train/feats_flip.csv.xz > f/feats.csv && python ambig.py flip_auroc f                     # = auroc_flip.txt
```

| sentence of the paper | `data/reference_outputs/ambig_train/` |
|---|---|
| 47% ambiguous; persistence 0.48 to 0.53, similarity statistics 0.51 to 0.53, cosine statistics 0.48 to 0.54, length 0.54 | `auroc_train.txt`: header; rows `w1`, `longbar`, `ltmax`; `nn_med`, `coherence`, `eig2`; `top1`, `mean_cos`, `std_cos`, `slope`; `nwords` (oriented) |
| persistence does not beat the similarity statistics (0.526 against 0.529) or its corpus-bearing control | `auroc_train.txt`: line `A1`; lines `real - corpus-bearing control` |
| original question and first reading: every statistic 0.49 to 0.53, only length differs | `auroc_flip.txt`, block `polysemic vs single` |
| joined query: W1 0.41 [0.40, 0.41], coverage gain 0.47, control 0.35, mean cosine 0.35 | `auroc_flip.txt`, block `multi-factual vs polysemic`: `w1`, `cov_gain_diff`, `w1_corp`, `mean_cos` |

In `auroc_flip.txt` the AUROC is P(the first-named type is larger), with a bootstrap over source questions.

### 5.3 What the longest bar measures (`code/ambig_train/isolate.py`)

Cutting the longest minimum-spanning-tree edge of the 50 unit offsets splits the neighbours in two; `isolate.py`
counts how often the smaller side holds three chunks or fewer, or one, and how often the longest bar equals the largest
nearest-neighbour distance. `data/reference_outputs/ambig_train/isolate_results.txt`: rows `ambig group 0/1` (clear /
ambiguous; "96% ... three chunks or fewer, 83% a single chunk") and `lenctl group Q1` (polysemic, Q1 against C3, g0:
95%) and `Q3_g0` (multi-factual, Q3 against C1: 80%), from `python isolate.py lenctl work/emb_<model>.npz`. The file
does not record which of the two models the lenctl rows used.

## Data and licences

- **Simulation tables**: generated by this code.
- **Nobel lectures**: copyrighted; **not redistributed** (no text, chunk text or embeddings). `lecture_manifest.csv`
  gives lecture id, laureate, prize year, title and token counts, to rebuild from the official public transcripts.
- **Wikipedia**: CC BY-SA 3.0 (AmbigQA 2020-01-20 dump). `selection_manifest.csv`: permutation rank, dump rowid, page
  id, title, characters, tokens of the 40 pages.
- **Essays (Gutenberg)**: public domain in the USA; the Project Gutenberg header and footer are stripped.
  `selection_manifest.csv`: Gutenberg id, title, LoC class, shuffle rank, body tokens, window start and length,
  SHA-256 of the window, catalogue date; `selection_log.csv`: the decision for every candidate walked. Selection rule:
  header of `build_gutenberg.py` (English text, LoC class Q or B outside BL–BX, no fiction/poetry/drama/periodicals/
  reference/juvenile/scripture, one book per author, RDF rights "Public domain in the USA", body ≥ 6,000 tokens; order
  = Text# sorted then `random.Random(0).shuffle`; first 40; window skips the first 10% and keeps ≤ 15,750 tokens).
  Authors are in the official catalogue under the same id.
- **Turing lectures**: copyrighted; obtained by the authors under ACM's terms for research and **not redistributed**
  (no text, chunk text or embeddings). `turing_manifest.csv` lists the DOI, the extracted title line, word count, the
  kept flag (≥ 3,000 extractable words) and the token count after cleaning and cap; rebuild from the DOIs.
- **Fields lectures**: from the IMU's freely available ICM proceedings; rights differ by volume (only the four ICM 2022
  lectures carry a CC BY 4.0 licence), so **no text is redistributed**. `manifest.csv` gives medal year, laureate, ICM
  year, title, PDF page span and source URL; `selection_log.csv` the decision for all 64 medallists; `candidates.csv` and
  `imu_talks.csv` the talks located and checked.
- **AmbigNQ** (section 5.2): questions and labels from AmbigQA, CC BY-SA 3.0; only per-question numbers are shipped
  (`feats_train.csv`: row `i` = position in `train.json`).
- **Embeddings**: none shipped, except `data/examples/crane_vectors.csv.xz` (all-MiniLM-L6-v2 vectors of the six crane sentences,
  their neighbours and 250 random chunks each, all from Wikipedia, CC BY-SA 3.0). Commercial embeddings cannot be
  redistributed; open-model embeddings regenerate from the shipped code (section 2.4; section 5: `build.py`,
  `ambig.py retrieve`).
- **Per-query feature tables** (`features_<model>.csv.xz`, clean text, five models): numbers and chunk identifiers
  only. Dropped to fit the size limit, none used by any reported result: `qg`, `cg`, `n_local`, `top1_same_lecture`;
  the 40 LT_max columns of the 20-draw nulls (`ltmax_enull00..19`, `ltmax_lnull00..19`), except that the corpus-null
  draws on the rows `bayes.py` reads (lodo Q250|C750, lodo_dens Q250|C250) ship in `ltmax_enull_<model>.csv.xz` (keys
  variant, cell, query_id), which `stats.py` never reads; the sorted-cosine and Gram columns (`c01..c50`, `g_*`,
  `nn_q10/med/q90`), except for text-embedding-3-small (the gradient-boosting rival), where they are kept on the rows
  that rival reads (primary and LODO Q250|C750, dens and lodo_dens Q250|C250) and blank elsewhere, and except
  `g_eig1`, `nn_med`, `nn_q10` on the two cells `sufficiency.py` reads (lodo Q250|C750, lodo_dens Q250|C250), kept for
  every shipped model. Prefix tables for the ε-sweep figure (not in the paper) are regenerated by `features.py`.

## Not included

Source texts and embeddings (except the crane vectors, section 4); the per-query tables of four of the nine models (section 2.4); the per-query vMF tables
(`vmf_<model>.csv.gz`, regenerated by `vmf.py`); the dose-response arms' feature tables; the columns listed above;
the 20-seed density control (not reported in the paper); rendered figures and statistics markdown (regenerated by the
commands above); section 5's embeddings and retrieval files (`emb_*.npz`, `retrieved.npz`, `corpus_emb.npy`) and the
registered corpus they search; the demonstration package's module that `ring_examples.py` imports.
