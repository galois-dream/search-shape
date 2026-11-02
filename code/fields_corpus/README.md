# Fields medalists' ICM lectures: the fifth real-text corpus

Built on day 1 at 21:16. Updated 21:18: English-only rule, decided before any feature was computed (§ Selection rule, step 5). Pages and statements below were read from the sites on day 1 at 20:55–21:15.
Purpose: the fifth corpus for the paper's real-text study. It holds one lecture per Fields medalist, given at an International Congress of Mathematicians (ICM).
It runs through the same pipeline as the Nobel, essays, Turing and Wikipedia corpora. The pipeline applies the 15,750-token cap, so no length cap is applied here.
The predictions in `PREDICTIONS.md` were written before this corpus was built, and the build did not change them.

## Result

- **40 lectures kept, all in English: 344,533 words** (whitespace tokens after cleaning).
  - Median 8,577 words; minimum 3,701; maximum 20,221 (Duminil-Copin).
- **24 of the 64 medalists (1936–2022) are not in the corpus:**
  - 11 had no lecture that passes the rule;
  - 2 qualify but their lecture is not in English: Faltings (German, ICM 1994) and Lafforgue (French, ICM 2002);
  - 9 qualify in English but fall outside the cap of 40;
  - for 2 (Ahlfors, Atiyah), the next talk the rule would reach was never examined, which the cap makes moot.
  - `selection_log.csv` gives the reason for each person.
- **Kept lectures span medal years 1974–2022** and ICMs 1974–2022.
- **Swap under the English-only rule:** Faltings and Lafforgue were replaced by the two most recent English qualifiers from the capped list, Bombieri (medal 1974, ICM 1974) and Mumford (medal 1974, ICM 2002).
- **Language** comes from a stop-word heuristic; the label for each person is in the `language` column of `selection_log.csv`.

## Source and licence

- **Source:** the IMU's ICM Proceedings page, https://www.mathunion.org/icm/proceedings, which links free PDFs of every volume, 1893–2022.
  - No login and no paywall. Every file came from `www.mathunion.org`.
- **The page carries no reuse licence of its own.** Its only rights line is the site footer, "© Copyright International Mathematical Union 2026".
- **The IMU's statements on access:**
  - IMU-Net 30, July 2008 (https://www.mathunion.org/imu-net/archive/2008/imu-net-030):
    > "One of the projects the CEIC is pursuing is the digitization of all the Proceedings of the International Congresses of Mathematicians to make them readily accessible to the public. […] After having collected the copyrights -- which we eventually got -- our production process goes by first finding a copy of the respective proceedings volumes […] we will also make the scanned images publicly available, and we will encourage sites to mirror the collection, as this will contribute to its longevity"
  - IMU Circular Letter 7/2019, by Helge Holden, IMU Secretary General (https://www.mathunion.org/fileadmin/IMU/Publications/CircularLetters/2019-2020/IMU%20AO%20CL%207_2019.pdf):
    > "Scanned copies of all ICM proceedings can be found at https://www.mathunion.org/icm/proceedings."
- **Rights differ by volume.** These are the statements printed in the volumes used here:
  - **ICM 2022** (EMS Press), vol. 1 imprint page:
    > "The content of this volume is licensed under the CC BY 4.0 license, with the exception of the logos and branding of the International Mathematical Union and EMS Press, and where otherwise noted."
  - **ICM 2018** (vols. 1–2):
    > "Copyright © 2018 by Sociedade Brasileira de Matemática and International Mathematical Union. […] All rights reserved."
  - **ICM 1936–2014:** these are IMU-digitised scans or publisher PDFs, with the copyright of the original publishers or authors.
    - The IMU digitisation project's metadata page (https://www.math.uni-bielefeld.de/~rehmann/icm_proceedings.html) records the permission it obtained for each year.
    - Examples: "AMS, granted" (1950, 1986); "rights with authors" (1954, 1970, 1983).
- **What this means here:** the texts are openly available for download for research, but only the four 2022 lectures carry an open reuse licence.
  - This corpus is used for text analysis only.
  - **The texts are not redistributed.** The only possible exception is the four ICM 2022 lectures, which are CC BY 4.0; anything else needs a rights check, and that call is the authors'.

## Selection rule (exact)

1. **Medalists:** all 64 Fields medalists 1936–2022, from a list of medalists by year (columns `Year`, `Laureate(s)`, as on the Fields Medal Wikipedia page; not shipped). It was complete and matches the IMU list; `selection_log.csv` names all 64. Grigori Perelman, who declined the 2006 medal, is included.
2. **Talks:** every ICM plenary or invited section talk listed for the medalist on the IMU "ICM Plenary and Invited Speakers" list (https://www.mathunion.org/icm-plenary-and-invited-speakers; the rows are saved in `imu_talks.csv`).
   - The list also has "ICM Abel Lecture" entries; these are not plenary or section talks and are excluded.
   - Two talks printed in the proceedings but missing from that list were added: Thompson 1970 "Quadratic Pairs" and the co-authored Milnor–Kervaire 1958 talk. Neither changed any decision.
   - Laudationes ("The work of X") and other talks about the medalist are never used.
3. **Order:** the talks are ranked by |ICM year − medal year|.
   - Ties go to the plenary over the section talk, then to the earlier congress.
4. **Walking the ranked list:**
   - A talk with no text in the proceedings is passed over. This covers a lecture not given, a manuscript not submitted, or a laudatio only.
   - A talk whose cleaned text has fewer than 3,000 words is also passed over.
   - The first talk with at least 3,000 words is the person's lecture: one per person.
5. **English only:** a qualifying lecture must be in English; otherwise the person is excluded, with no fall-back to another talk.
   - This was decided on day 1 after the first build and before any feature was computed.
   - It excluded Faltings (German) and Lafforgue (French).
6. **Cap:** keep **the 40 most recent English-language qualifiers** by medal year (ties by surname).
   - 46 English lectures qualified. The cut fell cleanly between medal years 1970 and 1974, so no tie-break was needed.
7. **Unexamined talks:** if the walk would reach a talk that was never examined, the case is settled only when the cap makes it moot. That applies to Ahlfors 1932 and Atiyah 1978, both from medal years before the 1974 cut-off. Otherwise the build stops and asks for the talk to be located.

## Text extraction (`build_corpus.py`)

- **Source text:** the volume's existing text layer, via `pdftotext` (poppler, default mode). No OCR was run here.
  - The pre-2006 `*.ocr.pdf` volumes carry the IMU's own OCR layer, so OCR errors in those texts are inherited.
  - Only the lecture's pages are used. `pages` in `manifest.csv` gives **PDF page indices within the volume at `source_url`**, not printed page numbers.
- **Running headers, footers and page numbers** are removed by one simple rule.
  - A line among the first or last 3 non-empty lines of a page is removed if either:
    - it is a bare number; or
    - its letters-only lowercase form (or a ≥ 0.85 fuzzy match of it) recurs at a page edge on at least 3 pages of the lecture (2 for lectures of up to 10 pages).
  - The first page is exempt, so the title and author survive.
  - Publisher and congress stamps are removed by a fixed regex list, for example "Proceedings of the International Congress…", "© … Birkhäuser", "Documenta Mathematica…", "P.I.C.M. – 2018" and "CC BY 4.0 license".
- **Reference lists:** the text is cut at the last "References/Bibliography/Bibliographie/Literatur" heading that falls in the final 40% of the text.
  - Reference labels left alone on a line ("[12]") are dropped.
  - Freedman 1983 has no detectable heading, so his reference list stays in.
- **Control bytes:** C0 control characters left by math-font encodings are removed.
- **Kept as extracted:** the body, including formulas.
  - **Formula extraction is lossy.** Equations come out as symbol runs ("p x x", ".log x/A", "Vg,n+1"), sub- and superscripts are flattened, and OCR'd formulas can be garbled.
  - Some figure labels and a few header fragments survive, for example "J." and "6 1 7" on Gowers's first page and initials on 2018 first pages.

## Files

- `texts/fields_<medal_year>_<surname>.txt`: one file per kept lecture, 40 in all (not shipped: see Source and licence).
- `manifest.csv` (in `data/realtext/fields/`): medal_year, laureate, icm_year, title, pages, source_url, words.
- `selection_log.csv`: all 64 medalists, with the decision, the ICM year used, language, words and the reason. Failed closer talks are listed too.
- `candidates.csv` (96 rows): the hand-located lectures, with the volume and PDF page span for each, plus the talks that were checked and found absent.
- `imu_talks.csv`: the medalists' rows from the IMU speaker list, fetched on day 1.
- `fetch_volume.sh`: stage 1. It downloads one volume (sleeping 1 s first), dumps the per-page text to a cache directory and deletes the PDF.
- `build_corpus.py`: stage 2. It applies the rule and writes the outputs.

**To rebuild:**
1. Run `fetch_volume.sh <volume path from candidates.csv> <cache>` for each volume.
2. Run `python3 -I build_corpus.py <cache> <fields_medalists.csv>`.

**Cost of this build:** 29 volumes, 1.84 GB downloaded one at a time. Each PDF was deleted right after its text dump. The per-page text cache is about 92 MB and sits outside the repo.

## Caveats and uncertainty

- **Distance from the medal:**
  - Most lectures are from the medal congress itself or 4 years from it.
  - Five are further away:
    - Mumford (ICM 2002, 28 years after his medal; his 1970 and 1962 talks had fewer than 3,000 words);
    - Margulis (ICM 1990, 12 years after);
    - Perelman (ICM 1994, 12 years before);
    - Freedman (ICM 1983, 3 years before);
    - Mirzakhani (ICM 2010; her 2014 plenary was not given).
- **Not examined under the English-only rule:** whether Lafforgue's 1998 section talk is in English and ≥ 3,000 words. Step 5 does not fall back to another talk.
- **Not checked word by word:**
  - The language label comes from a stop-word count.
  - "No text in proceedings" decisions rest on contents pages and name searches of the volume text, so an OCR-mangled name could hide a lecture.
  - The decisions affected are Kodaira 1954, Margulis 1974, Deligne 1974, Thurston 1978, Zelmanov 1994, Cohen and Novikov 1966, and Thom 1983.
- **Unconfirmed:** I believe Drinfeld's 1986 lecture was printed but not delivered in person. The rule takes the printed lecture.
- **Excluded by language or cap:**
  - Qualifiers excluded by the cap: Selberg, Hörmander, Grothendieck, Smale, Baker and Hironaka (English); Schwartz, Serre and Thom (French).
  - All are from the OCR'd 1950–1970 congresses.
