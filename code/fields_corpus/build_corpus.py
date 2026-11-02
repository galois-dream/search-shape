#!/usr/bin/env python3
# build_corpus.py -- written on day 1 at 21:15
# Updated on day 1 at 21:25: English-only rule (non-English qualifiers excluded before the 40 cap).
# Purpose: stage 2 of the Fields/ICM corpus build. Apply the selection rule (README.md) to every
# Fields medalist 1936-2022, using
#   candidates.csv  - hand-located ICM lectures (proceedings volume + PDF page span) per medalist
#   imu_talks.csv   - the IMU "ICM Plenary and Invited Speakers" rows for each medalist
#   <cache>/<volume-stem>/pNNNN.txt - per-page pdftotext output written by fetch_volume.sh
# and write texts/fields_<medal_year>_<surname>.txt, manifest.csv and selection_log.csv.
# Usage: python3 -I build_corpus.py <cache_dir> <medalists_csv>
import csv, os, re, sys, unicodedata, difflib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_WORDS = 3000
CAP = 40
BASE = "https://www.mathunion.org"

# --- boilerplate lines (publisher / congress stamps) dropped wherever they occur on a page ---
BOILER = re.compile(r"|".join([
    r"^Proceedings of the International Congr", r"^Proc\. Int\. Cong\. Math", r"^Actes,? Congr",
    r"^ICM 20\d\d\s*[•·]", r"^P\s*\.\s*I\s*\.\s*C\s*\.\s*M\s*\.", r"^Rio de Janeiro, Vol\.",
    r"^Doc(\.|umenta)\s*Math", r"^©", r"^\(c\) ", r"Birkhäuser Verlag, Basel", r"^Published by EMS Press",
    r"^and licensed under", r"^a CC BY 4\.0 license", r"^DOI 10\.", r"^Berkeley, California, USA, 19",
    r"^August 16-24,? 1983", r"^Hyderabad, India, 2010", r"^Madrid, Spain, 2006", r"^Vancouver, 1974",
    r"^Helsinki,? 1978", r"^of Mathematicians,? Kyoto", r"Mathematical Society of Japan, 199",
    r"^Extra Volume ICM", r"^P$", r"^\.I \.C$", r"^M \. – 20\d\d$", r"^D\s?o\s?c\s?\.$", r"^M\s?A\s?T\s?H\s?\.$",
    r"^D\s?M\s?V$", r"^Doc\. M\s?A\s?T\s?H", r"^© 20\d\d (European Mathematical Society|International Mathematical Union)",
]), re.I)
REFS = re.compile(r"^\s*(\d+\.?\s*)?(R\s?E\s?F\s?E\s?R\s?E\s?N\s?C\s?E\s?S|References|Bibliography|BIBLIOGRAPHY|"
                  r"Bibliographie|BIBLIOGRAPHIE|Références|RÉFÉRENCES|Literatur|LITERATUR|Literaturverzeichnis|"
                  r"Список литературы|ЛИТЕРАТУРА|Литература)[\s.:\d*]*$")  # OCR may append footnote digits
PAGENUM = re.compile(r"^\W{0,3}\d{1,4}\W{0,3}$")


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z]", "", s)


def ascii_fold(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def clean(pages):
    """pages: list of page texts. Returns (body_text, refs_cut: bool)."""
    P = [[l.rstrip() for l in p.split("\n")] for p in pages]
    # running headers/footers: normalised lines among the first/last 3 non-empty lines of a page
    # that recur at the same edge on >= 3 pages of the lecture (or >= 25% of pages for short ones)
    edge = Counter()
    for lines in P:
        ne = [l for l in lines if l.strip()]
        for l in set(ne[:3] + ne[-3:]):
            if norm(l):
                edge[norm(l)] += 1
    thr = max(2, min(3, round(0.25 * len(P))))
    frequent = [k for k, c in edge.items() if c >= thr]

    def is_header(l):
        n = norm(l)
        if not n:
            return PAGENUM.match(l.strip()) is not None
        if n in frequent:
            return True
        return any(difflib.SequenceMatcher(None, n, f).ratio() >= 0.85 for f in frequent if abs(len(f) - len(n)) <= 4)

    out = []
    for k, lines in enumerate(P):
        idx = [i for i, l in enumerate(lines) if l.strip()]
        drop = set()
        for i in idx[:3] + idx[-3:]:
            s = lines[i].strip()
            # first page: keep the title/author block (it repeats the running header); drop page numbers only
            if PAGENUM.match(s) or (k > 0 and is_header(s)):
                drop.add(i)
        for i in idx:
            if BOILER.search(lines[i].strip()):
                drop.add(i)
        out.append("\n".join(l for i, l in enumerate(lines) if i not in drop))
    text = "\n".join(out)
    # cut the reference list: last references-type heading in the final 40% of the text
    lines = text.split("\n")
    cut = None
    for i, l in enumerate(lines):
        if REFS.match(l.strip()) and i >= 0.6 * len(lines):
            cut = i
    if cut is not None:
        lines = lines[:cut]
    # reference labels left on their own line by two-column reference layouts, e.g. "[12]"
    lines = [l for l in lines if not re.fullmatch(r"\s*\[[^\]\s]{1,12}\]\s*", l)]
    while lines and (not lines[-1].strip() or PAGENUM.match(lines[-1].strip())):
        lines.pop()
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", "\n".join(lines))  # font-encoding control bytes
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text, cut is not None


def language(text):
    w = Counter(re.findall(r"[a-zà-ÿа-я]+", text.lower()))
    score = {
        "en": sum(w[x] for x in ("the", "and", "of", "is", "that", "we")),
        "fr": sum(w[x] for x in ("le", "la", "les", "des", "est", "une")),
        "de": sum(w[x] for x in ("der", "die", "und", "das", "ist", "nicht")),
        "ru": sum(w[x] for x in ("и", "в", "что", "на", "не")),
    }
    return max(score, key=score.get)


def main(cache, medalists_csv):
    cand = list(csv.DictReader(open(os.path.join(HERE, "candidates.csv"), encoding="utf-8")))
    talks = list(csv.DictReader(open(os.path.join(HERE, "imu_talks.csv"), encoding="utf-8")))
    # medalist roster from candidates.csv, cross-checked against the medalists CSV (64 people)
    roster = {}
    for r in cand:
        roster.setdefault(r["surname"], (int(r["medal_year"]), r["laureate"]))
    src = {int(r["Year"]): ascii_fold(r["Laureate(s)"]) for r in csv.DictReader(open(medalists_csv, encoding="utf-8"))}
    for s, (y, name) in roster.items():
        key = ascii_fold(name.split()[-1] if s != "ngo" else "ngo")
        assert key in src[y], f"{name} not found in medalists CSV for {y}"
    n_src = sum(len(v.split(";")) for v in src.values())
    assert len(roster) == n_src == 64, (len(roster), n_src)

    os.makedirs(os.path.join(HERE, "texts"), exist_ok=True)
    for f in os.listdir(os.path.join(HERE, "texts")):  # rebuild from scratch: no stale files
        if f.startswith("fields_") and f.endswith(".txt"):
            os.remove(os.path.join(HERE, "texts", f))
    manifest, log, unresolved = [], [], []
    for s, (my, name) in sorted(roster.items(), key=lambda kv: (kv[1][0], kv[0])):
        rows = [r for r in cand if r["surname"] == s and r["icm_year"]]
        # every IMU-listed plenary/section talk enters the ordering; one without a candidates.csv row is
        # "unlocated" and must be located if the rule ever reaches it
        for t in talks:
            if t["surname"] == s and "Abel Lecture" not in t["section_or_type"]:
                if not any(r["icm_year"] == t["icm_year"] for r in rows):
                    kind = "plenary" if t["section_or_type"].startswith("Plenary") else "section"
                    rows.append(dict(icm_year=t["icm_year"], kind=kind, status="unlocated"))
        order = sorted(rows, key=lambda r: (abs(int(r["icm_year"]) - my), r["kind"] != "plenary", int(r["icm_year"])))
        tried, chosen = [], None
        for r in order:
            if r["status"].startswith("absent"):
                tried.append(f"{r['icm_year']} {r['kind']}: {r['status'][7:]}")
                continue
            if r["status"] == "unlocated":
                # resolved only if needed: see the cap check below
                unresolved.append((my, name, f"{'; '.join(tried)}; next by the rule, {r['icm_year']} {r['kind']}, not examined"))
                tried = None
                break
            stem = os.path.basename(r["volume"])[:-4]
            a, b = int(r["pdf_first"]), int(r["pdf_last"])
            pages = [open(os.path.join(cache, stem, "p%04d.txt" % p), encoding="utf-8").read() for p in range(a, b + 1)]
            text, refcut = clean(pages)
            words = len(text.split())
            if words >= MIN_WORDS:
                chosen = (r, text, words, refcut)
                break
            tried.append(f"{r['icm_year']} {r['kind']}: {words} words < {MIN_WORDS}")
        if tried is None:
            continue
        if not rows:
            tried.append("no ICM plenary or invited talk in the IMU speaker list")
        if chosen:
            r, text, words, refcut = chosen
            manifest.append(dict(medal_year=my, laureate=name, icm_year=r["icm_year"], title=r["title"],
                                 pages=f"{r['pdf_first']}-{r['pdf_last']}",
                                 source_url=f"{BASE}{r['volume']}#page={r['pdf_first']}", words=words,
                                 _text=text, _s=s, _lang=language(text), _kind=r["kind"], _tried=tried, _refcut=refcut))
        else:
            log.append(dict(medal_year=my, laureate=name, decision="skipped",
                            reason="; ".join(tried) or "no lecture found", icm_year="", language="", words=""))
    # English only (rule change on day 1 at 21:25, before any feature was computed): a qualifier whose
    # selected lecture is not in English is excluded, without falling back to another talk
    for m in [m for m in manifest if m["_lang"] != "en"]:
        manifest.remove(m)
        log.append(dict(medal_year=m["medal_year"], laureate=m["laureate"], decision="skipped",
                        reason=f"excluded by the English-only rule: lecture ({m['icm_year']} {m['_kind']}, {m['words']} words) is in {m['_lang']}",
                        icm_year=m["icm_year"], language=m["_lang"], words=m["words"]))
    # cap: keep the CAP most recent medal years among English qualifiers (ties: alphabetical surname)
    manifest.sort(key=lambda m: (-m["medal_year"], m["_s"]))
    kept, over = manifest[:CAP], manifest[CAP:]
    cutoff = kept[-1]["medal_year"] if over else 0
    for my, name, why in unresolved:
        if my >= cutoff:
            raise SystemExit(f"{name}: {why} -- locate it, the cap does not settle this case")
        log.append(dict(medal_year=my, laureate=name, decision="skipped", icm_year="", language="", words="",
                        reason=f"{why}; moot: medal year older than the cap cut-off ({cutoff})"))
    for m in over:
        log.append(dict(medal_year=m["medal_year"], laureate=m["laureate"], decision="skipped",
                        reason=f"qualifies ({m['icm_year']} {m['_kind']}, {m['words']} words) but beyond the {CAP}-lecture cap (older medal year)",
                        icm_year=m["icm_year"], language=m["_lang"], words=m["words"]))
    for m in kept:
        fn = f"fields_{m['medal_year']}_{m['_s']}.txt"
        open(os.path.join(HERE, "texts", fn), "w", encoding="utf-8").write(m["_text"])
        why = f"kept: {m['icm_year']} {m['_kind']}" + (f" (closer talks failed: {'; '.join(m['_tried'])})" if m["_tried"] else "")
        why += "" if m["_refcut"] else "; no reference heading found, list kept"
        log.append(dict(medal_year=m["medal_year"], laureate=m["laureate"], decision="included", reason=why,
                        icm_year=m["icm_year"], language=m["_lang"], words=m["words"]))
    kept.sort(key=lambda m: (m["medal_year"], m["_s"]))
    with open(os.path.join(HERE, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["medal_year", "laureate", "icm_year", "title", "pages", "source_url", "words"], extrasaction="ignore")
        w.writeheader(); w.writerows(kept)
    log.sort(key=lambda l: (l["medal_year"], l["laureate"]))
    with open(os.path.join(HERE, "selection_log.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["medal_year", "laureate", "decision", "icm_year", "language", "words", "reason"])
        w.writeheader(); w.writerows(log)
    print(f"kept {len(kept)}, qualified {len(manifest)}, skipped {len(log) - len(kept)}, words {sum(m['words'] for m in kept)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
