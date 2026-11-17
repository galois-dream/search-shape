# Shipped copy of paper/trec_p2/fetch_topics.py; only this line added for the repo layout (PREREG.md deviation 7 = docs/analysis_plan_deviation7.md). The topic files themselves are not shipped.
# fetch_topics.py -- written 2026-11-14 09:55 CET. P2 (PREREG.md deviation 7): download the TREC Web track 2009-2012
# topic files from NIST and write TOPICS_TSV (number, query, type). RUN ONLY AFTER deviation 7 is filed on OSF, and record
# the download time in PREREG.md. Reads the query text and the NIST type only; subtopics are never written out.
# Usage: python fetch_topics.py OUT_DIR
import sys, re, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

URLS = {  # NIST's published topic files (the year's Web track pages on trec.nist.gov)
    2009: "https://trec.nist.gov/data/web/09/wt09.topics.full.xml",
    2010: "https://trec.nist.gov/data/web/10/wt2010-topics.xml",
    2011: "https://trec.nist.gov/data/web/11/full-topics.xml",
    2012: "https://trec.nist.gov/data/web/12/full-topics.xml",
}
# Usage 2 (NIST refuses scripted downloads): python fetch_topics.py OUT_DIR SAVED_DIR, where SAVED_DIR holds the four files
# saved from a browser as 2009.xml, 2010.xml, 2011.xml and 2012.xml (2011 and 2012 share the NIST name full-topics.xml).
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rows = []
saved = Path(sys.argv[2]) if len(sys.argv) > 2 else None
for year, url in URLS.items():
    raw = (saved / f"{year}.xml").read_bytes() if saved else urllib.request.urlopen(url, timeout=60).read()
    (out / f"topics_{year}.xml").write_bytes(raw)
    root = ET.fromstring(raw)
    for t in root.iter("topic"):
        rows.append((int(t.get("number")), " ".join(t.findtext("query").split()), t.get("type")))
rows.sort()
with open(out / "topics.tsv", "w") as f:
    f.write("number\tquery\ttype\n")
    for n, q, ty in rows:
        f.write(f"{n}\t{q}\t{ty}\n")
print(f"{len(rows)} topics; numbers {rows[0][0]}-{rows[-1][0]}", flush=True)
