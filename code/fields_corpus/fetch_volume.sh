#!/bin/bash
# fetch_volume.sh -- written on day 1 at 21:00
# Purpose: stage 1 of the Fields/ICM corpus build. Download ONE ICM proceedings volume from the
# IMU site, dump its existing text layer page by page (pdftotext, no OCR), record its page count,
# then DELETE the whole-volume PDF. Only the per-page text stays (in a cache dir outside the repo).
# Usage: fetch_volume.sh <IMU path, e.g. /fileadmin/ICM/Proceedings/ICM1974.1/ICM1974.1.ocr.pdf> <cache_dir>
# Output: <cache_dir>/<volume-stem>/p0001.txt ... (one file per PDF page), <cache_dir>/<stem>/NPAGES
set -euo pipefail
path="$1"; cache="$2"
stem=$(basename "$path" .pdf)
out="$cache/$stem"
if [ -f "$out/NPAGES" ]; then echo "cached: $stem"; exit 0; fi
mkdir -p "$out"
pdf="$cache/$stem.pdf"
sleep 1   # politeness: ~1 s between requests
curl -sSfL -A "Mozilla/5.0 (academic corpus build; one file at a time)" -o "$pdf" "https://www.mathunion.org$path"
n=$(pdfinfo "$pdf" | awk '/^Pages:/{print $2}')
# one pass over the whole volume; pdftotext ends every page with a form feed -> split on it
pdftotext -q -enc UTF-8 "$pdf" - | awk -v d="$out" 'BEGIN{RS="\f"; i=0} {i++; f=sprintf("%s/p%04d.txt",d,i); printf "%s", $0 > f; close(f)}'
echo "$n" > "$out/NPAGES"
echo "$(stat -f %z "$pdf") https://www.mathunion.org$path" >> "$cache/DOWNLOADED_BYTES"
rm -f "$pdf"
echo "done: $stem ($n pages)"
