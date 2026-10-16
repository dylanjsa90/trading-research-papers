"""Helpers shared by paperdigest-downloader.py and browser-download.py."""
import csv
import json
import os
import re
import unicodedata

import pymupdf

# Below this the PDF is likely an author template, a user guide, or a paper from the reference list.
MIN_TITLE_SCORE = 0.85

CSV_FIELDS = ["rank", "status", "title", "authors", "source", "date", "view_url", "pdf_url", "pdf_path", "note"]


def _words(s):
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())


def title_score(title, pdf_bytes):
    """Fraction of the title's words found on the PDF's first two pages (page 1 is often a publisher cover)."""
    want = [w for w in _words(re.sub(r"\[[^\]]*\]", " ", title)) if len(w) > 2]  # drop tags like [pre-publication]
    if not want:
        return 1.0
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = " ".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
    except Exception:
        return 0.0
    got = _words(text)
    words, joined = set(got), "".join(got)  # joined catches words split by hyphenation/line breaks
    return sum(w in words or w in joined for w in want) / len(want)


def save_manifest(outdir, rows):
    """Write manifest.json (source of truth) and manifest.csv from a list of row dicts."""
    rows = sorted(rows, key=lambda r: r["rank"])
    path = os.path.join(outdir, "manifest.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    path = os.path.join(outdir, "manifest.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)
