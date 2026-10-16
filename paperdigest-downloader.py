#!/usr/bin/env python3
"""
Download the papers listed on a Paper Digest "Recent Papers on ..." page.

Default page:
  https://resources.paperdigest.org/2020/04/recent-papers-on-algorithmic-trading-high-frequency-trading/

Each row on the page has a "View" link pointing at the paper's source. The
source varies (arXiv, doi.org -> many publishers, PubMed, OpenReview, NeurIPS,
AAAI OJS, ACL Anthology, ACM DL, GitHub ...), so every row is run through a
chain of resolvers that each propose candidate PDF URLs until one of them
actually returns a PDF:

  1. Source-specific rules   arXiv abs->pdf, OpenReview forum->pdf, ACL .pdf,
                             ACM /doi/pdf/, PubMed -> PMC (Europe PMC render),
                             NeurIPS virtual page -> OpenReview, GitHub README
                             -> arXiv link
  2. Unpaywall               legal open-access copies of DOIs (needs --email)
  3. Semantic Scholar        openAccessPdf / arXiv id
  4. Landing-page scrape     <meta name="citation_pdf_url">, PDF <link>s, etc.
  5. arXiv title search      preprint version of paywalled papers

Nothing tries to bypass paywalls. Papers with no open copy are recorded in
manifest.json as "not_found" together with their landing page, so you can
fetch them by hand (e.g. through your library).

Usage:
  uv run --env-file .env python paperdigest-downloader.py
  uv run python paperdigest-downloader.py --since 2026-01-01 --limit 50 --workers 4
  uv run python paperdigest-downloader.py --html saved_page.html   # if the site blocks scripts
  uv run python paperdigest-downloader.py --dry-run                # just list papers -> manifest

Every PDF must have the paper's title on its first two pages, otherwise it is
rejected (landing pages often link templates, guides or cited papers).

Re-running resumes: papers already downloaded are skipped, failed ones retried.
Results are merged into the existing manifest, so rows outside --since/--limit
and notes left by browser-download.py are kept.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from urllib.parse import parse_qs, quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import paperlib
from paperlib import MIN_TITLE_SCORE, title_score

DEFAULT_URL = (
    "https://resources.paperdigest.org/2020/04/"
    "recent-papers-on-algorithmic-trading-high-frequency-trading/"
)
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
)
TIMEOUT = 40
MAX_PDF_BYTES = 200 * 1024 * 1024

# Minimum seconds between requests to the same host (politeness).
HOST_DELAY = {
    "arxiv.org": 3.0,
    "export.arxiv.org": 3.0,
    "api.semanticscholar.org": 1.5,
    "eutils.ncbi.nlm.nih.gov": 0.4,
}
DEFAULT_DELAY = 1.0


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Paper:
    rank: int
    title: str
    authors: str
    source: str
    date: str
    view_url: str
    status: str = "pending"  # downloaded | not_found | error | listed
    pdf_path: str = ""
    pdf_url: str = ""
    note: str = ""
    tried: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
class Http:
    def __init__(self):
        self._local = threading.local()
        self._host_lock: dict[str, threading.Lock] = {}
        self._host_last: dict[str, float] = {}
        self._glock = threading.Lock()

    @property
    def s(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            s = requests.Session()
            s.headers.update(
                {
                    "User-Agent": UA,
                    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                }
            )
            self._local.s = s
        return self._local.s

    def _throttle(self, url: str):
        host = urlparse(url).hostname or ""
        with self._glock:
            lock = self._host_lock.setdefault(host, threading.Lock())
        with lock:
            delay = HOST_DELAY.get(host, DEFAULT_DELAY)
            wait = self._host_last.get(host, 0) + delay - time.time()
            if wait > 0:
                time.sleep(wait)
            self._host_last[host] = time.time()

    def get(self, url: str, **kw) -> requests.Response:
        self._throttle(url)
        kw.setdefault("timeout", TIMEOUT)
        kw.setdefault("allow_redirects", True)
        r = self.s.get(url, **kw)
        if r.status_code == 429:  # back off once
            time.sleep(int(r.headers.get("Retry-After", "10") or 10))
            self._throttle(url)
            r = self.s.get(url, **kw)
        return r

    def json(self, url: str, **kw):
        try:
            r = self.get(url, headers={"Accept": "application/json"}, **kw)
            return r.json() if r.ok else None
        except (requests.RequestException, ValueError):
            return None

    def html(self, url: str) -> tuple[str, str] | None:
        """Return (final_url, html) or None."""
        try:
            r = self.get(url)
        except requests.RequestException:
            return None
        if not r.ok or "html" not in r.headers.get("Content-Type", ""):
            return None
        return r.url, r.text


# --------------------------------------------------------------------------- #
# Listing parser
# --------------------------------------------------------------------------- #
def parse_listing(html: str) -> list[Paper]:
    soup = BeautifulSoup(html, "html.parser")
    papers = []
    for tr in soup.select("table tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 5:
            continue
        rank_txt = tds[0].get_text(strip=True)
        if not rank_txt.isdigit():
            continue
        title_el = tds[1].find("b") or tds[1].find("a")
        view = next(
            (
                a
                for a in tds[1].find_all("a")
                if a.get_text(strip=True).lower() == "view"
            ),
            None,
        )
        papers.append(
            Paper(
                rank=int(rank_txt),
                title=" ".join(title_el.get_text(" ", strip=True).split())
                if title_el
                else "",
                authors=" ".join(tds[2].get_text(" ", strip=True).split()),
                source=tds[3].get_text(strip=True),
                date=tds[4].get_text(strip=True),
                view_url=view["href"].strip() if view and view.has_attr("href") else "",
            )
        )
    return papers


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>?#]+", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/[0-9]{7})",
    re.IGNORECASE,
)


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def slugify(t: str, n: int = 80) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")
    return t[:n].rstrip("_") or "paper"


def doi_from_url(url: str) -> str | None:
    p = urlparse(url)
    if p.hostname and p.hostname.endswith("doi.org"):
        return p.path.lstrip("/") or None
    m = DOI_RE.search(url)
    return m.group(0).rstrip(".") if m else None


def pdf_links_in_html(html: str, base: str) -> list[str]:
    """Candidate PDF links advertised by a landing page (most reliable first)."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []

    def add(u):
        if u:
            u = urljoin(base, u.strip())
            if u.startswith("http") and u not in out:
                out.append(u)

    for m in soup.find_all(
        "meta",
        attrs={
            "name": re.compile(
                r"^(citation_pdf_url|eprints\.document_url|bepress_citation_pdf_url)$",
                re.IGNORECASE,
            )
        },
    ):
        add(m.get("content"))
    for l in soup.find_all("link", attrs={"type": "application/pdf"}):
        add(l.get("href"))
    for a in soup.find_all("a", href=True):
        href, txt = a["href"], a.get_text(" ", strip=True).lower()
        cls = " ".join(a.get("class", [])).lower()
        if (
            href.lower().split("?")[0].endswith(".pdf")
            or re.search(r"/(pdf|pdfdirect|epdf|download)(/|$|\?)", href, re.IGNORECASE)
            or ("pdf" in txt and len(txt) < 40)
            or "pdf" in cls
        ):
            if not href.startswith(("javascript:", "#", "mailto:")):
                add(href)
    return out[:8]


# --------------------------------------------------------------------------- #
# Resolvers: each yields candidate PDF URLs
# --------------------------------------------------------------------------- #
class Resolver:
    def __init__(self, http: Http, email: str | None, arxiv_search: bool):
        self.http, self.email, self.arxiv_search = http, email, arxiv_search

    # ---- entry point ------------------------------------------------------ #
    def candidates(self, p: Paper) -> Iterator[str]:
        url = p.view_url
        if not url:
            yield from self.by_title(p.title)
            return
        host = (urlparse(url).hostname or "").lower()
        seen_doi = None

        if "arxiv.org" in host:
            yield from self.arxiv(url)
        elif "openreview.net" in host:
            yield from self.openreview(url)
        elif "aclanthology.org" in host:
            yield url.rstrip("/").removesuffix(".pdf") + ".pdf"
        elif "pubmed.ncbi.nlm.nih.gov" in host:
            for item in self.pubmed(url):
                if item.startswith("doi:"):
                    seen_doi = item[4:]
                else:
                    yield item
        elif host.endswith("neurips.cc") or host.endswith("nips.cc"):
            yield from self.neurips(url)
        elif host == "github.com":
            yield from self.github(url)

        doi = seen_doi or doi_from_url(url)
        if doi:
            yield from self.doi(doi)
        elif not host.endswith("doi.org"):
            yield from self.landing(url)

        if self.arxiv_search:
            yield from self.by_title(p.title)

    # ---- source-specific -------------------------------------------------- #
    def arxiv(self, url):
        m = ARXIV_RE.search(url)
        if m:
            yield f"https://arxiv.org/pdf/{m.group(1)}"

    def openreview(self, url):
        q = parse_qs(urlparse(url).query)
        if "id" in q:
            yield f"https://openreview.net/pdf?id={q['id'][0]}"

    def pubmed(self, url):
        m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
        if not m:
            return
        pmid = m.group(1)
        data = self.http.json(
            "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
            f"?ids={pmid}&format=json&tool=paperdigest_dl"
            + (f"&email={quote(self.email)}" if self.email else "")
        )
        rec = (data or {}).get("records", [{}])[0] if data else {}
        pmcid, doi = rec.get("pmcid"), rec.get("doi")
        if pmcid:
            yield f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
            yield f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
        if not doi:
            page = self.http.html(url)
            if page:
                dm = re.search(r'name="citation_doi"\s+content="([^"]+)"', page[1])
                doi = dm.group(1) if dm else None
        if doi:
            yield "doi:" + doi

    def neurips(self, url):
        page = self.http.html(url)
        if not page:
            return
        final, html = page
        for m in re.finditer(
            r"https?://openreview\.net/(?:forum|pdf)\?id=[\w\-]+", html
        ):
            yield from self.openreview(m.group(0))
        for m in re.finditer(
            r'https?://(?:papers|proceedings)\.(?:nips|neurips)\.cc/[^"\'\s<>]+', html
        ):
            u = m.group(0)
            yield u if u.endswith(".pdf") else from_landing_marker(u)
        yield from (u for u in pdf_links_in_html(html, final))

    def github(self, url):
        page = self.http.html(url)
        if page:
            m = ARXIV_RE.search(page[1])
            if m:
                yield f"https://arxiv.org/pdf/{m.group(1)}"
            d = DOI_RE.search(page[1])
            if d:
                yield from self.doi(d.group(0).rstrip("."))

    # ---- DOI chain -------------------------------------------------------- #
    def doi(self, doi: str):
        d = doi.lower()
        if d.startswith("10.48550/arxiv."):
            yield f"https://arxiv.org/pdf/{doi.split('.', 2)[-1]}"
        if d.startswith("10.1145/"):  # ACM DL is open access
            yield f"https://dl.acm.org/doi/pdf/{doi}"
        if d.startswith("10.1109/"):  # IEEE: OA papers only
            pass  # handled by Unpaywall/S2

        if self.email:
            data = self.http.json(
                f"https://api.unpaywall.org/v2/{quote(doi)}?email={quote(self.email)}"
            )
            if data:
                locs = [data.get("best_oa_location")] + (data.get("oa_locations") or [])
                for loc in filter(None, locs):
                    if loc.get("url_for_pdf"):
                        yield loc["url_for_pdf"]
                for loc in filter(None, locs):
                    if loc.get("url_for_landing_page"):
                        yield from_landing_marker(loc["url_for_landing_page"])

        s2 = self.http.json(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi)}"
            "?fields=openAccessPdf,externalIds"
        )
        if s2:
            ax = (s2.get("externalIds") or {}).get("ArXiv")
            if ax:
                yield f"https://arxiv.org/pdf/{ax}"
            u = (s2.get("openAccessPdf") or {}).get("url")
            if u:
                yield u

        yield from self.landing(f"https://doi.org/{doi}")

    # ---- landing page scrape --------------------------------------------- #
    def landing(self, url):
        yield from_landing_marker(url)

    # ---- arXiv title search ---------------------------------------------- #
    def by_title(self, title: str):
        if not title:
            return
        q = re.sub(r"[^\w\s\-]", " ", title)
        q = " AND ".join(f"ti:{w}" for w in q.split() if len(w) > 2)[:900]
        try:
            r = self.http.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": q, "max_results": 5},
            )
            root = ET.fromstring(r.content)
        except (requests.RequestException, ET.ParseError):
            return
        ns = {"a": "http://www.w3.org/2005/Atom"}
        want = norm_title(title)
        for e in root.findall("a:entry", ns):
            t = e.findtext("a:title", "", ns)
            if norm_title(t) == want:
                m = ARXIV_RE.search(e.findtext("a:id", "", ns))
                if m:
                    yield f"https://arxiv.org/pdf/{m.group(1)}"
                    return


LANDING = "landing::"


def from_landing_marker(url: str) -> str:
    """Mark a URL as 'HTML page: fetch it and look for PDF links inside'."""
    return LANDING + url


# --------------------------------------------------------------------------- #
# Downloader
# --------------------------------------------------------------------------- #
def looks_like_pdf(first: bytes) -> bool:
    return b"%PDF" in first[:1024]


class Downloader:
    def __init__(self, http: Http, resolver: Resolver, outdir: str):
        self.http, self.resolver, self.outdir = http, resolver, outdir

    def fetch_pdf(
        self,
        url: str,
        dest: str,
        depth: int = 0,
        tried: list | None = None,
        title: str = "",
        rejected: list | None = None,
    ) -> str | None:
        """Download url to dest if it is a PDF of this title. If it's HTML, look one level deeper."""
        tried = tried if tried is not None else []
        rejected = rejected if rejected is not None else []
        if url in tried or len(tried) > 25:
            return None
        tried.append(url)
        try:
            r = self.http.get(url, stream=True)
        except requests.RequestException:
            return None
        try:
            if not r.ok:
                return None
            it = r.iter_content(65536)
            first = next(it, b"")
            if looks_like_pdf(first):
                tmp = dest + ".part"
                size = 0
                with open(tmp, "wb") as f:
                    f.write(first)
                    size += len(first)
                    for chunk in it:
                        size += len(chunk)
                        if size > MAX_PDF_BYTES:
                            raise OSError("PDF too large")
                        f.write(chunk)
                # A landing page's "PDF" link can be an author template, a user guide,
                # or a paper from the reference list, so check it is the right paper.
                with open(tmp, "rb") as f:
                    if title_score(title, f.read()) < MIN_TITLE_SCORE:
                        os.remove(tmp)
                        rejected.append(r.url)
                        return None
                os.replace(tmp, dest)
                return r.url
            ctype = r.headers.get("Content-Type", "")
            if depth < 1 and ("html" in ctype or first.lstrip()[:1] == b"<"):
                body = first + b"".join(it)
                html = body.decode(r.encoding or "utf-8", errors="replace")
                for u in pdf_links_in_html(html, r.url):
                    got = self.fetch_pdf(u, dest, depth + 1, tried, title, rejected)
                    if got:
                        return got
        except (OSError, requests.RequestException):
            pass
        finally:
            r.close()
        return None

    def process(self, p: Paper) -> Paper:
        fname = f"{p.rank:04d}_{slugify(p.title)}.pdf"
        dest = os.path.join(self.outdir, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            p.status, p.pdf_path = "downloaded", fname
            return p
        tried: list = []
        rejected: list = []
        try:
            for cand in self.resolver.candidates(p):
                if cand.startswith(LANDING):
                    url, depth = cand[len(LANDING) :], 0
                else:
                    url, depth = cand, 1
                got = self.fetch_pdf(url, dest, depth, tried, p.title, rejected)
                if got:
                    p.status, p.pdf_path, p.pdf_url = "downloaded", fname, got
                    break
            else:
                p.status = "not_found"
                p.note = "no open-access PDF found; open view_url manually"
                if rejected:
                    p.note += f" ({len(rejected)} PDF(s) rejected: title not on first pages)"
        except Exception as e:  # never let one row kill the run
            p.status, p.note = "error", f"{type(e).__name__}: {e}"
        p.tried = tried
        return p


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def load_manifest(path: str) -> dict[int, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {int(r["rank"]): r for r in json.load(f)}


def merge_result(old: dict | None, p: Paper) -> dict:
    """Row for p, keeping what earlier runs (incl. browser-download.py) learned when this run found nothing new."""
    row = asdict(p)
    if old and p.status != "downloaded":
        row["tried"] = old.get("tried", []) + [u for u in p.tried if u not in old.get("tried", [])]
        if old.get("note", "").startswith("browser:"):
            row["note"] = old["note"]
    return row


def save_manifest(outdir: str, prev: dict[int, dict], papers: list[Paper]):
    """Write prev with this run's papers merged in; rows this run didn't touch are kept as they were."""
    rows = dict(prev)
    for p in papers:
        rows[p.rank] = merge_result(prev.get(p.rank), p)
    paperlib.save_manifest(outdir, list(rows.values()))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=DEFAULT_URL, help="Paper Digest page URL")
    ap.add_argument(
        "--html", help="use a saved copy of the page instead of downloading it"
    )
    ap.add_argument(
        "--out", default="papers", help="output directory (default: ./papers)"
    )
    ap.add_argument(
        "--email",
        default=os.environ.get("UNPAYWALL_EMAIL"),
        help="your email for the Unpaywall API (or set UNPAYWALL_EMAIL). Strongly recommended.",
    )
    ap.add_argument("--since", help="only papers dated on/after YYYY-MM-DD")
    ap.add_argument(
        "--source", help="regex filter on the SOURCE column, e.g. 'arxiv|nips'"
    )
    ap.add_argument(
        "--limit", type=int, help="process at most N papers (after filters)"
    )
    ap.add_argument(
        "--workers", type=int, default=3, help="parallel downloads (default 3)"
    )
    ap.add_argument(
        "--no-arxiv-search",
        action="store_true",
        help="skip arXiv title-search fallback",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="only parse the list and write the manifest",
    )
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    http = Http()

    if a.html:
        with open(a.html, encoding="utf-8", errors="replace") as f:
            html = f.read()
    else:
        print(f"Fetching list: {a.url}")
        r = http.get(a.url)
        r.raise_for_status()
        html = r.text

    papers = parse_listing(html)
    if not papers:
        sys.exit(
            "No papers found on the page - has the layout changed? Try --html with a saved copy."
        )
    print(f"Found {len(papers)} papers on the page.")

    if a.since:
        papers = [p for p in papers if p.date >= a.since]
    if a.source:
        rx = re.compile(a.source, re.IGNORECASE)
        papers = [p for p in papers if rx.search(p.source) or rx.search(p.view_url)]
    if a.limit:
        papers = papers[: a.limit]

    # Resume: rows already downloaded (or added by hand) are kept exactly as they are.
    prev = load_manifest(os.path.join(a.out, "manifest.json"))
    done = []
    todo = []
    for p in papers:
        old = prev.get(p.rank)
        if (
            old
            and old.get("status") in ("downloaded", "recovered")
            and old.get("title") == p.title
            and os.path.exists(os.path.join(a.out, old.get("pdf_path", "")))
        ):
            done.append(p)
        else:
            todo.append(p)

    if a.dry_run:
        new = [p for p in todo if p.rank not in prev]
        for p in new:
            p.status = "listed"
        save_manifest(a.out, prev, new)
        print(f"Dry run: added {len(new)} new rows to {a.out}/manifest.json")
        return

    if not a.email:
        print(
            "Tip: pass --email you@example.com to enable Unpaywall (finds many more open-access PDFs)."
        )
    print(
        f"{len(done)} already downloaded, {len(todo)} to process with {a.workers} workers.\n"
    )

    dl = Downloader(http, Resolver(http, a.email, not a.no_arxiv_search), a.out)
    results: list[Paper] = []
    lock = threading.Lock()
    n = 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(dl.process, p): p for p in todo}
        for fut in as_completed(futs):
            p = fut.result()
            with lock:
                n += 1
                results.append(p)
                mark = {"downloaded": "OK ", "not_found": "-- ", "error": "ERR"}.get(
                    p.status, "?? "
                )
                print(f"[{n:4d}/{len(todo)}] {mark} #{p.rank:<4d} {p.title[:70]}")
                if n % 20 == 0:
                    save_manifest(a.out, prev, results)

    save_manifest(a.out, prev, results)
    counts: dict[str, int] = {"downloaded": len(done)} if done else {}
    for p in results:
        counts[p.status] = counts.get(p.status, 0) + 1
    print("\nSummary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"PDFs and manifest.csv are in: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
