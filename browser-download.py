"""Retry `not_found` manifest entries by opening each view_url in a real browser.

Usage: xvfb-run -a uv run python browser-download.py [--limit N] [--workers N] [--retry] [--ranks 1,2] [--headless]
"""

import argparse
import asyncio
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.async_api import Error as PWError
from playwright.async_api import async_playwright

from paperlib import MIN_TITLE_SCORE, save_manifest, title_score

PAPERS = Path("papers")
MANIFEST = PAPERS / "manifest.json"
NOTE_PREFIX = "browser:"
MIN_PDF_BYTES = 10_000

# JS run on a landing page to collect likely PDF links, best guesses first.
COLLECT_JS = r"""
() => {
  const out = [];
  const add = u => { if (u && !out.includes(u)) out.push(u); };
  document.querySelectorAll('meta[name="citation_pdf_url"], meta[name="eprints.document_url"]')
    .forEach(m => add(m.content));
  const anchors = [...document.querySelectorAll('a[href]')];
  for (const a of anchors) {
    const h = a.href, t = (a.innerText || a.title || a.getAttribute('aria-label') || '').trim();
    if (/^(javascript|mailto):/.test(h)) continue;
    if (/\.pdf(\?|#|$)|\/pdf(\/|\?|$)|pdfdirect|epdf|\/download\/|bitstream|delivery\.cfm|article\/download|viewcontent\.cgi/i.test(h)
        || /(download|view|full.?text|get)\s*(the\s*)?(article\s*)?pdf|^pdf$|download (this )?(paper|article)/i.test(t))
      add(h);
  }
  return out;
}
"""

# Pages that are only a hop to the real full text (PubMed -> PMC / publisher).
HOP_JS = r"""
() => [...document.querySelectorAll(
  '.full-text-links-list a[href], a[href*="pmc.ncbi.nlm.nih.gov/articles/PMC"], a[href*="europepmc.org"]')]
  .map(a => a.href)
"""


CHALLENGE_TITLES = re.compile(
    r"just a moment|client challenge|attention required|security check|are you a robot",
    re.IGNORECASE,
)


class BotCheck(Exception):
    pass


def pdf_name(entry):
    title = (
        unicodedata.normalize("NFKD", entry["title"]).encode("ascii", "ignore").decode()
    )
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:80].rstrip("_")
    return f"{entry['rank']:04d}_{slug}.pdf"


def publisher_guesses(url):
    """URL rewrites for publishers whose PDF link is not in the landing HTML."""
    p = urlparse(url)
    host, path = p.netloc, p.path
    guesses = []
    if "/doi/" in path and any(
        h in host
        for h in ("wiley", "tandfonline", "sagepub", "acm.org", "informs", "emerald")
    ):
        doi = (
            path.split("/doi/", 1)[1].split("/", 1)[-1]
            if re.match(r"/doi/(abs|full|epdf)/", path)
            else path.split("/doi/", 1)[1]
        )
        base = f"{p.scheme}://{host}/doi/"
        guesses += [base + "pdfdirect/" + doi, base + "pdf/" + doi]
    if "mdpi.com" in host and not path.endswith("/pdf"):
        guesses.append(url.rstrip("/") + "/pdf")
    if "arxiv.org" in host and "/abs/" in path:
        guesses.append(url.replace("/abs/", "/pdf/"))
    if "link.springer.com" in host and "/article/" in path:
        guesses.append(url.replace("/article/", "/content/pdf/") + ".pdf")
    if "openreview.net" in host and "forum" in path:
        guesses.append(url.replace("/forum", "/pdf"))
    if "sciencedirect.com" in host and "/pii/" in path:
        guesses.append(
            url.split("?")[0].rstrip("/") + "/pdfft?isDTMRedir=true&download=true"
        )
    return guesses


def is_pdf(body):
    return (
        body is not None
        and body[:1024].lstrip().startswith(b"%PDF")
        and len(body) >= MIN_PDF_BYTES
    )


async def fetch_pdf(context, page, url):
    """Try to get PDF bytes for url: first via the shared-cookie request API, then by navigating."""
    try:
        resp = await context.request.get(url, timeout=45_000, max_redirects=10)
        body = await resp.body()
        if is_pdf(body):
            return body, resp.url
    except PWError:
        pass
    # Navigation handles JS challenges (PMC proof-of-work, Cloudflare) and download-only links.
    # The PDF may arrive after a challenge page reloads itself, so watch every response/download
    # rather than trusting the response goto() returns.
    got = asyncio.get_running_loop().create_future()

    def offer(body, src):
        if is_pdf(body) and not got.done():
            got.set_result((body, src))

    async def on_response(resp):
        if "pdf" in (resp.headers.get("content-type") or ""):
            try:
                offer(await resp.body(), resp.url)
            except PWError:
                pass

    async def on_download(download):
        try:
            offer(Path(await download.path()).read_bytes(), download.url)
        except (PWError, OSError):
            pass

    page.on("response", on_response)
    page.on("download", on_download)
    try:
        try:
            await page.goto(url, wait_until="load", timeout=40_000)
        except PWError:
            pass  # "Download is starting" or a slow challenge; the listeners may still fire
        try:
            return await asyncio.wait_for(asyncio.shield(got), timeout=20)
        except TimeoutError:
            return None, None
    finally:
        page.remove_listener("response", on_response)
        page.remove_listener("download", on_download)


async def landing_candidates(page, url):
    resp = await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    try:
        await page.wait_for_load_state("networkidle", timeout=12_000)
    except PWError:
        pass
    # Bot-check interstitials sometimes clear themselves; otherwise flag the entry for a manual pass.
    for _ in range(10):
        if not CHALLENGE_TITLES.search(await page.title()):
            break
        await page.wait_for_timeout(2_000)
    else:
        raise BotCheck(page.url)
    final = page.url
    cands = await page.evaluate(COLLECT_JS)
    if resp and "pdf" in (resp.headers.get("content-type") or ""):
        cands.insert(0, resp.url)
    return final, cands + publisher_guesses(final)


async def process(context, entry):
    page = await context.new_page()
    tried, mismatched = [], []
    final = entry["view_url"]
    try:
        try:
            final, cands = await landing_candidates(page, entry["view_url"])
        except BotCheck as e:
            return None, None, tried, "blocked by bot check", str(e)
        except PWError as e:
            if "Download is starting" in str(e):
                cands = [entry["view_url"]]
            else:
                return (
                    None,
                    None,
                    tried,
                    f"landing failed: {str(e).splitlines()[0][:120]}",
                    final,
                )
        if not cands and "pubmed" in urlparse(final).netloc:
            for hop in (await page.evaluate(HOP_JS))[:3]:
                try:
                    hop_final, hop_cands = await landing_candidates(page, hop)
                except (PWError, BotCheck):
                    continue
                cands += [hop_final] + hop_cands
        for url in cands[:8]:
            url = urljoin(final, url)
            if url in tried:
                continue
            tried.append(url)
            body, src = await fetch_pdf(context, page, url)
            if body and title_score(entry["title"], body) >= MIN_TITLE_SCORE:
                return body, src, tried, "", final
            if body:
                mismatched.append(src)
        if mismatched:
            return (
                None,
                None,
                tried,
                f"PDFs found but none matched the title ({', '.join(mismatched[:2])})",
                final,
            )
        return None, None, tried, "no PDF reachable from landing page", final
    finally:
        await page.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--retry",
        action="store_true",
        help="also redo entries a previous browser run already tried",
    )
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--ranks",
        type=lambda s: {int(r) for r in s.split(",")},
        help="only these ranks, e.g. 18,20",
    )
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    todo = [
        e
        for e in manifest
        if e["status"] == "not_found"
        and (args.retry or not e["note"].startswith(NOTE_PREFIX))
        and (not args.ranks or e["rank"] in args.ranks)
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} entries to try")

    queue = asyncio.Queue()
    for e in todo:
        queue.put_nowait(e)
    lock = asyncio.Lock()
    counts = {"ok": 0, "fail": 0}

    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(
            str(Path(".browser-profile").resolve()),
            headless=args.headless,
            accept_downloads=True,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )

        async def worker():
            while not queue.empty():
                entry = queue.get_nowait()
                try:
                    body, src, tried, why, final = await asyncio.wait_for(
                        process(context, entry), timeout=300
                    )
                except Exception as e:  # one bad page must not stop the run
                    body, src, tried, why, final = (
                        None,
                        None,
                        [],
                        f"{type(e).__name__}: {str(e)[:120]}",
                        entry["view_url"],
                    )
                async with lock:
                    entry["tried"] += [u for u in tried if u not in entry["tried"]]
                    if body:
                        name = pdf_name(entry)
                        (PAPERS / name).write_bytes(body)
                        entry.update(
                            status="downloaded", pdf_path=name, pdf_url=src, note=""
                        )
                        counts["ok"] += 1
                        print(
                            f"[ok]   {entry['rank']:4d} {name} ({len(body) // 1024} KB)"
                        )
                    else:
                        entry["note"] = f"{NOTE_PREFIX} {why} ({final})"
                        counts["fail"] += 1
                        print(f"[miss] {entry['rank']:4d} {why} ({final})")
                    save_manifest(PAPERS, manifest)

        await asyncio.gather(*(worker() for _ in range(args.workers)))
        await context.close()

    print(f"done: {counts['ok']} downloaded, {counts['fail']} still not found")


if __name__ == "__main__":
    asyncio.run(main())
