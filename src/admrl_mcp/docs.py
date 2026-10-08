"""Search and read the public Admiral docs (https://docs.admrl.co) from the site's search index.

The docs site publishes ``/search-index.json`` (docusaurus-search-local): a list of
five parts, each ``{"documents": [...], "index": <lunr, ignored>}``:

0. pages     ``{i, t: title, u: path, b: breadcrumbs}``
1. headings  ``{i, t, u, h: "#anchor", p: page id}``
2. page descriptions ``{i, t: text, s: page title, u, p}``
3. keywords  (identical boilerplate on every page; ignored)
4. sections  ``{i, t: text, s: section title, u, h, p}``

SECURITY: this module never uses the Admiral client. Docs requests go through their
own ``httpx.Client`` with no auth and no Admiral headers, and only to the host of the
configured docs base. Nothing here may read Admiral credentials.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

DEFAULT_DOCS_BASE = "https://docs.admrl.co"
INDEX_PATH = "/search-index.json"
CACHE_TTL_SECONDS = 3600.0
MAX_PAGE_CHARS = 12000
MAX_SECTIONS_PER_PAGE = 2
SNIPPET_CHARS = 240
MAX_INDEX_BYTES = 20 * 1024 * 1024


class DocsError(RuntimeError):
    """The docs index could not be fetched or understood, or a page was not found."""


# ------------------------------------------------------------------ config ---

_docs_base: str | None = None  # set by browser.configure; stdio falls back to env
_transport_factory: Callable[[], httpx.BaseTransport | None] | None = None
_now: Callable[[], float] = time.monotonic  # tests replace this
_lock = threading.Lock()


def configure(docs_base: str | None = None, transport_factory: Callable[[], httpx.BaseTransport | None] | None = None) -> None:
    """Set the docs base and the transport selector. Drops any cached index."""
    global _docs_base, _transport_factory
    new_base = (docs_base or "").rstrip("/") or None
    if new_base != _docs_base:  # the browser calls configure() before every tool call; keep the cache
        clear_cache()
    _docs_base = new_base
    _transport_factory = transport_factory


def docs_base() -> str:
    base = _docs_base or os.environ.get("ADMRL_DOCS_BASE", "").strip() or DEFAULT_DOCS_BASE
    base = base.rstrip("/")
    parts = urlsplit(base)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise DocsError(f"Invalid docs base URL: {base!r}")
    return base


def _transport() -> httpx.BaseTransport | None:
    if _transport_factory is not None:
        return _transport_factory()
    if sys.platform == "emscripten":  # Pyodide: sync XHR, same as the Admiral client
        from .browser import XhrTransport

        return XhrTransport()
    return None


def _http_client() -> httpx.Client:
    # Deliberately bare: no auth, no Admiral headers, no shared client with the API.
    return httpx.Client(
        headers={"Accept": "application/json"},
        timeout=httpx.Timeout(30.0, connect=10.0),
        transport=_transport(),
        follow_redirects=False,
    )


def _allowed_url(url: str, base: str) -> str:
    want, got = urlsplit(base), urlsplit(url)
    if got.scheme != want.scheme or (got.hostname or "").lower() != (want.hostname or "").lower() or got.port != want.port:
        raise DocsError(f"Refusing to fetch {url!r}: only {want.scheme}://{want.netloc} is allowed.")
    return url


# ------------------------------------------------------------------- index ---

_STOP = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or that the this to was what when where which with you your".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(tok) > len(suffix) + 3 and tok.endswith(suffix):
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP]


@dataclass
class Section:
    page_id: int
    title: str  # section heading ("" for the page description)
    anchor: str  # "#frag" or ""
    text: str
    # per-field term frequencies + lengths (title = page title, heading = section title)
    tf_title: dict[str, int] = field(default_factory=dict)
    tf_head: dict[str, int] = field(default_factory=dict)
    tf_text: dict[str, int] = field(default_factory=dict)
    length: float = 0.0


@dataclass
class Page:
    id: int
    title: str
    path: str
    breadcrumbs: list[str]
    description: str = ""
    sections: list[Section] = field(default_factory=list)


@dataclass
class DocsIndex:
    base: str
    pages: dict[int, Page]
    by_path: dict[str, Page]
    sections: list[Section]
    df: dict[str, int]
    avg_len: float


def _counts(tokens: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in tokens:
        out[t] = out.get(t, 0) + 1
    return out


def _norm_path(path: str) -> str:
    path = urlsplit(path).path or "/"
    path = re.sub(r"(/index)?(\.html?)?$", "", path)
    return "/" + path.strip("/")


def build_index(raw: Any, base: str) -> DocsIndex:
    """Parse the docusaurus-search-local JSON into a searchable index."""
    if not isinstance(raw, list) or len(raw) < 5 or not all(isinstance(p, dict) for p in raw):
        raise DocsError("Docs search index has an unexpected shape.")
    docs = lambda n: [d for d in (raw[n].get("documents") or []) if isinstance(d, dict)]  # noqa: E731

    pages: dict[int, Page] = {}
    by_path: dict[str, Page] = {}
    for d in docs(0):
        page = Page(int(d["i"]), str(d.get("t") or ""), _norm_path(str(d.get("u") or "")), [str(b) for b in d.get("b") or []])
        pages[page.id] = page
        by_path[page.path] = page

    # Part 1 (headings) only supplies anchors/titles that part 4 also carries; part 4 has the text.
    for d in docs(2):
        page = pages.get(d.get("p"))
        if page is not None:
            page.description = str(d.get("t") or "")

    # Part 3 (keywords) is boilerplate repeated on every page: ignored on purpose.
    sections: list[Section] = []
    for page in pages.values():
        if page.description:
            page.sections.append(Section(page.id, "", "", page.description))
    for d in docs(4):
        page = pages.get(d.get("p"))
        if page is None:
            continue
        page.sections.append(Section(page.id, str(d.get("s") or ""), str(d.get("h") or ""), str(d.get("t") or "")))

    df: dict[str, int] = {}
    total = 0.0
    for page in pages.values():
        title_tf = _counts(tokenize(page.title))
        for sec in page.sections:
            sec.tf_title = title_tf
            sec.tf_head = _counts(tokenize(sec.title))
            sec.tf_text = _counts(tokenize(sec.text))
            sec.length = float(sum(sec.tf_text.values())) or 1.0
            total += sec.length
            for term in set(title_tf) | set(sec.tf_head) | set(sec.tf_text):
                df[term] = df.get(term, 0) + 1
            sections.append(sec)
    if not sections:
        raise DocsError("Docs search index contained no pages.")
    return DocsIndex(base, pages, by_path, sections, df, total / len(sections))


_cache: tuple[float, DocsIndex] | None = None


def clear_cache() -> None:
    global _cache
    _cache = None


def get_index() -> DocsIndex:
    """The parsed index, fetched at most once per TTL. A stale copy is served if a refresh fails."""
    global _cache
    base = docs_base()
    with _lock:
        if _cache is not None and _cache[1].base == base and _now() - _cache[0] < CACHE_TTL_SECONDS:
            return _cache[1]
        stale = _cache[1] if _cache is not None and _cache[1].base == base else None
        try:
            url = _allowed_url(base + INDEX_PATH, base)
            with _http_client() as client:
                response = client.get(url)
            if response.status_code != 200:
                raise DocsError(f"Docs index request failed: HTTP {response.status_code}")
            if len(response.content) > MAX_INDEX_BYTES:
                raise DocsError("Docs index is unexpectedly large.")
            index = build_index(response.json(), base)
        except DocsError as exc:
            if stale is not None:
                return stale
            raise DocsError(f"Could not load the Admiral docs ({base}): {exc}") from exc
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            if stale is not None:
                return stale
            raise DocsError(f"Could not load the Admiral docs ({base}): {type(exc).__name__}: {exc}") from exc
        _cache = (_now(), index)
        return index


# ----------------------------------------------------------------- scoring ---

K1, B = 1.2, 0.75
W_TITLE, W_HEAD, W_TEXT = 3.0, 2.5, 1.0


def _score(index: DocsIndex, terms: list[str], sec: Section) -> float:
    n = len(index.sections)
    norm = 1 - B + B * sec.length / index.avg_len
    score = 0.0
    for term in terms:
        df = index.df.get(term, 0)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        tf = W_TITLE * sec.tf_title.get(term, 0) + W_HEAD * sec.tf_head.get(term, 0) + W_TEXT * sec.tf_text.get(term, 0)
        if tf:
            score += idf * tf * (K1 + 1) / (tf + K1 * norm)
    return score


def make_snippet(text: str, terms: list[str], width: int = SNIPPET_CHARS) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= width:
        return text
    # Densest window: centre on the match whose neighbourhood has the most distinct query terms.
    spans = [(m.start(), _stem(m.group(0))) for m in _TOKEN_RE.finditer(text.lower())]
    hits = [pos for pos, tok in spans if tok in terms]
    start = 0
    if hits:
        best = max(hits, key=lambda p: len({tok for q, tok in spans if tok in terms and p - width // 2 <= q <= p + width // 2}))
        start = max(0, best - width // 3)
    end = min(len(text), start + width)
    start = max(0, end - width)
    if start > 0:  # snap to a word boundary
        sp = text.find(" ", start)
        start = sp + 1 if 0 <= sp < start + 20 else start
    snippet = text[start:end].strip()
    return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")


def search(query: str, limit: int = 5, index: DocsIndex | None = None) -> list[dict[str, Any]]:
    terms = tokenize(query)
    if not terms:
        return []
    index = index or get_index()
    limit = max(1, min(int(limit), 20))
    scored = sorted(
        ((s, sec) for sec in index.sections if (s := _score(index, terms, sec)) > 0),
        key=lambda x: -x[0],
    )
    out: list[dict[str, Any]] = []
    per_page: dict[int, int] = {}
    for score, sec in scored:
        if per_page.get(sec.page_id, 0) >= MAX_SECTIONS_PER_PAGE:
            continue
        per_page[sec.page_id] = per_page.get(sec.page_id, 0) + 1
        page = index.pages[sec.page_id]
        out.append(
            {
                "title": page.title,
                "section": sec.title or page.title,
                "url": index.base + page.path + sec.anchor,
                "breadcrumbs": page.breadcrumbs,
                "snippet": make_snippet(sec.text, terms),
                "score": round(score, 2),
            }
        )
        if len(out) >= limit:
            break
    return out


# -------------------------------------------------------------------- read ---


def find_page(index: DocsIndex, url_or_path: str) -> Page:
    value = (url_or_path or "").strip()
    if not value:
        raise DocsError("Pass a docs page URL or path, e.g. /advanced/ble.")
    if re.match(r"^[a-z][a-z0-9+.-]*://", value, re.I):
        _allowed_url(value, index.base)
    elif not value.startswith("/"):
        value = "/" + value
    page = index.by_path.get(_norm_path(value))
    if page is None:
        raise DocsError(f"No docs page at {url_or_path!r}. Use search_docs to find the right page.")
    return page


def read_page(url_or_path: str, max_chars: int = MAX_PAGE_CHARS, index: DocsIndex | None = None) -> dict[str, Any]:
    index = index or get_index()
    page = find_page(index, url_or_path)
    sections: list[dict[str, str]] = []
    used = 0
    truncated = False
    for sec in page.sections:
        if sec.text == page.description:
            continue  # returned separately as "description"
        room = max_chars - used
        if room <= 0:
            truncated = True
            break
        text = sec.text
        if len(text) > room:
            text, truncated = text[:room].rstrip() + "…", True
        used += len(text)
        sections.append({"section": sec.title, "url": index.base + page.path + sec.anchor, "text": text})
        if truncated:
            break
    return {
        "title": page.title,
        "url": index.base + page.path,
        "breadcrumbs": page.breadcrumbs,
        "description": page.description,
        "sections": sections,
        "truncated": truncated,
    }
