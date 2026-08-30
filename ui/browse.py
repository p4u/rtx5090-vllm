"""Web browsing tools for the chat tab, backed by obscura.

obscura (https://github.com/h4ckf0r0day/obscura) is a headless-browser
engine, not a search engine: `web_fetch` is a straight page fetch rendered to
markdown; `web_search` drives DuckDuckGo's HTML endpoint through it and
parses the results. Each call is one `docker run --rm h4ckf0r0day/obscura`
(the UI container already drives the host daemon; obscura lands on the
default bridge with outbound internet and — deliberately — obscura's default
private-network/SSRF protection left ON, so it cannot reach loopback- or
LAN-bound services like vLLM or this UI).

Every tool result carries a `debug` list: obscura's own stderr tracing
(RUST_LOG=obscura=info) plus one structured summary line — this is what the
chat's browsing-activity panel shows.
"""

import html as html_mod
import json
import re
import time
from urllib.parse import parse_qs, quote_plus, unquote, urlsplit

import vllm_mgr

OBSCURA_IMAGE = "h4ckf0r0day/obscura"
FETCH_TIMEOUT_MS = 20000
OUTER_TIMEOUT_S = 35
MAX_URL_LEN = 2000
SEARCH_JSON_CAP = 8000
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns a JSON list of results "
                           "(title, url, snippet). Use web_fetch to read a result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 8,
                                    "description": "how many results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a web page and return its content as markdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) URL to fetch"},
                    "max_chars": {"type": "integer", "minimum": 1000, "maximum": 30000,
                                  "description": "truncate content to this many characters (default 10000)"},
                },
                "required": ["url"],
            },
        },
    },
]

_avail_cache: tuple[float, bool] | None = None


async def available() -> bool:
    """Is the obscura image present? Cached 60s (checked per chat request)."""
    global _avail_cache
    now = time.time()
    if _avail_cache and now - _avail_cache[0] < 60:
        return _avail_cache[1]
    rc, _, _ = await vllm_mgr._run(
        ["docker", "image", "inspect", OBSCURA_IMAGE], timeout=10)
    _avail_cache = (now, rc == 0)
    return rc == 0


async def _obscura(url: str, dump: str) -> dict:
    """One obscura fetch. Returns {ok, stdout, debug, duration}."""
    scheme = urlsplit(url).scheme
    if scheme not in ("http", "https"):
        return {"ok": False, "stdout": "", "duration": 0.0,
                "debug": [f"rejected url (scheme {scheme!r}): only http/https allowed"]}
    if len(url) > MAX_URL_LEN:
        return {"ok": False, "stdout": "", "duration": 0.0,
                "debug": ["rejected url: longer than 2000 chars"]}
    t0 = time.time()
    rc, out, err = await vllm_mgr._run(
        ["docker", "run", "--rm", "-e", "RUST_LOG=obscura=info", OBSCURA_IMAGE,
         "fetch", url, "--dump", dump, "--timeout", str(FETCH_TIMEOUT_MS)],
        timeout=OUTER_TIMEOUT_S)
    dur = time.time() - t0
    debug = [ANSI_RE.sub("", l) for l in err.strip().splitlines()[-15:]]
    summary = f"fetch url={url} exit={rc} dur={dur:.1f}s stdout_bytes={len(out)}"
    debug.append(summary)
    print(f"[browse] {summary}", flush=True)
    return {"ok": rc == 0 and bool(out.strip()), "stdout": out,
            "duration": round(dur, 1), "debug": debug}


# ─── web_search ─────────────────────────────────────────────────────────────

_ANCHOR_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.S)
_SNIPPET_RE = re.compile(
    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', re.S)
_ANY_UDDG_RE = re.compile(r'<a[^>]*href="([^"]*uddg=[^"]+)"[^>]*>(.*?)</a>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(fragment: str) -> str:
    return html_mod.unescape(_TAG_RE.sub("", fragment)).strip()


def _unwrap_ddg(href: str) -> str | None:
    """//duckduckgo.com/l/?uddg=<pct-url>&rut=… → the real URL. None for ads."""
    href = html_mod.unescape(href)
    q = parse_qs(urlsplit(href).query)
    if "ad_domain" in q or "y.js" in href:
        return None
    if "uddg" in q:
        return unquote(q["uddg"][0])
    if href.startswith("http"):
        return href
    return None


def _parse_ddg_html(page: str, max_results: int) -> list[dict]:
    """Pure parser (unit-testable). Primary: result__a/result__snippet class
    pairs; drift fallback: any anchor carrying a uddg= redirect."""
    results = []
    anchors = _ANCHOR_RE.findall(page)
    snippets = [_clean_text(s) for s in _SNIPPET_RE.findall(page)]
    for i, (href, title_html) in enumerate(anchors):
        url = _unwrap_ddg(href)
        if not url:
            continue
        results.append({"title": _clean_text(title_html), "url": url,
                        "snippet": snippets[i] if i < len(snippets) else ""})
        if len(results) >= max_results:
            return results
    if results:
        return results
    # markup drifted — the /l/?uddg= redirect wrapper is structural
    seen = set()
    for href, text in _ANY_UDDG_RE.findall(page):
        url = _unwrap_ddg(href)
        if not url or url in seen:
            continue
        seen.add(url)
        title = _clean_text(text)
        if title:
            results.append({"title": title, "url": url, "snippet": ""})
        if len(results) >= max_results:
            break
    return results


async def web_search(args: dict) -> dict:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "empty query", "debug": []}
    try:
        max_results = max(1, min(8, int(args.get("max_results") or 5)))
    except (TypeError, ValueError):
        max_results = 5
    debug: list[str] = []
    for engine in (f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
                   f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"):
        r = await _obscura(engine, "html")
        debug += r["debug"]
        results = _parse_ddg_html(r["stdout"], max_results) if r["ok"] else []
        if results:
            payload = {"results": results}
            while len(json.dumps(payload)) > SEARCH_JSON_CAP and payload["results"]:
                payload["results"].pop()
            payload["debug"] = debug
            payload["duration"] = r["duration"]
            return payload
        debug.append(f"no results parsed from {urlsplit(engine).netloc}, "
                     f"{'trying fallback engine' if 'html.' in engine else 'giving up'}")
    return {"results": [], "error": "search returned no parseable results",
            "debug": debug, "duration": 0.0}


# ─── web_fetch ──────────────────────────────────────────────────────────────

async def web_fetch(args: dict) -> dict:
    url = str(args.get("url", "")).strip()
    if not url:
        return {"error": "empty url", "debug": []}
    try:
        max_chars = max(1000, min(30000, int(args.get("max_chars") or 10000)))
    except (TypeError, ValueError):
        max_chars = 10000
    r = await _obscura(url, "markdown")
    if not r["ok"]:
        return {"url": url, "error": "fetch failed (timeout, block, or private address)",
                "debug": r["debug"], "duration": r["duration"]}
    content = r["stdout"].strip()
    title_m = re.search(r"^#\s+(.+)$", content, re.M)
    title = title_m.group(1).strip() if title_m else None
    if title:  # headings are markdown — collapse [text](url) to text
        title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)[:200]
    truncated = len(content) > max_chars
    return {"url": url,
            "title": title,
            "content": content[:max_chars],
            "truncated": truncated,
            "debug": r["debug"], "duration": r["duration"]}


async def run_tool(name: str, args: dict) -> dict:
    if name == "web_search":
        return await web_search(args)
    if name == "web_fetch":
        return await web_fetch(args)
    return {"error": f"unknown tool {name!r}", "debug": []}
