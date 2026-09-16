"""
DuckDuckGo HTML web search for Streamlit and WebRTC chat apps.

No API key required. Parses html.duckduckgo.com results into title/url/snippet.
"""
from __future__ import annotations

import html
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

logger = logging.getLogger(__name__)

_SEARCH_PREFIX_RE = re.compile(r"^\s*/search\s+(.+)$", re.IGNORECASE | re.DOTALL)
_SEARCH_INTENT_RE = re.compile(
    r"(?i)\b("
    r"search\s+(the\s+)?(web|internet|online)\s+for"
    r"|search\s+for"
    r"|look\s+up"
    r"|google\s+"
    r"|bing\s+"
    r"|find\s+(online|on\s+the\s+web)"
    r"|what'?s\s+the\s+latest\s+on"
    r"|latest\s+news\s+on"
    r"|current\s+(status|price|version)\s+of"
    r")\b\s*(?P<q>.+)$"
)
_STRIP_FILLER_RE = re.compile(
    r"(?i)^(please\s+)?(can\s+you\s+|could\s+you\s+)?(help\s+me\s+)?"
)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    rank: int = 0

    def prompt_line(self) -> str:
        parts = [f"{self.rank}. {self.title or '(no title)'}"]
        if self.url:
            parts.append(f"   URL: {self.url}")
        if self.snippet:
            parts.append(f"   {self.snippet}")
        return "\n".join(parts)


def extract_search_query(text: str) -> Optional[str]:
    """Return a search query if the user message requests web search, else None."""
    if not text or not str(text).strip():
        return None
    raw = str(text).strip()

    m = _SEARCH_PREFIX_RE.match(raw)
    if m:
        q = m.group(1).strip()
        return q or None

    m = _SEARCH_INTENT_RE.search(raw)
    if m:
        q = (m.group("q") or "").strip()
        q = _STRIP_FILLER_RE.sub("", q).strip()
        q = q.strip(" \t\r\n\"'")
        q = re.sub(r"[?.!]+$", "", q).strip()
        if len(q) >= 2:
            return q
    return None


def _unwrap_ddg_redirect(href: str) -> str:
    if not href:
        return ""
    href = html.unescape(href.strip())
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urlparse(href)
        if "duckduckgo.com" in (parsed.netloc or "") and (
            parsed.path.startswith("/l/") or "uddg=" in (parsed.query or "")
        ):
            qs = parse_qs(parsed.query)
            uddg = qs.get("uddg") or qs.get("u")
            if uddg and uddg[0]:
                return unquote(uddg[0])
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return href
    except Exception:
        pass
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return ""


def parse_ddg_html(html_text: str, max_results: int = 5) -> List[SearchResult]:
    """Parse DuckDuckGo HTML search results."""
    if not html_text:
        return []

    results: List[SearchResult] = []
    block_re = re.compile(
        r'(?is)<div[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*'
        r'(?=<div[^>]*class="[^"]*result|</div>\s*</div>\s*<div id=|$)',
    )
    blocks = block_re.findall(html_text)
    if not blocks:
        blocks = re.split(r'(?i)(?=<a[^>]*class="[^"]*result__a)', html_text)

    link_re = re.compile(
        r'(?is)<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    )
    snip_re = re.compile(
        r'(?is)<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>'
        r'|<div[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>'
    )

    seen = set()
    for block in blocks:
        lm = link_re.search(block)
        if not lm:
            continue
        url = _unwrap_ddg_redirect(lm.group(1))
        if not url or url in seen:
            continue
        if "duckduckgo.com" in urlparse(url).netloc:
            continue
        title = re.sub(r"(?is)<[^>]+>", " ", lm.group(2))
        title = html.unescape(re.sub(r"\s+", " ", title)).strip()
        snippet = ""
        sm = snip_re.search(block)
        if sm:
            raw_snip = sm.group(1) if sm.group(1) is not None else sm.group(2) or ""
            snippet = re.sub(r"(?is)<[^>]+>", " ", raw_snip)
            snippet = html.unescape(re.sub(r"\s+", " ", snippet)).strip()
        seen.add(url)
        results.append(
            SearchResult(
                title=title or url,
                url=url,
                snippet=snippet,
                rank=len(results) + 1,
            )
        )
        if len(results) >= max_results:
            break

    if not results:
        for lm in link_re.finditer(html_text):
            url = _unwrap_ddg_redirect(lm.group(1))
            if not url or url in seen:
                continue
            if "duckduckgo.com" in urlparse(url).netloc:
                continue
            title = re.sub(r"(?is)<[^>]+>", " ", lm.group(2))
            title = html.unescape(re.sub(r"\s+", " ", title)).strip()
            seen.add(url)
            results.append(
                SearchResult(
                    title=title or url,
                    url=url,
                    snippet="",
                    rank=len(results) + 1,
                )
            )
            if len(results) >= max_results:
                break
    return results


class WebSearchTool:
    """DuckDuckGo HTML search client."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        max_results: Optional[int] = None,
        timeout: Optional[int] = None,
        user_agent: Optional[str] = None,
        app_name: str = "app",
    ):
        self.enabled = (
            enabled if enabled is not None else _env_bool("WEB_SEARCH_ENABLED", True)
        )
        self.app_name = app_name
        self.max_results = (
            max_results
            if max_results is not None
            else _env_int("WEB_SEARCH_MAX_RESULTS", 5)
        )
        self.timeout = (
            timeout if timeout is not None else _env_int("WEB_SEARCH_TIMEOUT", 15)
        )
        self.user_agent = user_agent or os.getenv(
            "WEB_SEARCH_USER_AGENT",
            "Mozilla/5.0 (compatible; ModelServing-WebSearch/1.0; "
            "+https://github.com/7sg-ai/model-serving)",
        )
        self.endpoint = os.getenv(
            "WEB_SEARCH_DDG_URL", "https://html.duckduckgo.com/html/"
        ).strip()
        self.searches = 0
        self.failures = 0
        self.last_error = ""
        self._lock = threading.Lock()

    def search(
        self, query: str, max_results: Optional[int] = None
    ) -> List[SearchResult]:
        if not self.enabled:
            self.last_error = "web search disabled"
            return []
        q = (query or "").strip()
        if not q:
            self.last_error = "empty query"
            return []
        limit = max_results if max_results is not None else self.max_results
        limit = max(1, min(int(limit), 10))

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            resp = requests.post(
                self.endpoint,
                data={"q": q, "b": "", "kl": "us-en"},
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = parse_ddg_html(resp.text, max_results=limit)
            with self._lock:
                self.searches += 1
                if not results:
                    self.failures += 1
                    self.last_error = "no results parsed (blocked or empty SERP)"
                else:
                    self.last_error = ""
            return results
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.failures += 1
                self.last_error = str(exc)
            logger.warning(
                "web search failed app=%s query=%r err=%s", self.app_name, q, exc
            )
            return []

    def format_context(
        self, query: str, results: List[SearchResult], error: str = ""
    ) -> str:
        lines = [
            "The user asked for web search. Results from DuckDuckGo HTML:",
            f"Query: {query}",
        ]
        if error and not results:
            lines.append(f"Search error: {error}")
            lines.append(
                "Answer from model knowledge and note that live search failed."
            )
            return "\n".join(lines)
        if not results:
            lines.append("No results returned.")
            return "\n".join(lines)
        lines.append("Results:")
        for r in results:
            lines.append(r.prompt_line())
        lines.append(
            "Use these results when answering. Cite titles/URLs when relevant."
        )
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "app_name": self.app_name,
            "endpoint": self.endpoint,
            "max_results": self.max_results,
            "timeout": self.timeout,
            "searches": self.searches,
            "failures": self.failures,
            "last_error": self.last_error or None,
        }


_SEARCH_TOOLS: Dict[str, WebSearchTool] = {}
_SEARCH_LOCK = threading.Lock()


def get_web_search_tool(app_name: str = "app") -> WebSearchTool:
    with _SEARCH_LOCK:
        tool = _SEARCH_TOOLS.get(app_name)
        if tool is None:
            tool = WebSearchTool(app_name=app_name)
            _SEARCH_TOOLS[app_name] = tool
        return tool
