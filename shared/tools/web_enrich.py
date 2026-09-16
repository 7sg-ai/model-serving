"""
Orchestrate web search + URL fetch for chat/voice apps.

1. Search when the user asks to search (or /search ...)
2. Fetch explicit URLs in the message (existing WebAccessTool)
3. Optionally fetch top search hit pages for richer context
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .web_access import FetchResult, WebAccessTool, extract_urls, get_web_access_tool
from .web_search import (
    SearchResult,
    WebSearchTool,
    extract_search_query,
    get_web_search_tool,
)


def _env_bool(name: str, default: bool) -> bool:
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
class WebEnrichment:
    """Combined search + fetch outcome for one user turn."""

    query: str = ""
    search_results: List[SearchResult] = field(default_factory=list)
    search_error: str = ""
    fetch_results: List[FetchResult] = field(default_factory=list)
    context: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.context)

    def ui_search_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "rank": r.rank,
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
            }
            for r in self.search_results
        ]

    def ui_fetch_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "url": wr.url,
                "ok": wr.ok,
                "status_code": wr.status_code,
                "title": wr.title,
                "log_path": wr.log_path,
                "content_sha256": wr.content_sha256,
                "raw_length": wr.raw_length,
                "error": wr.error,
            }
            for wr in self.fetch_results
        ]


class WebEnricher:
    def __init__(
        self,
        app_name: str = "app",
        access: Optional[WebAccessTool] = None,
        search: Optional[WebSearchTool] = None,
    ):
        self.app_name = app_name
        self.access = access or get_web_access_tool(app_name)
        self.search = search or get_web_search_tool(app_name)
        self.fetch_top_n = _env_int("WEB_SEARCH_FETCH_TOP", 2)
        self.fetch_search_pages = _env_bool("WEB_SEARCH_FETCH_PAGES", True)

    def enrich_user_text(self, text: str, session_id: str = "") -> WebEnrichment:
        out = WebEnrichment()
        if not text:
            return out

        blocks: List[str] = []

        query = extract_search_query(text)
        if query and self.search.enabled:
            out.query = query
            results = self.search.search(query)
            out.search_results = results
            out.search_error = self.search.last_error or ""
            blocks.append(
                self.search.format_context(query, results, error=out.search_error)
            )

            if (
                self.fetch_search_pages
                and results
                and self.access.enabled
                and not self.access.terminated
            ):
                top_urls = [
                    r.url for r in results[: max(0, self.fetch_top_n)] if r.url
                ]
                if top_urls:
                    page_results = self.access.fetch_urls(
                        top_urls, session_id=session_id
                    )
                    out.fetch_results.extend(page_results)
                    page_blocks = [
                        r.prompt_block(self.access.max_prompt_chars)
                        for r in page_results
                    ]
                    if page_blocks:
                        blocks.append(
                            "Fetched top search result page(s) for deeper context:\n\n"
                            + "\n\n".join(page_blocks)
                        )

        explicit = extract_urls(text)
        if explicit and self.access.enabled and not self.access.terminated:
            already = {r.url for r in out.fetch_results}
            new_urls = [u for u in explicit if u not in already]
            if new_urls:
                page_results = self.access.fetch_urls(
                    new_urls, session_id=session_id
                )
                out.fetch_results.extend(page_results)
                page_blocks = [
                    r.prompt_block(self.access.max_prompt_chars)
                    for r in page_results
                ]
                if page_blocks:
                    blocks.append(
                        "The user referenced web URL(s). Full page contents were "
                        "fetched and logged server-side. Use the following "
                        "extracted content:\n\n"
                        + "\n\n".join(page_blocks)
                    )

        out.context = "\n\n".join(blocks).strip()
        return out

    def enrich_messages(
        self,
        messages: List[Dict[str, str]],
        user_text: Optional[str] = None,
        session_id: str = "",
    ) -> Tuple[List[Dict[str, str]], WebEnrichment]:
        text = user_text
        if text is None:
            for msg in reversed(messages or []):
                if msg.get("role") == "user":
                    text = msg.get("content", "")
                    break
        enrichment = self.enrich_user_text(text or "", session_id=session_id)
        if not enrichment.context:
            return list(messages or []), enrichment
        enriched = list(messages or [])
        insert_at = len(enriched)
        for i in range(len(enriched) - 1, -1, -1):
            if enriched[i].get("role") == "user":
                insert_at = i
                break
        enriched.insert(
            insert_at,
            {"role": "system", "content": enrichment.context},
        )
        return enriched, enrichment

    def stats(self) -> Dict[str, Any]:
        return {
            "app_name": self.app_name,
            "search": self.search.stats(),
            "access": self.access.stats(),
            "fetch_top_n": self.fetch_top_n,
            "fetch_search_pages": self.fetch_search_pages,
        }


_ENRICHERS: Dict[str, WebEnricher] = {}


def get_web_enricher(app_name: str = "app") -> WebEnricher:
    if app_name not in _ENRICHERS:
        _ENRICHERS[app_name] = WebEnricher(app_name=app_name)
    return _ENRICHERS[app_name]
