from __future__ import annotations

import json
import re
from urllib.parse import quote
from urllib.parse import urlparse, urlunparse
from pathlib import Path
from typing import Any, Callable, Dict, List


ToolFunc = Callable[[Dict[str, Any]], Dict[str, Any]]
_LOW_SIGNAL_HOSTS = {
    "duckduckgo.com",
    "apps.apple.com",
    "play.google.com",
}
_LOW_SIGNAL_PATH_HINTS = {
    "/store/apps/",
    "/app/",
}


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _is_low_signal_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if any(value in host for value in _LOW_SIGNAL_HOSTS):
        return True
    return any(hint in path for hint in _LOW_SIGNAL_PATH_HINTS)


class LocalMCPToolServer:
    """Lightweight local MCP-style tool server used by planner-time logic."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolFunc] = {
            "list_ingested_files": self._list_ingested_files,
            "describe_file_manifest": self._describe_file_manifest,
            "recommend_processing_strategy": self._recommend_processing_strategy,
            "recommend_runtime_tools": self._recommend_runtime_tools,
            "web_research_playwright": self._web_research_playwright,
        }

    def available_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown MCP tool: {name}"}
        return tool(args)

    def _list_ingested_files(self, args: Dict[str, Any]) -> Dict[str, Any]:
        files = args.get("ingested_files", [])
        if not isinstance(files, list):
            files = []
        return {
            "count": len(files),
            "files": [str(v) for v in files],
        }

    def _describe_file_manifest(self, args: Dict[str, Any]) -> Dict[str, Any]:
        manifest_raw = str(args.get("file_manifest", "") or "")
        if not manifest_raw.strip():
            return {"summary": "No file manifest provided.", "types": {}}

        try:
            manifest = json.loads(manifest_raw)
        except json.JSONDecodeError:
            return {"summary": "Manifest was not valid JSON.", "types": {}}

        files = manifest.get("files", []) if isinstance(manifest, dict) else []
        if not isinstance(files, list):
            files = []

        type_counts: Dict[str, int] = {}
        for entry in files:
            if not isinstance(entry, dict):
                continue
            suffix = str(entry.get("suffix", "unknown") or "unknown")
            type_counts[suffix] = type_counts.get(suffix, 0) + 1

        return {
            "summary": f"Manifest contains {len(files)} file(s).",
            "types": type_counts,
        }

    def _recommend_processing_strategy(self, args: Dict[str, Any]) -> Dict[str, Any]:
        manifest_raw = str(args.get("file_manifest", "") or "")
        domain = str(args.get("domain", "") or "")

        strategies: List[str] = []
        if manifest_raw.strip():
            try:
                manifest = json.loads(manifest_raw)
            except json.JSONDecodeError:
                manifest = {}

            files = manifest.get("files", []) if isinstance(manifest, dict) else []
            suffixes = {
                str(entry.get("suffix", "")).lower()
                for entry in files
                if isinstance(entry, dict)
            }

            if ".csv" in suffixes or ".tsv" in suffixes:
                strategies.append("Use tabular profiling: row counts, column stats, numeric aggregates.")
            if ".json" in suffixes:
                strategies.append("Use schema extraction: keys/types, collection sizes, missing fields.")
            if not strategies:
                strategies.append("Use text summarization with line/token counts and keyword extraction.")
        else:
            strategies.append("No files detected: use domain-only planning strategy.")

        if "report" in domain.lower():
            strategies.append("Ensure final output includes a machine-readable report field.")

        return {
            "strategies": strategies,
        }

    def _recommend_runtime_tools(self, args: Dict[str, Any]) -> Dict[str, Any]:
        domain = str(args.get("domain", "") or "").lower()
        enable_bash_tooling = bool(args.get("enable_bash_tooling", True))

        tools: List[str] = [
            "run_python",
            "read_text_file",
            "read_json_file",
            "read_csv_preview",
            "scrape_url",
        ]
        if enable_bash_tooling:
            tools.append("run_bash")

        if "security" in domain:
            goals = [
                "Run dependency and static checks via run_python(['-m','pip', ...]) when needed.",
                "Use run_bash only for bounded shell checks with explicit command strings.",
                "Use scrape_url for threat-intel/context pulls from trusted documentation pages.",
            ]
        elif "data" in domain or "csv" in domain or "analysis" in domain:
            goals = [
                "Use run_python for deterministic Python tooling and local validation scripts.",
                "Prefer in-process Python parsing for input files before shell commands.",
                "Use read_csv_preview/read_json_file/read_text_file to enrich context before transformations.",
            ]
        else:
            goals = [
                "Prefer run_python for Python tooling commands.",
                "Use run_bash sparingly and only when shell semantics are required.",
                "Use scrape_url and file readers to build context before final task execution.",
            ]

        return {
            "runtime_tools": tools,
            "handoff_guidance": goals,
        }

    def _web_research_playwright(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query", "") or "").strip()
        max_results = int(args.get("max_results", 3) or 3)
        if not query:
            return {"error": "query is required"}

        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            return {
                "error": "playwright_not_installed",
                "hint": "Install dependency and browser: pip install playwright && playwright install chromium",
            }

        try:
            with sync_playwright() as p:
                chromium_path = Path(p.chromium.executable_path)
                if not chromium_path.exists():
                    return {
                        "error": "playwright_chromium_missing",
                        "hint": "Install browser binary first: playwright install chromium",
                    }

                browser = p.chromium.launch(headless=True)
                context = browser.new_context(ignore_https_errors=True)
                page = context.new_page()

                search_url = f"https://duckduckgo.com/?q={quote(query)}"
                page.goto(search_url, timeout=30000, wait_until="domcontentloaded")

                links = page.evaluate(
                    """
                    () => {
                      const anchors = Array.from(document.querySelectorAll('a'));
                      return anchors
                        .map(a => ({ href: a.href || '', text: (a.innerText || '').trim() }))
                        .filter(v => v.href.startsWith('http') && v.text.length > 10)
                        .slice(0, 60);
                    }
                    """
                )

                unique_urls: List[str] = []
                for item in links:
                    if not isinstance(item, dict):
                        continue
                    href = str(item.get("href", "")).strip()
                    if not href:
                        continue
                    href = _normalize_url(href)
                    if _is_low_signal_url(href):
                        continue
                    if href not in unique_urls:
                        unique_urls.append(href)
                    if len(unique_urls) >= max_results:
                        break

                findings: List[Dict[str, str]] = []
                for url in unique_urls:
                    try:
                        page.goto(url, timeout=25000, wait_until="domcontentloaded")
                        body = page.text_content("body") or ""
                        body = re.sub(r"\s+", " ", body).strip()
                        findings.append({"url": url, "snippet": body[:1200]})
                    except Exception as exc:
                        findings.append({"url": url, "snippet": f"Failed to read page: {exc}"})

                browser.close()
                return {
                    "query": query,
                    "sources": [f.get("url", "") for f in findings],
                    "findings": findings,
                }
        except Exception as exc:
            return {
                "error": "playwright_runtime_error",
                "details": str(exc),
                "hint": "Ensure Chromium is installed: playwright install chromium",
            }


def run_planner_mcp_tools(
    *,
    domain: str,
    enable_bash_tooling: bool,
    enabled: bool,
) -> Dict[str, Any]:
    """Execute planner-relevant MCP tools and return consolidated observations."""
    if not enabled:
        return {
            "tools_used": [],
            "observations": "MCP tools disabled for planner.",
        }

    server = LocalMCPToolServer()
    args = {"domain": domain, "enable_bash_tooling": enable_bash_tooling}

    tool_order = [
        "recommend_processing_strategy",
        "recommend_runtime_tools",
    ]
    outputs: List[str] = []
    runtime_tools: List[str] = []
    handoff_guidance: List[str] = []
    for name in tool_order:
        result = server.call_tool(name, args)
        outputs.append(f"{name}: {json.dumps(result, ensure_ascii=True)}")
        if name == "recommend_runtime_tools" and isinstance(result, dict):
            tools = result.get("runtime_tools", [])
            guidance = result.get("handoff_guidance", [])
            if isinstance(tools, list):
                runtime_tools = [str(v) for v in tools if str(v).strip()]
            if isinstance(guidance, list):
                handoff_guidance = [str(v) for v in guidance if str(v).strip()]

    return {
        "tools_used": tool_order,
        "observations": "\n".join(outputs),
        "runtime_tools": runtime_tools,
        "handoff_guidance": handoff_guidance,
    }


def run_research_mcp_tools(*, domain: str, enabled: bool) -> Dict[str, Any]:
    """Execute Playwright-backed web research via MCP-style tool calls."""
    if not enabled:
        return {
            "tools_used": [],
            "notes": "MCP tools disabled for research.",
            "sources": [],
        }

    server = LocalMCPToolServer()
    result = server.call_tool(
        "web_research_playwright",
        {
            "query": domain,
            "max_results": 3,
        },
    )

    if "error" in result:
        return {
            "tools_used": ["web_research_playwright"],
            "notes": f"Playwright MCP research unavailable: {result}",
            "sources": [],
        }

    findings = result.get("findings", []) if isinstance(result, dict) else []
    sources = result.get("sources", []) if isinstance(result, dict) else []
    notes_parts: List[str] = []
    if isinstance(findings, list):
        for item in findings:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", ""))
            snippet = str(item.get("snippet", ""))
            if _is_low_signal_url(url):
                continue
            cleaned = re.sub(r"\s+", " ", snippet).strip()
            notes_parts.append(f"Source: {url}\nSnippet: {cleaned[:350]}")
            if len(notes_parts) >= 3:
                break

    notes = "\n\n".join(notes_parts) if notes_parts else "No high-signal Playwright findings captured."
    filtered_sources = [
        str(v)
        for v in sources
        if isinstance(v, str) and v.strip() and not _is_low_signal_url(str(v))
    ]
    return {
        "tools_used": ["web_research_playwright"],
        "notes": notes,
        "sources": filtered_sources,
    }
