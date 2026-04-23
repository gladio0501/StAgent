from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
from urllib.parse import quote
from urllib.request import urlopen
from langchain_core.tools import tool

@tool
def inspect_file(filepath: str) -> str:
    """Reads a text or JSON file from the workspace to extract its exact content."""
    try:
        path = Path(filepath)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return f"Error: File not found at {path}"
        if path.stat().st_size > 500_000:
            return f"Error: File too large to read securely (>500KB)."
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading file: {exc}"

@tool
def search_web(query: str, max_results: int = 3) -> str:
    """Uses Wikipedia API and DuckDuckGo Lite to research factual questions on the web."""
    import urllib.request
    import urllib.parse
    import json
    import re
    from bs4 import BeautifulSoup
    
    parts = []
    
    # 1. Try Wikipedia REST API directly for factual queries
    try:
        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
        req = urllib.request.Request(wiki_url, headers={'User-Agent': 'StemAgent/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            search_results = data.get('query', {}).get('search', [])
            for res in search_results[:2]:  # Top 2 from Wikipedia
                snippet = re.sub(r'<[^>]+>', '', res.get('snippet', ''))
                parts.append(f"Wikipedia Source: {res.get('title')}\nSnippet: {snippet}")
    except Exception as e:
        parts.append(f"[Wikipedia Search Error: {e}]")
        
    # 2. Try DuckDuckGo Lite HTML fallback
    try:
        ddg_url = "https://lite.duckduckgo.com/lite/"
        data = urllib.parse.urlencode({'q': query}).encode('utf-8')
        req = urllib.request.Request(ddg_url, data=data, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')
            
            snippets = soup.find_all('td', class_='result-snippet')
            for i, snip in enumerate(snippets[:max_results]):
                parts.append(f"DuckDuckGo Result {i+1}:\nSnippet: {snip.text.strip()}")
    except Exception as e:
        parts.append(f"[DuckDuckGo Search Error: {e}]")
        
    if not parts or all("Error:" in p for p in parts):
        return f"No findings from web search. Logs: {' | '.join(parts)}"
        
    return "\n\n".join(parts)


TESTS_FILE = Path(__file__).resolve().parents[1] / "sandbox" / "tests.json"
_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError:\\s+No module named ['\"]([^'\"]+)['\"]")
_MODULE_PACKAGE_MAP = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
}

_TOOL_PACKAGE_MAP = {
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "pil": "pillow",
    "pillow": "pillow",
    "beautifulsoup": "beautifulsoup4",
    "bs4": "beautifulsoup4",
    "pip-audit": "pip-audit",
    "jsonschema": "jsonschema",
    "numpy": "numpy",
    "pandas": "pandas",
    "requests": "requests",
    "pytest": "pytest",
    "coverage": "coverage",
    "bandit": "bandit",
}

_NON_PACKAGE_TOOL_NAMES = {
    "run_python",
    "run_bash",
    "read_text_file",
    "read_json_file",
    "read_csv_preview",
    "scrape_url",
    "pathlib",
    "json",
    "csv",
    "statistics",
}


def strip_code_fences(content: str) -> str:
    """Extract executable Python from Markdown/prose responses."""
    text = content.strip()
    if not text:
        return ""

    fence_pattern = r"^```(?:python)?\s*\n(?P<code>[\s\S]*?)\n```\s*$"
    match = re.match(fence_pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group("code").strip() + "\n"

    block_pattern = r"```(?:python)?\s*\n([\s\S]*?)\n```"
    block_match = re.search(block_pattern, text, flags=re.IGNORECASE)
    if block_match:
        return block_match.group(1).strip() + "\n"

    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*```", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def extract_missing_module(error_trace: Optional[str]) -> str:
    if not error_trace:
        return ""
    match = _MISSING_MODULE_RE.search(error_trace)
    if not match:
        return ""
    return str(match.group(1)).strip()


def _module_to_package(module_name: str) -> str:
    root = module_name.split(".", 1)[0]
    return _MODULE_PACKAGE_MAP.get(root, root)


def auto_install_missing_dependency(module_name: str) -> tuple[bool, str]:
    root = module_name.split(".", 1)[0]
    if root in getattr(sys, "stdlib_module_names", set()):
        return False, f"'{root}' is part of the Python standard library."

    package_name = _module_to_package(module_name)
    install_cmd = [sys.executable, "-m", "pip", "install", package_name]
    try:
        result = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception as exc:
        return False, f"Auto-install failed for '{package_name}': {exc}"

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        return False, f"Auto-install failed for '{package_name}': {details[:800]}"

    return True, f"Auto-installed missing dependency '{package_name}'."


def deterministic_reflection(error_trace: str) -> str:
    """Return a concise root-cause critique directly from known traceback patterns."""
    trace = (error_trace or "").strip()
    if not trace:
        return (
            "1) Root cause\n"
            "No traceback was provided by runtime.\n\n"
            "2) Minimal fix strategy\n"
            "Capture and propagate stderr/stdout for failed executions before reflection.\n\n"
            "3) Regression risk to watch\n"
            "Low. Main risk is hiding real errors if execution metadata is incomplete."
        )

    if "ModuleNotFoundError" in trace:
        module_name = extract_missing_module(trace) or "a required package"
        return (
            "1) Root cause\n"
            f"Missing dependency at runtime: {module_name}.\n\n"
            "2) Minimal fix strategy\n"
            "Install the missing package and rerun. Keep dependency pinned in requirements.txt.\n\n"
            "3) Regression risk to watch\n"
            "Medium. New generated code may introduce different optional deps; keep auto-install + pinning in sync."
        )

    if "JSONDecodeError" in trace and "processing_plan" in trace:
        return (
            "1) Root cause\n"
            "Generated code treated STEM_FILE_PROCESSING_PLAN as JSON, but the value is plain text.\n\n"
            "2) Minimal fix strategy\n"
            "Read processing plan as opaque text. Only call json.loads when the string is explicitly JSON-formatted.\n\n"
            "3) Regression risk to watch\n"
            "Medium. Future generations may reintroduce strict JSON parsing assumptions for runtime env vars."
        )

    if "SyntaxError" in trace and "```" in trace:
        return (
            "1) Root cause\n"
            "Markdown fence markers leaked into executable Python.\n\n"
            "2) Minimal fix strategy\n"
            "Strip fenced markdown/prose and execute only extracted Python code.\n\n"
            "3) Regression risk to watch\n"
            "Low. Regression appears when model returns mixed prose + code and sanitization is bypassed."
        )

    if "Execution JSON missing required keys" in trace:
        return (
            "1) Root cause\n"
            "Script output violated the runtime JSON contract.\n\n"
            "2) Minimal fix strategy\n"
            "Ensure stdout is one JSON object containing status/domain/test_case_count on every success path.\n\n"
            "3) Regression risk to watch\n"
            "Medium. Partial exception handlers often return schema-incomplete JSON."
        )

    return (
        "1) Root cause\n"
        "Runtime/script failure detected from traceback.\n\n"
        "2) Minimal fix strategy\n"
        "Patch the failing line indicated in traceback, then rerun contract and domain tests.\n\n"
        "3) Regression risk to watch\n"
        "Medium. Fixes may mask adjacent input-validation and serialization paths."
    )


def validate_execution_output(output: Optional[str]) -> tuple[bool, str]:
    """Enforce the generated script output contract for orchestrator reliability."""
    if not output or not output.strip():
        return False, "Execution produced empty stdout; expected JSON output."

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"Execution stdout is not valid JSON: {exc}"

    if not isinstance(payload, dict):
        return False, "Execution JSON output must be an object."

    required_keys = ("status", "domain", "test_case_count")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        return False, f"Execution JSON missing required keys: {', '.join(missing)}"

    if payload.get("status") != "ok":
        return False, "Execution JSON has non-ok status."

    if not isinstance(payload.get("test_case_count"), int):
        return False, "Execution JSON field 'test_case_count' must be an integer."

    return True, ""


def extract_json_object(text: str) -> dict:
    """Extract a JSON object from plain text or fenced text responses."""
    candidate = strip_code_fences(text).strip()
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", candidate)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def get_llm() -> Optional[Any]:
    """Lazy-load OpenAI so local runs still work without credentials."""
    if not os.getenv("OPENAI_API_KEY"):
        return None

    try:
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, temperature=0)
    except Exception:
        return None


def fallback_requirements(domain: str) -> str:
    return (
        f"Domain: {domain}\n"
        "Goal: Build a reliable Python script for the domain task.\n"
        "Suggested libraries: json, csv, pathlib, statistics (and pandas if needed).\n"
        "Contract: read optional tests JSON path from argv and print a JSON summary.\n"
        "Checks: handle missing files, validate schema, produce deterministic outputs."
    )


def fallback_architecture(domain: str) -> dict:
    lower_domain = domain.lower()
    tools = ["json", "pathlib", "pytest"]
    strategy = "contract-tests + deterministic smoke tests"
    decision = "Single-pass JSON contract agent with iterative repair loop"

    if "security" in lower_domain:
        tools = ["bandit", "pip-audit", "json", "pathlib"]
        strategy = "policy checks + exploit regression tests"
        decision = "Policy-first security analysis agent"
    elif "qa" in lower_domain or "quality" in lower_domain:
        tools = ["pytest", "coverage", "json", "pathlib"]
        strategy = "coverage-gated unit + integration tests"
        decision = "Test-harness-centric QA agent"
    elif "research" in lower_domain:
        tools = ["json", "requests", "pathlib"]
        strategy = "evidence scoring + citation integrity checks"
        decision = "Retrieval-and-synthesis research agent"

    return {
        "architecture_decision": decision,
        "selected_tools": tools,
        "runtime_tooling": [
            "run_python",
            "run_bash",
            "read_text_file",
            "read_json_file",
            "read_csv_preview",
            "scrape_url",
        ],
        "test_strategy": strategy,
        "stop_threshold": 0.85,
    }


def materialize_runtime_tools(
    *,
    selected_tools: list[str],
    architect_runtime_tools: list[str],
    enable_bash_tooling: bool,
) -> list[str]:
    base_tools = [
        "run_python",
        "read_text_file",
        "read_json_file",
        "read_csv_preview",
        "scrape_url",
    ]
    if enable_bash_tooling:
        base_tools.append("run_bash")

    requested = [str(v).strip() for v in architect_runtime_tools if str(v).strip()]
    if not requested:
        requested = base_tools.copy()

    selected_joined = " ".join(str(v).lower() for v in selected_tools)
    if "request" in selected_joined or "http" in selected_joined or "scrap" in selected_joined:
        if "scrape_url" not in requested:
            requested.append("scrape_url")

    allowed = {
        "run_python",
        "run_bash",
        "read_text_file",
        "read_json_file",
        "read_csv_preview",
        "scrape_url",
    }
    final_tools: list[str] = []
    for name in requested:
        if name not in allowed:
            continue
        if name == "run_bash" and not enable_bash_tooling:
            continue
        if name not in final_tools:
            final_tools.append(name)

    for name in base_tools:
        if name == "run_bash" and not enable_bash_tooling:
            continue
        if name not in final_tools:
            final_tools.append(name)
    return final_tools


def build_runtime_tool_handoff(
    *,
    runtime_tools: list[str],
    handoff_guidance: list[str],
    enable_bash_tooling: bool,
) -> str:
    tools = [str(v) for v in runtime_tools if str(v).strip()]
    if not tools:
        tools = ["run_python", "read_text_file", "read_json_file", "read_csv_preview", "scrape_url"]
        if enable_bash_tooling:
            tools.append("run_bash")
    if not enable_bash_tooling:
        tools = []

    lines = [
        "Planner Runtime Tool Handoff",
        f"Runtime tools exposed to specialized agent: {', '.join(tools) if tools else 'none'}",
        "Usage priority: prefer run_python for Python tooling; use run_bash only for shell-specific needs.",
    ]
    if handoff_guidance:
        lines.append("Guidance:")
        for item in handoff_guidance[:6]:
            lines.append(f"- {item}")
    if not tools:
        lines.append("Tooling is disabled for this run; specialized agent must operate without runtime helpers.")
    return "\n".join(lines)


def compute_inter_agent_benchmarks(
    *,
    requirements: str,
    research_notes: str,
    research_sources: list[str],
    architecture_decision: str,
    selected_tools: list[str],
    test_strategy: str,
    planner_runtime_tool_handoff: str,
    code: str,
    execution_success: bool,
    execution_duration_seconds: float,
    hyper_score: float,
) -> Dict[str, float]:
    req = (requirements or "").strip()
    notes = (research_notes or "").strip()
    arch = (architecture_decision or "").strip()
    handoff = (planner_runtime_tool_handoff or "").strip().lower()

    planner_quality = 1.0 if len(req) >= 220 else (0.6 if len(req) >= 120 else 0.25)
    has_handoff_tools = 1.0 if "runtime tools exposed" in handoff else 0.0
    research_quality = min(1.0, len(notes) / 900.0)
    source_coverage = min(1.0, len(research_sources) / 3.0)
    architect_quality = 1.0 if arch and test_strategy.strip() else 0.4
    tool_plan_quality = min(1.0, len([t for t in selected_tools if str(t).strip()]) / 4.0)

    try:
        ast.parse(code or "")
        generator_syntax_valid = 1.0
    except SyntaxError:
        generator_syntax_valid = 0.0
    generator_contract_hint = 1.0 if all(k in (code or "") for k in ["status", "domain", "test_case_count"]) else 0.0

    runtime_success = 1.0 if execution_success else 0.0
    runtime_speed = max(0.0, min(1.0, 1.0 - (execution_duration_seconds / 30.0)))

    upstream_mean = (
        planner_quality
        + has_handoff_tools
        + research_quality
        + source_coverage
        + architect_quality
        + tool_plan_quality
        + generator_syntax_valid
        + generator_contract_hint
    ) / 8.0
    runtime_mean = (runtime_success + runtime_speed + hyper_score) / 3.0
    handoff_effectiveness = max(0.0, min(1.0, runtime_mean - (upstream_mean * 0.5) + (has_handoff_tools * 0.2)))

    return {
        "planner_quality": round(planner_quality, 4),
        "planner_tool_handoff": round(has_handoff_tools, 4),
        "research_quality": round(research_quality, 4),
        "research_source_coverage": round(source_coverage, 4),
        "architect_quality": round(architect_quality, 4),
        "tool_plan_quality": round(tool_plan_quality, 4),
        "generator_syntax_valid": round(generator_syntax_valid, 4),
        "generator_contract_hint": round(generator_contract_hint, 4),
        "runtime_success": round(runtime_success, 4),
        "runtime_speed": round(runtime_speed, 4),
        "upstream_mean": round(upstream_mean, 4),
        "runtime_mean": round(runtime_mean, 4),
        "handoff_effectiveness": round(handoff_effectiveness, 4),
    }


def compute_legacy_fitness(
    *,
    success: bool,
    execution_result: Optional[str],
    error_trace: Optional[str],
    domain: str,
    iteration_count: int,
    max_iterations: int,
) -> tuple[Dict[str, float], float]:
    """Legacy weighted score used as a baseline (before metric)."""
    process_success = 1.0 if success else 0.0
    contract_valid = 0.0
    domain_alignment = 0.0
    robustness = 1.0 if not error_trace else 0.0
    efficiency = max(0.0, 1.0 - (iteration_count / max(1, max_iterations)))

    payload = {}
    if execution_result:
        try:
            payload = json.loads(execution_result)
        except json.JSONDecodeError:
            payload = {}

    if isinstance(payload, dict):
        required = {"status", "domain", "test_case_count"}
        contract_valid = 1.0 if required.issubset(payload.keys()) and payload.get("status") == "ok" else 0.0
        result_domain = str(payload.get("domain", "")).lower()
        domain_alignment = 1.0 if domain.lower() in result_domain or result_domain in domain.lower() else 0.4

    metrics = {
        "process_success": process_success,
        "contract_valid": contract_valid,
        "domain_alignment": domain_alignment,
        "robustness": robustness,
        "efficiency": efficiency,
    }
    weighted_score = (
        0.30 * metrics["process_success"]
        + 0.30 * metrics["contract_valid"]
        + 0.20 * metrics["domain_alignment"]
        + 0.10 * metrics["robustness"]
        + 0.10 * metrics["efficiency"]
    )
    return metrics, round(weighted_score, 4)


def compute_domain_benchmark_score(
    *,
    domain: str,
    execution_result: Optional[str],
    success: bool,
    domain_class: str = "",
) -> tuple[str, float]:
    """Compute a domain-aware benchmark score in [0, 1] from runtime output.

    Important: ``success`` here is the *process* success flag (did the Python
    interpreter exit cleanly?), NOT whether the generated script completed the
    domain task correctly.  A script may run without crashing yet still emit
    ``{"status": "error", ...}`` because it could not locate required inputs
    or rubric data.  We must therefore check the payload contract status before
    awarding any domain metric score.
    """
    lower_domain = (domain or "").lower()
    payload: dict[str, Any] = {}
    if execution_result:
        try:
            parsed = json.loads(execution_result)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = {}

    # --- Contract guard ---------------------------------------------------
    # If the payload reports a non-ok status, the script did not complete the
    # domain task regardless of whether the process exited cleanly.  For all
    # metric types except pass_at_1 (which tests bare execution), we short-
    # circuit to 0.0 so that error-status runs cannot receive vacuous credit.
    payload_status_ok = str(payload.get("status", "")).strip().lower() == "ok"
    # Also require test_case_count > 0 so that a script that reports ok but
    # processed zero cases does not receive full credit.
    payload_cases = payload.get("test_case_count", 0)
    payload_has_work = isinstance(payload_cases, int) and payload_cases > 0

    cls = (domain_class or "").strip().lower()
    if cls in {"software_engineering", "software", "polyglot"} or any(
        key in lower_domain for key in ["software", "engineering", "polyglot", "code"]
    ):
        # pass@1: process-level success is the signal.
        return "pass_at_1", 1.0 if success else 0.0

    if cls in {"paper_review", "paper", "search_arena", "review"} or any(
        key in lower_domain for key in ["paper", "review", "search arena", "accept", "reject"]
    ):
        # overall_accuracy: must be explicitly present in the payload.
        # We do NOT fall back to `success` here — a crashed or errored script
        # with no accuracy field earns 0.0.
        for candidate in ["overall_accuracy", "accuracy", "acc"]:
            value = payload.get(candidate)
            if isinstance(value, (int, float)):
                return "overall_accuracy", max(0.0, min(1.0, float(value)))
        # No explicit accuracy field: if the payload is contract-valid and has
        # cases, try a prediction match; otherwise 0.0.
        if payload_status_ok and payload_has_work:
            predicted = payload.get("prediction") or payload.get("predicted_label")
            if isinstance(predicted, str):
                # Caller can supply expected via benchmark suite; here we can
                # only confirm presence of prediction output.
                return "overall_accuracy", 0.5  # partial credit for present-but-unverifiable
        return "overall_accuracy", 0.0

    if cls in {"olympiad_math", "olympiad", "imo", "math"} or any(
        key in lower_domain for key in ["olympiad", "imo", "proof", "math", "grading"]
    ):
        # points_percentage must be explicitly present in the payload.
        # An error-status run or a run with zero graded cases scores 0.0.
        if not payload_status_ok:
            return "points_percentage", 0.0
        percentage = payload.get("points_percentage")
        if isinstance(percentage, (int, float)):
            value = float(percentage)
            if value > 1.0:
                value = value / 100.0
            return "points_percentage", max(0.0, min(1.0, value))
        earned = payload.get("points_earned")
        total = payload.get("points_total")
        if isinstance(earned, (int, float)) and isinstance(total, (int, float)) and float(total) > 0:
            return "points_percentage", max(0.0, min(1.0, float(earned) / float(total)))
        # No grading data: if the script completed successfully but produced no
        # grading output, that is a task failure — score 0.0.
        return "points_percentage", 0.0

    if cls in {"general_assistant", "gaia", "assistant"} or any(
        key in lower_domain for key in ["general ai assistant", "web research", "gaia"]
    ):
        # overall_accuracy via normalised answer match.
        # The payload must contain an 'answer' field and status='ok'.
        # Actual comparison against the expected answer happens in evaluate_benchmark_case;
        # here we confirm the payload produced a non-empty answer string.
        if not payload_status_ok:
            return "overall_accuracy", 0.0
        answer = payload.get("answer")
        if isinstance(answer, str) and answer.strip():
            # Return 0.5 as a presence signal; evaluate_benchmark_case does the real match.
            return "overall_accuracy", 0.5
        return "overall_accuracy", 0.0

    # Generic fallback for domains without an explicit benchmark metric.
    return "generic_success", 1.0 if success else 0.0


def compute_hyperagent_score(
    *,
    success: bool,
    contract_valid: bool,
    domain_alignment: float,
    robustness: float,
    iteration_count: int,
    max_iterations: int,
    execution_duration_seconds: float,
    previous_best_fitness: float,
    benchmark_score: float,
) -> tuple[Dict[str, float], float]:
    """Hyperagents-inspired score with explicit progress and efficiency terms."""
    task_success = 1.0 if success else 0.0
    contract_compliance = 1.0 if contract_valid else 0.0
    efficiency = max(0.0, 1.0 - (iteration_count / max(1, max_iterations)))
    time_efficiency = max(0.0, min(1.0, 30.0 / max(1e-6, execution_duration_seconds * 10.0)))

    candidate_quality = (
        0.45 * task_success
        + 0.20 * contract_compliance
        + 0.20 * domain_alignment
        + 0.15 * robustness
    )
    improvement_delta = max(0.0, candidate_quality - previous_best_fitness)
    normalized_improvement = min(1.0, improvement_delta / 0.10) if improvement_delta > 0 else 0.0

    metrics = {
        "task_success": task_success,
        "contract_compliance": contract_compliance,
        "domain_alignment": domain_alignment,
        "robustness": robustness,
        "efficiency": efficiency,
        "time_efficiency": time_efficiency,
        "improvement_delta": round(improvement_delta, 4),
        "normalized_improvement": round(normalized_improvement, 4),
        "benchmark_score": max(0.0, min(1.0, benchmark_score)),
    }
    base_score = (
        0.55 * candidate_quality
        + 0.20 * normalized_improvement
        + 0.15 * efficiency
        + 0.10 * time_efficiency
    )
    score = (0.80 * base_score) + (0.20 * max(0.0, min(1.0, benchmark_score)))
    return metrics, round(score, 4)


def fetch_wikipedia_summary(domain: str) -> tuple[str, str]:
    """Fetch a compact external research signal from Wikipedia (no API key required)."""
    topic = quote(domain.strip())
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    try:
        with urlopen(url, timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
            extract = str(payload.get("extract") or "").strip()
            if extract:
                return extract[:600], url
    except Exception:
        pass
    return "", url


def compact_research_notes(text: str, max_chars: int = 1200) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    return cleaned[:max_chars]


def fallback_code(domain: str, requirements: str, error_trace: Optional[str]) -> str:
    error_hint = (error_trace or "none").replace('"""', "\"\"\"")
    req_text = requirements.replace('"""', "\"\"\"")
    domain_text = domain.replace('"""', "\"\"\"")
    return f'''import csv
import json
import os
import sys
from pathlib import Path

DOMAIN = """{domain_text}"""
REQUIREMENTS = """{req_text}"""
PREVIOUS_ERROR = """{error_hint}"""


def load_tests(path_arg: str | None) -> dict:
    if not path_arg:
        return {{"cases": []}}
    path = Path(path_arg)
    if not path.exists():
        return {{"cases": [], "note": f"tests file not found: {{path_arg}}"}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ingested_paths() -> list[str]:
    raw = os.getenv("STEM_INGESTED_FILES", "[]")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        values = []
    return [str(v) for v in values]


def load_processing_plan() -> str:
    return os.getenv("STEM_FILE_PROCESSING_PLAN", "").strip()


def _safe_float(value: str):
    try:
        return float(value)
    except Exception:
        return None


def _csv_report(path: Path) -> dict:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {{"path": str(path), "type": "csv", "row_count": 0, "column_count": 0, "numeric_summary": {{}}}}

    headers = list(rows[0].keys())
    numeric = {{h: [] for h in headers}}
    for row in rows:
        for h in headers:
            value = (row.get(h) or "").strip()
            as_num = _safe_float(value)
            if as_num is not None:
                numeric[h].append(as_num)

    numeric_summary = {{}}
    for h, vals in numeric.items():
        if vals:
            numeric_summary[h] = {{
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "mean": round(sum(vals) / len(vals), 4),
            }}

    return {{
        "path": str(path),
        "type": "csv",
        "row_count": len(rows),
        "column_count": len(headers),
        "columns": headers,
        "numeric_summary": numeric_summary,
    }}


def _build_reports(paths: list[str]) -> list[dict]:
    reports = []
    for value in paths:
        p = Path(value)
        if not p.exists() or not p.is_file():
            reports.append({{"path": str(p), "status": "missing"}})
            continue

        suffix = p.suffix.lower()
        if suffix in {{".csv", ".tsv"}}:
            reports.append(_csv_report(p))
        elif suffix == ".json":
            try:
                payload = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                reports.append({{
                    "path": str(p),
                    "type": "json",
                    "top_level_type": type(payload).__name__,
                    "keys": list(payload.keys())[:50] if isinstance(payload, dict) else [],
                }})
            except Exception as exc:
                reports.append({{"path": str(p), "type": "json", "error": str(exc)}})
        else:
            text = p.read_text(encoding="utf-8", errors="replace")
            reports.append({{"path": str(p), "type": "text", "line_count": len(text.splitlines())}})
    return reports


def solve(domain: str, tests: dict, ingested_paths: list[str], processing_plan: str) -> dict:
    reports = _build_reports(ingested_paths)
    return {{
        "domain": domain,
        "requirements_digest": REQUIREMENTS[:160],
        "previous_error_digest": PREVIOUS_ERROR[:160],
        "test_case_count": len(tests.get("cases", [])) if isinstance(tests, dict) else 0,
        "ingested_file_count": len(ingested_paths),
        "ingested_files": ingested_paths,
        "file_processing_plan": processing_plan,
        "report": reports,
        "status": "ok",
    }}


def main() -> None:
    tests_path = sys.argv[1] if len(sys.argv) > 1 else None
    tests = load_tests(tests_path)
    ingested_paths = load_ingested_paths()
    processing_plan = load_processing_plan()
    result = solve(DOMAIN, tests, ingested_paths, processing_plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def _tool_to_package_name(tool_name: str) -> str:
    candidate = str(tool_name or "").strip()
    if not candidate:
        return ""
    token = candidate.split()[0].strip().lower()
    token = token.replace(",", "")
    if token in _NON_PACKAGE_TOOL_NAMES:
        return ""
    if token in getattr(sys, "stdlib_module_names", set()):
        return ""
    return _TOOL_PACKAGE_MAP.get(token, token)


def ensure_selected_tool_dependencies(selected_tools: list[str]) -> tuple[list[str], list[str]]:
    """Install packages implied by architect-selected tools.

    Returns (installed_or_present_packages, failures).
    """
    packages: list[str] = []
    for tool in selected_tools:
        package = _tool_to_package_name(str(tool))
        if package and package not in packages:
            packages.append(package)

    installed_or_present: list[str] = []
    failures: list[str] = []
    for package in packages:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode == 0:
                installed_or_present.append(package)
            else:
                details = (result.stderr or result.stdout or "").strip()
                failures.append(f"{package}: {details[:240]}")
        except Exception as exc:
            failures.append(f"{package}: {exc}")

    return installed_or_present, failures
