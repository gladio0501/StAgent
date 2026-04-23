from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from sandbox.executor import execute_python_code
from stem.graph import app as graph_app
from stem.ingest import build_file_manifest, build_file_processing_plan, ingest_files
from stem.nodes_support import (
    TESTS_FILE,
    compute_domain_benchmark_score,
    compute_legacy_fitness,
    fallback_code,
    get_llm,
    strip_code_fences,
    validate_execution_output,
)
from stem.prompts import GENERATE_PROMPT
from stem.state import AgentState


class BenchmarkCase(TypedDict, total=False):
    id: str
    domain: str
    theme: str
    domain_class: str
    metric_type: str
    input_files: List[str]
    weight: float
    max_iterations: int
    stop_threshold: float
    patience: int
    open_run_mode: bool
    enable_bash_tooling: bool
    enable_mcp_tools: bool
    enable_staged_evaluation: bool
    staged_eval_ratio: float
    staged_eval_max_cases: int
    staged_eval_threshold: float
    staged_eval_fail_scale: float
    expected: Dict[str, Any]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark_suite(path: str) -> Dict[str, Any]:
    suite_path = Path(path)
    if not suite_path.exists():
        raise FileNotFoundError(f"Benchmark suite not found: {path}")
    payload = _load_json(suite_path)
    if not isinstance(payload, dict):
        raise ValueError("Benchmark suite must be a JSON object.")
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark suite must define a non-empty 'cases' array.")
    return payload


def _parse_execution_payload(final_state: Dict[str, Any]) -> Dict[str, Any]:
    raw = final_state.get("execution_result")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _metric_from_case(case: BenchmarkCase) -> str:
    metric = str(case.get("metric_type", "")).strip().lower()
    if metric:
        return metric

    cls = str(case.get("domain_class", "")).strip().lower()
    if cls in {"software_engineering", "software", "polyglot"}:
        return "pass_at_1"
    if cls in {"paper_review", "paper", "search_arena", "review"}:
        return "overall_accuracy"
    if cls in {"olympiad_math", "olympiad", "imo", "math"}:
        return "points_percentage"
    return "generic_success"


def _safe_ratio(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    v = float(value)
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))


def evaluate_benchmark_case(case: BenchmarkCase, final_state: Dict[str, Any]) -> Dict[str, Any]:
    metric_name = _metric_from_case(case)
    expected = case.get("expected", {})
    expected = expected if isinstance(expected, dict) else {}
    payload = _parse_execution_payload(final_state)
    success = bool(final_state.get("success"))

    score = 0.0
    # Determine whether the output payload is contract-valid so we can avoid
    # awarding vacuous credit when the script ran but did not complete the task.
    payload_status_ok = str(payload.get("status", "")).strip().lower() == "ok"
    payload_cases = payload.get("test_case_count", 0)
    payload_has_work = isinstance(payload_cases, int) and payload_cases > 0

    if metric_name == "pass_at_1":
        score = 1.0 if success else 0.0
    elif metric_name == "overall_accuracy":
        explicit = _safe_ratio(payload.get("overall_accuracy"))
        if explicit is not None:
            score = explicit
        else:
            # Generic dictionary evaluation for adaptable benchmarks
            if expected:
                all_match = True
                for k, expected_v in expected.items():
                    payload_v = payload.get(k)
                    if isinstance(payload_v, str) and isinstance(expected_v, str):
                        if payload_v.strip().lower() != expected_v.strip().lower():
                            all_match = False
                    elif payload_v != expected_v:
                        all_match = False
                score = 1.0 if all_match else 0.0
            else:
                score = 1.0 if success else 0.0

    elif metric_name == "points_percentage":
        if not payload_status_ok:
            # Non-ok status means the task was not completed; score 0.0.
            score = 0.0
        else:
            explicit = _safe_ratio(payload.get("points_percentage"))
            if explicit is not None:
                score = explicit
            else:
                earned = payload.get("points_earned")
                total = payload.get("points_total")
                if isinstance(earned, (int, float)) and isinstance(total, (int, float)) and float(total) > 0:
                    score = max(0.0, min(1.0, float(earned) / float(total)))
                else:
                    # No grading data in a status=ok payload: task was not
                    # completed properly; score 0.0 instead of vacuous 1.0.
                    score = 0.0
    else:
        score = 1.0 if success else 0.0

    return {
        "metric_name": metric_name,
        "score": round(float(score), 4),
        "success": success,
        "fitness_score": float(final_state.get("fitness_score", 0.0) or 0.0),
        "benchmark_score": float(final_state.get("benchmark_score", 0.0) or 0.0),
        "benchmark_metric_name": str(final_state.get("benchmark_metric_name", "") or ""),
        "staged_eval_passed": bool(final_state.get("staged_eval_passed", True)),
    }


def _build_report(
    *,
    suite_path: str,
    started_at: datetime,
    repeats: int,
    case_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    weighted_sum = 0.0
    weighted_den = 0.0
    by_domain: Dict[str, List[float]] = {}
    for row in case_results:
        score = float(row.get("result", {}).get("score", 0.0) if isinstance(row.get("result"), dict) else 0.0)
        weight = float(row.get("weight", 1.0) or 1.0)
        weighted_sum += score * weight
        weighted_den += weight
        cls = str(row.get("domain_class", "") or "generic")
        by_domain.setdefault(cls, []).append(score)

    domain_summary = {
        cls: {
            "count": len(values),
            "mean_score": round(sum(values) / max(1, len(values)), 4),
        }
        for cls, values in by_domain.items()
    }
    global_score = round(weighted_sum / max(1e-9, weighted_den), 4)

    return {
        "run_id": started_at.strftime("%Y%m%dT%H%M%SZ"),
        "started_at_utc": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "suite_path": str(Path(suite_path).resolve()),
        "repeats": max(1, repeats),
        "global_score": global_score,
        "domain_summary": domain_summary,
        "case_results": case_results,
    }


def _write_report(*, output_dir: str, file_prefix: str, report: Dict[str, Any]) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_prefix}_{report.get('run_id', 'run')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(out_path)


def run_llm_baseline_suite(
    *,
    suite_path: str,
    output_dir: str,
    repeats: int,
    open_run_mode: bool = False,
) -> Dict[str, Any]:
    suite = load_benchmark_suite(suite_path)
    cases_raw = suite.get("cases", [])
    assert isinstance(cases_raw, list)

    llm = get_llm()
    started_at = datetime.now(UTC)
    case_results: List[Dict[str, Any]] = []

    for case_item in cases_raw:
        if not isinstance(case_item, dict):
            continue
        case: BenchmarkCase = case_item  # type: ignore[assignment]
        case_id = str(case.get("id", "")).strip() or f"case_{len(case_results)+1:03d}"
        domain = str(case.get("domain", "")).strip() or "Benchmark case"
        theme = str(case.get("theme", "")).strip()
        domain_class = str(case.get("domain_class", "")).strip().lower()
        weight = float(case.get("weight", 1.0) or 1.0)

        for rep in range(max(1, repeats)):
            input_files = case.get("input_files", [])
            input_files = input_files if isinstance(input_files, list) else []
            ingested_files, _ingested_context, ingest_errors = ingest_files([str(v) for v in input_files])
            file_manifest = build_file_manifest(ingested_files)
            file_processing_plan = build_file_processing_plan(file_manifest)

            if llm is None:
                code = fallback_code(
                    domain=domain,
                    requirements=(
                        "Pure-LLM baseline mode fallback. "
                        "No planner/research/architect or iterative reflection is used."
                    ),
                    error_trace=None,
                )
            else:
                # Inject file contents directly so LLM can see the task data without runtime helpers
                file_content_block = ""
                for fpath in ingested_files:
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as _f:
                            raw = _f.read(8000)  # cap at 8k chars per file
                        file_content_block += f"\n--- File: {fpath} ---\n{raw}\n"
                    except Exception:
                        file_content_block += f"\n--- File: {fpath} (unreadable) ---\n"

                prompt = (
                    f"Domain:\n{domain}\n\n"
                    f"Theme constraints:\n{theme or 'none'}\n\n"
                    "Mode: pure LLM baseline; no planner/research/architect/reflection loops.\n"
                    "Runtime helper tools are disabled; do not use run_bash/run_python helpers.\n"
                    "Use only standard Python and deterministic logic.\n\n"
                    f"Runtime file processing plan:\n{file_processing_plan or 'none'}\n\n"
                    f"Input file paths (available as STEM_INGESTED_FILES global):\n{json.dumps(ingested_files)}\n\n"
                    + (f"Input file contents (read these to understand the task):\n{file_content_block}\n\n" if file_content_block else "")
                    + "IMPORTANT: If the file contents contain a question or task, answer it directly by hardcoding the answer into the output JSON. Do not attempt file I/O at runtime.\n\n"
                    "Return Python code only."
                )
                response = llm.invoke([SystemMessage(content=GENERATE_PROMPT), HumanMessage(content=prompt)])
                code = strip_code_fences(str(getattr(response, "content", "") or ""))

            started = time.perf_counter()
            result = execute_python_code(
                code,
                tests_file=str(TESTS_FILE),
                timeout_seconds=30,
                extra_env={
                    "STEM_INGESTED_FILES": json.dumps(ingested_files),
                    "STEM_FILE_PROCESSING_PLAN": file_processing_plan,
                    "STEM_RUNTIME_TOOLS": json.dumps([]),
                },
                enforce_policy=not open_run_mode,
                enforce_resource_limits=not open_run_mode,
                isolated_execution=not open_run_mode,
                enable_bash_tooling=False,
                runtime_tools=[],
            )
            duration = round(time.perf_counter() - started, 4)

            success = bool(result.get("success"))
            execution_result_raw = result.get("execution_result")
            error_trace_raw = result.get("error_trace")
            execution_result = execution_result_raw if isinstance(execution_result_raw, str) else None
            error_trace = error_trace_raw if isinstance(error_trace_raw, str) else None

            if not open_run_mode:
                valid, validation_error = validate_execution_output(execution_result)
                if not valid:
                    success = False
                    error_trace = validation_error if not error_trace else f"{error_trace}\n{validation_error}"

            metric_name, metric_score = compute_domain_benchmark_score(
                domain=domain,
                execution_result=execution_result,
                success=success,
                domain_class=domain_class,
            )
            _baseline_metrics, baseline_score = compute_legacy_fitness(
                success=success,
                execution_result=execution_result,
                error_trace=error_trace,
                domain=domain,
                iteration_count=1,
                max_iterations=1,
            )

            final_state = {
                "success": success,
                "fitness_score": float(baseline_score),
                "benchmark_score": float(metric_score),
                "benchmark_metric_name": metric_name,
                "staged_eval_passed": True,
                "iteration_count": 1,
                "execution_result": execution_result,
                "error_trace": error_trace,
                "execution_duration_seconds": duration,
            }
            eval_result = evaluate_benchmark_case(case, final_state)

            case_results.append(
                {
                    "id": case_id,
                    "repeat": rep + 1,
                    "domain": domain,
                    "domain_class": domain_class,
                    "weight": weight,
                    "ingest_errors": ingest_errors,
                    "result": eval_result,
                    "final_state": {
                        "success": bool(final_state.get("success")),
                        "fitness_score": float(final_state.get("fitness_score", 0.0) or 0.0),
                        "benchmark_score": float(final_state.get("benchmark_score", 0.0) or 0.0),
                        "benchmark_metric_name": str(final_state.get("benchmark_metric_name", "") or ""),
                        "iteration_count": int(final_state.get("iteration_count", 0) or 0),
                    },
                }
            )

    report = _build_report(
        suite_path=suite_path,
        started_at=started_at,
        repeats=repeats,
        case_results=case_results,
    )
    report["runner"] = "llm_baseline"
    report["output_file"] = _write_report(output_dir=output_dir, file_prefix="benchmark_llm", report=report)
    return report


def run_conversational_baseline_suite(
    *,
    suite_path: str,
    output_dir: str,
    repeats: int,
    max_turns: int = 4,
    open_run_mode: bool = False,
) -> Dict[str, Any]:
    """Multi-turn conversational baseline — no graph, no code execution, no tools.

    The LLM receives the domain, file contents, and output schema in the first
    turn. If its answer fails evaluation it receives feedback and retries, keeping
    full message history across turns (memory). Scored with the same
    evaluate_benchmark_case logic as all other runners.
    """
    from langchain_core.messages import AIMessage

    suite = load_benchmark_suite(suite_path)
    cases_raw = suite.get("cases", [])
    assert isinstance(cases_raw, list)

    llm = get_llm()
    started_at = datetime.now(UTC)
    case_results: List[Dict[str, Any]] = []

    SYSTEM_PROMPT = (
        "You are a precise factual-answering agent. "
        "You will be given a domain task description and the contents of any relevant files. "
        "Your job is to reason over the provided information and output exactly ONE JSON object "
        "that satisfies the domain output contract. No prose, no markdown fences, no extra fields "
        "beyond what the contract requires.\n\n"
        "Domain contract rules:\n"
        "- Paper Review: include 'prediction' ('accept' or 'reject', lowercase), 'test_case_count': 1\n"
        "- General Assistant / GAIA: include 'answer' (short exact string), 'test_case_count': 1\n"
        "- Software Engineering: include 'status': 'ok', 'test_case_count': <n>\n"
        "- Olympiad Math: include 'points_percentage' (float 0-1)\n"
        "Always include 'status': 'ok' and 'domain': '<domain class>' in every output."
    )

    for case_item in cases_raw:
        if not isinstance(case_item, dict):
            continue
        case: BenchmarkCase = case_item  # type: ignore[assignment]
        case_id = str(case.get("id", "")).strip() or f"case_{len(case_results)+1:03d}"
        domain = str(case.get("domain", "")).strip() or "Benchmark case"
        domain_class = str(case.get("domain_class", "")).strip().lower()
        expected = case.get("expected", {})
        weight = float(case.get("weight", 1.0) or 1.0)

        for rep in range(max(1, repeats)):
            input_files = case.get("input_files", [])
            input_files = input_files if isinstance(input_files, list) else []
            ingested_files, _ingested_context, ingest_errors = ingest_files([str(v) for v in input_files])

            # Ground-truth field names that must NEVER be shown to the evaluatee
            _ORACLE_KEYS = {"expected_answer", "label", "accepted", "answer"}

            # Build file content block — the LLM's only source of task data
            file_content_block = ""
            for fpath in ingested_files:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as _f:
                        raw = _f.read(8000)
                    # Strip oracle fields if this is a JSON file
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            sanitised = {k: v for k, v in parsed.items() if k not in _ORACLE_KEYS}
                            raw = json.dumps(sanitised, indent=2)
                    except json.JSONDecodeError:
                        pass  # not JSON — keep as-is
                    file_content_block += f"\n--- File: {fpath} ---\n{raw}\n"
                except Exception:
                    file_content_block += f"\n--- File: {fpath} (unreadable) ---\n"


            first_user_msg = (
                f"Domain task:\n{domain}\n\n"
                + (f"File contents:\n{file_content_block}\n" if file_content_block else "No files provided.\n")
                + "\nOutput a single JSON object satisfying the domain contract. No other text."
            )

            history: List[Any] = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=first_user_msg),
            ]

            score = 0.0
            execution_result: Optional[str] = None
            final_answer_payload: Dict[str, Any] = {}

            if llm is None:
                execution_result = json.dumps({"status": "error", "message": "No LLM available"})
            else:
                for turn in range(max_turns):
                    response = llm.invoke(history)
                    raw_text = str(getattr(response, "content", "") or "").strip()
                    history.append(AIMessage(content=raw_text))

                    # Try to parse the JSON answer
                    try:
                        # strip accidental fences
                        clean = raw_text.strip()
                        if clean.startswith("```"):
                            clean = "\n".join(clean.split("\n")[1:])
                            clean = clean.rstrip("`").strip()
                        payload = json.loads(clean)
                    except json.JSONDecodeError:
                        payload = {}

                    final_answer_payload = payload
                    execution_result = json.dumps(payload) if payload else raw_text

                    # Check contract validity ONLY — no correctness signal exposed
                    _required_fields = list(case.get("expected", {}).keys())
                    if not _required_fields:
                        _required_fields = ["status"]

                    contract_ok = isinstance(payload, dict) and payload.get("status") == "ok"
                    if contract_ok:
                        for req_f in _required_fields:
                            if req_f not in payload:
                                contract_ok = False
                                break

                    if contract_ok:
                        break  # contract satisfied — stop retrying

                    # Contract invalid — give format feedback only, no correctness hint
                    missing = [] if not isinstance(payload, dict) else [
                        f for f in (["status"] + _required_fields) if f not in payload
                    ]
                    feedback = (
                        "Your output did not satisfy the required JSON contract. "
                        + (f"Missing fields: {missing}. " if missing else "Could not parse as JSON. ")
                        + "Output ONLY a valid JSON object with the required fields. No prose."
                    )
                    history.append(HumanMessage(content=feedback))

            # Score correctness once, after all turns — no expected answer leaked during loop
            final_state: Dict[str, Any] = {
                "success": bool(final_answer_payload.get("status") == "ok"),
                "fitness_score": 0.0,
                "benchmark_score": 0.0,
                "benchmark_metric_name": _metric_from_case(case),
                "staged_eval_passed": True,
                "iteration_count": min(max_turns, len([m for m in history if isinstance(m, AIMessage)])),
                "execution_result": execution_result,
                "error_trace": None,
            }
            eval_result = evaluate_benchmark_case(case, final_state)

            case_results.append(
                {
                    "id": case_id,
                    "repeat": rep + 1,
                    "domain": domain,
                    "domain_class": domain_class,
                    "weight": weight,
                    "ingest_errors": ingest_errors,
                    "result": eval_result,
                    "final_state": {
                        "success": bool(final_state.get("success")),
                        "fitness_score": float(final_state.get("fitness_score", 0.0) or 0.0),
                        "benchmark_score": float(final_state.get("benchmark_score", 0.0) or 0.0),
                        "benchmark_metric_name": str(final_state.get("benchmark_metric_name", "") or ""),
                        "iteration_count": int(final_state.get("iteration_count", 0) or 0),
                        "execution_result": execution_result or "",
                    },

                }
            )

    report = _build_report(
        suite_path=suite_path,
        started_at=started_at,
        repeats=repeats,
        case_results=case_results,
    )
    report["runner"] = "conversational_baseline"
    report["output_file"] = _write_report(output_dir=output_dir, file_prefix="benchmark_conv", report=report)
    return report




def compare_benchmark_reports(
    *,
    agent_report: Dict[str, Any],
    llm_report: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:

    agent_rows = agent_report.get("case_results", []) if isinstance(agent_report, dict) else []
    llm_rows = llm_report.get("case_results", []) if isinstance(llm_report, dict) else []
    agent_rows = agent_rows if isinstance(agent_rows, list) else []
    llm_rows = llm_rows if isinstance(llm_rows, list) else []

    llm_by_key: Dict[str, Dict[str, Any]] = {}
    for row in llm_rows:
        if not isinstance(row, dict):
            continue
        key = f"{row.get('id','')}::{row.get('repeat', 1)}"
        llm_by_key[key] = row

    per_case: List[Dict[str, Any]] = []
    domain_delta: Dict[str, List[float]] = {}
    agent_wins = 0
    llm_wins = 0
    ties = 0

    for arow in agent_rows:
        if not isinstance(arow, dict):
            continue
        key = f"{arow.get('id','')}::{arow.get('repeat', 1)}"
        lrow = llm_by_key.get(key)
        if not isinstance(lrow, dict):
            continue

        a_score = float(arow.get("result", {}).get("score", 0.0) if isinstance(arow.get("result"), dict) else 0.0)
        l_score = float(lrow.get("result", {}).get("score", 0.0) if isinstance(lrow.get("result"), dict) else 0.0)
        delta = round(a_score - l_score, 4)
        domain_class = str(arow.get("domain_class", "generic") or "generic")
        domain_delta.setdefault(domain_class, []).append(delta)

        if delta > 0:
            agent_wins += 1
        elif delta < 0:
            llm_wins += 1
        else:
            ties += 1

        per_case.append(
            {
                "id": arow.get("id"),
                "repeat": arow.get("repeat", 1),
                "domain_class": domain_class,
                "agent_score": a_score,
                "llm_score": l_score,
                "delta_agent_minus_llm": delta,
            }
        )

    domain_summary = {
        cls: {
            "count": len(values),
            "mean_delta_agent_minus_llm": round(sum(values) / max(1, len(values)), 4),
        }
        for cls, values in domain_delta.items()
    }

    started_at = datetime.now(UTC)
    report = {
        "run_id": started_at.strftime("%Y%m%dT%H%M%SZ"),
        "started_at_utc": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent_report_file": agent_report.get("output_file", ""),
        "llm_report_file": llm_report.get("output_file", ""),
        "agent_global_score": float(agent_report.get("global_score", 0.0) or 0.0),
        "llm_global_score": float(llm_report.get("global_score", 0.0) or 0.0),
        "global_delta_agent_minus_llm": round(
            float(agent_report.get("global_score", 0.0) or 0.0)
            - float(llm_report.get("global_score", 0.0) or 0.0),
            4,
        ),
        "agent_wins": agent_wins,
        "llm_wins": llm_wins,
        "ties": ties,
        "domain_delta_summary": domain_summary,
        "per_case": per_case,
    }
    report["output_file"] = _write_report(output_dir=output_dir, file_prefix="benchmark_compare", report=report)
    return report


def run_benchmark_suite(
    *,
    suite_path: str,
    output_dir: str,
    state_builder: Callable[..., AgentState],
    repeats: int,
    default_max_iterations: int,
    default_stop_threshold: float,
    default_patience: int,
    open_run_mode: bool = False,
) -> Dict[str, Any]:
    suite = load_benchmark_suite(suite_path)
    cases_raw = suite.get("cases", [])
    assert isinstance(cases_raw, list)

    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    case_results: List[Dict[str, Any]] = []

    for case_item in cases_raw:
        if not isinstance(case_item, dict):
            continue
        case: BenchmarkCase = case_item  # type: ignore[assignment]
        case_id = str(case.get("id", "")).strip() or f"case_{len(case_results)+1:03d}"
        domain = str(case.get("domain", "")).strip() or "Benchmark case"
        theme = str(case.get("theme", "")).strip()
        domain_class = str(case.get("domain_class", "")).strip().lower()
        weight = float(case.get("weight", 1.0) or 1.0)

        for rep in range(max(1, repeats)):
            input_files = case.get("input_files", [])
            input_files = input_files if isinstance(input_files, list) else []
            ingested_files, ingested_context, ingest_errors = ingest_files([str(v) for v in input_files])
            file_manifest = build_file_manifest(ingested_files)
            file_processing_plan = build_file_processing_plan(file_manifest)

            state: AgentState = state_builder(
                domain=domain,
                user_theme=theme,
                max_iterations=int(case.get("max_iterations", default_max_iterations) or default_max_iterations),
                stop_threshold=float(case.get("stop_threshold", default_stop_threshold) or default_stop_threshold),
                early_stop_patience=int(case.get("patience", default_patience) or default_patience),
                ingested_files=ingested_files,
                ingested_context=ingested_context,
                file_manifest=file_manifest,
                file_processing_plan=file_processing_plan,
                open_run_mode=bool(case.get("open_run_mode", False)) or bool(open_run_mode),
                enable_bash_tooling=bool(case.get("enable_bash_tooling", True)),
                enable_mcp_tools=bool(case.get("enable_mcp_tools", True)),
                enable_staged_evaluation=bool(case.get("enable_staged_evaluation", True)),
                staged_eval_ratio=float(case.get("staged_eval_ratio", 0.10) or 0.10),
                staged_eval_max_cases=int(case.get("staged_eval_max_cases", 10) or 10),
                staged_eval_threshold=float(case.get("staged_eval_threshold", 0.20) or 0.20),
                staged_eval_fail_scale=float(case.get("staged_eval_fail_scale", 0.35) or 0.35),
                benchmark_domain_class=domain_class,
                benchmark_case_id=case_id,
            )

            snapshots: List[Dict[str, Any]] = []
            for snapshot in graph_app.stream(state, stream_mode="values"):
                snapshots.append(dict(snapshot))
            final_state = snapshots[-1] if snapshots else {}
            eval_result = evaluate_benchmark_case(case, final_state)

            case_results.append(
                {
                    "id": case_id,
                    "repeat": rep + 1,
                    "domain": domain,
                    "domain_class": domain_class,
                    "weight": weight,
                    "ingest_errors": ingest_errors,
                    "result": eval_result,
                    "final_state": {
                        "success": bool(final_state.get("success")),
                        "fitness_score": float(final_state.get("fitness_score", 0.0) or 0.0),
                        "benchmark_score": float(final_state.get("benchmark_score", 0.0) or 0.0),
                        "benchmark_metric_name": str(final_state.get("benchmark_metric_name", "") or ""),
                        "iteration_count": int(final_state.get("iteration_count", 0) or 0),
                    },
                }
            )

    report = _build_report(
        suite_path=suite_path,
        started_at=started_at,
        repeats=repeats,
        case_results=case_results,
    )
    report["runner"] = "agent"
    report["output_file"] = _write_report(output_dir=output_dir, file_prefix="benchmark", report=report)
    return report
