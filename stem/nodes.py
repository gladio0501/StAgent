from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sandbox.executor import execute_python_code
from stem.archive import archive_variant
from stem.ingest import build_file_processing_plan
from stem.mcp_tools import run_planner_mcp_tools, run_research_mcp_tools
from stem.nodes_support import (
    TESTS_FILE,
    auto_install_missing_dependency,
    build_runtime_tool_handoff,
    compact_research_notes,
    compute_hyperagent_score,
    compute_domain_benchmark_score,
    compute_inter_agent_benchmarks,
    compute_legacy_fitness,
    deterministic_reflection,
    extract_json_object,
    extract_missing_module,
    fallback_architecture,
    fallback_code,
    fallback_requirements,
    fetch_wikipedia_summary,
    get_llm,
    inspect_file,
    search_web,
    ensure_selected_tool_dependencies,
    materialize_runtime_tools,
    strip_code_fences,
    validate_execution_output,
)
from stem.prompts import ARCHITECT_PROMPT, GENERATE_PROMPT, PLAN_PROMPT, REFLECT_PROMPT, RESEARCH_PROMPT
from stem.state import AgentState


def plan(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    domain = state["domain"]
    theme = state.get("user_theme", "")
    file_processing_plan = state.get("file_processing_plan", "")
    if not file_processing_plan.strip():
        file_processing_plan = build_file_processing_plan(state.get("file_manifest", ""))
    mcp_enabled = bool(state.get("enable_mcp_tools", True))
    enable_bash_tooling = bool(state.get("enable_bash_tooling", True))
    mcp = run_planner_mcp_tools(
        domain=domain,
        enable_bash_tooling=enable_bash_tooling,
        enabled=mcp_enabled,
    )
    mcp_observations = str(mcp.get("observations", ""))
    mcp_tools_used = [str(v) for v in mcp.get("tools_used", [])]
    runtime_tools = [str(v) for v in mcp.get("runtime_tools", [])]
    handoff_guidance = [str(v) for v in mcp.get("handoff_guidance", [])]
    planner_runtime_tool_handoff = build_runtime_tool_handoff(
        runtime_tools=runtime_tools,
        handoff_guidance=handoff_guidance,
        enable_bash_tooling=enable_bash_tooling,
    )

    if llm is None:
        requirements = fallback_requirements(domain)
        if theme:
            requirements += f"\nTheme constraints: {theme}"
        if mcp_observations:
            requirements += f"\nPlanner MCP observations:\n{mcp_observations[:3000]}"
        requirements += (
            "\nPlanner file boundary: do not inspect or depend on file contents. "
            "The specialized runtime agent will read files via STEM_INGESTED_FILES and "
            "STEM_FILE_PROCESSING_PLAN."
        )
        requirements += f"\n\n{planner_runtime_tool_handoff}"
        history = state["history"] + [HumanMessage(content=f"Plan for domain: {domain}")]
        return {
            "requirements": requirements,
            "file_processing_plan": file_processing_plan,
            "mcp_tools_used": mcp_tools_used,
            "mcp_tool_observations": mcp_observations,
            "planner_runtime_tool_handoff": planner_runtime_tool_handoff,
            "history": history,
        }

    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(
            content=(
                f"Domain: {domain}\n"
                f"Theme constraints: {theme or 'none'}\n"
                f"Planner MCP observations:\n{mcp_observations[:3000] or 'none'}\n"
                f"Runtime tool handoff to specialized agent:\n{planner_runtime_tool_handoff}\n"
                "Planning rule: do not inspect or rely on ingested files; only define generic "
                "runtime behavior and rely on specialized agent file processing at execution."
            )
        ),
    ]
    response = llm.invoke(messages)
    history = state["history"] + messages + [AIMessage(content=response.content)]
    return {
        "requirements": str(response.content),
        "file_processing_plan": file_processing_plan,
        "mcp_tools_used": mcp_tools_used,
        "mcp_tool_observations": mcp_observations,
        "planner_runtime_tool_handoff": planner_runtime_tool_handoff,
        "history": history,
    }


def research(state: AgentState) -> Dict[str, Any]:
    """Agentic loop to explore context, read files, and search the web."""
    domain = state["domain"]
    requirements = state["requirements"]
    MAX_RESEARCH_TOOL_CALLS = 6  # hard cap — prevents runaway loops & rate-limit burns

    llm = get_llm()
    if llm is None:
        # Fallback if no LLM
        notes = "No LLM available for dynamic research."
        combined = f"{requirements}\n\nResearch Signals:\n{notes}"
        return {"research_notes": notes, "requirements": combined, "history": state["history"] + [AIMessage(content=notes)]}

    # Check hard cap first — force summary if already at limit
    tool_calls_so_far = state.get("research_tool_calls", 0)
    if tool_calls_so_far >= MAX_RESEARCH_TOOL_CALLS:
        # Collect all tool results we have gathered so far and summarise
        from langchain_core.messages import ToolMessage
        tool_results = [m.content for m in state["history"] if isinstance(m, ToolMessage)]
        summary = "Max research iterations reached. Collected findings:\n" + "\n---\n".join(tool_results[-4:])
        compact = compact_research_notes(summary, max_chars=1400)
        combined = f"{requirements}\n\nResearch Signals (compact):\n{compact}".strip()
        return {
            "research_notes": compact,
            "research_sources": [],
            "requirements": combined,
            "history": state["history"] + [AIMessage(content=summary)],
            "research_tool_calls": tool_calls_so_far,
        }

    llm_with_tools = llm.bind_tools([inspect_file, search_web])

    ingested = "\n".join(state.get("ingested_files", []))
    prompt_context = f"{RESEARCH_PROMPT}\n\nDomain: {domain}\nIngested Files:\n{ingested}"

    # We dynamically prepend our system prompt to the accumulated history
    messages = [SystemMessage(content=prompt_context)] + state["history"]

    response = llm_with_tools.invoke(messages)
    new_history = state["history"] + [response]

    if hasattr(response, "tool_calls") and response.tool_calls:
        # Detect repeated identical queries — force stop if all pending tool calls
        # duplicate a query already present in history
        from langchain_core.messages import ToolMessage
        past_queries = set()
        for m in state["history"]:
            if hasattr(m, "tool_calls"):
                for tc in (m.tool_calls or []):
                    past_queries.add(str(tc.get("args", {})))
        new_queries = [str(tc.get("args", {})) for tc in response.tool_calls]
        all_duplicates = all(q in past_queries for q in new_queries)

        if all_duplicates:
            # All requested tool calls are identical to past ones — force termination
            summary = "Repeated queries detected; halting research loop to avoid rate-limit burn."
            compact = compact_research_notes(summary, max_chars=1400)
            combined = f"{requirements}\n\nResearch Signals (compact):\n{compact}".strip()
            return {
                "research_notes": compact,
                "research_sources": [],
                "requirements": combined,
                "history": state["history"] + [AIMessage(content=summary)],
                "research_tool_calls": tool_calls_so_far,
            }

        # LLM decided to use a tool; update history so the ToolNode can process it
        return {"history": new_history, "research_tool_calls": tool_calls_so_far + 1}
    else:
        # Research complete
        notes = str(response.content)
        compact_notes = compact_research_notes(notes, max_chars=1400)
        combined_requirements = f"{requirements}\n\nResearch Signals (compact):\n{compact_notes}".strip()

        return {
            "research_notes": compact_notes,
            "research_sources": [],
            "requirements": combined_requirements,
            "history": new_history,
            "research_tool_calls": tool_calls_so_far,
        }


def architect(state: AgentState) -> Dict[str, Any]:
    """Choose architecture/tooling/test strategy before code generation."""
    llm = get_llm()
    domain = state["domain"]
    requirements = state["requirements"]
    research_notes = state.get("research_notes", "")
    user_theme = state.get("user_theme", "")

    if llm is None:
        arch = fallback_architecture(domain)
        history = state["history"] + [AIMessage(content=f"Architecture fallback: {arch}")]
        return {
            "architecture_decision": arch["architecture_decision"],
            "selected_tools": arch["selected_tools"],
            "architect_runtime_tools": [str(v) for v in arch.get("runtime_tooling", [])],
            "test_strategy": arch["test_strategy"],
            "stop_threshold": float(arch["stop_threshold"]),
            "history": history,
        }

    prompt = (
        f"Domain:\n{domain}\n\n"
        f"Requirements:\n{requirements}\n\n"
        f"Research signals:\n{research_notes}\n\n"
        f"Theme constraints:\n{user_theme or 'none'}\n"
    )
    messages = [SystemMessage(content=ARCHITECT_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    parsed = extract_json_object(str(response.content))
    fallback = fallback_architecture(domain)

    architecture_decision = str(parsed.get("architecture_decision") or fallback["architecture_decision"])
    selected_tools = parsed.get("selected_tools")
    if not isinstance(selected_tools, list) or not selected_tools:
        selected_tools = fallback["selected_tools"]
    selected_tools = [str(tool) for tool in selected_tools]

    runtime_tooling = parsed.get("runtime_tooling")
    if not isinstance(runtime_tooling, list) or not runtime_tooling:
        runtime_tooling = fallback.get("runtime_tooling", [])
    architect_runtime_tools = [str(tool) for tool in runtime_tooling]

    test_strategy = str(parsed.get("test_strategy") or fallback["test_strategy"])
    try:
        stop_threshold = float(parsed.get("stop_threshold", fallback["stop_threshold"]))
    except (TypeError, ValueError):
        stop_threshold = float(fallback["stop_threshold"])
    stop_threshold = min(0.99, max(0.70, stop_threshold))

    history = state["history"] + messages + [AIMessage(content=str(response.content))]
    return {
        "architecture_decision": architecture_decision,
        "selected_tools": selected_tools,
        "architect_runtime_tools": architect_runtime_tools,
        "test_strategy": test_strategy,
        "stop_threshold": stop_threshold,
        "history": history,
    }


def generate_code(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    domain = state["domain"]
    requirements = state["requirements"]
    error_trace = state.get("error_trace")
    architecture_decision = state.get("architecture_decision", "")
    selected_tools = state.get("selected_tools", [])
    test_strategy = state.get("test_strategy", "")
    user_theme = state.get("user_theme", "")
    planner_runtime_tool_handoff = state.get("planner_runtime_tool_handoff", "")
    enable_bash_tooling = bool(state.get("enable_bash_tooling", True))
    runtime_tools = materialize_runtime_tools(
        selected_tools=[str(v) for v in selected_tools],
        architect_runtime_tools=[str(v) for v in state.get("architect_runtime_tools", [])],
        enable_bash_tooling=enable_bash_tooling,
    )

    if error_trace and (
        "Execution JSON missing required keys" in error_trace
        or "Execution JSON has non-ok status" in error_trace
        or "Error tokenizing data" in error_trace
        or "not valid JSON" in error_trace
    ):
        script = fallback_code(domain, requirements, error_trace)
        archive_path = archive_variant(
            domain=domain,
            iteration_count=state["iteration_count"],
            stage="generated_fallback",
            payload={"code": script, "reason": error_trace},
            score=0.35,
        )
        history = state["history"] + [
            HumanMessage(content="Switched to deterministic JSON-contract fallback code")
        ]
        return {"current_code": script, "history": history, "best_variant_path": archive_path}

    if llm is None:
        script = fallback_code(domain, requirements, error_trace)
        archive_path = archive_variant(
            domain=domain,
            iteration_count=state["iteration_count"],
            stage="generated_no_llm",
            payload={"code": script, "reason": "llm_unavailable"},
            score=0.3,
        )
        history = state["history"] + [HumanMessage(content="Generated code with fallback path")]
        return {"current_code": script, "history": history, "best_variant_path": archive_path}

    error_block = error_trace or "None"
    prompt = (
        f"Domain:\n{domain}\n\n"
        f"Requirements:\n{requirements}\n\n"
        f"Theme constraints:\n{user_theme or 'none'}\n\n"
        f"Architecture decision:\n{architecture_decision}\n\n"
        f"Selected tools:\n{', '.join(selected_tools)}\n\n"
        f"Planner-to-specialized runtime tool handoff:\n{planner_runtime_tool_handoff or 'none'}\n\n"
        f"Runtime helper tools available:\n{', '.join(runtime_tools) or 'none'}\n\n"
        f"Test strategy:\n{test_strategy}\n\n"
        f"Previous error trace:\n{error_block}\n\n"
        "Return Python code only."
    )
    messages = [SystemMessage(content=GENERATE_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    history = state["history"] + messages + [AIMessage(content=response.content)]
    cleaned_code = strip_code_fences(str(response.content))
    archive_path = archive_variant(
        domain=domain,
        iteration_count=state["iteration_count"],
        stage="generated_llm",
        payload={"code": cleaned_code, "error_context": error_block},
        score=0.6,
    )

    return {"current_code": cleaned_code, "history": history, "best_variant_path": archive_path}


def execute_code(state: AgentState) -> Dict[str, Any]:
    code = strip_code_fences(state.get("current_code") or "")
    ingested_files = state.get("ingested_files", [])
    file_processing_plan = state.get("file_processing_plan", "")
    enforce_output_contract = bool(state.get("enforce_output_contract", True))
    enforce_sandbox_policy = bool(state.get("enforce_sandbox_policy", True))
    enforce_resource_limits = bool(state.get("enforce_resource_limits", True))
    isolated_execution = bool(state.get("isolated_execution", True))
    enable_bash_tooling = bool(state.get("enable_bash_tooling", True))
    runtime_tools = materialize_runtime_tools(
        selected_tools=[str(v) for v in state.get("selected_tools", [])],
        architect_runtime_tools=[str(v) for v in state.get("architect_runtime_tools", [])],
        enable_bash_tooling=enable_bash_tooling,
    )
    runtime_env = {
        "STEM_INGESTED_FILES": json.dumps(ingested_files),
        "STEM_FILE_PROCESSING_PLAN": file_processing_plan,
        "STEM_RUNTIME_TOOLS": json.dumps(runtime_tools),
    }

    architect_installed, architect_install_failures = ensure_selected_tool_dependencies(
        [str(v) for v in state.get("selected_tools", [])]
    )
    architect_install_note = ""
    if architect_installed:
        architect_install_note += (
            "Architect tool dependencies installed/verified: "
            + ", ".join(architect_installed)
            + "."
        )
    if architect_install_failures:
        failure_block = " ".join(architect_install_failures)
        architect_install_note += (
            (" " if architect_install_note else "")
            + "Architect dependency install failures: "
            + failure_block
        )

    enable_staged_evaluation = bool(state.get("enable_staged_evaluation", True))
    staged_eval_ratio = float(state.get("staged_eval_ratio", 0.10) or 0.10)
    staged_eval_max_cases = int(state.get("staged_eval_max_cases", 10) or 10)
    staged_eval_threshold = float(state.get("staged_eval_threshold", 0.20) or 0.20)
    staged_eval_fail_scale = float(state.get("staged_eval_fail_scale", 0.35) or 0.35)

    staged_eval_passed = True
    staged_eval_cases = 0
    benchmark_total_cases = 0
    benchmark_metric_name = "generic_success"
    benchmark_score = 0.0
    staged_filter_note = ""
    short_circuited = False
    staged_tests_path: str | None = None

    result: Dict[str, Any] = {"success": False, "execution_result": None, "error_trace": "Evaluation not started."}
    duration = 0.0

    def _execute_with_recoveries(exec_code: str, t_file: str | None, t_sec: int) -> tuple[str, dict, str, float]:
        inst_note = ""
        inst_time = 0.0
        t0 = time.perf_counter()
        res = execute_python_code(
            exec_code,
            tests_file=t_file,
            timeout_seconds=t_sec,
            extra_env=runtime_env,
            enforce_policy=enforce_sandbox_policy,
            enforce_resource_limits=enforce_resource_limits,
            isolated_execution=isolated_execution,
            enable_bash_tooling=enable_bash_tooling,
            runtime_tools=runtime_tools,
        )

        syn_err = str(res.get("error_trace") or "")
        if not bool(res.get("success")) and "SyntaxError" in syn_err:
            cl_code = strip_code_fences(exec_code)
            if cl_code != exec_code:
                exec_code = cl_code
                res = execute_python_code(
                    exec_code,
                    tests_file=t_file,
                    timeout_seconds=t_sec,
                    extra_env=runtime_env,
                    enforce_policy=enforce_sandbox_policy,
                    enforce_resource_limits=enforce_resource_limits,
                    isolated_execution=isolated_execution,
                    enable_bash_tooling=enable_bash_tooling,
                    runtime_tools=runtime_tools,
                )

        missing_mod = extract_missing_module(str(res.get("error_trace") or ""))
        if missing_mod:
            t_inst = time.perf_counter()
            installed, inst_note = auto_install_missing_dependency(missing_mod)
            inst_time = time.perf_counter() - t_inst
            if installed:
                res = execute_python_code(
                    exec_code,
                    tests_file=t_file,
                    timeout_seconds=t_sec,
                    extra_env=runtime_env,
                    enforce_policy=enforce_sandbox_policy,
                    enforce_resource_limits=enforce_resource_limits,
                    isolated_execution=isolated_execution,
                    enable_bash_tooling=enable_bash_tooling,
                    runtime_tools=runtime_tools,
                )

        exec_dur = round((time.perf_counter() - t0) - inst_time, 4)
        return exec_code, res, inst_note, exec_dur

    install_note = ""
    if enable_staged_evaluation and TESTS_FILE.exists():
        try:
            tests_payload = json.loads(TESTS_FILE.read_text(encoding="utf-8"))
            cases = tests_payload.get("cases", []) if isinstance(tests_payload, dict) else []
            if isinstance(cases, list) and cases:
                benchmark_total_cases = len(cases)
                ratio_count = max(1, int(math.ceil(benchmark_total_cases * max(0.01, staged_eval_ratio))))
                staged_eval_cases = min(benchmark_total_cases, max(1, staged_eval_max_cases), ratio_count)

                if staged_eval_cases < benchmark_total_cases:
                    subset_payload = {"cases": cases[:staged_eval_cases]}
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix="_staged_tests.json", delete=False, encoding="utf-8"
                    ) as staged_file:
                        json.dump(subset_payload, staged_file)
                        staged_tests_path = staged_file.name
                else:
                    staged_tests_path = str(TESTS_FILE)

                code, staged_result, staged_inst_note, staged_duration = _execute_with_recoveries(
                    code, staged_tests_path, 12
                )
                if staged_inst_note:
                    install_note = staged_inst_note
                staged_success = bool(staged_result.get("success"))
                staged_execution_raw = staged_result.get("execution_result")
                staged_execution = staged_execution_raw if isinstance(staged_execution_raw, str) else None

                # --- Contract-validity check for staged eval -----------------
                # `staged_success` only means the Python process exited cleanly.
                # A script can crash gracefully and still print {"status": "error"}.
                # We must check the payload's own status field and require that
                # at least one case was actually graded before awarding credit.
                staged_payload: dict = {}
                if staged_execution:
                    try:
                        staged_payload = json.loads(staged_execution)
                        if not isinstance(staged_payload, dict):
                            staged_payload = {}
                    except json.JSONDecodeError:
                        staged_payload = {}
                staged_contract_ok = (
                    str(staged_payload.get("status", "")).strip().lower() == "ok"
                    and isinstance(staged_payload.get("test_case_count"), int)
                    and staged_payload["test_case_count"] > 0
                )

                metric_name, metric_score = compute_domain_benchmark_score(
                    domain=state["domain"],
                    execution_result=staged_execution,
                    success=staged_success,
                    domain_class=str(state.get("benchmark_domain_class", "") or ""),
                )
                benchmark_metric_name = metric_name
                benchmark_score = metric_score
                # Pass only when the payload is contract-valid AND the metric
                # meets the threshold.  Process success alone is not enough.
                staged_eval_passed = staged_contract_ok and metric_score >= staged_eval_threshold

                if not staged_eval_passed:
                    short_circuited = True
                    scaled_score = max(0.0, min(1.0, metric_score * max(0.0, staged_eval_fail_scale)))
                    benchmark_score = scaled_score
                    benchmark_metric_name = f"{metric_name}_staged_scaled"
                    staged_filter_note = (
                        "Staged evaluation filter pruned candidate: "
                        f"{metric_name}={metric_score:.4f}, threshold={staged_eval_threshold:.4f}, "
                        f"scaled={scaled_score:.4f}, cases={staged_eval_cases}/{benchmark_total_cases}."
                    )
                    result = staged_result
                    duration = staged_duration
                else:
                    staged_filter_note = (
                        "Staged evaluation passed: "
                        f"{metric_name}={metric_score:.4f}, cases={staged_eval_cases}/{benchmark_total_cases}."
                    )
        except Exception as exc:
            staged_filter_note = f"Staged evaluation skipped due to setup error: {exc}"
        finally:
            if staged_tests_path and staged_tests_path != str(TESTS_FILE):
                try:
                    Path(staged_tests_path).unlink(missing_ok=True)
                except Exception:
                    pass

    if not short_circuited:
        code, result, full_inst_note, duration = _execute_with_recoveries(code, str(TESTS_FILE), 30)
        if full_inst_note and not install_note:
            install_note = full_inst_note

    success = bool(result["success"])
    execution_result_raw = result.get("execution_result")
    error_trace_raw = result.get("error_trace")
    execution_result = execution_result_raw if isinstance(execution_result_raw, str) else None
    error_trace = error_trace_raw if isinstance(error_trace_raw, str) else None
    # Preserve execution_result as strict JSON output; contract validation and benchmark
    # scoring parse this field directly and fail on any non-JSON suffix text.
    if install_note:
        error_trace = f"{error_trace}\n{install_note}" if error_trace else install_note
    if architect_install_note:
        error_trace = (
            f"{error_trace}\n{architect_install_note}" if error_trace else architect_install_note
        )
    if staged_filter_note:
        error_trace = f"{error_trace}\n{staged_filter_note}" if error_trace else staged_filter_note

    if execution_result:
        try:
            _payload = json.loads(execution_result.strip())
            if isinstance(_payload, dict) and str(_payload.get("status", "")).strip().lower() == "error":
                _msg = _payload.get("message", "unknown error")
                err_msg = f"Logical Script Error (from JSON payload): {_msg}"
                error_trace = f"{error_trace}\n{err_msg}" if error_trace else err_msg
        except Exception:
            pass

    if success and enforce_output_contract:
        valid, validation_error = validate_execution_output(execution_result)
        if not valid:
            success = False
            error_trace = validation_error

    if not short_circuited:
        metric_name, metric_score = compute_domain_benchmark_score(
            domain=state["domain"],
            execution_result=execution_result,
            success=success,
            domain_class=str(state.get("benchmark_domain_class", "") or ""),
        )
        benchmark_metric_name = metric_name
        benchmark_score = metric_score

    if benchmark_total_cases <= 0 and TESTS_FILE.exists():
        try:
            tests_payload = json.loads(TESTS_FILE.read_text(encoding="utf-8"))
            cases = tests_payload.get("cases", []) if isinstance(tests_payload, dict) else []
            benchmark_total_cases = len(cases) if isinstance(cases, list) else 0
        except Exception:
            benchmark_total_cases = 0
    if staged_eval_cases <= 0:
        staged_eval_cases = benchmark_total_cases

    baseline_metrics, baseline_score = compute_legacy_fitness(
        success=success,
        execution_result=execution_result,
        error_trace=error_trace,
        domain=state["domain"],
        iteration_count=state["iteration_count"] + 1,
        max_iterations=state["max_iterations"],
    )

    contract_valid = bool(baseline_metrics.get("contract_valid", 0.0) >= 1.0)
    domain_alignment = float(baseline_metrics.get("domain_alignment", 0.0))
    robustness = float(baseline_metrics.get("robustness", 0.0))
    previous_best = float(state.get("best_fitness_score", 0.0))
    hyper_metrics, hyper_score = compute_hyperagent_score(
        success=success,
        contract_valid=contract_valid,
        domain_alignment=domain_alignment,
        robustness=robustness,
        iteration_count=state["iteration_count"] + 1,
        max_iterations=state["max_iterations"],
        execution_duration_seconds=duration,
        previous_best_fitness=previous_best,
        benchmark_score=benchmark_score,
    )

    improved = hyper_score > previous_best
    best_fitness_score = max(previous_best, hyper_score)
    no_improvement_count = 0 if improved else int(state.get("no_improvement_count", 0)) + 1
    before_after_metrics = {
        "before_legacy_fitness": baseline_score,
        "after_hyperagent_fitness": hyper_score,
        "delta": round(hyper_score - baseline_score, 4),
        "execution_duration_seconds": duration,
        "benchmark_score": benchmark_score,
    }

    merged_metrics: Dict[str, float] = {}
    merged_metrics.update(baseline_metrics)
    merged_metrics.update({f"hyper_{k}": v for k, v in hyper_metrics.items()})
    merged_metrics.update(
        {
            "benchmark_score": round(benchmark_score, 4),
            "staged_eval_passed": 1.0 if staged_eval_passed else 0.0,
            "staged_eval_cases": float(staged_eval_cases),
            "benchmark_total_cases": float(benchmark_total_cases),
        }
    )

    inter_agent_benchmarks = compute_inter_agent_benchmarks(
        requirements=state.get("requirements", ""),
        research_notes=state.get("research_notes", ""),
        research_sources=state.get("research_sources", []),
        architecture_decision=state.get("architecture_decision", ""),
        selected_tools=state.get("selected_tools", []),
        test_strategy=state.get("test_strategy", ""),
        planner_runtime_tool_handoff=state.get("planner_runtime_tool_handoff", ""),
        code=code,
        execution_success=success,
        execution_duration_seconds=duration,
        hyper_score=hyper_score,
    )
    merged_metrics.update({f"bench_{k}": v for k, v in inter_agent_benchmarks.items()})

    merged_metrics.update(before_after_metrics)
    before_after_metrics.update(
        {
            "inter_agent_upstream_mean": inter_agent_benchmarks.get("upstream_mean", 0.0),
            "inter_agent_runtime_mean": inter_agent_benchmarks.get("runtime_mean", 0.0),
            "inter_agent_handoff_effectiveness": inter_agent_benchmarks.get("handoff_effectiveness", 0.0),
        }
    )

    archive_path = archive_variant(
        domain=state["domain"],
        iteration_count=state["iteration_count"] + 1,
        stage="executed",
        payload={
            "success": success,
            "execution_result": execution_result,
            "error_trace": error_trace,
            "fitness_metrics": merged_metrics,
            "fitness_score": hyper_score,
            "before_after_metrics": before_after_metrics,
            "best_fitness_score": best_fitness_score,
            "no_improvement_count": no_improvement_count,
            "benchmark_metric_name": benchmark_metric_name,
            "benchmark_score": benchmark_score,
            "staged_eval_passed": staged_eval_passed,
            "staged_eval_cases": staged_eval_cases,
            "benchmark_total_cases": benchmark_total_cases,
            "research_notes": state.get("research_notes", ""),
            "research_sources": state.get("research_sources", []),
            "architecture_decision": state.get("architecture_decision", ""),
            "selected_tools": state.get("selected_tools", []),
            "architect_runtime_tools": state.get("architect_runtime_tools", []),
            "test_strategy": state.get("test_strategy", ""),
            "planner_runtime_tool_handoff": state.get("planner_runtime_tool_handoff", ""),
            "inter_agent_benchmarks": inter_agent_benchmarks,
        },
        score=hyper_score,
    )

    return {
        "success": success,
        "execution_result": execution_result,
        "error_trace": error_trace,
        "fitness_metrics": merged_metrics,
        "fitness_score": hyper_score,
        "best_fitness_score": best_fitness_score,
        "no_improvement_count": no_improvement_count,
        "before_after_metrics": before_after_metrics,
        "inter_agent_benchmarks": inter_agent_benchmarks,
        "benchmark_metric_name": benchmark_metric_name,
        "benchmark_score": benchmark_score,
        "staged_eval_passed": staged_eval_passed,
        "staged_eval_cases": staged_eval_cases,
        "benchmark_total_cases": benchmark_total_cases,
        "iteration_count": state["iteration_count"] + 1,
        "best_variant_path": archive_path if success else state.get("best_variant_path"),
    }


def reflect(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    error_trace = state.get("error_trace") or "No trace provided"

    deterministic = deterministic_reflection(error_trace)
    if deterministic:
        history = state["history"] + [AIMessage(content=deterministic)]
        return {"history": history}

    if llm is None:
        critique = (
            "Root cause: runtime/script issue detected. "
            "Fix strategy: adjust generated script to handle reported failure path. "
            "Risk: regressions in argument parsing and output formatting."
        )
        history = state["history"] + [AIMessage(content=critique)]
        return {"history": history}

    messages = [
        SystemMessage(content=REFLECT_PROMPT),
        HumanMessage(content=f"Error trace:\n{error_trace}"),
    ]
    response = llm.invoke(messages)
    history = state["history"] + messages + [AIMessage(content=response.content)]
    return {"history": history}
