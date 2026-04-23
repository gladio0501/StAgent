from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sandbox.executor import execute_python_code
from stem.graph import app as graph_app
from stem.ingest import build_file_manifest, build_file_processing_plan, ingest_files, save_uploaded_files
from stem.nodes_support import materialize_runtime_tools
from stem.state import AgentState


ARCHIVE_DIR = Path(__file__).resolve().parent / "archive"


def _build_initial_state(
    *,
    domain: str,
    user_theme: str,
    max_iterations: int,
    stop_threshold: float,
    early_stop_patience: int,
    ingested_files: Optional[List[str]] = None,
    ingested_context: str = "",
    file_manifest: str = "",
    file_processing_plan: str = "",
    open_run_mode: bool = False,
    enable_bash_tooling: bool = True,
    enable_mcp_tools: bool = True,
    enable_staged_evaluation: bool = True,
    staged_eval_ratio: float = 0.10,
    staged_eval_max_cases: int = 10,
    staged_eval_threshold: float = 0.20,
    staged_eval_fail_scale: float = 0.35,
    benchmark_domain_class: str = "",
    benchmark_case_id: str = "",
) -> AgentState:
    return {
        "domain": domain,
        "user_theme": user_theme,
        "ingested_files": ingested_files or [],
        "ingested_context": ingested_context,
        "file_manifest": file_manifest,
        "file_processing_plan": file_processing_plan,
        "open_run_mode": open_run_mode,
        "enable_bash_tooling": enable_bash_tooling,
        "enable_mcp_tools": enable_mcp_tools,
        "mcp_tools_used": [],
        "mcp_tool_observations": "",
        "planner_runtime_tool_handoff": "",
        "enforce_output_contract": not open_run_mode,
        "enforce_sandbox_policy": not open_run_mode,
        "enforce_resource_limits": not open_run_mode,
        "isolated_execution": not open_run_mode,
        "enable_staged_evaluation": enable_staged_evaluation,
        "staged_eval_ratio": max(0.01, min(1.0, staged_eval_ratio)),
        "staged_eval_max_cases": max(1, staged_eval_max_cases),
        "staged_eval_threshold": max(0.0, min(1.0, staged_eval_threshold)),
        "staged_eval_fail_scale": max(0.0, min(1.0, staged_eval_fail_scale)),
        "requirements": "",
        "research_notes": "",
        "research_sources": [],
        "architecture_decision": "",
        "selected_tools": [],
        "architect_runtime_tools": [],
        "test_strategy": "",
        "stop_threshold": stop_threshold,
        "current_code": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "execution_result": None,
        "error_trace": None,
        "fitness_metrics": {},
        "fitness_score": 0.0,
        "best_fitness_score": 0.0,
        "no_improvement_count": 0,
        "early_stop_patience": early_stop_patience,
        "before_after_metrics": {},
        "inter_agent_benchmarks": {},
        "benchmark_metric_name": "",
        "benchmark_score": 0.0,
        "benchmark_domain_class": benchmark_domain_class,
        "benchmark_case_id": benchmark_case_id,
        "staged_eval_passed": True,
        "staged_eval_cases": 0,
        "benchmark_total_cases": 0,
        "history": [HumanMessage(content=f"Start domain: {domain}")],
        "best_variant_path": None,
        "success": False,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items() if k != "history"}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except Exception:
        return None


def _load_archive_records(*, domain: str, started_at: datetime) -> List[Dict[str, Any]]:
    if not ARCHIVE_DIR.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for path in ARCHIVE_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        record_domain = str(payload.get("domain", ""))
        if record_domain != domain:
            continue

        ts_raw = str(payload.get("timestamp_utc", ""))
        ts = _parse_timestamp(ts_raw)
        if ts is not None and ts < started_at:
            continue

        payload["_file"] = str(path)
        rows.append(payload)

    rows.sort(key=lambda r: (int(r.get("iteration_count", 0)), str(r.get("timestamp_utc", "")), str(r.get("stage", ""))))
    return rows


def _get_chat_llm() -> Optional[Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
    except Exception:
        return None


def _run_agent(
    *,
    domain: str,
    theme: str,
    max_iterations: int,
    stop_threshold: float,
    patience: int,
    input_paths: Optional[List[str]] = None,
    uploaded_files: Optional[List[Any]] = None,
    open_run_mode: bool = False,
    enable_bash_tooling: bool = True,
    enable_mcp_tools: bool = True,
    enable_staged_evaluation: bool = True,
    staged_eval_ratio: float = 0.10,
    staged_eval_max_cases: int = 10,
    staged_eval_threshold: float = 0.20,
    staged_eval_fail_scale: float = 0.35,
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    collected_paths: List[str] = []
    if input_paths:
        collected_paths.extend([str(v) for v in input_paths if str(v).strip()])

    uploaded_files = uploaded_files or []
    if uploaded_files:
        ingest_dir = Path(__file__).resolve().parent / "ingested_inputs"
        uploaded_paths = save_uploaded_files(ingest_dir, uploaded_files)
        collected_paths.extend(uploaded_paths)

    ingested_files, ingested_context, _ = ingest_files(collected_paths)
    file_manifest = build_file_manifest(ingested_files)
    file_processing_plan = build_file_processing_plan(file_manifest)

    state = _build_initial_state(
        domain=domain,
        user_theme=theme,
        max_iterations=max(1, max_iterations),
        stop_threshold=min(0.99, max(0.70, stop_threshold)),
        early_stop_patience=max(1, patience),
        ingested_files=ingested_files,
        ingested_context=ingested_context,
        file_manifest=file_manifest,
        file_processing_plan=file_processing_plan,
        open_run_mode=open_run_mode,
        enable_bash_tooling=enable_bash_tooling,
        enable_mcp_tools=enable_mcp_tools,
        enable_staged_evaluation=enable_staged_evaluation,
        staged_eval_ratio=staged_eval_ratio,
        staged_eval_max_cases=staged_eval_max_cases,
        staged_eval_threshold=staged_eval_threshold,
        staged_eval_fail_scale=staged_eval_fail_scale,
    )

    started_at = datetime.now(UTC)
    snapshots: List[Dict[str, Any]] = []
    for snapshot in graph_app.stream(state, stream_mode="values"):
        snapshots.append(dict(snapshot))

    final_state = snapshots[-1] if snapshots else None
    archive_records = _load_archive_records(domain=domain, started_at=started_at)
    return snapshots, final_state, archive_records


def _render_process_timeline(snapshots: List[Dict[str, Any]]) -> None:
    if not snapshots:
        st.info("No snapshots captured for this run.")
        return

    timeline: List[Dict[str, Any]] = []
    for i, snap in enumerate(snapshots):
        timeline.append(
            {
                "step": i,
                "iteration": snap.get("iteration_count"),
                "success": snap.get("success"),
                "fitness_score": snap.get("fitness_score"),
                "best_fitness_score": snap.get("best_fitness_score"),
                "no_improvement_count": snap.get("no_improvement_count"),
                "error": snap.get("error_trace"),
            }
        )

    st.dataframe(timeline, use_container_width=True)


def _render_metrics(final_state: Dict[str, Any]) -> None:
    before_after = final_state.get("before_after_metrics", {}) or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hyper Fitness", f"{float(final_state.get('fitness_score', 0.0)):.4f}")
    c2.metric("Best Fitness", f"{float(final_state.get('best_fitness_score', 0.0)):.4f}")
    c3.metric("Legacy Fitness", f"{float(before_after.get('before_legacy_fitness', 0.0)):.4f}")
    c4.metric("Delta", f"{float(before_after.get('delta', 0.0)):.4f}")
    c5.metric(
        "Benchmark",
        f"{float(final_state.get('benchmark_score', 0.0)):.4f}",
        help=str(final_state.get("benchmark_metric_name", "")),
    )

    st.caption(
        "Staged evaluation: "
        f"passed={bool(final_state.get('staged_eval_passed', True))}, "
        f"cases={int(final_state.get('staged_eval_cases', 0))}/"
        f"{int(final_state.get('benchmark_total_cases', 0))}, "
        f"metric={str(final_state.get('benchmark_metric_name', '')) or 'n/a'}"
    )

    st.subheader("All Metrics")
    st.json(_to_jsonable(final_state.get("fitness_metrics", {})))

    st.subheader("Before / After")
    st.json(_to_jsonable(before_after))

    inter_agent = final_state.get("inter_agent_benchmarks", {}) or {}
    if inter_agent:
        st.subheader("Inter-Agent Benchmarks")
        st.json(_to_jsonable(inter_agent))

    execution_result = final_state.get("execution_result")
    if isinstance(execution_result, str) and execution_result.strip():
        try:
            payload = json.loads(execution_result)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, dict) and "report" in payload:
            st.subheader("Runtime File Report")
            report = payload.get("report")
            if isinstance(report, list):
                st.dataframe(report, use_container_width=True)
            else:
                st.json(_to_jsonable(report))


def _render_archive(records: List[Dict[str, Any]]) -> None:
    st.subheader("Generation and Execution Artifacts")
    if not records:
        st.info("No archive records found for this run yet.")
        return

    summary = [
        {
            "timestamp_utc": row.get("timestamp_utc"),
            "iteration": row.get("iteration_count"),
            "stage": row.get("stage"),
            "score": row.get("score"),
            "file": row.get("_file"),
        }
        for row in records
    ]
    st.dataframe(summary, use_container_width=True)

    for row in records:
        stage = row.get("stage", "unknown")
        iteration = row.get("iteration_count", "?")
        label = f"iter {iteration} - {stage}"
        with st.expander(label):
            st.caption(str(row.get("_file", "")))
            payload = row.get("payload", {})
            code = payload.get("code") if isinstance(payload, dict) else None
            if isinstance(code, str) and code.strip():
                st.code(code, language="python")

            st.json(_to_jsonable(row))


def _render_final_state(final_state: Dict[str, Any]) -> None:
    st.subheader("Final Agent State")
    slim = dict(final_state)
    if "history" in slim:
        slim["history_count"] = len(slim.get("history", []))
        del slim["history"]
    st.json(_to_jsonable(slim))

    ingested_files = final_state.get("ingested_files", [])
    if ingested_files:
        st.subheader("Ingested Files")
        st.write(ingested_files)

    ingested_context = str(final_state.get("ingested_context", ""))
    if ingested_context.strip():
        st.subheader("Ingested Context Preview")
        st.code(ingested_context[:4000], language="text")

    file_manifest = str(final_state.get("file_manifest", "")).strip()
    if file_manifest:
        st.subheader("File Manifest (Runtime Input)")
        st.code(file_manifest[:4000], language="json")

    file_processing_plan = str(final_state.get("file_processing_plan", "")).strip()
    if file_processing_plan:
        st.subheader("File Processing Plan (Runtime Execution Plan)")
        st.code(file_processing_plan, language="text")

    mcp_tools_used = final_state.get("mcp_tools_used", [])
    if mcp_tools_used:
        st.subheader("Planner MCP Tools Used")
        st.write(mcp_tools_used)

    mcp_tool_observations = str(final_state.get("mcp_tool_observations", "")).strip()
    if mcp_tool_observations:
        st.subheader("Planner MCP Observations")
        st.code(mcp_tool_observations, language="text")

    planner_runtime_tool_handoff = str(final_state.get("planner_runtime_tool_handoff", "")).strip()
    if planner_runtime_tool_handoff:
        st.subheader("Planner to Specialized Tool Handoff")
        st.code(planner_runtime_tool_handoff, language="text")


def _run_specialized_runtime_code(
    *,
    code: str,
    state: Dict[str, Any],
    chat_file_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute user-provided snippet inside the specialized runtime sandbox."""
    merged_paths: List[str] = []
    state_paths = state.get("ingested_files", [])
    if isinstance(state_paths, list):
        merged_paths.extend([str(v) for v in state_paths if str(v).strip()])
    if chat_file_paths:
        merged_paths.extend([str(v) for v in chat_file_paths if str(v).strip()])
    # Preserve order while removing duplicates.
    merged_paths = list(dict.fromkeys(merged_paths))
    runtime_tools = materialize_runtime_tools(
        selected_tools=[str(v) for v in state.get("selected_tools", [])],
        architect_runtime_tools=[str(v) for v in state.get("architect_runtime_tools", [])],
        enable_bash_tooling=bool(state.get("enable_bash_tooling", True)),
    )

    runtime_env = {
        "STEM_INGESTED_FILES": json.dumps(merged_paths),
        "STEM_FILE_PROCESSING_PLAN": str(state.get("file_processing_plan", "") or ""),
        "STEM_RUNTIME_TOOLS": json.dumps(runtime_tools),
    }
    return execute_python_code(
        code,
        tests_file=None,
        timeout_seconds=30,
        extra_env=runtime_env,
        enforce_policy=bool(state.get("enforce_sandbox_policy", True)),
        enforce_resource_limits=bool(state.get("enforce_resource_limits", True)),
        isolated_execution=bool(state.get("isolated_execution", True)),
        enable_bash_tooling=bool(state.get("enable_bash_tooling", True)),
        runtime_tools=runtime_tools,
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    content = (text or "").strip()
    if not content:
        return {}

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, flags=re.IGNORECASE)
    if block:
        candidate = block.group(1).strip()
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

    obj_match = re.search(r"\{[\s\S]*\}", content)
    if not obj_match:
        return {}
    try:
        parsed = json.loads(obj_match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _run_continuous_specialized_task(
    *,
    task: str,
    llm: Any,
    state: Dict[str, Any],
    chat_file_paths: Optional[List[str]] = None,
    max_steps: int = 8,
) -> str:
    """Autonomous loop: choose tool actions and execute until final answer."""
    tool_names = materialize_runtime_tools(
        selected_tools=[str(v) for v in state.get("selected_tools", [])],
        architect_runtime_tools=[str(v) for v in state.get("architect_runtime_tools", [])],
        enable_bash_tooling=bool(state.get("enable_bash_tooling", True)),
    )
    direct_helper_actions = {
        name for name in tool_names if name in {"scrape_url", "read_text_file", "read_json_file", "read_csv_preview"}
    }
    steps: List[Dict[str, Any]] = []
    working_script = ""

    system = SystemMessage(
        content=(
            "You are the final specialized Stem Agent operating in autonomous loop mode. "
            "On each turn, return JSON only with one action.\n"
            f"Allowed action values: {', '.join(tool_names + ['final'])}.\n"
            "Schema:\n"
            "{\n"
            '  "action": "one allowed action value",\n'
            '  "reason": "short reason",\n'
            '  "message": "optional user-facing progress update",\n'
            '  "args": {"named": "arguments for direct helper actions"},\n'
            '  "code": "python code when action=run_python",\n'
            '  "script_mode": "replace|incremental" (optional, only for run_python),\n'
            '  "command": "bash command when action=run_bash",\n'
            '  "final": "final response when action=final"\n'
            "}\n"
            "Rules:\n"
            "- Use tools only when needed for execution, data retrieval, or verification.\n"
            "- If the user asks for explanation/summary and existing context is sufficient, use action=final directly.\n"
            "- Prefer run_python for Python/data work.\n"
            "- Use run_bash only when shell semantics are required.\n"
            "- For direct helper actions (scrape_url/read_*), provide parameters under `args` as a JSON object.\n"
            "- For run_python, prefer returning a complete executable script in `code`.\n"
            "- If `script_mode` is incremental, provide only additive/patch-like code that builds on prior working script.\n"
            "- Keep code deterministic and concise.\n"
            "- Never output markdown, prose, or extra keys outside JSON."
        )
    )

    def _build_structured_python_runner(script: str) -> str:
        payload = repr(script)
        return (
            "import contextlib\n"
            "import io\n"
            "import json\n"
            "import traceback\n"
            "source = " + payload + "\n"
            "ns = {}\n"
            "stdout_buffer = io.StringIO()\n"
            "try:\n"
            "    with contextlib.redirect_stdout(stdout_buffer):\n"
            "        exec(compile(source, '<autonomous_step>', 'exec'), ns, ns)\n"
            "    result = ns.get('RESULT', None)\n"
            "    print(json.dumps({'status': 'ok', 'result': result, 'stdout': stdout_buffer.getvalue()}, default=str, sort_keys=True))\n"
            "except Exception as exc:\n"
            "    print(json.dumps({'status': 'error', 'error': str(exc), 'stdout': stdout_buffer.getvalue(), 'traceback': traceback.format_exc()}, default=str, sort_keys=True))\n"
        )

    def _build_step_message(action: str, reason: str, stdout: str, stderr: str) -> str:
        payload = _extract_json_object(stdout)
        if action == "run_python" and isinstance(payload, dict):
            status = str(payload.get("status", "")).strip().lower()
            if status == "error":
                err = str(payload.get("error", "execution error")).strip()
                return f"Python step failed: {err}."

            nested_stdout = str(payload.get("stdout", "")).strip()
            nested_payload = _extract_json_object(nested_stdout) if nested_stdout else {}
            if isinstance(nested_payload, dict) and nested_payload:
                keys = ", ".join(list(nested_payload.keys())[:6])
                return f"Python step completed. Produced structured output with keys: {keys}."

            result = payload.get("result")
            if result is not None:
                return "Python step completed and returned a RESULT value."
            return "Python step completed successfully."

        if action == "run_bash":
            payload = _extract_json_object(stdout)
            if isinstance(payload, dict) and payload:
                ok = bool(payload.get("ok"))
                code = payload.get("exit_code")
                return f"Bash step {'succeeded' if ok else 'failed'} with exit code {code}."

        if action in direct_helper_actions:
            payload = _extract_json_object(stdout)
            if isinstance(payload, dict) and payload:
                if payload.get("ok") is False:
                    return f"{action} failed: {str(payload.get('error', 'runtime helper error'))[:140]}"
                keys = ", ".join(list(payload.keys())[:6])
                return f"{action} completed. Output keys: {keys}."

        if stderr and stderr.strip() and stderr.strip() != "(none)":
            return f"Step executed with stderr: {stderr[:160]}"
        return f"Step executed for: {reason}"

    def _format_trace(include_observations: bool = False) -> str:
        lines: List[str] = ["Execution summary:"]
        for item in steps:
            message = str(item.get("message", "")).strip() or str(item.get("reason", ""))
            lines.append(f"- Step {item['step']} [{item['action']}]: {message}")
            if include_observations:
                lines.append("  details:")
                lines.append("```text")
                lines.append(str(item.get("observation", "(none)"))[:2800])
                lines.append("```")
        return "\n".join(lines)

    # Step 0: deterministic schema probe for attached CSV files to reduce blind column assumptions.
    if chat_file_paths:
        probe_code = (
            "import csv\n"
            "import json\n"
            "import os\n"
            "from pathlib import Path\n"
            "paths = json.loads(os.getenv('STEM_INGESTED_FILES', '[]'))\n"
            "summaries = []\n"
            "for raw in paths:\n"
            "    p = Path(raw)\n"
            "    if not p.exists() or p.suffix.lower() not in {'.csv', '.tsv'}:\n"
            "        continue\n"
            "    try:\n"
            "        with p.open('r', encoding='utf-8', errors='replace') as f:\n"
            "            reader = csv.DictReader(f)\n"
            "            headers = reader.fieldnames or []\n"
            "            row_count = 0\n"
            "            for _ in reader:\n"
            "                row_count += 1\n"
            "                if row_count >= 50:\n"
            "                    break\n"
            "        summaries.append({'path': str(p), 'headers': headers, 'sample_row_count': row_count})\n"
            "    except Exception as exc:\n"
            "        summaries.append({'path': str(p), 'error': str(exc)})\n"
            "print(json.dumps({'status': 'ok', 'schema_probe': summaries}, sort_keys=True))\n"
        )
        probe_result = _run_specialized_runtime_code(
            code=probe_code,
            state=state,
            chat_file_paths=chat_file_paths,
        )
        probe_stdout = str(probe_result.get("execution_result") or "").strip()
        probe_stderr = str(probe_result.get("error_trace") or "").strip()
        steps.append(
            {
                "step": "0",
                "action": "run_python",
                "reason": "Automatic schema probe for attached CSV/TSV files.",
                "message": "Inspected attached tabular files and detected available columns.",
                "observation": (
                    f"success={bool(probe_result.get('success'))}\n"
                    f"stdout:\n{(probe_stdout or '(empty)')[:1800]}\n"
                    f"stderr:\n{(probe_stderr or '(none)')[:900]}"
                ),
            }
        )

    for idx in range(1, max(1, max_steps) + 1):
        transcript = []
        for item in steps:
            transcript.append(
                f"Step {item['step']} action={item['action']} reason={item['reason']}\n"
                f"Observation:\n{item['observation']}"
            )
        transcript_block = "\n\n".join(transcript) if transcript else "none"

        prompt = HumanMessage(
            content=(
                f"Task:\n{task}\n\n"
                f"Domain: {state.get('domain', '')}\n"
                f"Theme: {state.get('user_theme', '')}\n"
                f"Architecture: {state.get('architecture_decision', '')}\n"
                f"Planner runtime handoff: {state.get('planner_runtime_tool_handoff', '')}\n"
                f"Runtime helper tools available: {json.dumps(tool_names)}\n"
                f"Ingested files: {json.dumps(state.get('ingested_files', []))}\n"
                f"Chat-attached files: {json.dumps(chat_file_paths or [])}\n"
                f"Current working script:\n{working_script[:5000] or 'none'}\n\n"
                f"Previous execution result: {str(state.get('execution_result', ''))[:1000]}\n\n"
                f"Prior steps:\n{transcript_block}\n\n"
                f"Return the next action JSON for step {idx}."
            )
        )

        response = llm.invoke([system, prompt])
        action_payload = _extract_json_object(str(getattr(response, "content", "")))
        action = str(action_payload.get("action", "")).strip().lower()
        reason = str(action_payload.get("reason", "")).strip() or "none"
        agent_message = str(action_payload.get("message", "")).strip()

        if action == "final":
            final_text = str(action_payload.get("final", "")).strip()
            if not final_text:
                final_text = "Task loop ended without a final summary."
            if steps:
                return "\n\n".join([final_text, _format_trace(include_observations=False)])
            return final_text

        if action == "run_python":
            stdout = ""
            stderr = ""
            code = str(action_payload.get("code", "")).strip()
            if not code:
                observation = "Missing 'code' for run_python action."
            else:
                script_mode = str(action_payload.get("script_mode", "replace")).strip().lower()
                if script_mode == "incremental" and working_script.strip():
                    working_script = f"{working_script.rstrip()}\n\n{code}\n"
                else:
                    working_script = code

                runner_code = _build_structured_python_runner(working_script)
                result = _run_specialized_runtime_code(
                    code=runner_code,
                    state=state,
                    chat_file_paths=chat_file_paths,
                )
                stdout = str(result.get("execution_result") or "").strip()
                stderr = str(result.get("error_trace") or "").strip()
                observation = (
                    f"success={bool(result.get('success'))}\n"
                    f"stdout:\n{(stdout or '(empty)')[:1800]}\n"
                    f"stderr:\n{(stderr or '(none)')[:900]}"
                )
            step_message = agent_message or _build_step_message(action, reason, stdout, stderr)
            steps.append(
                {
                    "step": str(idx),
                    "action": action,
                    "reason": reason,
                    "message": step_message,
                    "observation": observation,
                }
            )
            continue

        if action == "run_bash":
            stdout = ""
            stderr = ""
            if not state.get("enable_bash_tooling", True):
                observation = "run_bash requested but runtime bash tooling is disabled."
            else:
                command = str(action_payload.get("command", "")).strip()
                if not command:
                    observation = "Missing 'command' for run_bash action."
                else:
                    runner_code = (
                        "import json\n"
                        f"out = run_bash({json.dumps(command)})\n"
                        "print(json.dumps(out, sort_keys=True))\n"
                    )
                    result = _run_specialized_runtime_code(
                        code=runner_code,
                        state=state,
                        chat_file_paths=chat_file_paths,
                    )
                    stdout = str(result.get("execution_result") or "").strip()
                    stderr = str(result.get("error_trace") or "").strip()
                    observation = (
                        f"success={bool(result.get('success'))}\n"
                        f"stdout:\n{(stdout or '(empty)')[:1800]}\n"
                        f"stderr:\n{(stderr or '(none)')[:900]}"
                    )
            step_message = agent_message or _build_step_message(action, reason, stdout, stderr)
            steps.append(
                {
                    "step": str(idx),
                    "action": action,
                    "reason": reason,
                    "message": step_message,
                    "observation": observation,
                }
            )
            continue

        if action in direct_helper_actions:
            stdout = ""
            stderr = ""
            args_payload = action_payload.get("args", {})
            if not isinstance(args_payload, dict):
                observation = "Invalid 'args' payload for direct helper action; expected object."
            else:
                helper_runner = (
                    "import json\n"
                    f"tool_name = {json.dumps(action)}\n"
                    f"kwargs = json.loads({json.dumps(json.dumps(args_payload))})\n"
                    "fn = globals().get(tool_name)\n"
                    "if not callable(fn):\n"
                    "    out = {'ok': False, 'error': f'tool {tool_name} is not available in runtime', 'tool': tool_name}\n"
                    "else:\n"
                    "    try:\n"
                    "        out = fn(**kwargs)\n"
                    "    except TypeError as exc:\n"
                    "        out = {'ok': False, 'error': f'invalid arguments for {tool_name}: {exc}', 'tool': tool_name, 'args': kwargs}\n"
                    "    except Exception as exc:\n"
                    "        out = {'ok': False, 'error': str(exc), 'tool': tool_name}\n"
                    "print(json.dumps(out, default=str, sort_keys=True))\n"
                )
                result = _run_specialized_runtime_code(
                    code=helper_runner,
                    state=state,
                    chat_file_paths=chat_file_paths,
                )
                stdout = str(result.get("execution_result") or "").strip()
                stderr = str(result.get("error_trace") or "").strip()
                observation = (
                    f"success={bool(result.get('success'))}\n"
                    f"stdout:\n{(stdout or '(empty)')[:1800]}\n"
                    f"stderr:\n{(stderr or '(none)')[:900]}"
                )

            step_message = agent_message or _build_step_message(action, reason, stdout, stderr)
            steps.append(
                {
                    "step": str(idx),
                    "action": action,
                    "reason": reason,
                    "message": step_message,
                    "observation": observation,
                }
            )
            continue

        invalid_payload = json.dumps(action_payload, ensure_ascii=True)[:500]
        steps.append(
            {
                "step": str(idx),
                "action": "invalid",
                "reason": "invalid_action",
                "message": "Received invalid tool instruction payload and skipped it.",
                "observation": f"Invalid action payload: {invalid_payload}",
            }
        )

    summary = [
        "Task loop reached max steps before a final action.",
        "",
        _format_trace(include_observations=False),
    ]
    return "\n".join(summary)


st.set_page_config(page_title="Stem Agent UI", layout="wide")
st.title("Stem Agent Dashboard")
st.caption("Run the agent, inspect iteration process, and view all metrics/artifacts.")

if "snapshots" not in st.session_state:
    st.session_state.snapshots = []
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "archive_records" not in st.session_state:
    st.session_state.archive_records = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_file_paths" not in st.session_state:
    st.session_state.chat_file_paths = []
if "chat_file_context" not in st.session_state:
    st.session_state.chat_file_context = ""

with st.sidebar:
    st.header("Run Settings")
    domain = st.text_input("Domain", value="Analyze CSV sales data")
    theme = st.text_area("Theme", value="finance-safe deterministic json", height=100)
    max_iterations = st.slider("Max iterations", min_value=1, max_value=10, value=4)
    stop_threshold = st.slider("Stop threshold", min_value=0.70, max_value=0.99, value=0.85, step=0.01)
    patience = st.slider("Early-stop patience", min_value=1, max_value=5, value=2)
    open_run_mode = st.checkbox(
        "Open run mode (remove restrictions)",
        value=False,
        help="Disables sandbox policy checks, contract validation, and strict stopping logic for this run.",
    )
    enable_bash_tooling = st.checkbox(
        "Enable runtime bash tooling",
        value=True,
        help="Injects a sandboxed `run_bash(command, timeout_seconds)` helper into specialized runtime code.",
    )
    enable_mcp_tools = st.checkbox(
        "Enable planner MCP tools",
        value=True,
        help="Runs domain-only MCP tools during planning. Planner does not ingest or read runtime input files.",
    )
    enable_staged_evaluation = st.checkbox(
        "Enable staged benchmark filter",
        value=True,
        help="Runs a quick Phase 1 subset evaluation before full execution to prune weak candidates.",
    )
    staged_eval_ratio = st.slider(
        "Staged eval ratio",
        min_value=0.05,
        max_value=1.0,
        value=0.10,
        step=0.05,
        help="Fraction of benchmark cases used in Phase 1 staged evaluation.",
    )
    staged_eval_max_cases = st.slider(
        "Staged eval max cases",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="Maximum number of benchmark cases to run in staged evaluation.",
    )
    staged_eval_threshold = st.slider(
        "Staged eval threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help="Minimum Phase 1 benchmark score required to continue to full evaluation.",
    )
    staged_eval_fail_scale = st.slider(
        "Staged fail score scale",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Scaling factor applied to benchmark score when staged evaluation fails.",
    )
    input_paths_text = st.text_area(
        "Runtime input file paths (one per line)",
        value="",
        help="Use existing local absolute/relative paths. These files are processed only by the specialized runtime agent at execution time.",
    )
    runtime_uploads = st.file_uploader(
        "Runtime input files (upload)",
        accept_multiple_files=True,
        key="runtime_file_uploads",
        help="Normal upload flow: files are saved locally and processed by the specialized runtime agent.",
    )

    st.caption(
        "Boundary: planner generates strategy only; runtime agent is the only stage that reads provided files."
    )

    run_clicked = st.button("Run Agent", type="primary", use_container_width=True)

if run_clicked:
    parsed_input_paths = [line.strip() for line in input_paths_text.splitlines() if line.strip()]
    with st.spinner("Running agent..."):
        snapshots, final_state, records = _run_agent(
            domain=domain,
            theme=theme,
            max_iterations=max_iterations,
            stop_threshold=stop_threshold,
            patience=patience,
            input_paths=parsed_input_paths,
            uploaded_files=runtime_uploads,
            open_run_mode=open_run_mode,
            enable_bash_tooling=enable_bash_tooling,
            enable_mcp_tools=enable_mcp_tools,
            enable_staged_evaluation=enable_staged_evaluation,
            staged_eval_ratio=staged_eval_ratio,
            staged_eval_max_cases=staged_eval_max_cases,
            staged_eval_threshold=staged_eval_threshold,
            staged_eval_fail_scale=staged_eval_fail_scale,
        )
        st.session_state.snapshots = snapshots
        st.session_state.final_state = final_state
        st.session_state.archive_records = records
        st.session_state.chat_messages = []

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Generation Process Timeline")
    _render_process_timeline(st.session_state.snapshots)

    if st.session_state.final_state:
        _render_metrics(st.session_state.final_state)

with right:
    _render_archive(st.session_state.archive_records)

if st.session_state.final_state:
    st.divider()
    _render_final_state(st.session_state.final_state)

    st.divider()
    st.subheader("Chat with Final Agent")
    chat_paths_text = st.text_area(
        "Chat context file paths (one per line)",
        value="",
        help="These files are added only to post-run final-agent chat context and are not used for generation/execution.",
    )
    chat_uploads = st.file_uploader(
        "Chat context files (upload)",
        accept_multiple_files=True,
        key="chat_file_uploads",
        help="Normal upload flow: files are loaded only into final-agent chat context.",
    )
    load_chat_files = st.button("Load Chat Files", use_container_width=False)
    if load_chat_files:
        chat_paths = [line.strip() for line in chat_paths_text.splitlines() if line.strip()]
        collected_chat_paths: List[str] = list(chat_paths)
        if chat_uploads:
            chat_ingest_dir = Path(__file__).resolve().parent / "chat_inputs"
            collected_chat_paths.extend(save_uploaded_files(chat_ingest_dir, chat_uploads))

        paths, context, errors = ingest_files(collected_chat_paths)
        st.session_state.chat_file_paths = paths
        st.session_state.chat_file_context = context

        if paths:
            st.success(f"Loaded {len(paths)} chat file(s).")
        if errors:
            for err in errors:
                st.warning(err)

    if st.session_state.chat_file_paths:
        st.caption("Chat files loaded:")
        st.write(st.session_state.chat_file_paths)
    if st.session_state.chat_file_context:
        with st.expander("Chat File Context Preview"):
            st.code(st.session_state.chat_file_context[:4000], language="text")

    llm = _get_chat_llm()
    if llm is None:
        st.info("Set OPENAI_API_KEY in environment to enable chat.")
    else:
        st.caption(
            "Autonomous mode: normal messages run continuous tool-driven execution. "
            "Direct commands: `/run <python_code>` or `/runbash <bash_command>`."
        )
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask the final agent about this run")
        if question:
            st.session_state.chat_messages.append({"role": "user", "content": question})

            state = st.session_state.final_state

            stripped = question.strip()
            if stripped.startswith("/run "):
                snippet = stripped[len("/run ") :].strip()
                if not snippet:
                    answer = "`/run` requires Python code after the command."
                else:
                    exec_result = _run_specialized_runtime_code(
                        code=snippet,
                        state=state,
                        chat_file_paths=st.session_state.chat_file_paths,
                    )
                    stdout = str(exec_result.get("execution_result") or "").strip()
                    stderr = str(exec_result.get("error_trace") or "").strip()
                    answer = (
                        "Specialized runtime execution result:\n"
                        f"- success: `{bool(exec_result.get('success'))}`\n"
                        f"- stdout:```text\n{stdout or '(empty)'}\n```\n"
                        f"- stderr:```text\n{stderr or '(none)'}\n```"
                    )
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                st.rerun()

            if stripped.startswith("/runbash "):
                command = stripped[len("/runbash ") :].strip()
                if not command:
                    answer = "`/runbash` requires a bash command after the command."
                elif not bool(state.get("enable_bash_tooling", True)):
                    answer = "Runtime bash tooling is disabled for this run. Enable it in Run Settings and rerun."
                else:
                    runner = (
                        "import json\n"
                        f"out = run_bash({json.dumps(command)})\n"
                        "print(json.dumps(out, sort_keys=True))\n"
                    )
                    exec_result = _run_specialized_runtime_code(
                        code=runner,
                        state=state,
                        chat_file_paths=st.session_state.chat_file_paths,
                    )
                    stdout = str(exec_result.get("execution_result") or "").strip()
                    stderr = str(exec_result.get("error_trace") or "").strip()
                    answer = (
                        "Specialized runtime bash result:\n"
                        f"- success: `{bool(exec_result.get('success'))}`\n"
                        f"- stdout:```text\n{stdout or '(empty)'}\n```\n"
                        f"- stderr:```text\n{stderr or '(none)'}\n```"
                    )
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                st.rerun()

            with st.spinner("Running autonomous specialized agent..."):
                answer = _run_continuous_specialized_task(
                    task=question,
                    llm=llm,
                    state=state,
                    chat_file_paths=st.session_state.chat_file_paths,
                    max_steps=8,
                )
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()
