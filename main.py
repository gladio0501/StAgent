from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from stem.benchmarks import compare_benchmark_reports, run_benchmark_suite, run_conversational_baseline_suite, run_llm_baseline_suite
from stem.graph import app
from stem.ingest import build_file_manifest, build_file_processing_plan, ingest_files
from stem.state import AgentState


def _get_chat_llm() -> Optional[Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, temperature=0.2)
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stem Agent with optional interactive console controls.")
    parser.add_argument("--domain", type=str, default=None, help="Task domain (e.g., Analyze CSV sales data)")
    parser.add_argument("--theme", type=str, default=None, help="Theme/style constraints to inject into planning and generation")
    parser.add_argument("--max-iterations", type=int, default=4, help="Maximum mutation iterations")
    parser.add_argument("--stop-threshold", type=float, default=0.85, help="Target hyperagent score for stop")
    parser.add_argument("--patience", type=int, default=2, help="Early-stop patience for no improvement")
    parser.add_argument(
        "--open-run",
        action="store_true",
        help="Remove run restrictions (sandbox policy checks, output contract enforcement, strict stopping).",
    )
    parser.add_argument(
        "--disable-mcp-tools",
        action="store_true",
        help="Disable planner MCP tool calls.",
    )
    parser.add_argument(
        "--disable-bash-tooling",
        action="store_true",
        help="Disable specialized runtime bash tooling bridge (run_bash helper).",
    )
    parser.add_argument(
        "--disable-staged-evaluation",
        action="store_true",
        help="Disable staged benchmark filter before full evaluation.",
    )
    parser.add_argument(
        "--staged-eval-ratio",
        type=float,
        default=0.10,
        help="Phase 1 subset ratio used for staged evaluation.",
    )
    parser.add_argument(
        "--staged-eval-max-cases",
        type=int,
        default=10,
        help="Maximum number of staged benchmark cases.",
    )
    parser.add_argument(
        "--staged-eval-threshold",
        type=float,
        default=0.20,
        help="Minimum staged benchmark score required to continue to full evaluation.",
    )
    parser.add_argument(
        "--staged-eval-fail-scale",
        type=float,
        default=0.35,
        help="Score scale applied when staged evaluation fails (early prune).",
    )
    parser.add_argument(
        "--ingest-files",
        type=str,
        nargs="*",
        default=[],
        help="Optional file paths for specialized runtime-agent processing",
    )
    parser.add_argument("--no-console", action="store_true", help="Disable interactive console prompts")
    parser.add_argument("--no-chat", action="store_true", help="Skip post-run chat with final agent")
    parser.add_argument(
        "--benchmark-suite",
        type=str,
        default="",
        help="Run benchmark suite JSON and exit after writing benchmark report.",
    )
    parser.add_argument(
        "--benchmark-output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory to write benchmark run reports.",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help="Number of repeats per benchmark case.",
    )
    parser.add_argument(
        "--benchmark-runner",
        type=str,
        choices=["agent", "llm", "conv", "both"],
        default="agent",
        help="Benchmark execution mode: agent pipeline, pure LLM baseline, conversational baseline (no tools), or both (agent+llm) with comparison.",
    )
    return parser.parse_args()


def _interactive_or_default(args: argparse.Namespace) -> tuple[str, str]:
    default_domain = "Analyze CSV sales data"
    domain = args.domain or default_domain
    theme = args.theme or ""

    if args.no_console or not sys.stdin.isatty():
        return domain, theme

    entered_domain = input(f"Domain [{domain}]: ").strip()
    if entered_domain:
        domain = entered_domain

    entered_theme = input(
        "Theme constraints (press Enter to skip, e.g. 'finance-safe, deterministic, concise JSON outputs'): "
    ).strip()
    if entered_theme:
        theme = entered_theme

    return domain, theme


def build_initial_state(
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
        "research_tool_calls": 0,
        "best_variant_path": None,
        "success": False,
    }


def _print_final_metrics(state: AgentState) -> None:
    before_after = state.get("before_after_metrics", {})
    if before_after:
        print("\nBefore/After metrics:")
        print(json.dumps(before_after, indent=2))

    print("\nFinal run summary:")
    print(
        json.dumps(
            {
                "success": state.get("success"),
                "iteration_count": state.get("iteration_count"),
                "fitness_score": state.get("fitness_score"),
                "best_fitness_score": state.get("best_fitness_score"),
                "no_improvement_count": state.get("no_improvement_count"),
                "stop_threshold": state.get("stop_threshold"),
                "early_stop_patience": state.get("early_stop_patience"),
                "best_variant_path": state.get("best_variant_path"),
            },
            indent=2,
        )
    )


def _chat_with_final_agent(state: AgentState) -> None:
    llm = _get_chat_llm()
    if llm is None:
        print("\nChat unavailable: OPENAI_API_KEY missing or model init failed.")
        return

    if not sys.stdin.isatty():
        print("\nChat skipped: non-interactive terminal.")
        return

    print("\nInteractive chat with final agent (type 'exit' to quit).")
    system = SystemMessage(
        content=(
            "You are the final specialized Stem Agent. Answer as an implementation-aware agent using "
            "the run context below. Keep responses practical and concise.\n\n"
            f"Domain: {state.get('domain', '')}\n"
            f"Theme constraints: {state.get('user_theme', '') or 'none'}\n"
            f"Architecture decision: {state.get('architecture_decision', '')}\n"
            f"Selected tools: {', '.join(state.get('selected_tools', []))}\n"
            f"Test strategy: {state.get('test_strategy', '')}\n"
            f"Requirements: {state.get('requirements', '')[:1200]}\n"
            f"Ingested files: {', '.join(state.get('ingested_files', []))}\n"
            f"Execution result: {str(state.get('execution_result', ''))[:1200]}\n"
            f"Before/after metrics: {json.dumps(state.get('before_after_metrics', {}))}\n"
        )
    )
    messages: List[Any] = [system]

    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            break

        messages.append(HumanMessage(content=user_text))
        response = llm.invoke(messages)
        content = str(getattr(response, "content", "")).strip()
        print(f"agent> {content}\n")
        messages.append(response)


def main() -> None:
    args = _parse_args()

    if args.benchmark_suite:
        repeats = max(1, int(args.benchmark_repeats))
        default_max_iterations = max(1, int(args.max_iterations))
        default_stop_threshold = min(0.99, max(0.70, float(args.stop_threshold)))
        default_patience = max(1, int(args.patience))

        if args.benchmark_runner == "agent":
            report = run_benchmark_suite(
                suite_path=str(args.benchmark_suite),
                output_dir=str(args.benchmark_output_dir),
                state_builder=build_initial_state,
                repeats=repeats,
                default_max_iterations=default_max_iterations,
                default_stop_threshold=default_stop_threshold,
                default_patience=default_patience,
                open_run_mode=bool(args.open_run),
            )
            print("\nBenchmark run complete.")
            print(
                json.dumps(
                    {
                        "runner": report.get("runner", "agent"),
                        "run_id": report.get("run_id"),
                        "global_score": report.get("global_score"),
                        "domain_summary": report.get("domain_summary"),
                        "output_file": report.get("output_file"),
                    },
                    indent=2,
                )
            )
            return

        if args.benchmark_runner == "llm":
            report = run_llm_baseline_suite(
                suite_path=str(args.benchmark_suite),
                output_dir=str(args.benchmark_output_dir),
                repeats=repeats,
                open_run_mode=bool(args.open_run),
            )
            print("\nLLM baseline benchmark run complete.")
            print(
                json.dumps(
                    {
                        "runner": report.get("runner", "llm_baseline"),
                        "run_id": report.get("run_id"),
                        "global_score": report.get("global_score"),
                        "domain_summary": report.get("domain_summary"),
                        "output_file": report.get("output_file"),
                    },
                    indent=2,
                )
            )
            return

        if args.benchmark_runner == "conv":
            report = run_conversational_baseline_suite(
                suite_path=str(args.benchmark_suite),
                output_dir=str(args.benchmark_output_dir),
                repeats=repeats,
                max_turns=4,
                open_run_mode=bool(args.open_run),
            )
            print("\nConversational baseline benchmark run complete.")
            print(
                json.dumps(
                    {
                        "runner": report.get("runner", "conversational_baseline"),
                        "run_id": report.get("run_id"),
                        "global_score": report.get("global_score"),
                        "domain_summary": report.get("domain_summary"),
                        "output_file": report.get("output_file"),
                    },
                    indent=2,
                )
            )
            return

        agent_report = run_benchmark_suite(
            suite_path=str(args.benchmark_suite),
            output_dir=str(args.benchmark_output_dir),
            state_builder=build_initial_state,
            repeats=repeats,
            default_max_iterations=default_max_iterations,
            default_stop_threshold=default_stop_threshold,
            default_patience=default_patience,
            open_run_mode=bool(args.open_run),
        )
        llm_report = run_llm_baseline_suite(
            suite_path=str(args.benchmark_suite),
            output_dir=str(args.benchmark_output_dir),
            repeats=repeats,
            open_run_mode=bool(args.open_run),
        )
        comparison = compare_benchmark_reports(
            agent_report=agent_report,
            llm_report=llm_report,
            output_dir=str(args.benchmark_output_dir),
        )
        print("\nBenchmark comparison run complete.")
        print(
            json.dumps(
                {
                    "agent_global_score": agent_report.get("global_score"),
                    "llm_global_score": llm_report.get("global_score"),
                    "global_delta_agent_minus_llm": comparison.get("global_delta_agent_minus_llm"),
                    "agent_report_file": agent_report.get("output_file"),
                    "llm_report_file": llm_report.get("output_file"),
                    "comparison_file": comparison.get("output_file"),
                },
                indent=2,
            )
        )
        return

    domain, theme = _interactive_or_default(args)
    ingested_files, ingested_context, ingest_errors = ingest_files(args.ingest_files)
    file_manifest = build_file_manifest(ingested_files)
    file_processing_plan = build_file_processing_plan(file_manifest)

    state = build_initial_state(
        domain=domain,
        user_theme=theme,
        max_iterations=max(1, args.max_iterations),
        stop_threshold=min(0.99, max(0.70, args.stop_threshold)),
        early_stop_patience=max(1, args.patience),
        ingested_files=ingested_files,
        ingested_context=ingested_context,
        file_manifest=file_manifest,
        file_processing_plan=file_processing_plan,
        open_run_mode=bool(args.open_run),
        enable_bash_tooling=not bool(args.disable_bash_tooling),
        enable_mcp_tools=not bool(args.disable_mcp_tools),
        enable_staged_evaluation=not bool(args.disable_staged_evaluation),
        staged_eval_ratio=float(args.staged_eval_ratio),
        staged_eval_max_cases=int(args.staged_eval_max_cases),
        staged_eval_threshold=float(args.staged_eval_threshold),
        staged_eval_fail_scale=float(args.staged_eval_fail_scale),
    )

    print("\nRunning Stem Agent...")
    if theme:
        print(f"Theme injected: {theme}")
    if args.open_run:
        print("Open run mode: enabled (restrictions removed)")
    if ingested_files:
        print(f"Ingested files: {len(ingested_files)}")
        for file_path in ingested_files:
            print(f" - {file_path}")
    if ingest_errors:
        print("Ingestion warnings:")
        for msg in ingest_errors:
            print(f" - {msg}")

    final_state: Optional[Dict[str, Any]] = None
    for snapshot in app.stream(state, stream_mode="values"):
        final_state = snapshot
        print(
            {
                "iteration": snapshot.get("iteration_count"),
                "success": snapshot.get("success"),
                "fitness": snapshot.get("fitness_score"),
                "node_error": snapshot.get("error_trace"),
            }
        )

    print("\nRun complete.")
    if final_state:
        _print_final_metrics(final_state)  # type: ignore[arg-type]
        if not args.no_chat:
            _chat_with_final_agent(final_state)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
