from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State tracked across each LangGraph step."""

    domain: str
    user_theme: str
    ingested_files: List[str]
    ingested_context: str
    file_manifest: str
    file_processing_plan: str
    open_run_mode: bool
    enable_bash_tooling: bool
    enable_mcp_tools: bool
    mcp_tools_used: List[str]
    mcp_tool_observations: str
    planner_runtime_tool_handoff: str
    enforce_output_contract: bool
    enforce_sandbox_policy: bool
    enforce_resource_limits: bool
    isolated_execution: bool
    enable_staged_evaluation: bool
    staged_eval_ratio: float
    staged_eval_max_cases: int
    staged_eval_threshold: float
    staged_eval_fail_scale: float
    requirements: str
    research_notes: str
    research_sources: List[str]
    architecture_decision: str
    selected_tools: List[str]
    architect_runtime_tools: List[str]
    test_strategy: str
    stop_threshold: float
    current_code: Optional[str]
    iteration_count: int
    max_iterations: int
    execution_result: Optional[str]
    error_trace: Optional[str]
    fitness_metrics: Dict[str, float]
    fitness_score: float
    best_fitness_score: float
    no_improvement_count: int
    early_stop_patience: int
    before_after_metrics: Dict[str, float]
    inter_agent_benchmarks: Dict[str, float]
    benchmark_metric_name: str
    benchmark_score: float
    benchmark_domain_class: str
    benchmark_case_id: str
    staged_eval_passed: bool
    staged_eval_cases: int
    benchmark_total_cases: int
    history: List[BaseMessage]
    research_tool_calls: int  # tracks tool call iterations in research react loop
    best_variant_path: Optional[str]
    success: bool
