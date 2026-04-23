from langgraph.graph import END, StateGraph
from langchain_core.messages import ToolMessage

from stem.nodes import architect, execute_code, generate_code, plan, reflect, research
from stem.nodes_support import inspect_file, search_web
from stem.state import AgentState

def research_tools_condition(state: AgentState) -> str:
    history = state.get("history", [])
    if history and hasattr(history[-1], "tool_calls") and history[-1].tool_calls:
        return "tools"
    return "__end__"

def research_tools_node(state: AgentState):
    history = state.get("history", [])
    if not history or not hasattr(history[-1], "tool_calls"):
        return {"history": history}
        
    last_msg = history[-1]
    new_history = list(history)
    
    for tool_call in last_msg.tool_calls:
        try:
            if tool_call["name"] == "inspect_file":
                res = inspect_file.invoke(tool_call["args"])
            elif tool_call["name"] == "search_web":
                res = search_web.invoke(tool_call["args"])
            else:
                res = f"Error: unknown tool {tool_call['name']}"
        except Exception as e:
            res = f"Tool execution failed: {e}"
            
        new_history.append(
            ToolMessage(content=str(res), name=tool_call["name"], tool_call_id=tool_call["id"])
        )
        
    return {"history": new_history}

workflow = StateGraph(AgentState)

workflow.add_node("plan", plan)
workflow.add_node("research", research)
workflow.add_node("research_tools", research_tools_node)
workflow.add_node("architect", architect)
workflow.add_node("generate", generate_code)
workflow.add_node("execute", execute_code)
workflow.add_node("reflect", reflect)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "research")
workflow.add_conditional_edges("research", research_tools_condition, {"tools": "research_tools", "__end__": "architect"})
workflow.add_edge("research_tools", "research")
workflow.add_edge("architect", "generate")
workflow.add_edge("generate", "execute")


def route_execution(state: AgentState):
    open_run_mode = bool(state.get("open_run_mode", False))
    threshold = state.get("stop_threshold", 0.85)
    patience = state.get("early_stop_patience", 2)
    stagnation = state.get("no_improvement_count", 0)
    best_score = state.get("best_fitness_score", 0.0)
    max_iterations = int(state.get("max_iterations", 0))

    if open_run_mode:
        if state["success"]:
            import json as _json
            _raw = state.get("execution_result") or ""
            try:
                _payload = _json.loads(_raw.strip()) if isinstance(_raw, str) and _raw.strip() else {}
                _contract_ok = str(_payload.get("status", "")).strip().lower() == "ok"
            except Exception:
                _contract_ok = False
            
            # In open run, we stil want to iterate if the score is low (e.g., correct=0).
            if _contract_ok:
                if state.get("fitness_score", 0.0) >= threshold:
                    return END
                if best_score >= 0.75 and stagnation >= patience:
                    return END

        if max_iterations > 0 and state["iteration_count"] >= max_iterations:
            return END
        return "reflect"

    if state["success"] and state.get("fitness_score", 0.0) >= threshold:
        return END
    if best_score >= 0.75 and stagnation >= patience:
        return END
    if state["iteration_count"] >= max_iterations:
        return END
    return "reflect"


workflow.add_conditional_edges("execute", route_execution)
workflow.add_edge("reflect", "generate")

app = workflow.compile()
