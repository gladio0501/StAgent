# Stem Agent

**A self-specializing, fitness-guided agentic loop powered by LangGraph and OpenAI.**

Stem Agent autonomously plans, researches, architects, generates, executes, and reflects to produce working Python solutions for arbitrary task domains — without hand-crafted domain logic. It uses fitness scoring and iterative self-improvement to converge on high-quality outputs, with a sandboxed execution environment for safety.

---

## Features

- **Multi-stage agentic loop** — Plan → Research → Architect → Generate → Execute → Reflect
- **LLM-driven architecture selection** — Agent picks its own tool stack and strategy per task
- **Fitness-based stopping** — Multi-metric scoring (process, contract, domain, robustness, efficiency)
- **Sandboxed execution** — Restricted subprocess with import blocklisting and POSIX resource limits
- **MCP tool integration** — Planner consults local MCP strategy tools before requirements are finalized
- **File ingestion** — Runtime agents can process ingested CSV, text, or JSON files
- **Dynamic Benchmarking suite** — Uniquely adaptable harness that evaluates agent vs raw LLM schemas dynamically based on JSON definitions
- **Streamlit UI** — Visual interface for running the agent, viewing results, and comparing benchmarks
- **Post-run chat** — Interactive conversation with the final specialized agent after each run

---

## Architecture

```
main.py / streamlit_app.py
        │
        ▼
  stem/graph.py  ──── LangGraph compiled app
        │
        ├── [plan]       stem/nodes.py  ─── Planner LLM + MCP tools
        ├── [research]   stem/nodes.py  ─── Wikipedia + heuristic signals
        ├── [architect]  stem/nodes.py  ─── Architecture JSON selection
        ├── [generate]   stem/nodes.py  ─── Code generation LLM
        ├── [execute]    stem/nodes.py  ─── sandbox/executor.py
        └── [reflect]    stem/nodes.py  ─── LLM critique & guidance
                │
                ▼
         route_execution
           ├── STOP  (success + score ≥ threshold)
           ├── STOP  (max iterations reached / stagnation)
           └── LOOP  reflect → generate → execute
```

### Key Modules

| File | Role |
|------|------|
| `main.py` | CLI entry-point, argument parsing, benchmark orchestration |
| `streamlit_app.py` | Streamlit web UI |
| `stem/graph.py` | LangGraph graph definition and routing logic |
| `stem/nodes.py` | All agent node implementations and fitness scoring |
| `stem/nodes_support.py` | Node helper utilities |
| `stem/state.py` | `AgentState` TypedDict — full runtime state model |
| `stem/prompts.py` | LLM prompts (planner, architect, generator, reflector) |
| `stem/ingest.py` | File ingestion, manifest and processing plan builder |
| `stem/archive.py` | Per-iteration JSON artifact archiving |
| `stem/mcp_tools.py` | Local MCP tool implementations for planner |
| `stem/benchmarks.py` | Benchmark suite runner and agent vs. LLM comparison |
| `sandbox/executor.py` | Sandboxed subprocess executor with safety gates |
| `benchmarks/suites/` | Benchmark suite definitions (JSON) |
| `benchmarks/results/` | Benchmark run reports (gitignored) |
| `archive/` | Per-run execution artifacts (gitignored) |

### Fitness Score Formula

```
fitness_score =
    0.30 × process_success   (process ran + output contract passed)
  + 0.30 × contract_valid    (required JSON keys, status correctness)
  + 0.20 × domain_alignment  (output domain matches requested domain)
  + 0.10 × robustness        (no error trace)
  + 0.10 × efficiency        (fewer iterations = better)
```

### Stop Conditions

1. **Target reached** — `success == true` and `fitness_score ≥ stop_threshold`
2. **Stagnation** — `no_improvement_count ≥ early_stop_patience`
3. **Hard cap** — `iteration_count ≥ max_iterations`

---

## Setup

### 1. Prerequisites

- Python 3.11+
- An OpenAI API key

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini          # optional, defaults to gpt-4o-mini
LANGSMITH_API_KEY=ls__...         # optional, enables LangSmith tracing
```

Load the env before running:

```bash
set -a && source .env && set +a
```

### 5. Running the Application

You are ready to run the agent. Web research (Wikipedia/DuckDuckGo Lite APIs) runs natively without heavy dependencies like Playwright or headless browsers.

---

## Running the Agent

### Interactive CLI (default)

Prompts for domain and theme at startup:

```bash
set -a && source .env && set +a
./.venv/bin/python main.py
```

### Non-interactive CLI

```bash
set -a && source .env && set +a
./.venv/bin/python main.py \
  --domain "Analyze CSV sales data" \
  --theme "finance-safe deterministic json" \
  --max-iterations 4 \
  --stop-threshold 0.85 \
  --patience 2 \
  --no-console \
  --no-chat
```

### With file ingestion

```bash
./.venv/bin/python main.py \
  --domain "Summarize sales trends" \
  --ingest-files sample_sales.csv \
  --no-console
```

### Open run mode (remove sandbox restrictions)

```bash
./.venv/bin/python main.py --open-run --no-console --no-chat
```

---

## Streamlit UI

```bash
set -a && source .env && set +a
./.venv/bin/streamlit run streamlit_app.py
```

The Streamlit app provides:
- Domain and theme input
- Real-time streaming of agent node events
- Fitness score visualization
- Archive viewer for past runs
- Benchmark comparison charts

---

## Benchmarks

### Run the full benchmark suite (agent + LLM baseline comparison)

```bash
set -a && source .env && set +a
./.venv/bin/python main.py \
  --benchmark-suite benchmarks/suites/default_suite.json \
  --benchmark-output-dir benchmarks/results \
  --benchmark-repeats 1 \
  --benchmark-runner both \
  --open-run \
  --no-console --no-chat
```

### Runner options

| Flag | Description |
|------|-------------|
| `--benchmark-runner agent` | Full ReAct agentic pipeline only |
| `--benchmark-runner llm` | Raw LLM baseline (zero-shot static script evaluation) |
| `--benchmark-runner conv` | Conversational baseline (multi-turn no tools) |
| `--benchmark-runner both` | Both + side-by-side comparison report |

Results are written as JSON to `benchmarks/results/`. The harness supports **Dynamic Exact-Match Scoring** — it automatically pulls parsing rules from whatever schema is defined in the `expected` objects of your `.json` test suite instead of relying on hardcoded Python rules.

---

## Sandboxed Execution

Generated code runs in a restricted subprocess (`sandbox/executor.py`):

- **Import blocklist** — `socket`, `subprocess`, `requests`, `httpx`, `urllib`, `ftplib`, `paramiko`
- **Call blocklist** — `eval`, `exec`, `compile`, `os.system`, `popen`, `fork`, `spawn`
- **Isolated Python** — `python -I -B` (no user site, no .pyc writes)
- **POSIX resource limits** — CPU time, memory, file size, core dumps disabled
- **Temp workspace cleanup** after each run

Disable restrictions with `--open-run` for trusted/development use.

---

## Project Structure

```
StemAgent/
├── main.py                     # CLI entry-point
├── streamlit_app.py            # Streamlit web UI
├── requirements.txt
├── sample_sales.csv            # Example input file
├── stem/                       # Agent core
│   ├── graph.py                # LangGraph graph + routing
│   ├── nodes.py                # Node implementations
│   ├── nodes_support.py        # Node helpers
│   ├── state.py                # AgentState type
│   ├── prompts.py              # LLM prompts
│   ├── ingest.py               # File ingestion
│   ├── archive.py              # Artifact archiver
│   ├── mcp_tools.py            # MCP tool implementations
│   └── benchmarks.py           # Benchmark runner
├── sandbox/                    # Execution sandbox
│   ├── executor.py             # Sandboxed runner
│   └── tests.json              # Sandbox test cases
├── benchmarks/
│   ├── suites/                 # Suite definitions
│   │   └── default_suite.json
│   ├── thresholds.json         # Score thresholds
│   └── results/                # Run reports (gitignored)
├── archive/                    # Run artifacts (gitignored)
├── chat_inputs/                # Chat context files (gitignored)
└── ingested_inputs/            # Ingested file copies (gitignored)
```

---

## Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--domain` | interactive prompt | Task domain description |
| `--theme` | `""` | Style/constraint hints for generation |
| `--max-iterations` | `4` | Max generation-execution cycles |
| `--stop-threshold` | `0.85` | Fitness score target for early stop |
| `--patience` | `2` | Stagnation stop patience |
| `--open-run` | off | Disable sandbox + contract enforcement |
| `--no-console` | off | Skip interactive prompts |
| `--no-chat` | off | Skip post-run agent chat |
| `--disable-mcp-tools` | off | Disable planner MCP tool calls |
| `--ingest-files` | `[]` | Files to pass to runtime agent |
| `--benchmark-suite` | `""` | Path to benchmark suite JSON |
| `--benchmark-runner` | `agent` | `agent`, `llm`, `conv`, or `both` |

---

## Dependencies

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent graph runtime
- [LangChain OpenAI](https://github.com/langchain-ai/langchain) — LLM integration
- [Streamlit](https://streamlit.io/) — web UI
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data processing in generated agents
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) / [urllib](https://docs.python.org/3/library/urllib.html) — lightweight open-web scraping
- [NLTK](https://www.nltk.org/) — text processing utilities
