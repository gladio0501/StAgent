# Benchmarks

This directory contains full benchmark suites and generated reports for StemAgent.

## Run full benchmark suite

```bash
./.venv/bin/python main.py \
  --benchmark-suite benchmarks/suites/default_suite.json \
  --benchmark-output-dir benchmarks/results \
  --benchmark-repeats 1 \
  --no-console --no-chat
```

## Run pure LLM baseline (no agent pipeline)

```bash
./.venv/bin/python main.py \
  --benchmark-suite benchmarks/suites/default_suite.json \
  --benchmark-output-dir benchmarks/results \
  --benchmark-repeats 1 \
  --benchmark-runner llm \
  --no-console --no-chat
```

## Run side-by-side comparison (agent vs pure LLM)

```bash
./.venv/bin/python main.py \
  --benchmark-suite benchmarks/suites/default_suite.json \
  --benchmark-output-dir benchmarks/results \
  --benchmark-repeats 1 \
  --benchmark-runner both \
  --no-console --no-chat
```

## Run benchmarks in open-run mode

Use `--open-run` with any benchmark runner mode to disable sandbox/output contract restrictions globally for benchmark cases.

```bash
./.venv/bin/python main.py \
  --benchmark-suite benchmarks/suites/default_suite.json \
  --benchmark-output-dir benchmarks/results \
  --benchmark-repeats 1 \
  --benchmark-runner both \
  --open-run \
  --no-console --no-chat
```

## Suite format

Each case supports:

- `id`: unique case id
- `domain`: task/domain prompt used by agent pipeline
- `theme`: optional theme constraints
- `domain_class`: `software_engineering|paper_review|olympiad_math|...`
- `metric_type`: `pass_at_1|overall_accuracy|points_percentage|generic_success`
- `weight`: weighted contribution to global score
- `input_files`: optional runtime input files
- staged-eval controls: `enable_staged_evaluation`, `staged_eval_ratio`, `staged_eval_max_cases`, `staged_eval_threshold`, `staged_eval_fail_scale`

## Output report

Reports are written to `benchmarks/results/benchmark_<timestamp>.json` and include:

- `global_score`
- `domain_summary`
- per-case `result` and selected final-state metrics

Additional report files:

- `benchmark_llm_<timestamp>.json` for pure LLM baseline runs
- `benchmark_compare_<timestamp>.json` for agent-vs-LLM delta analysis
