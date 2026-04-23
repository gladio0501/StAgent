from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional


def _policy_violation(code: str) -> Optional[str]:
    """Block obviously unsafe capabilities before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    blocked_modules = {
        "socket",
        "subprocess",
        "requests",
        "httpx",
        "urllib",
        "ftplib",
        "paramiko",
    }
    blocked_calls = {
        "eval",
        "exec",
        "compile",
        "system",
        "popen",
        "fork",
        "spawn",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in blocked_modules:
                    return f"Policy violation: blocked import '{root}'."
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in blocked_modules:
                return f"Policy violation: blocked import '{root}'."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
                return f"Policy violation: blocked call '{node.func.id}'."
            if isinstance(node.func, ast.Attribute) and node.func.attr in blocked_calls:
                return f"Policy violation: blocked call '{node.func.attr}'."
    return None


def _resource_limiter(timeout_seconds: int):
    """Return a POSIX pre-exec hook that applies conservative resource limits."""

    def _limit() -> None:
        try:
            import resource

            cpu_cap = max(1, timeout_seconds)
            memory_cap = 512 * 1024 * 1024
            file_cap = 2 * 1024 * 1024

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_cap, cpu_cap))
            resource.setrlimit(resource.RLIMIT_AS, (memory_cap, memory_cap))
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_cap, file_cap))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            # Failing closed here may break portability; timeout remains the fallback.
            pass

    return _limit


def execute_python_code(
    code: str,
    tests_file: Optional[str] = None,
    timeout_seconds: int = 30,
    extra_env: Optional[Dict[str, str]] = None,
    enforce_policy: bool = True,
    enforce_resource_limits: bool = True,
    isolated_execution: bool = True,
    enable_bash_tooling: bool = True,
    runtime_tools: Optional[List[str]] = None,
) -> Dict[str, Optional[str] | bool]:
    """Execute generated Python in an isolated temp file and capture outputs."""
    if enforce_policy:
        violation = _policy_violation(code)
        if violation:
            return {
                "success": False,
                "execution_result": None,
                "error_trace": violation,
            }

    temp_path = None
    runner_path = None
    work_dir = None
    try:
        work_dir = tempfile.mkdtemp(prefix="stem_sandbox_")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            temp_path = tmp.name

        default_runtime_tools = ["run_python", "read_text_file", "read_json_file", "read_csv_preview", "scrape_url"]
        if enable_bash_tooling:
            default_runtime_tools.append("run_bash")

        requested_runtime_tools = [str(v) for v in (runtime_tools or default_runtime_tools) if str(v).strip()]
        allowed_runtime_tools = {
            "run_python",
            "run_bash",
            "read_text_file",
            "read_json_file",
            "read_csv_preview",
            "scrape_url",
        }
        effective_runtime_tools: List[str] = []
        for tool_name in requested_runtime_tools:
            if tool_name not in allowed_runtime_tools:
                continue
            if tool_name == "run_bash" and not enable_bash_tooling:
                continue
            if tool_name not in effective_runtime_tools:
                effective_runtime_tools.append(tool_name)

        if effective_runtime_tools:
            runner_code = textwrap.dedent(
                """
                import csv
                import json
                import os
                import subprocess
                import sys
                from urllib.parse import urlparse
                from urllib.request import Request, urlopen
                from pathlib import Path

                def run_bash(command: str, timeout_seconds: int = 20) -> dict:
                    if not isinstance(command, str) or not command.strip():
                        return {
                            "ok": False,
                            "exit_code": -1,
                            "stdout": "",
                            "stderr": "command must be a non-empty string",
                            "command": str(command),
                        }

                    max_timeout = int(os.getenv("STEM_BASH_MAX_TIMEOUT", "20"))
                    timeout = max(1, min(int(timeout_seconds), max_timeout))
                    cwd = os.getenv("STEM_BASH_WORKDIR") or os.getcwd()

                    try:
                        proc = subprocess.run(
                            ["/bin/bash", "-lc", command],
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                            cwd=cwd,
                            env={
                                "PATH": os.getenv("PATH", ""),
                                "HOME": os.getenv("HOME", ""),
                                "LANG": os.getenv("LANG", "C.UTF-8"),
                                "LC_ALL": os.getenv("LC_ALL", "C.UTF-8"),
                            },
                        )
                        return {
                            "ok": proc.returncode == 0,
                            "exit_code": proc.returncode,
                            "stdout": proc.stdout,
                            "stderr": proc.stderr,
                            "command": command,
                        }
                    except subprocess.TimeoutExpired as exc:
                        return {
                            "ok": False,
                            "exit_code": -1,
                            "stdout": exc.stdout or "",
                            "stderr": f"bash timeout after {timeout}s",
                            "command": command,
                        }
                    except Exception as exc:
                        return {
                            "ok": False,
                            "exit_code": -1,
                            "stdout": "",
                            "stderr": str(exc),
                            "command": command,
                        }

                def run_python(args: list[str], timeout_seconds: int = 20) -> dict:
                    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                        return {
                            "ok": False,
                            "exit_code": -1,
                            "stdout": "",
                            "stderr": "args must be a list[str]",
                            "command": str(args),
                        }

                    max_timeout = int(os.getenv("STEM_BASH_MAX_TIMEOUT", "20"))
                    timeout = max(1, min(int(timeout_seconds), max_timeout))
                    cwd = os.getenv("STEM_BASH_WORKDIR") or os.getcwd()
                    cmd = [sys.executable] + args

                    try:
                        proc = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                            cwd=cwd,
                            env={
                                "PATH": os.getenv("PATH", ""),
                                "HOME": os.getenv("HOME", ""),
                                "LANG": os.getenv("LANG", "C.UTF-8"),
                                "LC_ALL": os.getenv("LC_ALL", "C.UTF-8"),
                            },
                        )
                        return {
                            "ok": proc.returncode == 0,
                            "exit_code": proc.returncode,
                            "stdout": proc.stdout,
                            "stderr": proc.stderr,
                            "command": " ".join(cmd),
                        }
                    except subprocess.TimeoutExpired as exc:
                        return {
                            "ok": False,
                            "exit_code": -1,
                            "stdout": exc.stdout or "",
                            "stderr": f"python timeout after {timeout}s",
                            "command": " ".join(cmd),
                        }
                    except Exception as exc:
                        return {
                            "ok": False,
                            "exit_code": -1,
                            "stdout": "",
                            "stderr": str(exc),
                            "command": " ".join(cmd),
                        }

                def read_text_file(path: str, max_chars: int = 20000) -> dict:
                    try:
                        text = Path(path).read_text(encoding="utf-8", errors="replace")
                        clipped = text[: max(1, int(max_chars))]
                        return {
                            "ok": True,
                            "path": path,
                            "content": clipped,
                            "truncated": len(clipped) < len(text),
                        }
                    except Exception as exc:
                        return {"ok": False, "path": path, "error": str(exc)}

                def read_json_file(path: str, max_chars: int = 30000) -> dict:
                    text_result = read_text_file(path, max_chars=max_chars)
                    if not text_result.get("ok"):
                        return text_result
                    try:
                        parsed = json.loads(text_result.get("content") or "")
                        return {
                            "ok": True,
                            "path": path,
                            "data": parsed,
                            "truncated": bool(text_result.get("truncated")),
                        }
                    except Exception as exc:
                        return {
                            "ok": False,
                            "path": path,
                            "error": f"json parse failed: {exc}",
                            "content": text_result.get("content", ""),
                        }

                def read_csv_preview(path: str, max_rows: int = 25) -> dict:
                    try:
                        rows = []
                        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
                            reader = csv.DictReader(handle)
                            for idx, row in enumerate(reader):
                                if idx >= max(1, int(max_rows)):
                                    break
                                rows.append(dict(row))
                        return {
                            "ok": True,
                            "path": path,
                            "row_count": len(rows),
                            "rows": rows,
                        }
                    except Exception as exc:
                        return {"ok": False, "path": path, "error": str(exc)}

                def scrape_url(url: str, timeout_seconds: int = 15, max_chars: int = 12000) -> dict:
                    try:
                        parsed = urlparse(str(url))
                        if parsed.scheme not in ("http", "https"):
                            return {"ok": False, "url": str(url), "error": "only http/https URLs are allowed"}

                        timeout = max(1, min(int(timeout_seconds), 30))
                        request = Request(
                            str(url),
                            headers={"User-Agent": "StemAgentRuntime/1.0"},
                        )
                        with urlopen(request, timeout=timeout) as response:
                            content_type = str(response.headers.get("Content-Type", ""))
                            payload = response.read(max(1, int(max_chars))).decode("utf-8", errors="replace")
                        return {
                            "ok": True,
                            "url": str(url),
                            "content_type": content_type,
                            "content": payload,
                        }
                    except Exception as exc:
                        return {"ok": False, "url": str(url), "error": str(exc)}

                _requested_tools = os.getenv("STEM_RUNTIME_TOOLS", "[]")
                try:
                    AVAILABLE_RUNTIME_TOOLS = json.loads(_requested_tools)
                    if not isinstance(AVAILABLE_RUNTIME_TOOLS, list):
                        AVAILABLE_RUNTIME_TOOLS = []
                except Exception:
                    AVAILABLE_RUNTIME_TOOLS = []

                def list_runtime_tools() -> list[str]:
                    return list(AVAILABLE_RUNTIME_TOOLS)

                def main() -> None:
                    if len(sys.argv) < 2:
                        raise RuntimeError("missing user script path")

                    script_path = sys.argv[1]
                    passthrough = sys.argv[2:]
                    # argv shim: if ingested files are present, expose the first one as argv[1]
                    # so generated code that reads sys.argv[1] gets the task file, not the
                    # internal test-harness JSON.
                    _ingested = json.loads(os.getenv("STEM_INGESTED_FILES", "[]") or "[]")
                    if isinstance(_ingested, list) and _ingested:
                        sys.argv = [script_path] + [str(p) for p in _ingested]
                    else:
                        sys.argv = [script_path] + passthrough


                    user_globals = {
                        "__name__": "__main__",
                        "__file__": script_path,
                        "AVAILABLE_RUNTIME_TOOLS": AVAILABLE_RUNTIME_TOOLS,
                        "list_runtime_tools": list_runtime_tools,
                        # Inject STEM env vars as Python globals so generated code
                        # can access ingested files directly without env var parsing.
                        "STEM_INGESTED_FILES": json.loads(
                            os.getenv("STEM_INGESTED_FILES", "[]") or "[]"
                        ),
                        "STEM_FILE_PROCESSING_PLAN": os.getenv(
                            "STEM_FILE_PROCESSING_PLAN", ""
                        ),
                    }
                    tool_registry = {
                        "run_bash": run_bash,
                        "run_python": run_python,
                        "read_text_file": read_text_file,
                        "read_json_file": read_json_file,
                        "read_csv_preview": read_csv_preview,
                        "scrape_url": scrape_url,
                    }
                    for name in AVAILABLE_RUNTIME_TOOLS:
                        fn = tool_registry.get(name)
                        if fn is not None:
                            user_globals[name] = fn
                    source = Path(script_path).read_text(encoding="utf-8")
                    exec(compile(source, script_path, "exec"), user_globals, user_globals)

                if __name__ == "__main__":
                    main()
                """
            )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as runner:
                runner.write(runner_code)
                runner_path = runner.name

        command = [sys.executable]
        if isolated_execution:
            command.extend(["-I", "-B"])
        if runner_path:
            command.append(runner_path)
        command.append(temp_path)
        if tests_file:
            command.append(tests_file)

        # Inherit host env so PATH/venv tools remain available inside run_bash.
        exec_env = dict(os.environ)
        exec_env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONNOUSERSITE": "1",
                "STEM_BASH_MAX_TIMEOUT": "20",
                "STEM_BASH_WORKDIR": work_dir,
                "STEM_RUNTIME_TOOLS": json.dumps(effective_runtime_tools),
            }
        )
        if extra_env:
            exec_env.update({str(k): str(v) for k, v in extra_env.items()})

        preexec = (
            _resource_limiter(timeout_seconds)
            if os.name == "posix" and enforce_resource_limits
            else None
        )

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=work_dir,
            env=exec_env,
            preexec_fn=preexec,
        )

        if result.returncode == 0:
            return {
                "success": True,
                "execution_result": result.stdout,
                "error_trace": None,
            }
        return {
            "success": False,
            "execution_result": result.stdout,
            "error_trace": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "execution_result": None,
            "error_trace": "Execution timed out.",
        }
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
        if runner_path:
            try:
                Path(runner_path).unlink(missing_ok=True)
            except Exception:
                pass
        if work_dir:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
