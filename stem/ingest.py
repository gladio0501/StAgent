from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Iterable, List, Tuple


MAX_FILE_BYTES = 256 * 1024
MAX_TEXT_CHARS = 6000


def _safe_read_text(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > MAX_FILE_BYTES:
        raw = raw[:MAX_FILE_BYTES]
    return raw.decode("utf-8", errors="replace")


def _summarize_json(text: str) -> str:
    try:
        payload = json.loads(text)
    except Exception:
        return text[:MAX_TEXT_CHARS]

    if isinstance(payload, dict):
        keys = list(payload.keys())[:25]
        return f"JSON object keys: {keys}\nPreview: {json.dumps(payload, ensure_ascii=True)[:MAX_TEXT_CHARS]}"
    if isinstance(payload, list):
        sample = payload[:3]
        return f"JSON list length={len(payload)}\nSample: {json.dumps(sample, ensure_ascii=True)[:MAX_TEXT_CHARS]}"
    return f"JSON scalar: {str(payload)[:MAX_TEXT_CHARS]}"


def _summarize_csv(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return "CSV appears empty."
    preview = lines[:8]
    return "CSV preview:\n" + "\n".join(preview)[:MAX_TEXT_CHARS]


def summarize_file(path: Path) -> str:
    text = _safe_read_text(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        body = _summarize_json(text)
    elif suffix in {".csv", ".tsv"}:
        body = _summarize_csv(text)
    else:
        body = text[:MAX_TEXT_CHARS]

    return f"File: {path.name}\nPath: {path}\nType: {suffix or 'unknown'}\n{body}".strip()


def ingest_files(paths: Iterable[str]) -> Tuple[List[str], str, List[str]]:
    ingested: List[str] = []
    summaries: List[str] = []
    errors: List[str] = []

    for item in paths:
        candidate = Path(item).expanduser().resolve()
        if not candidate.exists() or not candidate.is_file():
            errors.append(f"Missing file: {item}")
            continue

        try:
            summary = summarize_file(candidate)
            ingested.append(str(candidate))
            summaries.append(summary)
        except Exception as exc:
            errors.append(f"Failed to ingest {candidate}: {exc}")

    context = "\n\n---\n\n".join(summaries)
    return ingested, context, errors


def build_file_manifest(paths: Iterable[str]) -> str:
    """Create a compact metadata-only manifest for planning and runtime contracts."""
    entries: List[dict] = []
    for item in paths:
        candidate = Path(item).expanduser().resolve()
        if not candidate.exists() or not candidate.is_file():
            continue

        entry = {
            "path": str(candidate),
            "name": candidate.name,
            "suffix": candidate.suffix.lower() or "unknown",
            "size_bytes": candidate.stat().st_size,
        }

        try:
            if candidate.suffix.lower() in {".csv", ".tsv"}:
                text = _safe_read_text(candidate)
                lines = [line for line in text.splitlines() if line.strip()]
                delimiter = "\t" if candidate.suffix.lower() == ".tsv" else ","
                if lines:
                    reader = csv.reader([lines[0]], delimiter=delimiter)
                    header = next(reader, [])
                    entry["columns"] = header[:50]
                    entry["row_preview_count"] = max(0, len(lines) - 1)
            elif candidate.suffix.lower() == ".json":
                text = _safe_read_text(candidate)
                payload = json.loads(text)
                if isinstance(payload, dict):
                    entry["json_keys"] = list(payload.keys())[:50]
                    entry["json_type"] = "object"
                elif isinstance(payload, list):
                    entry["json_type"] = "array"
                    entry["json_length"] = len(payload)
                else:
                    entry["json_type"] = type(payload).__name__
        except Exception:
            pass

        entries.append(entry)

    return json.dumps({"files": entries}, ensure_ascii=True, indent=2)


def build_file_processing_plan(file_manifest: str) -> str:
    """Build a deterministic runtime file-work plan for the specialized agent only."""
    if not file_manifest.strip():
        return "No input files provided."
    try:
        payload = json.loads(file_manifest)
    except json.JSONDecodeError:
        return "Input files provided but manifest could not be parsed."

    files = payload.get("files", []) if isinstance(payload, dict) else []
    if not isinstance(files, list) or not files:
        return "No input files provided."

    instructions: List[str] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "unknown"))
        suffix = str(entry.get("suffix", "unknown"))
        path = str(entry.get("path", ""))
        if suffix in {".csv", ".tsv"}:
            cols = entry.get("columns", [])
            instructions.append(
                f"File {name} at {path}: parse tabular data, compute row count, column count, and numeric-column summaries; produce concise report. Columns: {cols}."
            )
        elif suffix == ".json":
            instructions.append(
                f"File {name} at {path}: parse JSON and summarize schema keys/types and top-level counts."
            )
        else:
            instructions.append(
                f"File {name} at {path}: read as text and summarize key points and line counts."
            )
    return "\n".join(instructions)


def save_uploaded_files(base_dir: Path, uploaded_files: Iterable[Any]) -> List[str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for file_obj in uploaded_files:
        name = Path(getattr(file_obj, "name", "upload.bin")).name
        target = base_dir / name
        data = file_obj.getvalue()
        target.write_bytes(data)
        saved_paths.append(str(target.resolve()))

    return saved_paths
