from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import ARTIFACT_FILENAMES
from .ids import sha256_file
from .transcript import load_transcript


def inspect_job(value: Path) -> dict[str, Any]:
    manifest_path = _manifest_path(value.expanduser().resolve())
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or not manifest.get("job_id"):
        raise ValueError(f"Not a Whisper Lab manifest: {manifest_path}")

    root = manifest_path.parent.resolve()
    artifacts = []
    all_valid = True
    raw_artifacts = manifest.get("artifacts", [])
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raw_artifacts = []
        all_valid = False
    tracked_paths: set[str] = set()
    for record in raw_artifacts:
        if not isinstance(record, dict):
            artifacts.append({"path": "<invalid>", "status": "invalid record"})
            all_valid = False
            continue
        relative = Path(str(record.get("path", "")))
        relative_name = relative.as_posix()
        if relative_name in tracked_paths:
            artifacts.append({"path": relative_name, "status": "duplicate record"})
            all_valid = False
            continue
        tracked_paths.add(relative_name)
        path = (root / relative).resolve()
        if relative.is_absolute() or (path != root and root not in path.parents):
            artifacts.append({"path": str(relative), "status": "unsafe path"})
            all_valid = False
            continue
        if not path.is_file():
            artifacts.append({"path": str(relative), "status": "missing"})
            all_valid = False
            continue
        expected_size = record.get("bytes")
        actual_size = path.stat().st_size
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            artifacts.append({"path": relative_name, "status": "invalid size"})
            all_valid = False
            continue
        if expected_size != actual_size:
            artifacts.append(
                {"path": relative_name, "status": "size mismatch", "bytes": actual_size}
            )
            all_valid = False
            continue
        expected = record.get("sha256")
        actual = sha256_file(path)
        status = "ok" if expected == actual else "hash mismatch"
        artifacts.append({"path": relative_name, "status": status, "bytes": actual_size})
        all_valid = all_valid and status == "ok"

    for name in ARTIFACT_FILENAMES:
        path = root / name
        if path.is_file() and name not in tracked_paths:
            artifacts.append({"path": name, "status": "untracked"})
            all_valid = False
    if "transcript.json" not in tracked_paths:
        if not any(item["path"] == "transcript.json" for item in artifacts):
            artifacts.append({"path": "transcript.json", "status": "missing from manifest"})
        all_valid = False

    transcript_path = root / "transcript.json"
    try:
        transcript = load_transcript(transcript_path)
        segment_count = len(transcript.segments)
    except ValueError:
        segment_count = None
        all_valid = False
    return {
        "valid": all_valid,
        "job_id": manifest["job_id"],
        "manifest": str(manifest_path),
        "source": manifest.get("source", {}),
        "model": manifest.get("model", {}),
        "options": manifest.get("options", {}),
        "segments": segment_count,
        "artifacts": artifacts,
    }


def _manifest_path(path: Path) -> Path:
    if path.is_dir():
        return path / "manifest.json"
    if path.name == "transcript.json":
        return path.with_name("manifest.json")
    return path
