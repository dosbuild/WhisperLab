from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from .ids import sha256_file

ARTIFACT_FILENAMES = (
    "transcript.json",
    "transcript.txt",
    "transcript.srt",
    "transcript.vtt",
)
JOB_FILENAMES = (*ARTIFACT_FILENAMES, "manifest.json")


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^\w]+", "-", value.lower(), flags=re.UNICODE).strip("-_")
    return slug or "media"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def artifact_record(path: Path, *, root: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve())
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def output_dir(root: Path, source_stem: str, job_id: str) -> Path:
    return root / f"{safe_slug(source_stem)}--{job_id[:12]}"
