from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def job_id(*, source_sha256: str, model: dict[str, Any], options: dict[str, Any]) -> str:
    """Return the content/config identity used for output naming and resume."""

    return stable_hash(
        {
            "source_sha256": source_sha256,
            "model": model,
            "options": options,
        }
    )
