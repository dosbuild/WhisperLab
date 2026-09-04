from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    """The only two writable roots used by Whisper Lab."""

    output_dir: Path
    model_dir: Path

    @classmethod
    def from_values(
        cls,
        *,
        output_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        cwd: Path | None = None,
    ) -> RuntimePaths:
        base = (cwd or Path.cwd()).resolve()
        output_value = output_dir or os.environ.get("WHISPERLAB_OUTPUT_DIR", "outputs")
        model_value = model_dir or os.environ.get("WHISPERLAB_MODEL_DIR", ".cache/models")
        return cls(
            output_dir=_resolve_from(base, output_value),
            model_dir=_resolve_from(base, model_value),
        )


def _resolve_from(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()
