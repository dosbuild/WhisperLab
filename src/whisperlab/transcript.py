from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Segment:
    id: int
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    metadata: dict[str, Any]
    backend: dict[str, Any]
    model: dict[str, Any]
    language: dict[str, Any]
    timings: dict[str, Any]
    segments: list[Segment]
    words: list[dict[str, Any]] | None = None
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Transcript:
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported transcript schema: {value.get('schema_version')!r}")
        raw_segments = value.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("Transcript segments must be a list")
        try:
            segments = [
                Segment(
                    id=int(item["id"]),
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item["text"]),
                )
                for item in raw_segments
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid transcript segment: {exc}") from exc
        words = value.get("words")
        if words is not None and not isinstance(words, list):
            raise ValueError("Transcript words must be a list or null")
        warnings = value.get("warnings", [])
        if not isinstance(warnings, list):
            raise ValueError("Transcript warnings must be a list")
        return cls(
            metadata=_mapping_field(value, "metadata"),
            backend=_mapping_field(value, "backend"),
            model=_mapping_field(value, "model"),
            language=_mapping_field(value, "language"),
            timings=_mapping_field(value, "timings"),
            segments=segments,
            words=words,
            warnings=[str(item) for item in warnings],
        )


def load_transcript(path: Path) -> Transcript:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read transcript {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Transcript must contain a JSON object: {path}")
    return Transcript.from_dict(value)


def render_txt(transcript: Transcript) -> str:
    lines = [segment.text.strip() for segment in transcript.segments if segment.text.strip()]
    return "\n".join(lines) + ("\n" if lines else "")


def render_srt(transcript: Transcript) -> str:
    blocks = []
    for index, segment in enumerate(_nonempty_segments(transcript), start=1):
        blocks.append(
            f"{index}\n{_timestamp(segment.start, ',')} --> {_timestamp(segment.end, ',')}\n"
            f"{segment.text.strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(transcript: Transcript) -> str:
    blocks = [
        f"{_timestamp(segment.start, '.')} --> {_timestamp(segment.end, '.')}\n"
        f"{segment.text.strip()}"
        for segment in _nonempty_segments(transcript)
    ]
    body = "\n\n".join(blocks)
    return "WEBVTT\n\n" + body + ("\n" if body else "")


def _nonempty_segments(transcript: Transcript) -> list[Segment]:
    return [segment for segment in transcript.segments if segment.text.strip()]


def _mapping_field(value: dict[str, Any], name: str) -> dict[str, Any]:
    field = value.get(name, {})
    if not isinstance(field, dict):
        raise ValueError(f"Transcript {name} must be an object")
    return dict(field)


def _timestamp(seconds: float, decimal: str) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{decimal}{milliseconds:03d}"
