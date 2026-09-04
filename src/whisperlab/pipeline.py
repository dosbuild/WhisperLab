from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    ARTIFACT_FILENAMES,
    JOB_FILENAMES,
    artifact_record,
    atomic_write_json,
    atomic_write_text,
    output_dir,
)
from .engine import transcribe_audio
from .ids import job_id, sha256_file
from .inspect import inspect_job
from .models import ModelResolution, resolve_model
from .transcript import load_transcript, render_srt, render_txt, render_vtt

SUPPORTED_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
ALLOWED_FORMATS = ("txt", "srt", "vtt", "json")
Progress = Callable[[str], None]


@dataclass(frozen=True)
class DecodeOptions:
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = False

    def __post_init__(self) -> None:
        if self.beam_size < 1:
            raise ValueError("beam size must be at least 1")

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "word_timestamps": self.word_timestamps,
        }


@dataclass(frozen=True)
class JobResult:
    source: Path
    directory: Path
    transcript: Path
    manifest: Path
    resumed: bool


def discover_inputs(target: Path) -> list[Path]:
    target = target.expanduser()
    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported media type '{target.suffix}'. Supported: {supported}")
        return [target.resolve()]
    if not target.exists():
        raise FileNotFoundError(f"Input does not exist: {target}")
    if not target.is_dir():
        raise ValueError(f"Input is neither a file nor a directory: {target}")
    return [
        path.resolve()
        for path in sorted(target.rglob("*"))
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(target).parts)
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def normalize_formats(formats: Iterable[str]) -> list[str]:
    normalized = []
    for value in formats:
        name = value.strip().lower()
        if not name or name in normalized:
            continue
        if name not in ALLOWED_FORMATS:
            raise ValueError(
                f"Unsupported output format '{name}'. Choose from: {', '.join(ALLOWED_FORMATS)}"
            )
        normalized.append(name)
    if not normalized:
        raise ValueError("At least one output format is required")
    return normalized


def export_transcript(transcript_path: Path, formats: Iterable[str]) -> list[Path]:
    transcript_path = transcript_path.expanduser().resolve()
    transcript = load_transcript(transcript_path)
    selected_formats = normalize_formats(formats)
    managed_manifest = _managed_manifest_for_export(transcript_path, transcript.metadata)
    renderers = {"txt": render_txt, "srt": render_srt, "vtt": render_vtt}
    render_formats = list(selected_formats)
    if managed_manifest is not None:
        for output_format in renderers:
            if (
                transcript_path.with_suffix(f".{output_format}").is_file()
                and output_format not in render_formats
            ):
                render_formats.append(output_format)

    outputs_by_format: dict[str, Path] = {"json": transcript_path}
    for output_format in render_formats:
        if output_format == "json":
            continue
        destination = transcript_path.with_suffix(f".{output_format}")
        atomic_write_text(destination, renderers[output_format](transcript))
        outputs_by_format[output_format] = destination

    if managed_manifest is not None:
        manifest_path, manifest_data = managed_manifest
        _write_manifest(manifest_path, manifest_data, transcript_path.parent)
    return [outputs_by_format[output_format] for output_format in selected_formats]


def run_transcription(
    *,
    target: Path,
    output_root: Path,
    model_name: str,
    model_dir: Path,
    options: DecodeOptions,
    formats: Iterable[str],
    allow_download: bool,
    resume: bool,
    device: str,
    compute_type: str,
    progress: Progress | None = None,
) -> list[JobResult]:
    report = progress or (lambda _message: None)
    files = discover_inputs(target)
    if not files:
        raise RuntimeError(f"No supported media files found under {target}")
    selected_formats = normalize_formats(formats)
    report(f"Resolving model {model_name}")
    model = resolve_model(model_name, model_dir, allow_download=allow_download)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    report(f"Found {len(files)} media file(s); model {model.repo}@{model.revision[:12]}")
    results = []
    for index, source in enumerate(files, start=1):
        report(f"[{index}/{len(files)}] Hashing {source.name}")
        results.append(
            _run_one(
                source=source,
                output_root=output_root,
                model=model,
                options=options,
                formats=selected_formats,
                resume=resume,
                device=device,
                compute_type=compute_type,
                progress=report,
                position=f"[{index}/{len(files)}]",
            )
        )
    return results


def _run_one(
    *,
    source: Path,
    output_root: Path,
    model: ModelResolution,
    options: DecodeOptions,
    formats: list[str],
    resume: bool,
    device: str,
    compute_type: str,
    progress: Progress,
    position: str,
) -> JobResult:
    source_hash = sha256_file(source)
    identity_options = {
        **options.identity,
        "device": device,
        "compute_type": compute_type,
        "schema_version": "1.0",
    }
    identity = job_id(
        source_sha256=source_hash,
        model=model.identity,
        options=identity_options,
    )
    preferred = output_dir(output_root, source.stem, identity)
    existing = _find_resumable_job(output_root, preferred, identity) if resume else None
    if existing:
        progress(f"{position} Resuming {source.name}")
        transcript_path = existing / "transcript.json"
        export_transcript(transcript_path, formats)
        manifest_path = existing / "manifest.json"
        return JobResult(source, existing, transcript_path, manifest_path, True)

    progress(f"{position} Transcribing {source.name}")
    transcript_path = preferred / "transcript.json"
    manifest_path = preferred / "manifest.json"
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    source_metadata = {
        "name": source.name,
        "bytes": source.stat().st_size,
        "sha256": source_hash,
    }
    metadata = {"job_id": identity, "created_at": created_at, "source": source_metadata}
    transcript = transcribe_audio(
        source,
        model=model,
        options=options.identity,
        device=device,
        compute_type=compute_type,
        metadata=metadata,
    )
    preferred.mkdir(parents=True, exist_ok=True)
    _remove_generated_job_files(preferred)
    atomic_write_json(transcript_path, transcript.to_dict())
    export_transcript(transcript_path, formats)
    manifest_data = {
        "schema_version": "1.0",
        "job_id": identity,
        "created_at": created_at,
        "source": source_metadata,
        "model": model.identity,
        "options": identity_options,
        "artifacts": [],
    }
    _write_manifest(manifest_path, manifest_data, preferred)
    progress(f"{position} Wrote {preferred}")
    return JobResult(source, preferred, transcript_path, manifest_path, False)


def _find_resumable_job(output_root: Path, preferred: Path, expected_job_id: str) -> Path | None:
    candidates = [preferred, *sorted(output_root.glob(f"*--{expected_job_id[:12]}"))]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        manifest_path = candidate / "manifest.json"
        transcript_path = candidate / "transcript.json"
        if not manifest_path.is_file() or not transcript_path.is_file():
            continue
        try:
            report = inspect_job(candidate)
        except (OSError, ValueError):
            continue
        if report["valid"] and report["job_id"] == expected_job_id:
            return candidate
    return None


def _write_manifest(path: Path, value: dict[str, Any], job_root: Path) -> None:
    artifact_paths = [job_root / name for name in ARTIFACT_FILENAMES if (job_root / name).is_file()]
    value["artifacts"] = [artifact_record(item, root=job_root) for item in artifact_paths]
    atomic_write_json(path, value)


def _managed_manifest_for_export(
    transcript_path: Path, metadata: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    manifest_path = transcript_path.with_name("manifest.json")
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Not a Whisper Lab manifest: {manifest_path}")
    transcript_job_id = metadata.get("job_id")
    if not transcript_job_id or manifest.get("job_id") != transcript_job_id:
        return None

    report = inspect_job(manifest_path)
    canonical_records = [item for item in report["artifacts"] if item["path"] == "transcript.json"]
    if len(canonical_records) != 1 or canonical_records[0]["status"] != "ok":
        raise ValueError("Managed transcript.json does not match its manifest")
    return manifest_path, manifest


def _remove_generated_job_files(job_root: Path) -> None:
    for name in JOB_FILENAMES:
        path = job_root / name
        if path.is_file() or path.is_symlink():
            path.unlink()
