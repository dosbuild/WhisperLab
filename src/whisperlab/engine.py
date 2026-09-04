from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
from typing import Any

from .models import ModelResolution
from .transcript import Segment, Transcript


def transcribe_audio(
    input_path: Path,
    *,
    model: ModelResolution,
    options: dict[str, Any],
    device: str,
    compute_type: str,
    metadata: dict[str, Any],
) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "faster-whisper is unavailable; install the project with: pip install -e ."
        ) from exc

    load_started = time.perf_counter()
    try:
        runtime = WhisperModel(
            str(model.path),
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load model '{model.requested}': {exc}") from exc
    load_finished = time.perf_counter()

    try:
        stream, info = runtime.transcribe(
            str(input_path),
            language=options["language"],
            beam_size=options["beam_size"],
            vad_filter=options["vad_filter"],
            word_timestamps=options["word_timestamps"],
        )
        segments = []
        words = [] if options["word_timestamps"] else None
        for index, item in enumerate(stream, start=1):
            segments.append(
                Segment(
                    id=index,
                    start=float(item.start or 0.0),
                    end=float(item.end or 0.0),
                    text=(item.text or "").strip(),
                )
            )
            if words is not None:
                for word in item.words or []:
                    words.append(
                        {
                            "start": float(word.start or 0.0),
                            "end": float(word.end or 0.0),
                            "word": str(word.word),
                            "probability": float(word.probability),
                        }
                    )
    except Exception as exc:
        raise RuntimeError(f"Transcription failed for {input_path.name}: {exc}") from exc
    inference_finished = time.perf_counter()

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    inference_seconds = inference_finished - load_finished
    return Transcript(
        metadata=metadata,
        backend={
            "name": "faster-whisper",
            "version": _package_version("faster-whisper"),
            "ctranslate2_version": _package_version("ctranslate2"),
            "device": device,
            "compute_type": compute_type,
        },
        model={"repo": model.repo, "revision": model.revision},
        language={
            "code": getattr(info, "language", options["language"]),
            "confidence": getattr(info, "language_probability", None),
        },
        timings={
            "duration_seconds": duration,
            "model_load_seconds": load_finished - load_started,
            "inference_seconds": inference_seconds,
            "real_time_factor": inference_seconds / duration if duration else None,
        },
        segments=segments,
        words=words,
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
