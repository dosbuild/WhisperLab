from __future__ import annotations

import json
import queue
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from ..pipeline import DecodeOptions, run_transcription
from ..transcript import load_transcript, render_txt
from .api import (
    MAX_DOWNLOAD_BYTES,
    TelegramAPIError,
    TelegramClient,
    TelegramFileTooLargeError,
)
from .updates import MediaRequest, model_label

_STOP = object()
_TEXT_RESULT_LIMIT = 3800


class TelegramWorker:
    def __init__(
        self,
        work_queue: queue.Queue[MediaRequest | object],
        client: TelegramClient,
        *,
        output_dir: Path,
        model_dir: Path,
        model: str,
        options: DecodeOptions,
        device: str,
        compute_type: str,
    ) -> None:
        self._queue = work_queue
        self._client = client
        self._output_dir = output_dir
        self._model_dir = model_dir
        self._model = model
        self._options = options
        self._device = device
        self._compute_type = compute_type
        self._thread = threading.Thread(
            target=self._run,
            name="whisperlab-telegram-worker",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        """Discard pending jobs, then wait for the active job to finish."""

        discarded = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
            if item is not _STOP:
                discarded += 1
        self._queue.put_nowait(_STOP)
        self._thread.join()
        return discarded

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, MediaRequest)
                try:
                    self._process(item)
                except Exception as exc:
                    # One malformed artifact or unexpected implementation error must not
                    # permanently kill the only transcription worker.
                    self._terminal_error(item, exc)
                    self._send_failure(
                        item,
                        "Local transcription failed. Check the WhisperLab terminal for details.",
                    )
            finally:
                self._queue.task_done()

    def _process(self, request: MediaRequest) -> None:
        self._send_status(request)
        try:
            with tempfile.TemporaryDirectory(prefix="whisperlab-telegram-") as raw:
                temporary_root = Path(raw)
                info = self._client.get_file(request.file_id)
                _check_reported_size(info.get("file_size"))
                file_path = info.get("file_path")
                if not isinstance(file_path, str) or not file_path:
                    raise TelegramAPIError("Telegram getFile returned no file_path")

                # file_path and the original file_name are never joined onto this directory.
                # Whisper Lab constructs the complete destination from trusted values only.
                destination = temporary_root / (f"update-{request.update_id}{request.extension}")
                self._client.download_file(file_path, destination)
                results = run_transcription(
                    target=destination,
                    output_root=self._output_dir,
                    model_name=self._model,
                    model_dir=self._model_dir,
                    options=self._options,
                    formats=("txt", "srt", "vtt"),
                    allow_download=False,
                    resume=True,
                    device=self._device,
                    compute_type=self._compute_type,
                    progress=lambda message: print(
                        f"telegram update {request.update_id}: {message}", file=sys.stderr
                    ),
                )
                if len(results) != 1:
                    raise RuntimeError("Telegram media did not produce exactly one job")
                result = results[0]
        except TelegramFileTooLargeError:
            self._send_failure(
                request,
                "That file is over Telegram's standard Bot API 20 MB download limit.",
            )
            return
        except TelegramAPIError as exc:
            self._terminal_error(request, exc)
            self._send_failure(request, "Telegram could not download that media. Please resend it.")
            return
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            self._terminal_error(request, exc)
            self._send_failure(request, _pipeline_error_message(exc, self._model))
            return

        try:
            self._deliver_result(request, result.transcript)
        except (TelegramAPIError, OSError, ValueError, json.JSONDecodeError) as exc:
            self._terminal_error(request, exc)

    def _send_status(self, request: MediaRequest) -> None:
        try:
            self._client.send_message(
                request.chat_id,
                f"Transcribing locally with {model_label(self._model)}…",
                reply_to=request.message_id,
            )
        except TelegramAPIError as exc:
            self._terminal_error(request, exc)

    def _send_failure(self, request: MediaRequest, message: str) -> None:
        try:
            self._client.send_message(request.chat_id, message, reply_to=request.message_id)
        except TelegramAPIError as exc:
            self._terminal_error(request, exc)

    def _deliver_result(self, request: MediaRequest, transcript_path: Path) -> None:
        transcript = load_transcript(transcript_path)
        text = render_txt(transcript).strip()
        summary = _result_summary(transcript, self._model)
        if not text:
            self._client.send_message(
                request.chat_id,
                f"Transcription completed, but no speech was detected.\n\n{summary}",
                reply_to=request.message_id,
            )
            return
        message = f"{text}\n\n{summary}"
        if _telegram_text_length(message) <= _TEXT_RESULT_LIMIT:
            self._client.send_message(
                request.chat_id,
                message,
                reply_to=request.message_id,
            )
            return
        document = transcript_path.with_suffix(".txt")
        self._client.send_document(
            request.chat_id,
            document,
            caption=summary,
            reply_to=request.message_id,
        )

    @staticmethod
    def _terminal_error(request: MediaRequest, error: BaseException) -> None:
        print(f"telegram update {request.update_id} failed: {error}", file=sys.stderr)


def _check_reported_size(value: Any) -> None:
    if isinstance(value, int) and not isinstance(value, bool) and value > MAX_DOWNLOAD_BYTES:
        raise TelegramFileTooLargeError(
            "Telegram media exceeds the standard Bot API 20 MB download limit"
        )


def _pipeline_error_message(error: BaseException, model: str) -> str:
    detail = str(error)
    if "not available locally" in detail:
        if model in {"tiny", "small", "turbo", "large-v3"}:
            return f"The local model is unavailable. Operator: whisperlab models download {model}"
        return "The configured local model is unavailable. Ask the bot operator to run doctor."
    if "faster-whisper is unavailable" in detail:
        return "Local transcription is not installed correctly. Ask the bot operator to run doctor."
    return "Local transcription failed. Check the WhisperLab terminal for details."


def _result_summary(transcript: Any, model: str) -> str:
    language = str(transcript.language.get("code") or "unknown")
    duration = _format_duration(transcript.timings.get("duration_seconds"))
    job_id = str(transcript.metadata.get("job_id") or "unknown")[:12]
    return f"Local • {language} • {duration} • {model_label(model)} • job {job_id}"


def _format_duration(value: object) -> str:
    try:
        seconds = max(0, round(float(value)))
    except (TypeError, ValueError):
        return "unknown duration"
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def _telegram_text_length(value: str) -> int:
    """Conservatively count UTF-16 code units, as Telegram clients do."""

    return len(value.encode("utf-16-le")) // 2
