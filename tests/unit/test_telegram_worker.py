from __future__ import annotations

import json
import queue
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from whisperlab.pipeline import DecodeOptions, JobResult
from whisperlab.telegram.api import MAX_DOWNLOAD_BYTES
from whisperlab.telegram.updates import MediaRequest
from whisperlab.telegram.worker import TelegramWorker
from whisperlab.transcript import Segment, Transcript


class FakeClient:
    def __init__(self, *, file_path: str = "voice/remote.oga", file_size: int = 5) -> None:
        self.file_path = file_path
        self.file_size = file_size
        self.messages = []
        self.documents = []
        self.destinations = []

    def send_message(self, chat_id, text, *, reply_to=None):
        self.messages.append((chat_id, text, reply_to))

    def send_document(self, chat_id, document, *, caption, reply_to=None):
        self.documents.append((chat_id, document, caption, reply_to))

    def get_file(self, file_id):
        return {"file_path": self.file_path, "file_size": self.file_size}

    def download_file(self, file_path, destination):
        self.destinations.append((file_path, destination))
        destination.write_bytes(b"audio")
        return 5


def media_request(update_id: int = 810000001) -> MediaRequest:
    return MediaRequest(
        update_id=update_id,
        chat_id=123456789,
        message_id=41,
        user_id=123456789,
        file_id="telegram-file-id",
        file_size=5,
        extension=".ogg",
        media_kind="voice",
    )


def write_result(root: Path, *, text: str = "hello locally") -> JobResult:
    job = root / "job--abc123"
    job.mkdir(parents=True, exist_ok=True)
    transcript_path = job / "transcript.json"
    transcript = Transcript(
        metadata={"job_id": "a" * 64},
        backend={"name": "test"},
        model={"repo": "test", "revision": "test"},
        language={"code": "en", "confidence": 1.0},
        timings={"duration_seconds": 65.0},
        segments=[Segment(id=1, start=0.0, end=65.0, text=text)],
    )
    transcript_path.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    transcript_path.with_suffix(".txt").write_text(text, encoding="utf-8")
    manifest = job / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return JobResult(Path("source.ogg"), job, transcript_path, manifest, False)


def worker(client, work_queue, root: Path) -> TelegramWorker:
    return TelegramWorker(
        work_queue,
        client,
        output_dir=root / "outputs",
        model_dir=root / "models",
        model="tiny",
        options=DecodeOptions(language="en", beam_size=1, vad_filter=True),
        device="cpu",
        compute_type="int8",
    )


class TelegramWorkerTests(unittest.TestCase):
    def test_download_reaches_existing_pipeline_with_safe_temporary_path(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            client = FakeClient(file_path="../../untrusted/telegram-name.oga")
            target_during_call = None

            def fake_run(**kwargs):
                nonlocal target_during_call
                target_during_call = kwargs["target"]
                self.assertTrue(target_during_call.is_file())
                self.assertEqual(target_during_call.name, "update-810000001.ogg")
                self.assertNotIn("untrusted", str(target_during_call))
                self.assertFalse(kwargs["allow_download"])
                self.assertTrue(kwargs["resume"])
                self.assertEqual(kwargs["model_name"], "tiny")
                self.assertEqual(kwargs["formats"], ("txt", "srt", "vtt"))
                self.assertEqual(kwargs["device"], "cpu")
                return [write_result(root)]

            with patch("whisperlab.telegram.worker.run_transcription", side_effect=fake_run):
                worker(client, queue.Queue(), root)._process(media_request())

            self.assertFalse(target_during_call.exists())
            self.assertEqual(client.destinations[0][0], "../../untrusted/telegram-name.oga")
            self.assertEqual(client.messages[-1][1].splitlines()[0], "hello locally")
            self.assertIn("job aaaaaaaaaaaa", client.messages[-1][1])

    def test_long_transcript_uses_existing_text_export_as_document(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            client = FakeClient()
            result = write_result(root, text="x" * 3900)
            with patch("whisperlab.telegram.worker.run_transcription", return_value=[result]):
                worker(client, queue.Queue(), root)._process(media_request())

            self.assertEqual(len(client.documents), 1)
            self.assertEqual(client.documents[0][1], result.transcript.with_suffix(".txt"))
            self.assertNotIn(str(root), client.documents[0][2])

    def test_non_bmp_text_uses_conservative_telegram_length(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            client = FakeClient()
            result = write_result(root, text="🙂" * 2000)
            with patch("whisperlab.telegram.worker.run_transcription", return_value=[result]):
                worker(client, queue.Queue(), root)._process(media_request())

            self.assertEqual(len(client.documents), 1)

    def test_empty_transcript_has_explicit_reply(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            client = FakeClient()
            result = write_result(root, text="")
            with patch("whisperlab.telegram.worker.run_transcription", return_value=[result]):
                worker(client, queue.Queue(), root)._process(media_request())

            self.assertIn("no speech was detected", client.messages[-1][1])

    def test_reported_overflow_never_reaches_pipeline(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            client = FakeClient(file_size=MAX_DOWNLOAD_BYTES + 1)
            with patch("whisperlab.telegram.worker.run_transcription") as run:
                worker(client, queue.Queue(), root)._process(media_request())

            run.assert_not_called()
            self.assertIn("20 MB", client.messages[-1][1])
            self.assertEqual(client.destinations, [])

    def test_pipeline_failure_is_actionable_without_stack_trace_in_chat(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            client = FakeClient()
            with (
                patch(
                    "whisperlab.telegram.worker.run_transcription",
                    side_effect=RuntimeError(
                        "Model 'tiny' is not available locally. "
                        "Download it first: whisperlab models download tiny"
                    ),
                ),
                redirect_stderr(StringIO()),
            ):
                worker(client, queue.Queue(), root)._process(media_request())

            self.assertIn("whisperlab models download tiny", client.messages[-1][1])
            self.assertNotIn("RuntimeError", client.messages[-1][1])
            self.assertFalse(client.destinations[0][1].exists())

    def test_unexpected_job_failure_does_not_kill_the_worker(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            client = FakeClient()
            work = queue.Queue(maxsize=16)
            runner = worker(client, work, root)
            result = write_result(root)

            with (
                patch(
                    "whisperlab.telegram.worker.run_transcription",
                    side_effect=[TypeError("unexpected test failure"), [result]],
                ) as run,
                redirect_stderr(StringIO()),
            ):
                runner.start()
                work.put_nowait(media_request(1))
                work.put_nowait(media_request(2))
                work.join()
                runner.stop()

            self.assertEqual(run.call_count, 2)
            self.assertTrue(
                any("Local transcription failed" in item[1] for item in client.messages)
            )
            self.assertIn("hello locally", client.messages[-1][1])
            self.assertTrue(all(not destination.exists() for _, destination in client.destinations))

    def test_shutdown_discards_pending_but_finishes_active_job(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            client = FakeClient()
            work = queue.Queue(maxsize=16)
            runner = worker(client, work, root)
            started = threading.Event()
            release = threading.Event()
            stopped = threading.Event()
            result = write_result(root)

            def slow_run(**_kwargs):
                started.set()
                release.wait(timeout=2)
                return [result]

            with patch("whisperlab.telegram.worker.run_transcription", side_effect=slow_run) as run:
                runner.start()
                work.put_nowait(media_request(1))
                self.assertTrue(started.wait(timeout=1))
                work.put_nowait(media_request(2))
                outcome = {}

                def stop():
                    outcome["discarded"] = runner.stop()
                    stopped.set()

                stopper = threading.Thread(target=stop)
                stopper.start()
                self.assertFalse(stopped.wait(timeout=0.05))
                release.set()
                stopper.join(timeout=1)

            self.assertTrue(stopped.is_set())
            self.assertEqual(outcome["discarded"], 1)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
