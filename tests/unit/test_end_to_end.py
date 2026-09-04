from __future__ import annotations

import json
import queue
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from whisperlab.cli import main
from whisperlab.inspect import inspect_job
from whisperlab.models import ModelResolution
from whisperlab.pipeline import DecodeOptions
from whisperlab.telegram.server import WEBHOOK_PATH, WebhookApplication
from whisperlab.telegram.worker import TelegramWorker
from whisperlab.transcript import Segment, Transcript

SECRET = "s" * 32
USER_ID = 123456789
FIXTURES = Path(__file__).parents[1] / "fixtures"


def _model(root: Path) -> ModelResolution:
    return ModelResolution(
        requested="tiny",
        repo="Systran/faster-whisper-tiny",
        path=root / "model-snapshot",
        revision="local-test-revision",
    )


def _transcript(metadata: dict) -> Transcript:
    return Transcript(
        metadata=metadata,
        backend={"name": "faster-whisper-test-double"},
        model={"repo": "Systran/faster-whisper-tiny", "revision": "local-test-revision"},
        language={"code": "en", "confidence": 0.99},
        timings={"duration_seconds": 1.25},
        segments=[Segment(id=1, start=0.0, end=1.25, text="a local transcript")],
    )


class IntegrationTelegramClient:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str, int | None]] = []
        self.documents: list[tuple[int, Path, str, int | None]] = []
        self.download_destination: Path | None = None

    def get_file(self, file_id: str) -> dict:
        if file_id != "AwACAgQAAxkBAAIBvoice":
            raise AssertionError(f"unexpected file id: {file_id}")
        return {"file_path": "voice/telegram-generated-name.oga", "file_size": 12}

    def download_file(self, file_path: str, destination: Path) -> int:
        if file_path != "voice/telegram-generated-name.oga":
            raise AssertionError(f"unexpected remote path: {file_path}")
        self.download_destination = destination
        destination.write_bytes(b"fixture audio")
        return destination.stat().st_size

    def send_message(self, chat_id: int, text: str, *, reply_to: int | None = None) -> None:
        self.messages.append((chat_id, text, reply_to))

    def send_document(
        self,
        chat_id: int,
        document: Path,
        *,
        caption: str,
        reply_to: int | None = None,
    ) -> None:
        self.documents.append((chat_id, document, caption, reply_to))


class EndToEndTests(unittest.TestCase):
    def test_cli_to_pipeline_writes_an_inspectable_job_without_downloading(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "sample.wav"
            source.write_bytes(b"fixture audio")
            model = _model(root)
            stdout = StringIO()
            stderr = StringIO()

            def fake_inference(_source: Path, **kwargs) -> Transcript:
                return _transcript(kwargs["metadata"])

            with (
                patch("whisperlab.pipeline.resolve_model", return_value=model) as resolve,
                patch("whisperlab.pipeline.transcribe_audio", side_effect=fake_inference),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                status = main(
                    [
                        "transcribe",
                        str(source),
                        "--model",
                        "tiny",
                        "--model-dir",
                        str(root / "models"),
                        "--output-dir",
                        str(root / "outputs"),
                        "--device",
                        "cpu",
                        "--compute-type",
                        "int8",
                        "--quiet",
                    ]
                )

            self.assertEqual(status, 0, stderr.getvalue())
            resolve.assert_called_once()
            model_args, model_kwargs = resolve.call_args
            self.assertEqual(model_args[0], "tiny")
            self.assertEqual(model_args[1].resolve(), (root / "models").resolve())
            self.assertEqual(model_kwargs, {"allow_download": False})
            job = Path(stdout.getvalue().strip())
            report = inspect_job(job)
            self.assertTrue(report["valid"])
            self.assertEqual(report["segments"], 1)
            self.assertEqual(
                {item["path"] for item in report["artifacts"]},
                {"transcript.json", "transcript.txt", "transcript.srt", "transcript.vtt"},
            )
            self.assertEqual(
                (job / "transcript.txt").read_text(encoding="utf-8"),
                "a local transcript\n",
            )

    def test_webhook_crosses_real_queue_pipeline_artifacts_and_reply(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            client = IntegrationTelegramClient()
            work: queue.Queue = queue.Queue(maxsize=16)
            worker = TelegramWorker(
                work,
                client,
                output_dir=root / "outputs",
                model_dir=root / "models",
                model="tiny",
                options=DecodeOptions(language="en", beam_size=1),
                device="cpu",
                compute_type="int8",
            )
            application = WebhookApplication(
                webhook_secret=SECRET,
                allowed_user_ids=frozenset({USER_ID}),
                model="tiny",
                work_queue=work,
            )
            body = (FIXTURES / "telegram_voice_update.json").read_bytes()
            inference_sources: list[Path] = []

            def fake_inference(source: Path, **kwargs) -> Transcript:
                self.assertTrue(source.is_file())
                self.assertEqual(source.name, "update-810000001.ogg")
                inference_sources.append(source)
                return _transcript(kwargs["metadata"])

            with (
                patch("whisperlab.pipeline.resolve_model", return_value=_model(root)) as resolve,
                patch(
                    "whisperlab.pipeline.transcribe_audio", side_effect=fake_inference
                ) as inference,
                redirect_stderr(StringIO()),
            ):
                worker.start()
                try:
                    response = application.handle(
                        "POST",
                        WEBHOOK_PATH,
                        {"X-Telegram-Bot-Api-Secret-Token": SECRET},
                        body,
                    )
                    self.assertEqual(response.status, 200)
                    self.assertIn("queued", json.loads(response.body)["text"])
                    work.join()

                    duplicate = application.handle(
                        "POST",
                        WEBHOOK_PATH,
                        {"X-Telegram-Bot-Api-Secret-Token": SECRET},
                        body,
                    )
                    self.assertEqual(duplicate.status, 200)
                    self.assertEqual(duplicate.body, b"")
                finally:
                    worker.stop()

            resolve.assert_called_once()
            model_args, model_kwargs = resolve.call_args
            self.assertEqual(model_args[0], "tiny")
            self.assertEqual(model_args[1].resolve(), (root / "models").resolve())
            self.assertEqual(model_kwargs, {"allow_download": False})
            inference.assert_called_once()
            self.assertEqual(len(inference_sources), 1)
            self.assertFalse(inference_sources[0].exists())
            self.assertIsNotNone(client.download_destination)
            self.assertFalse(client.download_destination.exists())
            self.assertEqual(client.documents, [])
            self.assertIn("Transcribing locally with tiny", client.messages[0][1])
            self.assertIn("a local transcript", client.messages[-1][1])
            self.assertIn("Local • en • 0:01 • tiny • job ", client.messages[-1][1])

            jobs = list((root / "outputs").iterdir())
            self.assertEqual(len(jobs), 1)
            report = inspect_job(jobs[0])
            self.assertTrue(report["valid"])
            self.assertEqual(report["source"]["name"], "update-810000001.ogg")
            self.assertNotIn("telegram-generated-name", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
