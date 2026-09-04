from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from whisperlab.telegram.api import (
    TelegramAPIError,
    TelegramClient,
    TelegramFileTooLargeError,
    redact_telegram_error,
    validate_bot_token,
    validate_webhook_url,
)

TOKEN = "123456:secret-token"


class FakeResponse:
    def __init__(self, payload: bytes, *, headers: dict[str, str] | None = None) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


class FakeOpener:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        return self.responses.pop(0)


def api_response(result) -> FakeResponse:
    return FakeResponse(json.dumps({"ok": True, "result": result}).encode())


class TelegramApiTests(unittest.TestCase):
    def test_json_api_operations_build_expected_requests(self):
        opener = FakeOpener(
            api_response({"id": 1, "username": "bot"}),
            api_response({"file_path": "voice/file.oga", "file_size": 42}),
            api_response(True),
            api_response({"url": "https://example.test/telegram/webhook"}),
            api_response(True),
            api_response({"message_id": 1}),
        )
        client = TelegramClient(TOKEN, urlopen=opener)

        self.assertEqual(client.get_me()["username"], "bot")
        self.assertEqual(client.get_file("file-id")["file_size"], 42)
        client.set_webhook(
            "https://example.test/telegram/webhook",
            "a" * 32,
            drop_pending_updates=True,
        )
        self.assertIn("url", client.get_webhook_info())
        client.delete_webhook(drop_pending_updates=False)
        client.send_message(123, "hello", reply_to=9)

        methods = [item[0].full_url.rsplit("/", maxsplit=1)[-1] for item in opener.requests]
        self.assertEqual(
            methods,
            ["getMe", "getFile", "setWebhook", "getWebhookInfo", "deleteWebhook", "sendMessage"],
        )
        set_payload = json.loads(opener.requests[2][0].data)
        self.assertEqual(set_payload["allowed_updates"], ["message"])
        self.assertEqual(set_payload["max_connections"], 1)
        self.assertTrue(set_payload["drop_pending_updates"])
        send_payload = json.loads(opener.requests[5][0].data)
        self.assertEqual(send_payload["reply_parameters"], {"message_id": 9})

    def test_send_document_uses_multipart_without_exposing_path(self):
        with tempfile.TemporaryDirectory() as raw:
            transcript = Path(raw) / "transcript.txt"
            transcript.write_text("hello", encoding="utf-8")
            opener = FakeOpener(api_response({"message_id": 1}))
            TelegramClient(TOKEN, urlopen=opener).send_document(
                123,
                transcript,
                caption="Local • en",
                reply_to=9,
            )

        request = opener.requests[0][0]
        self.assertIn("multipart/form-data", request.headers["Content-type"])
        self.assertIn(b'filename="transcript.txt"', request.data)
        self.assertNotIn(str(transcript.parent).encode(), request.data)

    def test_streamed_download_is_bounded_and_removes_partial_file(self):
        opener = FakeOpener(FakeResponse(b"123456"))
        client = TelegramClient(TOKEN, urlopen=opener)
        with (
            tempfile.TemporaryDirectory() as raw,
            patch("whisperlab.telegram.api.MAX_DOWNLOAD_BYTES", 5),
        ):
            destination = Path(raw) / "safe.ogg"
            with self.assertRaises(TelegramFileTooLargeError):
                client.download_file("voice/remote.oga", destination)
            self.assertFalse(destination.exists())

    def test_remote_file_path_is_never_treated_as_local_destination(self):
        opener = FakeOpener()
        client = TelegramClient(TOKEN, urlopen=opener)
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "update-1.ogg"
            with self.assertRaisesRegex(TelegramAPIError, "invalid file_path"):
                client.download_file("../../private.txt", destination)
            self.assertFalse(destination.exists())
        self.assertEqual(opener.requests, [])

    def test_token_and_token_bearing_urls_are_redacted(self):
        error = urllib.error.URLError(f"https://api.telegram.org/bot{TOKEN}/getMe leaked {TOKEN}")

        def fail(*_args, **_kwargs):
            raise error

        with self.assertRaises(TelegramAPIError) as raised:
            TelegramClient(TOKEN, urlopen=fail).get_me()

        message = str(raised.exception)
        self.assertNotIn(TOKEN, message)
        self.assertNotIn("api.telegram.org/bot", message)
        self.assertIn("REDACTED", redact_telegram_error(error, TOKEN))

    def test_token_cannot_inject_into_bot_api_url(self):
        malicious = TOKEN + "/getMe?secret=leak"
        with self.assertRaisesRegex(ValueError, "invalid format") as raised:
            validate_bot_token(malicious)
        self.assertNotIn(malicious, str(raised.exception))

        with self.assertRaisesRegex(ValueError, "invalid format"):
            TelegramClient(malicious)

    def test_webhook_url_validation_is_strict(self):
        validate_webhook_url("https://example.test/telegram/webhook")
        for value in (
            "http://example.test/telegram/webhook",
            "https://user:pass@example.test/telegram/webhook",
            "https://example.test/telegram/webhook?token=x",
            "https://example.test/telegram/webhook#fragment",
            "https://example.test/telegram/\nwebhook",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_webhook_url(value)


if __name__ == "__main__":
    unittest.main()
