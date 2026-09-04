from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from whisperlab.telegram.api import MAX_DOWNLOAD_BYTES
from whisperlab.telegram.config import (
    ALLOWED_USER_IDS_ENV,
    BOT_TOKEN_ENV,
    WEBHOOK_SECRET_ENV,
    TelegramConfig,
    validate_webhook_secret,
)
from whisperlab.telegram.updates import decide_update

SECRET = "a" * 32
USER_ID = 123456789


def fixture(name: str) -> dict:
    return json.loads(Path(f"tests/fixtures/{name}").read_text(encoding="utf-8"))


class TelegramConfigTests(unittest.TestCase):
    def test_configuration_comes_from_environment_without_secret_repr(self):
        environment = {
            BOT_TOKEN_ENV: "123456:token-value",
            WEBHOOK_SECRET_ENV: SECRET,
            ALLOWED_USER_IDS_ENV: "123456789, 987654321,123456789",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = TelegramConfig.from_environment()

        self.assertEqual(config.allowed_user_ids, frozenset({123456789, 987654321}))
        self.assertNotIn("token-value", repr(config))
        self.assertNotIn(SECRET, repr(config))

    def test_whisperlab_deliberately_requires_more_than_telegram_minimum(self):
        with self.assertRaisesRegex(RuntimeError, "32-256"):
            validate_webhook_secret("a")  # Telegram accepts this; Whisper Lab does not.
        validate_webhook_secret(SECRET)

    def test_webhook_secret_character_set_and_allowlist_are_strict(self):
        with self.assertRaisesRegex(RuntimeError, "only A-Z"):
            validate_webhook_secret("a" * 31 + ".")
        with (
            patch.dict(
                os.environ,
                {
                    BOT_TOKEN_ENV: "token",
                    WEBHOOK_SECRET_ENV: SECRET,
                    ALLOWED_USER_IDS_ENV: "name",
                },
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "numeric user IDs"),
        ):
            TelegramConfig.from_environment()

    def test_bot_token_rejects_url_control_characters_without_echoing_them(self):
        malicious = "123456:safe/../../token-leak"
        with (
            patch.dict(
                os.environ,
                {
                    BOT_TOKEN_ENV: malicious,
                    WEBHOOK_SECRET_ENV: SECRET,
                },
                clear=True,
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            TelegramConfig.from_environment()

        self.assertIn("invalid format", str(raised.exception))
        self.assertNotIn(malicious, str(raised.exception))


class TelegramUpdateTests(unittest.TestCase):
    def test_voice_is_authorized_and_parsed(self):
        decision = decide_update(
            fixture("telegram_voice_update.json"),
            allowed_user_ids=frozenset({USER_ID}),
            model="tiny",
        )

        self.assertEqual(decision.media.extension, ".ogg")
        self.assertEqual(decision.media.file_id, "AwACAgQAAxkBAAIBvoice")
        self.assertIn("queued", decision.response["text"])

    def test_supported_telegram_media_shapes(self):
        examples = {
            "audio": ({"file_id": "a", "mime_type": "audio/flac"}, ".flac"),
            "video_note": ({"file_id": "v"}, ".mp4"),
            "video": ({"file_id": "v", "file_name": "clip.webm"}, ".webm"),
            "document": ({"file_id": "d", "file_name": "talk.m4a"}, ".m4a"),
        }
        for kind, (media, extension) in examples.items():
            with self.subTest(kind=kind):
                update = fixture("telegram_voice_update.json")
                update["message"].pop("voice")
                update["message"][kind] = media
                decision = decide_update(
                    update,
                    allowed_user_ids=frozenset({USER_ID}),
                    model="tiny",
                )
                self.assertEqual(decision.media.extension, extension)

    def test_untrusted_document_name_only_contributes_safe_extension(self):
        decision = decide_update(
            fixture("telegram_document_update.json"),
            allowed_user_ids=frozenset({USER_ID}),
            model="tiny",
        )

        self.assertEqual(decision.media.extension, ".mp3")
        self.assertNotIn("private", repr(decision.media))
        self.assertNotIn("..", repr(decision.media))

    def test_unsupported_document_and_message_get_useful_response(self):
        update = fixture("telegram_document_update.json")
        update["message"]["document"] = {
            "file_id": "pdf",
            "file_name": "notes.pdf",
            "mime_type": "application/pdf",
        }
        decision = decide_update(
            update,
            allowed_user_ids=frozenset({USER_ID}),
            model="tiny",
        )

        self.assertIsNone(decision.media)
        self.assertIn("supported media", decision.response["text"])

    def test_oversized_media_is_rejected_before_queueing(self):
        update = fixture("telegram_voice_update.json")
        update["message"]["voice"]["file_size"] = MAX_DOWNLOAD_BYTES + 1
        decision = decide_update(
            update,
            allowed_user_ids=frozenset({USER_ID}),
            model="tiny",
        )

        self.assertIsNone(decision.media)
        self.assertIn("20 MB", decision.response["text"])

    def test_unauthorized_sender_cannot_create_media_work(self):
        decision = decide_update(
            fixture("telegram_voice_update.json"),
            allowed_user_ids=frozenset(),
            model="tiny",
        )

        self.assertIsNone(decision.media)
        self.assertIn(str(USER_ID), decision.response["text"])

    def test_whoami_is_available_without_authorization(self):
        update = fixture("telegram_voice_update.json")
        update["message"].pop("voice")
        update["message"]["text"] = "/whoami@WhisperLabBot"
        decision = decide_update(update, allowed_user_ids=frozenset(), model="tiny")

        self.assertIn(f"user ID: {USER_ID}", decision.response["text"])
        self.assertIn(f"chat ID: {USER_ID}", decision.response["text"])

    def test_private_chat_policy_and_help(self):
        update = fixture("telegram_voice_update.json")
        update["message"]["chat"] = {"id": -1001, "type": "supergroup"}
        decision = decide_update(
            update,
            allowed_user_ids=frozenset({USER_ID}),
            model="tiny",
        )
        self.assertIsNone(decision.media)
        self.assertIn("private chats only", decision.response["text"])

        update = fixture("telegram_voice_update.json")
        update["message"].pop("voice")
        update["message"]["text"] = "/help"
        decision = decide_update(update, allowed_user_ids=frozenset(), model="tiny")
        self.assertIn("runs locally", decision.response["text"])


if __name__ == "__main__":
    unittest.main()
