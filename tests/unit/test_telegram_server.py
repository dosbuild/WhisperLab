from __future__ import annotations

import json
import queue
import threading
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from whisperlab.telegram.api import TelegramAPIError
from whisperlab.telegram.server import (
    HEALTH_PATH,
    WEBHOOK_PATH,
    UpdateDeduplicator,
    WebhookApplication,
    _best_effort_bot_name,
)

SECRET = "s" * 32
USER_ID = 123456789
HEADERS = {"X-Telegram-Bot-Api-Secret-Token": SECRET}


def fixture_body(name: str = "telegram_voice_update.json") -> bytes:
    return Path(f"tests/fixtures/{name}").read_bytes()


def application(work_queue=None, deduplicator=None):
    return WebhookApplication(
        webhook_secret=SECRET,
        allowed_user_ids=frozenset({USER_ID}),
        model="tiny",
        work_queue=work_queue or queue.Queue(maxsize=16),
        deduplicator=deduplicator,
    )


class TelegramServerTests(unittest.TestCase):
    def test_valid_secret_admits_media_then_records_update(self):
        work = queue.Queue(maxsize=16)
        app = application(work)

        response = app.handle("POST", WEBHOOK_PATH, HEADERS, fixture_body())

        self.assertEqual(response.status, 200)
        self.assertIn("queued", json.loads(response.body)["text"])
        self.assertEqual(work.qsize(), 1)
        self.assertTrue(app.deduplicator.contains(810000001))

    def test_missing_and_invalid_secrets_are_indistinguishable(self):
        app = application()
        missing = app.handle("POST", WEBHOOK_PATH, {}, fixture_body())
        invalid = app.handle(
            "POST",
            WEBHOOK_PATH,
            {"X-Telegram-Bot-Api-Secret-Token": "wrong"},
            fixture_body(),
        )

        self.assertEqual((missing.status, missing.body), (invalid.status, invalid.body))
        self.assertEqual(missing.status, 403)

    def test_queue_full_returns_200_and_does_not_mark_update(self):
        work = queue.Queue(maxsize=1)
        work.put_nowait(object())
        app = application(work)

        response = app.handle("POST", WEBHOOK_PATH, HEADERS, fixture_body())

        self.assertEqual(response.status, 200)
        self.assertIn("busy", json.loads(response.body)["text"])
        self.assertFalse(app.deduplicator.contains(810000001))

    def test_unauthorized_media_never_reaches_queue(self):
        work = queue.Queue(maxsize=16)
        app = WebhookApplication(
            webhook_secret=SECRET,
            allowed_user_ids=frozenset(),
            model="tiny",
            work_queue=work,
        )

        response = app.handle("POST", WEBHOOK_PATH, HEADERS, fixture_body())

        self.assertEqual(response.status, 200)
        self.assertIn("not authorized", json.loads(response.body)["text"])
        self.assertTrue(work.empty())

    def test_pre_admission_failure_does_not_mark_update(self):
        class BrokenQueue(queue.Queue):
            def put_nowait(self, item):
                raise RuntimeError("broken")

        app = application(BrokenQueue())
        response = app.handle("POST", WEBHOOK_PATH, HEADERS, fixture_body())

        self.assertEqual(response.status, 500)
        self.assertFalse(app.deduplicator.contains(810000001))

    def test_duplicate_update_is_queued_only_once(self):
        work = queue.Queue(maxsize=16)
        app = application(work)

        first = app.handle("POST", WEBHOOK_PATH, HEADERS, fixture_body())
        second = app.handle("POST", WEBHOOK_PATH, HEADERS, fixture_body())

        self.assertTrue(first.body)
        self.assertEqual(second.body, b"")
        self.assertEqual(work.qsize(), 1)

    def test_concurrent_duplicates_have_atomic_queue_admission(self):
        work = queue.Queue(maxsize=16)
        app = application(work)
        barrier = threading.Barrier(8)
        responses = []

        def invoke():
            barrier.wait()
            responses.append(app.handle("POST", WEBHOOK_PATH, HEADERS, fixture_body()))

        threads = [threading.Thread(target=invoke) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(work.qsize(), 1)
        self.assertEqual(sum(bool(response.body) for response in responses), 1)

    def test_command_is_handled_once_without_queueing(self):
        update = json.loads(fixture_body())
        update["message"].pop("voice")
        update["message"]["text"] = "/whoami"
        body = json.dumps(update).encode()
        work = queue.Queue()
        app = application(work)

        first = app.handle("POST", WEBHOOK_PATH, HEADERS, body)
        second = app.handle("POST", WEBHOOK_PATH, HEADERS, body)

        self.assertIn("user ID", json.loads(first.body)["text"])
        self.assertEqual(second.body, b"")
        self.assertTrue(work.empty())

    def test_health_paths_methods_and_invalid_json_are_minimal(self):
        app = application()
        health = app.handle("GET", HEALTH_PATH, {}, b"")
        wrong_method = app.handle("POST", HEALTH_PATH, {}, b"")
        missing = app.handle("GET", "/", {}, b"")
        malformed = app.handle("POST", WEBHOOK_PATH, HEADERS, b"not-json")

        self.assertEqual(json.loads(health.body), {"status": "ok"})
        self.assertEqual(wrong_method.status, 405)
        self.assertEqual(missing.status, 404)
        self.assertEqual(malformed.status, 400)

    def test_shutdown_closes_admission(self):
        app = application()
        app.stop_accepting()

        response = app.handle("POST", WEBHOOK_PATH, HEADERS, fixture_body())

        self.assertEqual(response.status, 503)
        self.assertFalse(app.deduplicator.contains(810000001))

    def test_deduplicator_is_bounded(self):
        deduplicator = UpdateDeduplicator(capacity=2)
        for update_id in (1, 2, 3):
            self.assertTrue(deduplicator.mark_handled(update_id))

        self.assertFalse(deduplicator.contains(1))
        self.assertTrue(deduplicator.contains(2))
        self.assertTrue(deduplicator.contains(3))

    def test_get_me_failure_is_a_sanitized_best_effort_warning(self):
        class OfflineClient:
            def get_me(self):
                raise TelegramAPIError("Telegram API getMe is unavailable: offline")

        stderr = StringIO()
        with redirect_stderr(stderr):
            name = _best_effort_bot_name(OfflineClient())

        self.assertIn("unverified", name)
        self.assertIn("continuing", stderr.getvalue())
        self.assertNotIn("token", stderr.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
