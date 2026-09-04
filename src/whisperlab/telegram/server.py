from __future__ import annotations

import hmac
import ipaddress
import json
import queue
import sys
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..models import resolve_model
from ..pipeline import DecodeOptions
from .api import TelegramAPIError, TelegramClient
from .config import TelegramConfig
from .updates import MediaRequest, decide_update, webhook_message
from .worker import TelegramWorker

WEBHOOK_PATH = "/telegram/webhook"
HEALTH_PATH = "/healthz"
MAX_WEBHOOK_BODY_BYTES = 1024 * 1024
DEFAULT_QUEUE_SIZE = 16
_BUSY_MESSAGE = "Local transcription is busy; resend this media later."


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    body: bytes = b""
    content_type: str = "application/json; charset=utf-8"


class UpdateDeduplicator:
    def __init__(self, *, capacity: int = 4096) -> None:
        self._capacity = capacity
        self._seen: set[int] = set()
        self._order: deque[int] = deque()
        self._lock = threading.Lock()
        self._accepting = True

    def admit_media(
        self,
        update_id: int,
        request: MediaRequest,
        work_queue: queue.Queue[MediaRequest | object],
    ) -> str:
        """Atomically queue media, recording its ID only after successful admission."""

        with self._lock:
            if update_id in self._seen:
                return "duplicate"
            if not self._accepting:
                return "closed"
            try:
                work_queue.put_nowait(request)
            except queue.Full:
                # Deliberate backpressure: the caller returns 200 with a busy message.
                # This update is not recorded as accepted.
                return "full"
            self._remember(update_id)
            return "admitted"

    def mark_handled(self, update_id: int) -> bool:
        with self._lock:
            if not self._accepting or update_id in self._seen:
                return False
            self._remember(update_id)
            return True

    def stop_accepting(self) -> None:
        with self._lock:
            self._accepting = False

    def contains(self, update_id: int) -> bool:
        with self._lock:
            return update_id in self._seen

    def _remember(self, update_id: int) -> None:
        self._seen.add(update_id)
        self._order.append(update_id)
        while len(self._order) > self._capacity:
            self._seen.discard(self._order.popleft())


class WebhookApplication:
    def __init__(
        self,
        *,
        webhook_secret: str,
        allowed_user_ids: frozenset[int],
        model: str,
        work_queue: queue.Queue[MediaRequest | object],
        deduplicator: UpdateDeduplicator | None = None,
    ) -> None:
        self._secret = webhook_secret
        self._allowed_user_ids = allowed_user_ids
        self._model = model
        self._queue = work_queue
        self.deduplicator = deduplicator or UpdateDeduplicator()

    def authorized(self, headers: Mapping[str, str]) -> bool:
        normalized = {key.lower(): value for key, value in headers.items()}
        supplied = normalized.get("x-telegram-bot-api-secret-token", "")
        return hmac.compare_digest(supplied, self._secret)

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> HTTPResponse:
        if path == HEALTH_PATH:
            if method != "GET":
                return _json_response(405, {"error": "method not allowed"})
            return _json_response(200, {"status": "ok"})
        if path != WEBHOOK_PATH:
            return _json_response(404, {"error": "not found"})
        if method != "POST":
            return _json_response(405, {"error": "method not allowed"})
        if not self.authorized(headers):
            return _json_response(403, {"error": "forbidden"})
        if len(body) > MAX_WEBHOOK_BODY_BYTES:
            return _json_response(413, {"error": "request too large"})
        try:
            update = json.loads(body.decode("utf-8"))
            if not isinstance(update, dict):
                raise ValueError("Telegram update must be a JSON object")
            decision = decide_update(
                update,
                allowed_user_ids=self._allowed_user_ids,
                model=self._model,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return _json_response(400, {"error": "invalid update"})

        if decision.media is not None:
            try:
                admission = self.deduplicator.admit_media(
                    decision.update_id,
                    decision.media,
                    self._queue,
                )
            except Exception:
                return _json_response(500, {"error": "queue admission failed"})
            if admission == "duplicate":
                return HTTPResponse(200)
            if admission == "closed":
                return _json_response(503, {"error": "shutting down"})
            if admission == "full":
                response = webhook_message(
                    decision.media.chat_id,
                    _BUSY_MESSAGE,
                    reply_to=decision.media.message_id,
                )
                return _json_response(200, response)
            return _json_response(200, decision.response or {})

        if not self.deduplicator.mark_handled(decision.update_id):
            return HTTPResponse(200)
        return _json_response(200, decision.response or {})

    def stop_accepting(self) -> None:
        self.deduplicator.stop_accepting()


class _WebhookHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve_telegram(
    *,
    host: str,
    port: int,
    config: TelegramConfig,
    output_dir: Path,
    model_dir: Path,
    model: str,
    options: DecodeOptions,
    device: str,
    compute_type: str,
) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    resolution = resolve_model(model, model_dir, allow_download=False)
    client = TelegramClient(config.bot_token)
    diagnostic_client = TelegramClient(config.bot_token, timeout=5.0)
    bot_name = _best_effort_bot_name(diagnostic_client)

    work_queue: queue.Queue[MediaRequest | object] = queue.Queue(maxsize=DEFAULT_QUEUE_SIZE)
    worker = TelegramWorker(
        work_queue,
        client,
        output_dir=output_dir,
        model_dir=model_dir,
        model=model,
        options=options,
        device=device,
        compute_type=compute_type,
    )
    application = WebhookApplication(
        webhook_secret=config.webhook_secret,
        allowed_user_ids=config.allowed_user_ids,
        model=model,
        work_queue=work_queue,
    )
    handler = _handler_for(application)
    server = _WebhookHTTPServer((host, port), handler)
    worker.start()

    policy = (
        f"{len(config.allowed_user_ids)} authorized user(s)"
        if config.allowed_user_ids
        else "transcription disabled (empty allowlist; /whoami available)"
    )
    print(f"Telegram bot: {bot_name}")
    print(f"Local receiver: http://{host}:{port}")
    print(f"Webhook path: {WEBHOOK_PATH}")
    print(f"Health check: http://{host}:{port}{HEALTH_PATH}")
    print(f"Model: {model} ({resolution.repo}@{resolution.revision[:12]})")
    print(f"Authorization: {policy}")
    print("Inference: local faster-whisper only; Telegram transports messages and media")
    print("Expose only the webhook through your chosen public HTTPS bridge.")
    if not _is_loopback(host):
        print(
            "warning: receiver is bound beyond loopback; protect it with a firewall and proxy",
            file=sys.stderr,
        )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping Telegram receiver…", file=sys.stderr)
    finally:
        application.stop_accepting()
        server.server_close()
        discarded = worker.stop()
        if discarded:
            print(f"Discarded {discarded} queued job(s) during shutdown.", file=sys.stderr)


def _handler_for(application: WebhookApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "WhisperLab"
        sys_version = ""

        def version_string(self) -> str:
            return self.server_version

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch(b"")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != WEBHOOK_PATH:
                self._dispatch(b"")
                return
            if not application.authorized(self.headers):
                self._write(application.handle("POST", self.path, self.headers, b""))
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._write(_json_response(400, {"error": "invalid content length"}))
                return
            if length < 0:
                self._write(_json_response(400, {"error": "invalid content length"}))
                return
            if length > MAX_WEBHOOK_BODY_BYTES:
                self._write(_json_response(413, {"error": "request too large"}))
                return
            self._dispatch(self.rfile.read(length))

        def do_PUT(self) -> None:  # noqa: N802
            self._dispatch(b"")

        def do_DELETE(self) -> None:  # noqa: N802
            self._dispatch(b"")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._dispatch(b"")

        def do_PATCH(self) -> None:  # noqa: N802
            self._dispatch(b"")

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _dispatch(self, body: bytes) -> None:
            self._write(application.handle(self.command, self.path, self.headers, body))

        def _write(self, response: HTTPResponse) -> None:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if response.body:
                self.wfile.write(response.body)

    return Handler


def _json_response(status: int, value: dict[str, Any]) -> HTTPResponse:
    return HTTPResponse(
        status,
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )


def _best_effort_bot_name(client: TelegramClient) -> str:
    try:
        me = client.get_me()
    except TelegramAPIError as exc:
        print(f"warning: getMe failed; continuing with local receiver: {exc}", file=sys.stderr)
        return "unverified (Telegram unavailable during startup)"
    username = me.get("username")
    if isinstance(username, str) and username:
        return f"@{username}"
    return "verified token (bot has no username)"


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
