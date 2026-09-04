from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

TELEGRAM_API_ROOT = "https://api.telegram.org"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
_TOKEN_URL = re.compile(r"https?://api\.telegram\.org/(?:file/)?bot[^/\s]+")
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9:_-]+")


class TelegramAPIError(RuntimeError):
    """A secret-safe Telegram transport or API error."""


class TelegramFileTooLargeError(TelegramAPIError):
    """A Telegram file exceeds the standard Bot API download limit."""


def redact_telegram_error(value: object, token: str) -> str:
    message = str(value).replace(token, "[REDACTED]") if token else str(value)
    return _TOKEN_URL.sub("[REDACTED TELEGRAM URL]", message)


class TelegramClient:
    def __init__(
        self,
        token: str,
        *,
        timeout: float = 30.0,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        validate_bot_token(token)
        self._token = token
        self._timeout = timeout
        self._urlopen = urlopen

    def __repr__(self) -> str:
        return "TelegramClient(token=[REDACTED])"

    def get_me(self) -> dict[str, Any]:
        return self._object_result("getMe")

    def get_file(self, file_id: str) -> dict[str, Any]:
        return self._object_result("getFile", {"file_id": file_id})

    def send_message(self, chat_id: int, text: str, *, reply_to: int | None = None) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to is not None:
            payload["reply_parameters"] = {"message_id": reply_to}
        self._call("sendMessage", payload)

    def send_document(
        self,
        chat_id: int,
        document: Path,
        *,
        caption: str,
        reply_to: int | None = None,
    ) -> None:
        try:
            size = document.stat().st_size
            if size > MAX_DOCUMENT_BYTES:
                raise TelegramAPIError("Transcript document exceeds Telegram's 50 MB upload limit")
            content = document.read_bytes()
        except OSError as exc:
            raise TelegramAPIError(f"Could not read transcript document: {exc}") from exc
        if len(content) > MAX_DOCUMENT_BYTES:
            raise TelegramAPIError("Transcript document exceeds Telegram's 50 MB upload limit")

        boundary = f"whisperlab-{uuid.uuid4().hex}"
        fields: dict[str, str] = {"chat_id": str(chat_id), "caption": caption}
        if reply_to is not None:
            fields["reply_parameters"] = json.dumps({"message_id": reply_to})
        body = _multipart_body(boundary, fields, document.name, content)
        self._call_bytes(
            "sendDocument",
            body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )

    def set_webhook(
        self,
        url: str,
        secret: str,
        *,
        drop_pending_updates: bool,
    ) -> None:
        validate_webhook_url(url)
        self._call(
            "setWebhook",
            {
                "url": url,
                "secret_token": secret,
                "allowed_updates": ["message"],
                "max_connections": 1,
                "drop_pending_updates": drop_pending_updates,
            },
        )

    def get_webhook_info(self) -> dict[str, Any]:
        return self._object_result("getWebhookInfo")

    def delete_webhook(self, *, drop_pending_updates: bool) -> None:
        self._call("deleteWebhook", {"drop_pending_updates": drop_pending_updates})

    def download_file(self, file_path: str, destination: Path) -> int:
        remote_path = _validated_remote_path(file_path)
        quoted_path = urllib.parse.quote(remote_path, safe="/")
        url = f"{TELEGRAM_API_ROOT}/file/bot{self._token}/{quoted_path}"
        request = urllib.request.Request(url, method="GET")
        written = 0
        try:
            with self._urlopen(request, timeout=self._timeout) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
                    raise TelegramFileTooLargeError(_too_large_message())
                with destination.open("wb") as handle:
                    while chunk := response.read(64 * 1024):
                        written += len(chunk)
                        if written > MAX_DOWNLOAD_BYTES:
                            raise TelegramFileTooLargeError(_too_large_message())
                        handle.write(chunk)
        except TelegramFileTooLargeError:
            with suppress(FileNotFoundError):
                destination.unlink()
            raise
        except (OSError, ValueError, urllib.error.URLError) as exc:
            with suppress(FileNotFoundError):
                destination.unlink()
            detail = redact_telegram_error(exc, self._token)
            raise TelegramAPIError(f"Telegram file download failed: {detail}") from exc
        return written

    def _object_result(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self._call(method, payload)
        if not isinstance(result, dict):
            raise TelegramAPIError(f"Telegram API {method} returned an invalid result")
        return result

    def _call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload or {}, separators=(",", ":")).encode("utf-8")
        return self._call_bytes(method, data, content_type="application/json")

    def _call_bytes(self, method: str, data: bytes, *, content_type: str) -> Any:
        url = f"{TELEGRAM_API_ROOT}/bot{self._token}/{method}"
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with self._urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            description = _http_error_description(exc)
            detail = redact_telegram_error(description, self._token)
            raise TelegramAPIError(
                f"Telegram API {method} failed (HTTP {exc.code}): {detail}"
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            detail = redact_telegram_error(exc, self._token)
            raise TelegramAPIError(f"Telegram API {method} is unavailable: {detail}") from exc
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TelegramAPIError(f"Telegram API {method} returned invalid JSON") from exc
        if not isinstance(value, dict) or value.get("ok") is not True:
            description = (
                value.get("description", "request was rejected")
                if isinstance(value, dict)
                else "invalid response"
            )
            detail = redact_telegram_error(description, self._token)
            raise TelegramAPIError(f"Telegram API {method} failed: {detail}")
        return value.get("result")


def validate_webhook_url(value: str) -> None:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Telegram webhook URL must not contain control characters")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Telegram webhook URL must be an absolute HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("Telegram webhook URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Telegram webhook URL must not contain a query or fragment")


def validate_bot_token(value: str) -> None:
    """Keep the secret safe to embed in Telegram's mandatory token-bearing URL."""

    if not value or _SAFE_TOKEN.fullmatch(value) is None:
        raise ValueError("Telegram bot token has an invalid format")


def _validated_remote_path(value: str) -> str:
    if not value or any(ord(character) < 32 for character in value):
        raise TelegramAPIError("Telegram getFile returned an invalid file_path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise TelegramAPIError("Telegram getFile returned an invalid file_path")
    return path.as_posix()


def _multipart_body(
    boundary: str,
    fields: dict[str, str],
    filename: str,
    content: bytes,
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            (
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            )
        )
    safe_filename = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
    chunks.extend(
        (
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="document"; filename="{safe_filename}"\r\n'
            ).encode(),
            b"Content-Type: text/plain; charset=utf-8\r\n\r\n",
            content,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return b"".join(chunks)


def _http_error_description(error: urllib.error.HTTPError) -> str:
    try:
        value = json.loads(error.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return str(error.reason or "request failed")
    if isinstance(value, dict):
        return str(value.get("description", error.reason or "request failed"))
    return str(error.reason or "request failed")


def _too_large_message() -> str:
    return "Telegram media exceeds the standard Bot API 20 MB download limit"
