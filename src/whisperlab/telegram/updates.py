from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..pipeline import SUPPORTED_EXTENSIONS
from .api import MAX_DOWNLOAD_BYTES

_MIME_EXTENSIONS = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/x-aiff": ".aiff",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
}


@dataclass(frozen=True)
class MediaRequest:
    update_id: int
    chat_id: int
    message_id: int
    user_id: int
    file_id: str
    file_size: int | None
    extension: str
    media_kind: str


@dataclass(frozen=True)
class UpdateDecision:
    update_id: int
    response: dict[str, Any] | None
    media: MediaRequest | None = None


def decide_update(
    update: dict[str, Any], *, allowed_user_ids: frozenset[int], model: str
) -> UpdateDecision:
    update_id = _required_integer(update.get("update_id"), "update_id")
    message = update.get("message")
    if not isinstance(message, dict):
        return UpdateDecision(update_id, None)

    chat = message.get("chat")
    sender = message.get("from")
    if not isinstance(chat, dict) or not isinstance(sender, dict):
        return UpdateDecision(update_id, None)
    chat_id = _required_integer(chat.get("id"), "message.chat.id")
    user_id = _required_integer(sender.get("id"), "message.from.id")
    message_id = _required_integer(message.get("message_id"), "message.message_id")

    if chat.get("type") != "private":
        return UpdateDecision(
            update_id,
            webhook_message(chat_id, "Whisper Lab's Telegram bot works in private chats only."),
        )

    command = _command(message.get("text"))
    if command == "/whoami":
        return UpdateDecision(
            update_id,
            webhook_message(
                chat_id,
                f"Your Telegram user ID: {user_id}\nYour chat ID: {chat_id}",
                reply_to=message_id,
            ),
        )
    if command in {"/start", "/help"}:
        return UpdateDecision(
            update_id,
            webhook_message(chat_id, _help_text(), reply_to=message_id),
        )
    if command:
        return UpdateDecision(
            update_id,
            webhook_message(chat_id, "Unknown command. Use /help for the supported workflow."),
        )

    if user_id not in allowed_user_ids:
        return UpdateDecision(
            update_id,
            webhook_message(
                chat_id,
                f"You are not authorized to run local transcription. Your user ID is {user_id}.",
                reply_to=message_id,
            ),
        )

    media = _media_request(update_id, message_id, chat_id, user_id, message)
    if media is None:
        return UpdateDecision(
            update_id,
            webhook_message(
                chat_id,
                "Send a voice note, audio file, video note, video, or supported media document.",
                reply_to=message_id,
            ),
        )
    if media.file_size is not None and media.file_size > MAX_DOWNLOAD_BYTES:
        return UpdateDecision(
            update_id,
            webhook_message(
                chat_id,
                "That file is over Telegram's standard Bot API 20 MB download limit.",
                reply_to=message_id,
            ),
        )
    return UpdateDecision(
        update_id,
        webhook_message(
            chat_id,
            f"Received — queued for local transcription with {model_label(model)}.",
            reply_to=message_id,
        ),
        media,
    )


def webhook_message(chat_id: int, text: str, *, reply_to: int | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {"method": "sendMessage", "chat_id": chat_id, "text": text}
    if reply_to is not None:
        response["reply_parameters"] = {"message_id": reply_to}
    return response


def _media_request(
    update_id: int,
    message_id: int,
    chat_id: int,
    user_id: int,
    message: dict[str, Any],
) -> MediaRequest | None:
    for kind in ("voice", "audio", "video_note", "video", "document"):
        candidate = message.get(kind)
        if not isinstance(candidate, dict):
            continue
        file_id = candidate.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            return None
        extension = _media_extension(kind, candidate)
        if extension is None:
            return None
        return MediaRequest(
            update_id=update_id,
            chat_id=chat_id,
            message_id=message_id,
            user_id=user_id,
            file_id=file_id,
            file_size=_optional_size(candidate.get("file_size")),
            extension=extension,
            media_kind=kind,
        )
    return None


def _media_extension(kind: str, value: dict[str, Any]) -> str | None:
    if kind == "voice":
        return ".ogg"
    if kind == "video_note":
        return ".mp4"

    # Telegram's untrusted file_name is inspected only for a whitelisted suffix.
    # It is never retained or interpreted as a local path.
    filename = value.get("file_name")
    if isinstance(filename, str):
        suffix = _filename_suffix(filename)
        if suffix in SUPPORTED_EXTENSIONS:
            return suffix
    mime_type = value.get("mime_type")
    if isinstance(mime_type, str) and mime_type.lower() in _MIME_EXTENSIONS:
        return _MIME_EXTENSIONS[mime_type.lower()]
    if kind == "audio":
        return ".mp3"
    if kind == "video":
        return ".mp4"
    return None


def _command(value: object) -> str | None:
    if not isinstance(value, str) or not value.startswith("/"):
        return None
    first = value.strip().split(maxsplit=1)[0].lower()
    return first.split("@", maxsplit=1)[0]


def _required_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Telegram update has no valid {field}")
    return value


def _optional_size(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def model_label(value: str) -> str:
    label = re.split(r"[/\\]", value)[-1]
    label = "".join(character for character in label if character.isprintable()).strip()
    return label[:80] or "configured model"


def _filename_suffix(value: str) -> str:
    # Split both POSIX and Windows-looking names without ever resolving or joining a path.
    basename = re.split(r"[/\\]", value)[-1]
    dot = basename.rfind(".")
    return basename[dot:].lower() if dot >= 0 else ""


def _help_text() -> str:
    return (
        "Send a voice note, audio file, video note, video, or supported media document. "
        "Speech recognition runs locally on the owner's Mac with faster-whisper. "
        "Use /whoami to see your numeric Telegram ID."
    )
