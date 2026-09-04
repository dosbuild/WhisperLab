from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .api import validate_bot_token

BOT_TOKEN_ENV = "WHISPERLAB_TELEGRAM_BOT_TOKEN"
WEBHOOK_SECRET_ENV = "WHISPERLAB_TELEGRAM_WEBHOOK_SECRET"
ALLOWED_USER_IDS_ENV = "WHISPERLAB_TELEGRAM_ALLOWED_USER_IDS"

# Telegram accepts 1-256 characters. Whisper Lab deliberately requires at least
# 32 characters because this value authenticates requests that can consume local compute.
MIN_WEBHOOK_SECRET_LENGTH = 32
MAX_WEBHOOK_SECRET_LENGTH = 256
_WEBHOOK_SECRET = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str = field(repr=False)
    webhook_secret: str = field(repr=False)
    allowed_user_ids: frozenset[int]

    @classmethod
    def from_environment(cls) -> TelegramConfig:
        return cls(
            bot_token=bot_token_from_environment(),
            webhook_secret=webhook_secret_from_environment(),
            allowed_user_ids=allowed_user_ids_from_environment(),
        )


def bot_token_from_environment() -> str:
    token = os.environ.get(BOT_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(f"{BOT_TOKEN_ENV} is required")
    try:
        validate_bot_token(token)
    except ValueError as exc:
        raise RuntimeError(f"{BOT_TOKEN_ENV} has an invalid format") from exc
    return token


def webhook_secret_from_environment() -> str:
    secret = os.environ.get(WEBHOOK_SECRET_ENV, "").strip()
    validate_webhook_secret(secret)
    return secret


def validate_webhook_secret(secret: str) -> None:
    if not secret:
        raise RuntimeError(f"{WEBHOOK_SECRET_ENV} is required")
    if not MIN_WEBHOOK_SECRET_LENGTH <= len(secret) <= MAX_WEBHOOK_SECRET_LENGTH:
        raise RuntimeError(
            f"{WEBHOOK_SECRET_ENV} must be {MIN_WEBHOOK_SECRET_LENGTH}-"
            f"{MAX_WEBHOOK_SECRET_LENGTH} characters"
        )
    if _WEBHOOK_SECRET.fullmatch(secret) is None:
        raise RuntimeError(
            f"{WEBHOOK_SECRET_ENV} may contain only A-Z, a-z, 0-9, underscore, and hyphen"
        )


def allowed_user_ids_from_environment() -> frozenset[int]:
    raw = os.environ.get(ALLOWED_USER_IDS_ENV, "").strip()
    if not raw:
        return frozenset()
    values: set[int] = set()
    for item in raw.split(","):
        candidate = item.strip()
        try:
            user_id = int(candidate)
        except ValueError as exc:
            raise RuntimeError(f"{ALLOWED_USER_IDS_ENV} must contain numeric user IDs") from exc
        if user_id <= 0:
            raise RuntimeError(f"{ALLOWED_USER_IDS_ENV} must contain positive user IDs")
        values.add(user_id)
    return frozenset(values)


def configuration_checks() -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    try:
        bot_token_from_environment()
    except RuntimeError as exc:
        checks.append(("telegram token", "fail", str(exc)))
    else:
        checks.append(("telegram token", "pass", "configured"))

    try:
        webhook_secret_from_environment()
    except RuntimeError as exc:
        checks.append(("telegram webhook secret", "fail", str(exc)))
    else:
        checks.append(("telegram webhook secret", "pass", "configured (32+ characters)"))

    try:
        allowed = allowed_user_ids_from_environment()
    except RuntimeError as exc:
        checks.append(("telegram allowlist", "fail", str(exc)))
    else:
        if allowed:
            checks.append(("telegram allowlist", "pass", f"{len(allowed)} user(s) configured"))
        else:
            checks.append(
                (
                    "telegram allowlist",
                    "warn",
                    "empty; transcription is disabled but /whoami remains available",
                )
            )
    return checks
