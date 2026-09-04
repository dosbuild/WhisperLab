from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import RuntimePaths
from .doctor import has_failures, run_doctor
from .inspect import inspect_job
from .models import download_model, model_rows
from .pipeline import DecodeOptions, export_transcript, normalize_formats, run_transcription


def _print_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def _formats(value: str) -> list[str]:
    try:
        return normalize_formats(value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _paths(args: argparse.Namespace) -> RuntimePaths:
    return RuntimePaths.from_values(
        output_dir=getattr(args, "output_dir", None),
        model_dir=getattr(args, "model_dir", None),
    )


def command_doctor(args: argparse.Namespace) -> int:
    paths = _paths(args)
    checks = run_doctor(
        model=args.model,
        model_dir=paths.model_dir,
        output_dir=paths.output_dir,
        strict=args.strict,
        telegram=args.telegram,
    )
    _print_rows(
        [{"check": check.name, "status": check.status, "detail": check.message} for check in checks]
    )
    return 1 if has_failures(checks) else 0


def command_models(args: argparse.Namespace) -> int:
    paths = _paths(args)
    if args.models_command == "list":
        _print_rows(model_rows(paths.model_dir))
        return 0
    print(f"Downloading {args.model}…", file=sys.stderr)
    resolution = download_model(args.model, paths.model_dir)
    print(resolution.path)
    return 0


def command_transcribe(args: argparse.Namespace) -> int:
    paths = _paths(args)
    language = None if args.language.lower() == "auto" else args.language
    results = run_transcription(
        target=Path(args.input),
        output_root=paths.output_dir,
        model_name=args.model,
        model_dir=paths.model_dir,
        options=DecodeOptions(
            language=language,
            beam_size=args.beam_size,
            vad_filter=args.vad,
            word_timestamps=args.word_timestamps,
        ),
        formats=args.formats,
        allow_download=args.download,
        resume=args.resume,
        device=args.device,
        compute_type=args.compute_type,
        progress=None if args.quiet else lambda message: print(message, file=sys.stderr),
    )
    for result in results:
        print(result.directory)
    return 0


def command_export(args: argparse.Namespace) -> int:
    for output in export_transcript(Path(args.transcript), args.formats):
        print(output)
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    report = inspect_job(Path(args.job))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        rows = [
            {"artifact": item["path"], "status": item["status"]} for item in report["artifacts"]
        ]
        print(f"job       {report['job_id']}")
        print(f"manifest  {report['manifest']}")
        print(f"segments  {report['segments']}")
        print(f"valid     {str(report['valid']).lower()}")
        print()
        _print_rows(rows)
    return 0 if report["valid"] else 1


def command_ui(args: argparse.Namespace) -> int:
    from .ui import serve_ui

    paths = _paths(args)
    serve_ui(
        host=args.host,
        port=args.port,
        output_dir=paths.output_dir,
        model_dir=paths.model_dir,
    )
    return 0


def command_telegram_serve(args: argparse.Namespace) -> int:
    from .telegram.config import TelegramConfig
    from .telegram.server import serve_telegram

    paths = _paths(args)
    language = None if args.language.lower() == "auto" else args.language
    serve_telegram(
        host=args.host,
        port=args.port,
        config=TelegramConfig.from_environment(),
        output_dir=paths.output_dir,
        model_dir=paths.model_dir,
        model=args.model,
        options=DecodeOptions(
            language=language,
            beam_size=args.beam_size,
            vad_filter=args.vad,
        ),
        device=args.device,
        compute_type=args.compute_type,
    )
    return 0


def command_telegram_webhook(args: argparse.Namespace) -> int:
    from .telegram.api import TelegramClient, redact_telegram_error
    from .telegram.config import bot_token_from_environment, webhook_secret_from_environment

    token = bot_token_from_environment()
    client = TelegramClient(token)
    if args.webhook_command == "set":
        client.set_webhook(
            args.url,
            webhook_secret_from_environment(),
            drop_pending_updates=args.drop_pending_updates,
        )
        print(f"Webhook registered: {args.url}")
        return 0
    if args.webhook_command == "delete":
        client.delete_webhook(drop_pending_updates=args.drop_pending_updates)
        print("Webhook deleted.")
        return 0

    info = client.get_webhook_info()
    last_error = redact_telegram_error(info.get("last_error_message", ""), token)
    allowed_updates = info.get("allowed_updates")
    allowed_updates_text = (
        ", ".join(str(item) for item in allowed_updates)
        if isinstance(allowed_updates, list)
        else "default"
    )
    _print_rows(
        [
            {"setting": "url", "value": info.get("url") or "not configured"},
            {"setting": "pending updates", "value": info.get("pending_update_count", 0)},
            {"setting": "max connections", "value": info.get("max_connections", "")},
            {
                "setting": "allowed updates",
                "value": allowed_updates_text or "default",
            },
            {"setting": "last error", "value": last_error or "none"},
        ]
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisperlab",
        description="Local, resumable speech-to-text with inspectable artifacts.",
        epilog="Start with: whisperlab models download tiny",
    )
    parser.add_argument("--version", action="version", version=f"whisperlab {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="check runtime and model readiness")
    doctor.add_argument("--model", default="small", help="model preset, Hub ID, or local directory")
    doctor.add_argument("--model-dir", help="model cache (default: .cache/models)")
    doctor.add_argument("--output-dir", help="artifact root (default: outputs)")
    doctor.add_argument(
        "--telegram",
        action="store_true",
        help="also check local Telegram environment configuration",
    )
    doctor.add_argument("--strict", action="store_true", help="treat warnings as failures")
    doctor.set_defaults(func=command_doctor)

    models = subcommands.add_parser("models", help="list or download models")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    model_list = model_commands.add_parser("list", help="show the curated model presets")
    model_list.add_argument("--model-dir", help="model cache (default: .cache/models)")
    model_list.set_defaults(func=command_models)
    model_download = model_commands.add_parser("download", help="download a model explicitly")
    model_download.add_argument("model", help="preset such as tiny, or a Hugging Face model ID")
    model_download.add_argument("--model-dir", help="model cache (default: .cache/models)")
    model_download.set_defaults(func=command_models)

    transcribe = subcommands.add_parser(
        "transcribe", help="transcribe one media file or every supported file in a directory"
    )
    transcribe.add_argument("input", help="media file or directory")
    transcribe.add_argument(
        "--model", default="small", help="preset, Hub ID, or local model directory"
    )
    transcribe.add_argument("--language", default="auto", help="language code, or auto (default)")
    transcribe.add_argument(
        "--beam-size", type=_positive_int, default=5, help="decoder beam size (default: 5)"
    )
    transcribe.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="filter non-speech regions",
    )
    transcribe.add_argument(
        "--word-timestamps", action="store_true", help="include word timing and probability data"
    )
    transcribe.add_argument(
        "--formats",
        type=_formats,
        default=_formats("txt,srt,vtt"),
        help="comma-separated txt,srt,vtt,json (default: txt,srt,vtt)",
    )
    transcribe.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    transcribe.add_argument("--compute-type", default="auto", help="CTranslate2 compute type")
    transcribe.add_argument("--output-dir", help="artifact root (default: outputs)")
    transcribe.add_argument("--model-dir", help="model cache (default: .cache/models)")
    transcribe.add_argument(
        "--download", action="store_true", help="allow an explicit model download if missing"
    )
    transcribe.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse a completed content/config-identical job",
    )
    transcribe.add_argument("--quiet", action="store_true", help="suppress progress messages")
    transcribe.set_defaults(func=command_transcribe)

    export = subcommands.add_parser("export", help="render text or subtitles from transcript.json")
    export.add_argument("transcript")
    export.add_argument(
        "--formats",
        type=_formats,
        default=_formats("txt,srt,vtt"),
        help="comma-separated txt,srt,vtt,json (default: txt,srt,vtt)",
    )
    export.set_defaults(func=command_export)

    inspect = subcommands.add_parser("inspect", help="verify a job manifest and artifact hashes")
    inspect.add_argument("job", help="job directory, manifest.json, or transcript.json")
    inspect.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    inspect.set_defaults(func=command_inspect)

    ui = subcommands.add_parser("ui", help="serve the small local transcription UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--model-dir", help="model cache (default: .cache/models)")
    ui.add_argument("--output-dir", help="artifact root (default: outputs)")
    ui.set_defaults(func=command_ui)

    telegram = subcommands.add_parser(
        "telegram", help="receive Telegram media for local transcription"
    )
    telegram_commands = telegram.add_subparsers(dest="telegram_command", required=True)
    telegram_serve = telegram_commands.add_parser(
        "serve", help="serve the local Telegram webhook receiver"
    )
    telegram_serve.add_argument(
        "--model", default="small", help="preset, Hub ID, or local model directory"
    )
    telegram_serve.add_argument(
        "--language", default="auto", help="language code, or auto (default)"
    )
    telegram_serve.add_argument(
        "--beam-size", type=_positive_int, default=5, help="decoder beam size (default: 5)"
    )
    telegram_serve.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="filter non-speech regions",
    )
    telegram_serve.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    telegram_serve.add_argument("--compute-type", default="auto", help="CTranslate2 compute type")
    telegram_serve.add_argument("--host", default="127.0.0.1")
    telegram_serve.add_argument("--port", type=int, default=8787)
    telegram_serve.add_argument("--model-dir", help="model cache (default: .cache/models)")
    telegram_serve.add_argument("--output-dir", help="artifact root (default: outputs)")
    telegram_serve.set_defaults(func=command_telegram_serve)

    webhook = telegram_commands.add_parser("webhook", help="manage the Telegram webhook")
    webhook_commands = webhook.add_subparsers(dest="webhook_command", required=True)
    webhook_set = webhook_commands.add_parser("set", help="register the public HTTPS webhook")
    webhook_set.add_argument("url")
    webhook_set.add_argument(
        "--drop-pending-updates",
        action="store_true",
        help="discard updates Telegram is already holding",
    )
    webhook_set.set_defaults(func=command_telegram_webhook)
    webhook_info = webhook_commands.add_parser("info", help="show Telegram webhook status")
    webhook_info.set_defaults(func=command_telegram_webhook)
    webhook_delete = webhook_commands.add_parser("delete", help="remove the Telegram webhook")
    webhook_delete.add_argument(
        "--drop-pending-updates",
        action="store_true",
        help="discard updates Telegram is already holding",
    )
    webhook_delete.set_defaults(func=command_telegram_webhook)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
