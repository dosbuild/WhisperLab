from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from whisperlab.cli import build_parser, main
from whisperlab.config import RuntimePaths


class ConfigAndCliTests(unittest.TestCase):
    def test_relative_runtime_paths_resolve_from_cwd(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = RuntimePaths.from_values(cwd=root, output_dir="results", model_dir="models")

        self.assertEqual(paths.output_dir, (root / "results").resolve())
        self.assertEqual(paths.model_dir, (root / "models").resolve())

    def test_transcribe_defaults_are_useful_and_explicit(self):
        args = build_parser().parse_args(["transcribe", "sample.wav"])

        self.assertEqual(args.model, "small")
        self.assertEqual(args.language, "auto")
        self.assertEqual(args.formats, ["txt", "srt", "vtt"])
        self.assertTrue(args.vad)
        self.assertTrue(args.resume)
        self.assertFalse(args.download)

    def test_invalid_format_is_a_usage_error(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["transcribe", "sample.wav", "--formats", "docx"])

    def test_missing_input_has_friendly_runtime_error(self):
        with tempfile.TemporaryDirectory() as raw:
            missing = Path(raw) / "missing.wav"
            stderr = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(stderr):
                code = main(["transcribe", str(missing), "--model", "tiny"])

        self.assertEqual(code, 1)
        self.assertIn("Input does not exist", stderr.getvalue())

    def test_cli_exposes_one_transcription_command(self):
        parser = build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if action.dest == "command"  # noqa: SLF001
        )

        self.assertIn("transcribe", subparsers.choices)
        self.assertNotIn("batch", subparsers.choices)

    def test_telegram_serve_defaults_are_local_and_do_not_download(self):
        args = build_parser().parse_args(["telegram", "serve"])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8787)
        self.assertEqual(args.model, "small")
        self.assertEqual(args.language, "auto")
        self.assertTrue(args.vad)
        self.assertFalse(hasattr(args, "download"))

    def test_telegram_webhook_command_shape(self):
        set_args = build_parser().parse_args(
            [
                "telegram",
                "webhook",
                "set",
                "https://example.test/telegram/webhook",
                "--drop-pending-updates",
            ]
        )
        info_args = build_parser().parse_args(["telegram", "webhook", "info"])
        delete_args = build_parser().parse_args(["telegram", "webhook", "delete"])

        self.assertEqual(set_args.webhook_command, "set")
        self.assertTrue(set_args.drop_pending_updates)
        self.assertEqual(info_args.webhook_command, "info")
        self.assertEqual(delete_args.webhook_command, "delete")


if __name__ == "__main__":
    unittest.main()
