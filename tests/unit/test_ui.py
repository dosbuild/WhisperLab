from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

from whisperlab.cli import build_parser
from whisperlab.ui import (
    MAX_FORM_BYTES,
    _claim_run,
    _command_from_form,
    _content_length,
    _safe_artifact,
    _valid_csrf,
)


class UiTests(unittest.TestCase):
    def test_ui_binds_to_loopback_by_default(self):
        args = build_parser().parse_args(["ui"])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)

    def test_ui_constructs_the_same_explicit_cli_contract(self):
        command = _command_from_form(
            {
                "target": ["recording.m4a"],
                "model": ["tiny"],
                "language": ["en"],
                "formats": ["txt,vtt"],
                "beam_size": ["1"],
                "resume": ["on"],
                "vad": ["on"],
            },
            output_dir=Path("outputs"),
            model_dir=Path("models"),
        )

        self.assertEqual(command[:4], [sys.executable, "-m", "whisperlab", "transcribe"])
        self.assertIn("--resume", command)
        self.assertIn("--vad", command)
        self.assertNotIn("--download", command)

    def test_artifact_route_stays_inside_output_root(self):
        root = Path("outputs").resolve()

        self.assertEqual(_safe_artifact(root, "job/transcript.txt"), root / "job/transcript.txt")
        with self.assertRaisesRegex(ValueError, "outside"):
            _safe_artifact(root, "../README.md")

    def test_csrf_token_is_required_and_compared_exactly(self):
        self.assertFalse(_valid_csrf({}, "expected"))
        self.assertFalse(_valid_csrf({"csrf_token": ["wrong"]}, "expected"))
        self.assertTrue(_valid_csrf({"csrf_token": ["expected"]}, "expected"))

    def test_content_length_rejects_missing_invalid_negative_and_oversized_values(self):
        for value in (None, "", "invalid", "-1", str(MAX_FORM_BYTES + 1)):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "content length|large"),
            ):
                _content_length(value)
        self.assertEqual(_content_length("0"), 0)
        self.assertEqual(_content_length(str(MAX_FORM_BYTES)), MAX_FORM_BYTES)

    def test_only_one_concurrent_ui_run_is_admitted(self):
        state = {"running": False}
        lock = threading.Lock()
        barrier = threading.Barrier(8)
        outcomes = []

        def claim(index):
            barrier.wait()
            outcomes.append(_claim_run(state, lock, ["whisperlab", str(index)]))

        threads = [threading.Thread(target=claim, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(outcomes.count(True), 1)
        self.assertEqual(outcomes.count(False), 7)


if __name__ == "__main__":
    unittest.main()
