from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from whisperlab.artifacts import artifact_record
from whisperlab.inspect import inspect_job


class InspectTests(unittest.TestCase):
    def test_inspect_verifies_hashes_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            job = Path(raw)
            transcript = job / "transcript.json"
            transcript.write_text(
                Path("tests/fixtures/transcript.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            manifest = {
                "schema_version": "1.0",
                "job_id": "b" * 64,
                "source": {"name": "sample.wav"},
                "model": {"repo": "test", "revision": "test"},
                "options": {},
                "artifacts": [artifact_record(transcript, root=job)],
            }
            (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            before = inspect_job(job)
            changed = transcript.read_text(encoding="utf-8").replace("Whisper Lab", "Whisper Lob")
            transcript.write_text(changed, encoding="utf-8")
            after = inspect_job(job)

        self.assertTrue(before["valid"])
        self.assertEqual(before["segments"], 2)
        self.assertFalse(after["valid"])
        self.assertEqual(after["artifacts"][0]["status"], "hash mismatch")

    def test_inspect_rejects_artifact_path_traversal(self):
        with tempfile.TemporaryDirectory() as raw:
            job = Path(raw) / "job"
            job.mkdir()
            transcript = job / "transcript.json"
            transcript.write_text(
                Path("tests/fixtures/transcript.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            manifest = {
                "schema_version": "1.0",
                "job_id": "b" * 64,
                "artifacts": [{"path": "../private.txt", "sha256": "x"}],
            }
            (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = inspect_job(job)

        self.assertFalse(report["valid"])
        self.assertEqual(report["artifacts"][0]["status"], "unsafe path")

    def test_inspect_reports_malformed_artifact_records(self):
        with tempfile.TemporaryDirectory() as raw:
            job = Path(raw)
            (job / "transcript.json").write_text(
                Path("tests/fixtures/transcript.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (job / "manifest.json").write_text(
                json.dumps({"schema_version": "1.0", "job_id": "b" * 64, "artifacts": [7]}),
                encoding="utf-8",
            )

            report = inspect_job(job)

        self.assertFalse(report["valid"])
        self.assertIn({"path": "<invalid>", "status": "invalid record"}, report["artifacts"])

    def test_inspect_rejects_empty_or_untracked_canonical_membership(self):
        with tempfile.TemporaryDirectory() as raw:
            job = Path(raw)
            transcript = job / "transcript.json"
            transcript.write_text(
                Path("tests/fixtures/transcript.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            manifest_path = job / "manifest.json"
            manifest_path.write_text(
                json.dumps({"schema_version": "1.0", "job_id": "b" * 64, "artifacts": []}),
                encoding="utf-8",
            )

            empty = inspect_job(job)
            text = job / "transcript.txt"
            text.write_text("untracked\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "job_id": "b" * 64,
                        "artifacts": [artifact_record(text, root=job)],
                    }
                ),
                encoding="utf-8",
            )
            untracked = inspect_job(job)

        self.assertFalse(empty["valid"])
        self.assertIn({"path": "transcript.json", "status": "untracked"}, empty["artifacts"])
        self.assertFalse(untracked["valid"])
        self.assertTrue(
            any(
                item["path"] == "transcript.json" and item["status"] == "untracked"
                for item in untracked["artifacts"]
            )
        )

    def test_inspect_checks_recorded_size(self):
        with tempfile.TemporaryDirectory() as raw:
            job = Path(raw)
            transcript = job / "transcript.json"
            transcript.write_text(
                Path("tests/fixtures/transcript.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            record = artifact_record(transcript, root=job)
            record["bytes"] += 1
            (job / "manifest.json").write_text(
                json.dumps({"schema_version": "1.0", "job_id": "b" * 64, "artifacts": [record]}),
                encoding="utf-8",
            )

            report = inspect_job(job)

        self.assertFalse(report["valid"])
        self.assertEqual(report["artifacts"][0]["status"], "size mismatch")


if __name__ == "__main__":
    unittest.main()
