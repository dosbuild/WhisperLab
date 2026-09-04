from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from whisperlab.artifacts import artifact_record
from whisperlab.inspect import inspect_job
from whisperlab.pipeline import export_transcript, normalize_formats
from whisperlab.transcript import load_transcript, render_srt, render_txt, render_vtt


class TranscriptTests(unittest.TestCase):
    def test_renderers_round_only_at_the_output_edge(self):
        transcript = load_transcript(Path("tests/fixtures/transcript.json"))

        self.assertEqual(transcript.segments[0].end, 1.23456)
        self.assertEqual(render_txt(transcript), "Привет мир\nWhisper Lab\n")
        self.assertIn("00:00:00,000 --> 00:00:01,235", render_srt(transcript))
        self.assertIn("00:00:01.250 --> 00:00:02.500", render_vtt(transcript))

    def test_export_is_explicit_and_does_not_duplicate_json(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                Path("tests/fixtures/transcript.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            outputs = export_transcript(transcript_path, ["txt", "json"])

            self.assertEqual(
                outputs, [(root / "transcript.txt").resolve(), transcript_path.resolve()]
            )
            self.assertEqual(
                sorted(path.name for path in root.iterdir()), ["transcript.json", "transcript.txt"]
            )

    def test_formats_are_normalized_and_deduplicated(self):
        self.assertEqual(normalize_formats([" TXT ", "txt", "VTT"]), ["txt", "vtt"])

    def test_managed_job_export_updates_and_preserves_consistent_artifacts(self):
        with tempfile.TemporaryDirectory() as raw:
            job = Path(raw)
            transcript_path = job / "transcript.json"
            transcript_path.write_text(
                Path("tests/fixtures/transcript.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            manifest = {
                "schema_version": "1.0",
                "job_id": "b" * 64,
                "artifacts": [artifact_record(transcript_path, root=job)],
            }
            (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            export_transcript(transcript_path, ["txt"])
            (job / "transcript.txt").write_text("tampered", encoding="utf-8")
            export_transcript(transcript_path, ["srt"])
            report = inspect_job(job)

        self.assertTrue(report["valid"])
        self.assertEqual(
            [item["path"] for item in report["artifacts"]],
            ["transcript.json", "transcript.txt", "transcript.srt"],
        )

    def test_malformed_structured_fields_raise_a_public_value_error(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "transcript.json"
            value = json.loads(Path("tests/fixtures/transcript.json").read_text(encoding="utf-8"))
            value["metadata"] = None
            path.write_text(json.dumps(value), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "metadata must be an object"):
                load_transcript(path)


if __name__ == "__main__":
    unittest.main()
