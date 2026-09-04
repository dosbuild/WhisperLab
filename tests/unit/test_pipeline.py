from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from whisperlab.ids import job_id
from whisperlab.inspect import inspect_job
from whisperlab.models import ModelResolution
from whisperlab.pipeline import DecodeOptions, discover_inputs, run_transcription
from whisperlab.transcript import Segment, Transcript


class PipelineTests(unittest.TestCase):
    def test_discovery_is_recursive_sorted_and_ignores_hidden_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "b.mp3").write_bytes(b"b")
            (root / "nested").mkdir()
            (root / "nested" / "a.WAV").write_bytes(b"a")
            (root / ".hidden.wav").write_bytes(b"hidden")
            (root / "notes.txt").write_text("notes", encoding="utf-8")

            inputs = discover_inputs(root)

        self.assertEqual([path.name for path in inputs], ["b.mp3", "a.WAV"])

    def test_job_identity_changes_with_model_or_options(self):
        common = {"source_sha256": "a" * 64}
        first = job_id(
            **common,
            model={"repo": "model", "revision": "one"},
            options={"language": "en"},
        )
        changed_model = job_id(
            **common,
            model={"repo": "model", "revision": "two"},
            options={"language": "en"},
        )
        changed_options = job_id(
            **common,
            model={"repo": "model", "revision": "one"},
            options={"language": "fr"},
        )

        self.assertNotEqual(first, changed_model)
        self.assertNotEqual(first, changed_options)

    def test_pipeline_writes_portable_artifacts_and_resumes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "Interview.wav"
            source.write_bytes(b"deliberate fixture audio")
            output = root / "outputs"
            cache = root / "models"
            model = ModelResolution(
                requested="tiny",
                repo="Systran/faster-whisper-tiny",
                path=root / "snapshot",
                revision="revision-123",
            )
            fake = Transcript(
                metadata={},
                backend={"name": "test"},
                model=model.identity,
                language={"code": "en", "confidence": 1.0},
                timings={"duration_seconds": 1.0},
                segments=[Segment(id=1, start=0.0, end=1.0, text="hello")],
            )

            def fake_transcribe(_source, **kwargs):
                fake.metadata = kwargs["metadata"]
                return fake

            with (
                patch("whisperlab.pipeline.resolve_model", return_value=model),
                patch("whisperlab.pipeline.transcribe_audio", side_effect=fake_transcribe) as run,
            ):
                first = run_transcription(
                    target=source,
                    output_root=output,
                    model_name="tiny",
                    model_dir=cache,
                    options=DecodeOptions(language="en"),
                    formats=["txt", "srt"],
                    allow_download=False,
                    resume=True,
                    device="cpu",
                    compute_type="int8",
                )
                second = run_transcription(
                    target=source,
                    output_root=output,
                    model_name="tiny",
                    model_dir=cache,
                    options=DecodeOptions(language="en"),
                    formats=["txt", "srt", "vtt"],
                    allow_download=False,
                    resume=True,
                    device="cpu",
                    compute_type="int8",
                )

            self.assertEqual(run.call_count, 1)
            self.assertFalse(first[0].resumed)
            self.assertTrue(second[0].resumed)
            self.assertEqual(first[0].directory, second[0].directory)
            self.assertTrue((first[0].directory / "transcript.vtt").is_file())
            manifest = json.loads(first[0].manifest.read_text(encoding="utf-8"))
            artifact_paths = [record["path"] for record in manifest["artifacts"]]
            self.assertEqual(
                artifact_paths,
                ["transcript.json", "transcript.txt", "transcript.srt", "transcript.vtt"],
            )
            self.assertTrue(all(not Path(path).is_absolute() for path in artifact_paths))
            self.assertNotIn("host", manifest)
            self.assertNotIn(str(root), json.dumps(manifest))

    def test_empty_directory_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with self.assertRaisesRegex(RuntimeError, "No supported media files"):
                run_transcription(
                    target=root,
                    output_root=root / "out",
                    model_name="tiny",
                    model_dir=root / "models",
                    options=DecodeOptions(),
                    formats=["txt"],
                    allow_download=False,
                    resume=True,
                    device="auto",
                    compute_type="auto",
                )

    def test_tampered_canonical_transcript_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "sample.wav"
            source.write_bytes(b"fixture audio")
            model = ModelResolution("tiny", "test/model", root / "snapshot", "revision")

            def fake_transcribe(_source, **kwargs):
                return Transcript(
                    metadata=kwargs["metadata"],
                    backend={"name": "test"},
                    model=model.identity,
                    language={"code": "en"},
                    timings={"duration_seconds": 1.0},
                    segments=[Segment(id=1, start=0.0, end=1.0, text="trusted")],
                )

            arguments = {
                "target": source,
                "output_root": root / "outputs",
                "model_name": "tiny",
                "model_dir": root / "models",
                "options": DecodeOptions(language="en"),
                "formats": ["txt"],
                "allow_download": False,
                "resume": True,
                "device": "cpu",
                "compute_type": "int8",
            }
            with (
                patch("whisperlab.pipeline.resolve_model", return_value=model),
                patch("whisperlab.pipeline.transcribe_audio", side_effect=fake_transcribe) as run,
            ):
                first = run_transcription(**arguments)[0]
                value = json.loads(first.transcript.read_text(encoding="utf-8"))
                value["segments"][0]["text"] = "tampered"
                first.transcript.write_text(json.dumps(value), encoding="utf-8")
                second = run_transcription(**arguments)[0]

            self.assertEqual(run.call_count, 2)
            self.assertFalse(second.resumed)
            self.assertTrue(inspect_job(second.directory)["valid"])
            self.assertEqual(
                (second.directory / "transcript.txt").read_text(encoding="utf-8"),
                "trusted\n",
            )

    def test_no_resume_removes_only_stale_generated_formats(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "sample.wav"
            source.write_bytes(b"fixture audio")
            model = ModelResolution("tiny", "test/model", root / "snapshot", "revision")

            def fake_transcribe(_source, **kwargs):
                return Transcript(
                    metadata=kwargs["metadata"],
                    backend={"name": "test"},
                    model=model.identity,
                    language={"code": "en"},
                    timings={"duration_seconds": 1.0},
                    segments=[Segment(id=1, start=0.0, end=1.0, text="fresh")],
                )

            common = {
                "target": source,
                "output_root": root / "outputs",
                "model_name": "tiny",
                "model_dir": root / "models",
                "options": DecodeOptions(),
                "allow_download": False,
                "resume": False,
                "device": "cpu",
                "compute_type": "int8",
            }
            with (
                patch("whisperlab.pipeline.resolve_model", return_value=model),
                patch("whisperlab.pipeline.transcribe_audio", side_effect=fake_transcribe),
            ):
                first = run_transcription(formats=["txt", "srt"], **common)[0]
                unrelated = first.directory / "notes.md"
                unrelated.write_text("keep me", encoding="utf-8")
                second = run_transcription(formats=["vtt"], **common)[0]

            self.assertFalse((second.directory / "transcript.txt").exists())
            self.assertFalse((second.directory / "transcript.srt").exists())
            self.assertTrue((second.directory / "transcript.vtt").is_file())
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep me")
            self.assertEqual(
                [item["path"] for item in inspect_job(second.directory)["artifacts"]],
                ["transcript.json", "transcript.vtt"],
            )


if __name__ == "__main__":
    unittest.main()
