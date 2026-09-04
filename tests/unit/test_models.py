from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from whisperlab.models import find_local_model, repo_for, resolve_model


def make_snapshot(cache: Path, repo: str, revision: str = "a1b2c3") -> Path:
    repo_dir = cache / ("models--" + repo.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(revision, encoding="utf-8")
    for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.json"):
        (snapshot / name).write_text(name, encoding="utf-8")
    return snapshot


class ModelTests(unittest.TestCase):
    def test_presets_map_to_real_repository_ids(self):
        self.assertEqual(repo_for("tiny"), "Systran/faster-whisper-tiny")
        self.assertEqual(repo_for("turbo"), "mobiuslabsgmbh/faster-whisper-large-v3-turbo")
        self.assertEqual(repo_for("organization/custom-model"), "organization/custom-model")

    def test_hugging_face_snapshot_is_resolved_by_ref(self):
        with tempfile.TemporaryDirectory() as raw:
            cache = Path(raw)
            snapshot = make_snapshot(cache, repo_for("tiny"))
            result = find_local_model("tiny", cache)

        self.assertIsNotNone(result)
        self.assertEqual(result.path, snapshot.resolve())
        self.assertEqual(result.revision, "a1b2c3")

    def test_missing_model_never_downloads_implicitly(self):
        with (
            tempfile.TemporaryDirectory() as raw,
            self.assertRaisesRegex(RuntimeError, "models download tiny"),
        ):
            resolve_model("tiny", Path(raw), allow_download=False)

    def test_incomplete_local_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            model = Path(raw) / "model"
            model.mkdir()
            (model / "model.bin").write_bytes(b"weights")

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                find_local_model(str(model), Path(raw) / "cache")


if __name__ == "__main__":
    unittest.main()
