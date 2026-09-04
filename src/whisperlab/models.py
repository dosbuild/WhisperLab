from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ids import sha256_file, stable_hash


@dataclass(frozen=True)
class ModelPreset:
    name: str
    repo: str
    description: str


@dataclass(frozen=True)
class ModelResolution:
    requested: str
    repo: str
    path: Path
    revision: str

    @property
    def identity(self) -> dict[str, str]:
        return {"repo": self.repo, "revision": self.revision}


PRESETS = (
    ModelPreset("tiny", "Systran/faster-whisper-tiny", "Fast smoke tests and rough drafts"),
    ModelPreset("small", "Systran/faster-whisper-small", "Balanced everyday transcription"),
    ModelPreset(
        "turbo",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "High quality with lower latency than large-v3",
    ),
    ModelPreset("large-v3", "Systran/faster-whisper-large-v3", "Highest-quality baseline"),
)
PRESET_BY_NAME = {preset.name: preset for preset in PRESETS}


def repo_for(model: str) -> str:
    preset = PRESET_BY_NAME.get(model)
    return preset.repo if preset else model


def find_local_model(model: str, cache_dir: Path) -> ModelResolution | None:
    direct = Path(model).expanduser()
    if direct.is_dir():
        direct = direct.resolve()
        if not _is_complete(direct):
            raise RuntimeError(f"Local model directory is incomplete: {direct}")
        model_file = direct / "model.bin"
        revision = stable_hash(
            {
                "config_sha256": sha256_file(direct / "config.json"),
                "model_bytes": model_file.stat().st_size,
                "model_mtime_ns": model_file.stat().st_mtime_ns,
            }
        )
        return ModelResolution(model, f"local/{direct.name}", direct, revision)

    repo = repo_for(model)
    repo_dir = cache_dir / ("models--" + repo.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    candidates: list[Path] = []
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(snapshots / revision)
    if snapshots.is_dir():
        candidates.extend(
            sorted(
                (path for path in snapshots.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_complete(candidate):
            return ModelResolution(model, repo, candidate, candidate.name)
    return None


def resolve_model(model: str, cache_dir: Path, *, allow_download: bool) -> ModelResolution:
    local = find_local_model(model, cache_dir)
    if local:
        return local
    if not allow_download:
        raise RuntimeError(
            f"Model '{model}' is not available locally. "
            f"Download it first: whisperlab models download {model}"
        )
    return download_model(model, cache_dir)


def download_model(model: str, cache_dir: Path) -> ModelResolution:
    try:
        from faster_whisper.utils import download_model as faster_whisper_download
    except Exception as exc:
        raise RuntimeError(
            "faster-whisper is unavailable; install the project with: pip install -e ."
        ) from exc

    if Path(model).expanduser().is_dir():
        resolution = find_local_model(model, cache_dir)
        assert resolution is not None
        return resolution

    cache_dir.mkdir(parents=True, exist_ok=True)
    repo = repo_for(model)
    try:
        downloaded = Path(faster_whisper_download(repo, cache_dir=str(cache_dir)))
    except Exception as exc:
        raise RuntimeError(f"Could not download model '{model}': {exc}") from exc
    if not _is_complete(downloaded):
        raise RuntimeError(f"Downloaded model is incomplete: {downloaded}")
    return ModelResolution(model, repo, downloaded.resolve(), downloaded.name)


def model_rows(cache_dir: Path) -> list[dict[str, str]]:
    rows = []
    for preset in PRESETS:
        resolution = find_local_model(preset.name, cache_dir)
        rows.append(
            {
                "model": preset.name,
                "status": "ready" if resolution else "not downloaded",
                "description": preset.description,
            }
        )
    return rows


def _is_complete(path: Path) -> bool:
    required = ("model.bin", "config.json", "tokenizer.json")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        return False
    return any(item.is_file() for item in path.glob("vocabulary.*"))
