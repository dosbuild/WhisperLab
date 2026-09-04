from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from .models import find_local_model


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str


def run_doctor(
    *,
    model: str,
    model_dir: Path,
    output_dir: Path,
    strict: bool,
    telegram: bool = False,
) -> list[Check]:
    checks = [_python_check(), _package_check("faster-whisper"), _package_check("ctranslate2")]
    checks.append(_path_check("model cache", model_dir))
    checks.append(_path_check("output", output_dir))
    try:
        local = find_local_model(model, model_dir)
        if local:
            checks.append(Check("model", "pass", f"{local.repo}@{local.revision[:12]}"))
        else:
            checks.append(
                Check("model", "warn", f"not downloaded; run: whisperlab models download {model}")
            )
    except RuntimeError as exc:
        checks.append(Check("model", "fail", str(exc)))
    if telegram:
        from .telegram.config import configuration_checks

        checks.extend(Check(*item) for item in configuration_checks())
    if strict:
        return [
            Check(check.name, "fail" if check.status == "warn" else check.status, check.message)
            for check in checks
        ]
    return checks


def has_failures(checks: list[Check]) -> bool:
    return any(check.status == "fail" for check in checks)


def _python_check() -> Check:
    version = platform.python_version()
    supported = sys.version_info[:2] >= (3, 10)
    return Check("python", "pass" if supported else "fail", version)


def _package_check(distribution: str) -> Check:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return Check(distribution, "fail", "not installed; run: pip install -e .")
    module = distribution.replace("-", "_")
    try:
        importlib.import_module(module)
    except Exception as exc:
        return Check(distribution, "fail", f"{version} is installed but import failed: {exc}")
    return Check(distribution, "pass", version)


def _path_check(name: str, path: Path) -> Check:
    parent = path.expanduser().resolve()
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    writable = parent.is_dir() and os.access(parent, os.W_OK)
    return Check(name, "pass" if writable else "fail", str(path))
