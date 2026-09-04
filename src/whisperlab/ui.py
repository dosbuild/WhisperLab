from __future__ import annotations

import hmac
import html
import secrets
import shlex
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .models import PRESETS

MAX_FORM_BYTES = 64 * 1024


def serve_ui(*, host: str, port: int, output_dir: Path, model_dir: Path) -> None:
    state: dict[str, Any] = {
        "running": False,
        "command": "",
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    state_lock = threading.Lock()
    csrf_token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        server_version = "WhisperLab"
        sys_version = ""

        def version_string(self) -> str:
            return self.server_version

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/artifact?"):
                self._render_artifact()
            else:
                self._render_index()

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/run":
                self.send_error(404)
                return
            try:
                length = _content_length(self.headers.get("Content-Length"))
                form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
                command = _command_from_form(form, output_dir=output_dir, model_dir=model_dir)
            except (UnicodeDecodeError, ValueError) as exc:
                self.send_error(400, str(exc))
                return
            if not _valid_csrf(form, csrf_token):
                self.send_error(403, "invalid form token")
                return
            if not _claim_run(state, state_lock, command):
                self._redirect_home()
                return
            threading.Thread(
                target=_run_command,
                args=(command, state, state_lock),
                daemon=True,
            ).start()
            self._redirect_home()

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _redirect_home(self) -> None:
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def _render_index(self) -> None:
            with state_lock:
                view = dict(state)
            options = "".join(
                f'<option value="{html.escape(preset.name)}"'
                f"{' selected' if preset.name == 'small' else ''}>"
                f"{html.escape(preset.name)} — {html.escape(preset.description)}</option>"
                for preset in PRESETS
            )
            status = (
                "running"
                if view["running"]
                else ("ready" if view["returncode"] is None else f"exit {view['returncode']}")
            )
            refresh = '<meta http-equiv="refresh" content="2">' if view["running"] else ""
            links = _job_links(str(view["stdout"]), output_dir)
            button_disabled = "disabled" if view["running"] else ""
            button_label = "Transcribing…" if view["running"] else "Transcribe"
            details_open = "open" if view["stderr"] else ""
            progress_text = html.escape(str(view["stderr"]))
            body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh}
  <title>Whisper Lab</title>
  <style>
    :root {{
      color-scheme: light dark;
      --accent: #6e5ae6;
      --panel: color-mix(in srgb, Canvas 94%, #6e5ae6 6%);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font: 16px/1.5 ui-sans-serif, system-ui, sans-serif;
      margin: 0 auto;
      max-width: 880px;
      padding: 48px 24px;
    }}
    h1 {{ font-size: clamp(2.4rem, 7vw, 4.5rem); letter-spacing: -.06em; margin: 0; }}
    .lede {{ color: GrayText; font-size: 1.1rem; margin: .25rem 0 2rem; }}
    form, .result {{
      background: var(--panel);
      border: 1px solid color-mix(in srgb, CanvasText 14%, transparent);
      border-radius: 16px;
      padding: 24px;
    }}
    .grid {{ display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }}
    label {{
      display: block;
      font-size: .82rem;
      font-weight: 700;
      letter-spacing: .04em;
      margin-bottom: 5px;
      text-transform: uppercase;
    }}
    input, select, button {{
      border: 1px solid color-mix(in srgb, CanvasText 22%, transparent);
      border-radius: 8px;
      font: inherit;
      padding: 10px 12px;
      width: 100%;
    }}
    .wide {{ grid-column: 1 / -1; }}
    .checks {{ display: flex; flex-wrap: wrap; gap: 18px; margin-top: 18px; }}
    .checks label {{
      align-items: center;
      display: flex;
      gap: 7px;
      margin: 0;
      text-transform: none;
    }}
    .checks input {{ width: auto; }}
    button {{
      background: var(--accent);
      border: 0;
      color: white;
      cursor: pointer;
      font-weight: 700;
      margin-top: 20px;
    }}
    button:disabled {{ cursor: wait; opacity: .6; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: .85rem; }}
    pre {{ overflow: auto; white-space: pre-wrap; }}
    .result {{ margin-top: 20px; }}
    .badge {{ border: 1px solid; border-radius: 999px; display: inline-block; padding: 2px 9px; }}
    a {{ color: var(--accent); }}
    @media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Whisper Lab</h1>
  <p class="lede">A small local workbench for repeatable speech-to-text.</p>
  <form action="/run" method="post">
    <input name="csrf_token" type="hidden" value="{html.escape(csrf_token)}">
    <div class="grid">
      <div class="wide">
        <label for="target">Media file or directory</label>
        <input id="target" name="target" value="." required>
      </div>
      <div><label for="model">Model</label><select id="model" name="model">{options}</select></div>
      <div>
        <label for="language">Language</label>
        <input id="language" name="language" value="auto">
      </div>
      <div>
        <label for="formats">Formats</label>
        <input id="formats" name="formats" value="txt,srt,vtt">
      </div>
      <div>
        <label for="beam_size">Beam size</label>
        <input id="beam_size" name="beam_size" type="number" min="1" value="5">
      </div>
    </div>
    <div class="checks">
      <label><input name="resume" type="checkbox" checked> Resume identical work</label>
      <label><input name="vad" type="checkbox" checked> Voice activity filter</label>
      <label><input name="word_timestamps" type="checkbox"> Word timestamps</label>
      <label><input name="download" type="checkbox"> Download if missing</label>
    </div>
    <button type="submit" {button_disabled}>{button_label}</button>
  </form>
  <section class="result">
    <span class="badge">{html.escape(status)}</span>
    <p><code>{html.escape(str(view["command"]))}</code></p>
    {links}
    <details {details_open}>
      <summary>Progress</summary>
      <pre>{progress_text}</pre>
    </details>
  </section>
</body>
</html>"""
            self._send_html(body)

        def _render_artifact(self) -> None:
            query = urllib.parse.urlparse(self.path).query
            value = urllib.parse.parse_qs(query).get("path", [""])[0]
            try:
                path = _safe_artifact(output_dir, value)
                content = path.read_text(encoding="utf-8")
            except (OSError, ValueError) as exc:
                self.send_error(404, str(exc))
                return
            self._send_html(f"<pre>{html.escape(content)}</pre>")

        def _send_html(self, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; "
                "form-action 'self'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Whisper Lab UI: http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _command_from_form(
    form: dict[str, list[str]], *, output_dir: Path, model_dir: Path
) -> list[str]:
    target = form.get("target", [""])[0].strip()
    if not target:
        raise ValueError("media path is required")
    model = form.get("model", ["small"])[0] or "small"
    language = form.get("language", ["auto"])[0] or "auto"
    formats = form.get("formats", ["txt,srt,vtt"])[0] or "txt,srt,vtt"
    beam_size = form.get("beam_size", ["5"])[0] or "5"
    command = [
        sys.executable,
        "-m",
        "whisperlab",
        "transcribe",
        target,
        "--model",
        model,
        "--language",
        language,
        "--formats",
        formats,
        "--beam-size",
        beam_size,
        "--output-dir",
        str(output_dir),
        "--model-dir",
        str(model_dir),
    ]
    command.append("--resume" if "resume" in form else "--no-resume")
    command.append("--vad" if "vad" in form else "--no-vad")
    if "word_timestamps" in form:
        command.append("--word-timestamps")
    if "download" in form:
        command.append("--download")
    return command


def _run_command(command: list[str], state: dict[str, Any], state_lock: threading.Lock) -> None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        result = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except OSError as exc:
        result = {"returncode": 1, "stdout": "", "stderr": str(exc)}
    with state_lock:
        state.update(result)
        state["running"] = False


def _content_length(value: str | None) -> int:
    try:
        length = int(value or "")
    except ValueError as exc:
        raise ValueError("invalid content length") from exc
    if length < 0:
        raise ValueError("invalid content length")
    if length > MAX_FORM_BYTES:
        raise ValueError("form is too large")
    return length


def _valid_csrf(form: dict[str, list[str]], expected: str) -> bool:
    supplied = form.get("csrf_token", [""])[0]
    return hmac.compare_digest(supplied, expected)


def _claim_run(state: dict[str, Any], state_lock: threading.Lock, command: list[str]) -> bool:
    with state_lock:
        if state["running"]:
            return False
        state.update(
            {
                "running": True,
                "command": shlex.join(command),
                "returncode": None,
                "stdout": "",
                "stderr": "",
            }
        )
        return True


def _safe_artifact(root: Path, value: str) -> Path:
    root = root.resolve()
    path = (root / value).resolve()
    if root not in path.parents or path.suffix.lower() not in {".json", ".txt", ".srt", ".vtt"}:
        raise ValueError("artifact path is outside the output directory")
    return path


def _job_links(stdout: str, output_dir: Path) -> str:
    links = []
    for line in stdout.splitlines():
        job_dir = Path(line.strip())
        try:
            relative_job = job_dir.resolve().relative_to(output_dir.resolve())
        except ValueError:
            continue
        for name in (
            "transcript.txt",
            "transcript.srt",
            "transcript.vtt",
            "transcript.json",
            "manifest.json",
        ):
            path = job_dir / name
            if path.is_file():
                relative = relative_job / name
                href = "/artifact?" + urllib.parse.urlencode({"path": str(relative)})
                links.append(f'<li><a href="{href}">{html.escape(str(relative))}</a></li>')
    return "<ul>" + "".join(links) + "</ul>" if links else ""
