# Whisper Lab

**Local speech-to-text that leaves an audit trail you can actually read.**

Whisper Lab is a compact reference pipeline around
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Give it one media file—or a
directory—and it writes a canonical transcript, plain text, subtitles, and a hash-checked
manifest into one stable job folder.

It is deliberately smaller than a transcription platform. There is one backend, one primary
command, no service stack, and no database. The interesting parts are the boundaries: model
downloads are explicit, valid repeat work resumes by content and settings, outputs are portable,
and every artifact in a managed job can be verified later.

```text
media file(s) ──► PyAV decode + CTranslate2 inference ──► transcript.json
                                                        ├── transcript.txt
                                                        ├── transcript.srt
                                                        ├── transcript.vtt
                                                        └── manifest.json
```

Telegram can also act as a secure transport and trigger while inference remains on the local
machine. It feeds downloaded media into this same pipeline rather than a separate bot-specific
transcriber.

## Quick start

Python 3.10 or newer is required. The default `auto` device uses the CTranslate2 runtime
available on the machine; CPU can be selected explicitly, and CUDA can be used where it is
configured. System FFmpeg is **not** required; faster-whisper decodes media with the FFmpeg
libraries bundled by PyAV.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

whisperlab doctor
whisperlab models download tiny
whisperlab transcribe path/to/recording.m4a --model tiny
```

The final command prints the job directory. Progress goes to stderr, so stdout remains easy to
use in scripts.

## The everyday workflow

`small` is the default and a good general-purpose starting point:

```bash
whisperlab models download small
whisperlab transcribe recordings/ --language auto
```

Directory inputs are scanned recursively for common audio and video formats. Each file becomes
an independent job:

```text
outputs/
  interview--62ab9f206ca1/
    transcript.json     canonical, machine-readable transcript
    transcript.txt      one segment per line
    transcript.srt      SubRip subtitles
    transcript.vtt      WebVTT subtitles
    manifest.json       source/config identity and artifact hashes
```

Running the same command again resumes the existing job when its manifest and tracked artifacts
still verify. A damaged or modified canonical transcript is treated as non-resumable and inference
runs again. The identity includes the source content hash, immutable model snapshot revision,
decoding options, device/compute mode, and schema version. A different filename does not force
duplicate inference; a changed setting does create a distinct job.

Verify a job at any time:

```bash
whisperlab inspect outputs/interview--62ab9f206ca1
whisperlab inspect outputs/interview--62ab9f206ca1 --json
```

## Models

Whisper Lab offers four memorable presets while still accepting any compatible Hugging Face
model ID or local CTranslate2 directory.

| Preset | Use it for |
| --- | --- |
| `tiny` | fast smoke tests and rough drafts |
| `small` | balanced everyday transcription |
| `turbo` | high quality with lower latency than `large-v3` |
| `large-v3` | the highest-quality baseline |

```bash
whisperlab models list
whisperlab models download turbo
whisperlab models download Systran/faster-whisper-medium
```

Downloads are never an accidental side effect. `transcribe` uses local model files and tells you
the exact download command when they are absent. For an intentionally combined operation, pass
`--download`:

```bash
whisperlab transcribe sample.wav --model tiny --download
```

The default cache is `.cache/models/` and is ignored by Git. Override it with `--model-dir` or
`WHISPERLAB_MODEL_DIR`.

## Useful options

```bash
# Force a known language; auto-detection is the default.
whisperlab transcribe talk.mp3 --language en

# Fast greedy decoding without the voice activity filter.
whisperlab transcribe talk.mp3 --beam-size 1 --no-vad

# Include word-level timing/probability records in transcript.json.
whisperlab transcribe talk.mp3 --word-timestamps

# CUDA example. Hardware libraries are managed by CTranslate2, not this repo.
whisperlab transcribe talk.mp3 --device cuda --compute-type float16

# Choose exports or regenerate them from an existing canonical transcript.
# Exports beside a managed-job manifest keep that manifest in sync.
whisperlab transcribe talk.mp3 --formats txt,json
whisperlab export outputs/talk--<job-id>/transcript.json --formats srt,vtt

# Re-run inference even when an identical complete job exists.
whisperlab transcribe talk.mp3 --no-resume
```

Use `whisperlab transcribe --help` for the complete command contract.

## Telegram doorway

Whisper Lab can accept private Telegram voice notes and media through an authenticated webhook,
queue them for one local worker, and return the transcript to the originating chat:

```bash
export WHISPERLAB_TELEGRAM_BOT_TOKEN="..."
export WHISPERLAB_TELEGRAM_WEBHOOK_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export WHISPERLAB_TELEGRAM_ALLOWED_USER_IDS="123456789"

whisperlab telegram serve --model tiny
whisperlab telegram webhook set https://PUBLIC_HOST/telegram/webhook
```

The receiver binds to `127.0.0.1` by default; supply the public HTTPS bridge with your own tunnel
or reverse proxy. Speech recognition runs locally with the configured faster-whisper model, but
Telegram still transports and hosts the bot messages and media.

See [docs/telegram.md](docs/telegram.md) for BotFather setup, `/whoami`, webhook security, file
limits, tunnel setup, privacy, and the complete shutdown workflow.

## Local UI

The optional UI uses only Python’s standard library and invokes the same CLI command shown in
the terminal workflow:

```bash
whisperlab ui
```

Open <http://127.0.0.1:8765>. It admits one local job at a time, protects submissions with a
per-process form token, and links to the artifacts it produced. The page shows the exact CLI
command and final command output; use the CLI directly when live terminal progress matters. It
binds to loopback by default; do not expose it to an untrusted network.

## Paths and configuration

There is no hidden configuration file and no setup-generated state.

| Setting | CLI option | Environment variable | Default |
| --- | --- | --- | --- |
| artifact root | `--output-dir` | `WHISPERLAB_OUTPUT_DIR` | `outputs/` |
| model cache | `--model-dir` | `WHISPERLAB_MODEL_DIR` | `.cache/models/` |

Relative paths are resolved from the directory where the command is run. Both runtime roots are
created when needed. Inputs can live anywhere; there is no required inbox folder.

## What is worth reusing

- `src/whisperlab/models.py` resolves named, Hub-hosted, and local model snapshots without
  allowing silent downloads.
- `src/whisperlab/pipeline.py` shows a small content-addressed batch/resume workflow.
- `src/whisperlab/transcript.py` keeps canonical timestamps unrounded and renders TXT/SRT/VTT at
  the edge.
- `src/whisperlab/artifacts.py` demonstrates atomic writes and portable artifact records.
- `src/whisperlab/inspect.py` verifies paths and hashes without relying on a secondary index.
- `src/whisperlab/ui.py` is a dependency-free local control surface over the CLI contract.
- `src/whisperlab/telegram/` isolates authenticated webhook transport and a single local worker
  from the transcription pipeline.

The manifest is the source of truth. It intentionally stores artifact paths relative to the job
folder and omits hostnames, usernames, cache paths, and other machine-local details.

## Development

```bash
make setup
make check
```

`make check` runs Ruff and the standard-library unit suite. Tests mock model inference while
exercising discovery, content identity, resume, export, manifest creation, artifact verification,
CLI parsing, UI command construction, and Telegram security/queue behavior. CI runs the same
checks on Python 3.10, 3.12, and 3.13.

Real transcription is intentionally not part of the unit suite: it would require downloading a
large third-party model and would make the result hardware-dependent. `whisperlab doctor` is the
preflight for the actual machine and selected model.

## Scope

This project transcribes local media with faster-whisper. It does not implement diarization,
speaker labeling, forced alignment, translation workflows, a general-purpose job system,
collaborative review, or a hosted API. The Telegram receiver is a deliberately narrow local
doorway, not a bot platform.
