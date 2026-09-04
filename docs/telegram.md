# Telegram doorway

Whisper Lab can receive a Telegram voice note or media file and transcribe it with the same
local faster-whisper pipeline used by `whisperlab transcribe`.

```text
Telegram ──HTTPS──► tunnel or reverse proxy ──► 127.0.0.1:8787
                                                       │
                                                       ▼
                                              one local worker
                                                       │
                                                       ▼
                                      Whisper Lab → faster-whisper
```

Speech recognition and model inference run locally on the MacBook. Telegram still transports
and hosts the messages and media involved in the bot conversation. This workflow is therefore
not fully offline, and the audio is not confined to the Mac.

## 1. Create the bot

Open a chat with [@BotFather](https://t.me/BotFather), run `/newbot`, and follow its prompts.
BotFather returns a token. Treat it as a password: do not put it in a command-line option,
configuration file, transcript, screenshot, or commit.

Create a local environment and download a model explicitly:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

whisperlab models download tiny
```

## 2. Configure the receiver

The bot token and webhook secret are environment-only secrets. Generate a URL-safe secret:

```bash
export WHISPERLAB_TELEGRAM_BOT_TOKEN="PASTE_THE_BOTFATHER_TOKEN_HERE"
export WHISPERLAB_TELEGRAM_WEBHOOK_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

Telegram accepts webhook `secret_token` values containing `A-Z`, `a-z`, `0-9`, `_`, and `-`
with a length of 1–256 characters. Whisper Lab deliberately requires 32–256 characters as a
stronger local security policy.

If you already know your numeric Telegram user ID, configure it now:

```bash
export WHISPERLAB_TELEGRAM_ALLOWED_USER_IDS="123456789"
```

Multiple users are comma-separated. Usernames are never used for authorization. An empty
allowlist leaves `/whoami` available but fails closed for all transcription work.

Check local configuration without contacting Telegram:

```bash
whisperlab doctor --telegram --model tiny
```

## 3. Start the local receiver

```bash
whisperlab telegram serve --model tiny
```

The receiver binds to `127.0.0.1:8787` by default and prints its local address, webhook path,
model revision, allowlist policy, and local-inference guarantee. It never prints either secret.
A best-effort `getMe` check displays the bot username when Telegram is reachable; a temporary
Telegram outage produces a warning but does not prevent the local receiver from starting.

The only routes are:

- `POST /telegram/webhook` for authenticated Telegram updates;
- `GET /healthz` for a minimal, secret-free health response.

## 4. Supply public HTTPS

Telegram cannot reach localhost. Expose port 8787 through an HTTPS tunnel or reverse proxy of
your choice, such as Cloudflare Tunnel, ngrok, a conventional reverse proxy, or Tailscale
Funnel. Whisper Lab does not install, configure, or depend on any of them.

Register the complete public webhook URL returned by that bridge:

```bash
whisperlab telegram webhook set https://PUBLIC_HOST/telegram/webhook
whisperlab telegram webhook info
```

The registration limits Telegram to ordinary `message` updates, requests one webhook
connection, and configures `X-Telegram-Bot-Api-Secret-Token`. URLs must use HTTPS and cannot
contain embedded credentials, a query, or a fragment.

To intentionally discard updates Telegram is already holding during first-time setup:

```bash
whisperlab telegram webhook set https://PUBLIC_HOST/telegram/webhook \
  --drop-pending-updates
```

## 5. Discover your user ID

Send `/whoami` to the bot in a private chat. The response contains only your own numeric user
and chat IDs. If you started with an empty allowlist:

1. stop the receiver with Ctrl-C;
2. export `WHISPERLAB_TELEGRAM_ALLOWED_USER_IDS` using the returned user ID;
3. restart `whisperlab telegram serve --model tiny`.

## 6. Transcribe

Send the bot a voice note, audio file, video note, video, or an audio/video document in a format
already accepted by Whisper Lab. The bot acknowledges queue admission immediately, then one
worker downloads and transcribes the media:

```text
Received — queued for local transcription with tiny.
Transcribing locally with tiny…

<transcript>

Local • en • 0:42 • tiny • job 62ab9f206ca1
```

Normal transcripts return as plain text. Longer transcripts return as the existing
`transcript.txt` artifact rather than dozens of messages. JSON, TXT, SRT, VTT, and the manifest
remain in the normal `outputs/<name>--<job-id>/` structure.

The standard hosted Bot API allows `getFile` downloads only up to 20 MB. Whisper Lab rejects
larger media from update metadata, rechecks the `getFile` result, and bounds the download stream.

Telegram filenames are untrusted. Neither `file_name` nor Telegram's remote `file_path` is ever
used as a local destination. Downloads receive an internal name such as `update-810000001.ogg`
inside an operating-system temporary directory, which is removed after success or failure.

## Queue, retries, and shutdown

One worker runs model jobs sequentially so multiple messages do not compete for local RAM or
accelerator resources. The in-memory queue holds at most 16 jobs.

Telegram may redeliver webhook updates. Whisper Lab remembers 4,096 accepted update IDs for the
life of the process. An ID is remembered only after successful queue admission. If the queue is
full, the webhook deliberately returns HTTP 200 with a “busy; resend later” response and does not
mark the media as admitted. Existing content-addressed artifacts provide resume protection after
a restart.

On Ctrl-C, the receiver stops accepting work, discards queued jobs that have not started, lets
the active transcription finish, removes its temporary media, and exits. It does not transcribe
the entire backlog during shutdown.

## Stop and remove the webhook

Stop the local process with Ctrl-C, then remove the public webhook registration:

```bash
whisperlab telegram webhook delete
whisperlab telegram webhook info
```

To discard pending Telegram updates at the same time:

```bash
whisperlab telegram webhook delete --drop-pending-updates
```

The bot leaves no Telegram-specific database, log directory, model cache, or media inbox. Only
the configured Whisper Lab model cache and normal transcription artifacts remain.

## Troubleshooting

- **Model unavailable:** run `whisperlab models download <model>` locally, then restart.
- **403 from the webhook:** register it again with the same webhook-secret environment value
  used by the running receiver.
- **No updates:** run `whisperlab telegram webhook info` and inspect Telegram's last error.
- **Unauthorized:** use `/whoami`, update the numeric allowlist, and restart the receiver.
- **20 MB rejection:** the hosted Bot API cannot provide that file to the bot; send a smaller
  recording or use the ordinary local `whisperlab transcribe` command.
- **Temporary Telegram outage at startup:** the receiver continues listening; webhook management
  remains strict so it can diagnose the network or token problem.
