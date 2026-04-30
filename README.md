# ollama_voice

Voice (or text) chat for the Jetson Orin Nano. Prompts go to a local Ollama
model (`gemma4:e2b`); responses stream to the terminal **and** speak through
the USB speaker via Piper TTS. Press Enter at an empty prompt to **speak**
your message — it's captured from the C920 webcam mic and transcribed with
Whisper. With `--wake`, just say "hey jarvis" to start a turn — no need to
press Enter.

Sentences are synthesized as they arrive from the LLM, so audio starts
playing before generation finishes.

A pinned status bar at the bottom of the terminal shows live state with a
braille throbber: `thinking` (waiting on Ollama), `synthesizing` (Piper is
preparing audio), `speaking (N queued)` (audio is playing), or `idle`. If a
gap between sentences ever drops the bar back to `idle`, that's the source
of any stutter you hear — the speech queue ran dry while Piper was still
generating the next sentence.

## Requirements

These are already set up on this Jetson, but listed for reference:

- **Ollama** running at `http://localhost:11434` with `gemma4:e2b` pulled
- **Piper TTS** at `~/.local/bin/piper` (the `piper-tts` pip package)
- **Voice model:** `en_US-amy-medium.onnx` (+ `.onnx.json`) under
  `~/Programming/Piper/voices/`
- **USB speaker** as ALSA card 1 (`plughw:1,0` — confirm with `aplay -l`)
- **Whisper** (`openai-whisper` pip package) with the `tiny` model cached
  under `~/.cache/whisper/` — for speech input
- **C920 webcam mic** as ALSA card 0 (`plughw:0,0` — confirm with
  `arecord -l`)
- **openwakeword** with the `hey_jarvis_v0.1` model downloaded — for the
  wake-word feature (only loaded when `--wake` is passed)
- Python 3.10 with `requests`, `numpy`

> The Jetson's system Python has a numpy/pandas ABI mismatch that breaks
> openwakeword and `tflite_runtime`. Run from the venv at
> `~/Programming/Piper/oww-env/` (the shebang in `chat.py` already points
> there, so `./chat.py` works). Whisper has been installed inside that venv.

## Run

```bash
./chat.py                                          # uses oww-env via shebang
/home/dingo/Programming/Piper/oww-env/bin/python chat.py    # explicit
```

With a starting volume:

```bash
python3 chat.py --volume 70      # 70%
python3 chat.py --volume 1.5     # 150% (may clip)
```

Other CLI flags:

```bash
./chat.py --no-mic                       # disable speech input
./chat.py --mic-device plughw:0,0        # override mic ALSA device
./chat.py --whisper-model base.en        # swap whisper model
./chat.py --wake                         # enable wake-word ('hey jarvis')
./chat.py --wake --wake-threshold 0.7    # require a stronger detection
./chat.py --wake --wake-model alexa_v0.1 # try a different built-in word
```

## REPL commands

| Command       | Effect                                                |
| ------------- | ----------------------------------------------------- |
| `<Enter>`     | (empty input) Push-to-talk: record from mic and send  |
| `:say`        | Same as pressing Enter on an empty prompt             |
| `:q`          | Quit                                                  |
| `:reset`      | Clear conversation history                            |
| `:vol`        | Show current volume                                   |
| `:vol N`      | Set volume. `N` is 0–200 percent, or 0.0–2.0 mult.    |
| `Ctrl-C`      | Interrupt current speech (drops queue, stays in REPL) |
| `Ctrl-D`      | Quit                                                  |

### Speech input

Press Enter on an empty prompt to start recording. The status bar shows
`calibrating mic` → `listening` → `recording` → `transcribing`, with a
braille throbber. Recording stops automatically after ~1.5 s of silence
(or 30 s hard cap). The transcript is echoed as `you (spoken)> ...` and
sent to Ollama like a typed prompt.

The Whisper model loads lazily on the first speech attempt (~73 MB for
`tiny`, a few seconds on Jetson CPU), so users in text-only mode pay
nothing.

### Wake word

`./chat.py --wake` starts an always-on background listener. Say
`"hey jarvis"` (the default model). The status bar moves through
`waiting for wake word` → `wake heard (0.78)` → `recording` →
`transcribing` → `thinking` → `speaking`. The transcript is echoed as
`you (wake)> ...`.

- **One mic, one stream.** The wake listener owns the C920 mic; once
  triggered, it captures the following utterance off the same `arecord`
  process — no device-busy errors, no missed first syllable.
- **Self-trigger guard.** Detection is suppressed while the assistant is
  speaking, so the model doesn't trigger on its own TTS playback.
- **Mixed input.** While `--wake` is active, you can still type
  commands at the prompt — `:vol`, `:reset`, `:q` all work. (Trade-off:
  `readline` line-editing — backspace/arrow keys — is unavailable in
  wake mode because input is read line-buffered for the poll loop.)
- **Built-in models** (auto-downloaded on first use): `hey_jarvis_v0.1`
  (default), `hey_mycroft_v0.1`, `alexa_v0.1`, `hey_rhasspy_v0.1`,
  `timer_v0.1`, `weather_v0.1`. Pick with `--wake-model`.
- **Threshold** defaults to `0.5`. Bump it (`--wake-threshold 0.7`) if
  you get false positives; lower it if "hey jarvis" doesn't trigger
  reliably.

Volume changes apply to the **next** spoken sentence; anything already queued
keeps its prior level. `:vol 0` mutes synthesis without breaking the chat.

## Tweaks (top of `chat.py`)

- `MODEL` — swap to any installed Ollama model (`ollama list` to see them)
- `VOICE` — point to another `.onnx` file; `en_US-hfc_female-medium.onnx`
  is also installed
- `AUDIO_DEVICE` — ALSA device string; `plughw:1,0` is the USB speaker
- `MIC_DEVICE` — ALSA capture device; `plughw:0,0` is the C920 mic
- `WHISPER_MODEL` — `tiny`, `base.en`, etc. Bigger = slower but better
- `WAKE_MODEL_NAME` — default openwakeword model used by `--wake`
- `WAKE_THRESHOLD` — default detection score (0..1)
- `SYSTEM_PROMPT` — nudges the model toward concise, spoken-friendly replies

## License

GPL-3.0 — see [LICENSE](LICENSE).

## Notes

- **GPU memory is tight.** `gemma4:e2b` is ~7 GB on an 8 GB Jetson. If you
  hit a CUDA OOM, kill any stale `ollama run` interactive sessions
  (`pgrep -a ollama`) so the server can reload the model cleanly.
- **No streaming inside a sentence.** Piper synthesizes a full sentence
  per invocation; expect a small lag between the *first* token printing and
  the first sound, then continuous speech as the LLM keeps generating.
