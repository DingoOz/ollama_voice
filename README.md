# ollama_voice

Voice (or text) chat for the Jetson Orin Nano. Prompts go to a local Ollama
model (`gemma4:e2b`); responses stream to the terminal **and** speak through
the USB speaker via Piper TTS. Press Enter at an empty prompt to **speak**
your message — it's captured from the C920 webcam mic and transcribed with
Whisper.

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
- Python 3.10 with `requests`, `numpy`

## Run

```bash
python3 chat.py
```

With a starting volume:

```bash
python3 chat.py --volume 70      # 70%
python3 chat.py --volume 1.5     # 150% (may clip)
```

Other CLI flags:

```bash
python3 chat.py --no-mic                       # disable speech input
python3 chat.py --mic-device plughw:0,0        # override mic ALSA device
python3 chat.py --whisper-model base.en        # swap whisper model
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

Volume changes apply to the **next** spoken sentence; anything already queued
keeps its prior level. `:vol 0` mutes synthesis without breaking the chat.

## Tweaks (top of `chat.py`)

- `MODEL` — swap to any installed Ollama model (`ollama list` to see them)
- `VOICE` — point to another `.onnx` file; `en_US-hfc_female-medium.onnx`
  is also installed
- `AUDIO_DEVICE` — ALSA device string; `plughw:1,0` is the USB speaker
- `MIC_DEVICE` — ALSA capture device; `plughw:0,0` is the C920 mic
- `WHISPER_MODEL` — `tiny`, `base.en`, etc. Bigger = slower but better
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
