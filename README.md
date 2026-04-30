# ollama_voice

Type-to-voice chat for the Jetson Orin Nano. Prompts go to a local Ollama
model (`gemma4:e2b`); responses stream to the terminal **and** speak through
the USB speaker via Piper TTS.

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
- Python 3.10 with `requests`

## Run

```bash
python3 chat.py
```

With a starting volume:

```bash
python3 chat.py --volume 70      # 70%
python3 chat.py --volume 1.5     # 150% (may clip)
```

## REPL commands

| Command       | Effect                                               |
| ------------- | ---------------------------------------------------- |
| `:q`          | Quit                                                 |
| `:reset`      | Clear conversation history                           |
| `:vol`        | Show current volume                                  |
| `:vol N`      | Set volume. `N` is 0–200 percent, or 0.0–2.0 mult.   |
| `Ctrl-C`      | Interrupt current speech (drops queue, stays in REPL)|
| `Ctrl-D`      | Quit                                                 |

Volume changes apply to the **next** spoken sentence; anything already queued
keeps its prior level. `:vol 0` mutes synthesis without breaking the chat.

## Tweaks (top of `chat.py`)

- `MODEL` — swap to any installed Ollama model (`ollama list` to see them)
- `VOICE` — point to another `.onnx` file; `en_US-hfc_female-medium.onnx`
  is also installed
- `AUDIO_DEVICE` — ALSA device string; `plughw:1,0` is the USB speaker
- `SYSTEM_PROMPT` — nudges the model toward concise, spoken-friendly replies

## Notes

- **GPU memory is tight.** `gemma4:e2b` is ~7 GB on an 8 GB Jetson. If you
  hit a CUDA OOM, kill any stale `ollama run` interactive sessions
  (`pgrep -a ollama`) so the server can reload the model cleanly.
- **No streaming inside a sentence.** Piper synthesizes a full sentence
  per invocation; expect a small lag between the *first* token printing and
  the first sound, then continuous speech as the LLM keeps generating.
