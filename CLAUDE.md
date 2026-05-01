# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run / dev

This is a single-file Python program (`chat.py`). There is no test suite, no linter config, no build system, no package manifest — running the script *is* the dev loop.

- `./chat.py` — preferred. The shebang points at `/home/dingo/Programming/Piper/oww-env/bin/python`. `python3 chat.py` also works because `_reexec_in_venv_if_needed()` (chat.py:937) auto-switches to that venv.
- Smoke test the wake path: `./chat.py --wake` (say "hey jarvis").
- Useful flags: `--wake`, `--no-mic`, `--volume`, `--mic-device`, `--whisper-model`, `--wake-model`, `--wake-threshold`.

The venv at `~/Programming/Piper/oww-env/` is the canonical environment — `whisper`, `openwakeword`, and a numpy build with the right ABI all live there. **Do not** add a `requirements.txt` or recommend a fresh `pip install`; reuse that venv.

## Architecture

One process, main thread runs the REPL, three background threads handle audio. Audio I/O is done by piping to/from short-lived `piper`, `aplay`, and `arecord` subprocesses — no Python audio libraries.

- **`Speaker` (chat.py:65)** — daemon thread consuming a `queue.Queue[str|None]` of sentences. Per item: spawns `piper` and `aplay`, wires `piper.stdout → aplay.stdin`, writes text to `piper.stdin`, waits for `aplay`. `interrupt()` drains the queue and `terminate()`s the live `aplay` (Ctrl-C handler).
- **`Listener` (chat.py:180)** — push-to-talk capture. Spawns its own `arecord`, runs an energy-VAD loop (calibrate → preroll → record-until-silence), then calls `whisper.transcribe`. Whisper is lazy-loaded on first use so text-only sessions pay nothing.
- **`WakeListener` (chat.py:346)** — when `--wake` is set, owns the single `arecord` stream and runs a state machine: 1280-sample frames go to `openwakeword`; when a score crosses threshold it *flips into capture mode on the same stream* and pushes the transcribed utterance onto a `commands` queue the REPL polls. Reuses the `Listener` instance so Whisper loads once.
- **`Vision` (after `WakeListener`)** — opt-in object detection. `capture_frame()` shells out to `ffmpeg -f v4l2 ... -frames:v 1` to grab one JPG from `VIDEO_DEVICE`, then `detect()` runs `ultralytics.YOLO` (lazy-loaded, weights cached at `<repo>/yolov8n.pt`). The REPL matches `VISION_TRIGGER` (or `:see`) on user input *before* sending to Ollama and bypasses the LLM — `describe_detections()` formats COCO labels into a spoken sentence and `speaker.say()`s it. The exchange is still appended to `history` so follow-up questions ("tell me more about the laptop") have context.
- **`StatusBar` (chat.py:573)** — daemon thread, 10 fps repaint of the pinned bottom row. Sets an ANSI scroll region (`\033[1;{rows-1}r`) at start so normal stdout writes stay above the bar.
- **`run_repl` (chat.py:746)** — orchestrator. Streaming pipeline: `stream_chat()` (chat.py:705) yields chunks from Ollama → `split_sentences()` (chat.py:722) cuts complete sentences off the buffer using `SENTENCE_END` → completed sentences are pushed to `Speaker.q` *while the next chunk is still streaming*. This is why audio starts before the LLM finishes generating — don't refactor the pipeline to wait for full responses.

## Invariants (breaking these has bitten us before)

1. **One mic, one stream.** The C920 can't be opened twice. In `--wake` mode `WakeListener` owns the only `arecord`; the push-to-talk path is bypassed. New audio-capture features must route through `WakeListener`, not spawn a parallel `arecord`.
2. **Self-trigger guard.** `WakeListener._run` (chat.py:482) suppresses detection while `speaker.is_playing()` or `speaker.pending() > 0` and `oww.reset()`s afterward. Without this, the assistant wakes itself off its own TTS playback.
3. **Venv re-exec is load-bearing.** System Python on this Jetson has a numpy/pandas ABI mismatch that breaks `openwakeword` and `tflite_runtime`. Don't remove `_reexec_in_venv_if_needed()`. Note it compares `sys.executable` to the literal venv path — *don't* switch to `realpath`, the venv python is a symlink to the system python and `realpath` collapses them.
4. **StatusBar must pause around `input()`.** The bar uses raw ANSI escapes; repainting during `input()` clobbers the user's typing. The wake path uses `select(stdin)` instead of `input()` because `input()`/`readline` would block the queue poll.
5. **Volume changes are queue-deferred.** `Speaker.set_volume` only affects the next sentence pulled off the queue; already-queued items keep their old volume. Intentional.
6. **Vision and audio capture share the C920 USB device but different kernel nodes.** The C920 mic is ALSA `plughw:0,0`; the camera is `/dev/video0`. They can be opened concurrently — the one-mic invariant only applies to `arecord` consumers, not to v4l2.

## Hardware coupling

Top-of-file constants (chat.py:31–43) target this specific Jetson Orin Nano:

- `OLLAMA_URL`, `MODEL` (`gemma4:e2b`)
- `VOICE` — Piper `.onnx` path
- `AUDIO_DEVICE = "plughw:1,0"` — USB speaker (card 1)
- `MIC_DEVICE = "plughw:0,0"` — C920 mic (card 0)
- `WHISPER_MODEL`, `WAKE_MODEL_NAME`, `WAKE_THRESHOLD`, `SYSTEM_PROMPT`
- `VIDEO_DEVICE = "/dev/video0"` — C920 v4l2 capture node
- `YOLO_MODEL` — defaults to `<repo>/yolov8n.pt`; ultralytics auto-downloads on first call

If the hardware setup changes, verify ALSA card numbers with `aplay -l` / `arecord -l` and update these constants. Most are also overridable via CLI flags.
