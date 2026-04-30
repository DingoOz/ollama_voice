#!/home/dingo/Programming/Piper/oww-env/bin/python
"""Type a prompt -> Ollama (gemma4:e2b) -> Piper TTS -> USB speaker.

Streams the LLM response and speaks it sentence-by-sentence so audio starts
playing before generation finishes. Optional voice input via push-to-talk
(empty Enter / ':say') and wake-word activation (--wake, default 'hey jarvis').

Run from the oww-env venv so all native deps line up:
    /home/dingo/Programming/Piper/oww-env/bin/python chat.py [...]
The shebang above makes `./chat.py` work directly.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e2b"
VOICE = Path("/home/dingo/Programming/Piper/voices/en_US-amy-medium.onnx")
AUDIO_DEVICE = "plughw:1,0"   # USB speaker (card 1)
MIC_DEVICE = "plughw:0,0"     # C920 webcam mic (card 0)
WHISPER_MODEL = "tiny"        # tiny ~73MB, fast on Jetson CPU
WAKE_MODEL_NAME = "hey_jarvis_v0.1"  # built-in openwakeword model
WAKE_THRESHOLD = 0.5          # 0..1; raise for fewer false positives
SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your replies will be spoken aloud, "
    "so keep them concise and conversational. Avoid markdown, code fences, "
    "lists with bullets, or symbols that don't read well out loud."
)

PIPER_BIN = shutil.which("piper") or str(Path.home() / ".local/bin/piper")
SENTENCE_END = re.compile(r"([.!?]+[\")\]]?\s+|\n{2,})")


def voice_sample_rate(voice_path: Path) -> int:
    config = json.loads((voice_path.with_suffix(".onnx.json")).read_text())
    return int(config["audio"]["sample_rate"])


def clean_for_speech(text: str) -> str:
    # Strip markdown noise that sounds awful spoken.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"#{1,6}\s+", "", text)
    return text.strip()


class Speaker:
    """Background thread that speaks queued text chunks one at a time."""

    def __init__(self, voice: Path, device: str, volume: float = 1.0):
        self.voice = voice
        self.device = device
        self.sample_rate = voice_sample_rate(voice)
        self.q: "queue.Queue[str | None]" = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._current: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._volume = max(0.0, volume)
        self.thread.start()

    @property
    def volume(self) -> float:
        with self._lock:
            return self._volume

    def set_volume(self, value: float) -> float:
        with self._lock:
            self._volume = max(0.0, value)
            return self._volume

    def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if text:
            self.q.put(text)

    def wait_until_idle(self) -> None:
        self.q.join()

    def is_playing(self) -> bool:
        with self._lock:
            return self._current is not None and self._current.poll() is None

    def pending(self) -> int:
        return self.q.qsize()

    def interrupt(self) -> None:
        # Drop pending utterances and kill the one currently playing.
        try:
            while True:
                self.q.get_nowait()
                self.q.task_done()
        except queue.Empty:
            pass
        with self._lock:
            if self._current and self._current.poll() is None:
                self._current.terminate()

    def shutdown(self) -> None:
        self.interrupt()
        self.q.put(None)
        self.thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            try:
                self._speak_blocking(item)
            except Exception as e:  # noqa: BLE001
                print(f"\n[tts error] {e}", file=sys.stderr)
            finally:
                self.q.task_done()

    def _speak_blocking(self, text: str) -> None:
        vol = self.volume
        if vol <= 0.0:
            return
        piper = subprocess.Popen(
            [
                PIPER_BIN,
                "-m", str(self.voice),
                "--output-raw",
                "--sentence-silence", "0.2",
                "--volume", f"{vol:.3f}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        aplay = subprocess.Popen(
            [
                "aplay",
                "-D", self.device,
                "-r", str(self.sample_rate),
                "-f", "S16_LE",
                "-c", "1",
                "-t", "raw",
                "-q",
            ],
            stdin=piper.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Let aplay own the read end of the pipe.
        assert piper.stdout is not None
        piper.stdout.close()
        with self._lock:
            self._current = aplay
        try:
            assert piper.stdin is not None
            piper.stdin.write(text.encode("utf-8"))
            piper.stdin.close()
            aplay.wait()
            piper.wait()
        finally:
            with self._lock:
                self._current = None


class Listener:
    """Capture from a mic with simple energy VAD, then transcribe with Whisper.

    Whisper model loads lazily on first use (~73MB for 'tiny', a few seconds
    on Jetson CPU) so users who never invoke voice input pay no cost.
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 30
    FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000          # 480
    FRAME_BYTES = FRAME_SAMPLES * 2                          # int16 mono
    CALIBRATE_FRAMES = 10                                    # ~300 ms
    SPEECH_TRIGGER_FRAMES = 3                                # ~90 ms above threshold to start
    SILENCE_HANG_FRAMES = 50                                 # ~1.5 s of silence ends utterance
    PREROLL_FRAMES = 6                                       # keep ~180 ms before trigger
    MIN_RMS_THRESHOLD = 800.0                                # floor for very quiet rooms
    MAX_RECORD_SECONDS = 30
    PRE_TRIGGER_TIMEOUT_S = 8                                # give up if no speech detected

    def __init__(self, mic_device: str = MIC_DEVICE, model_name: str = WHISPER_MODEL):
        self.mic_device = mic_device
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self, status: "StatusBar | None" = None):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if status:
                    status.set_activity(f"loading whisper {self.model_name}")
                else:
                    print(f"[loading whisper {self.model_name}...]", flush=True)
                import whisper  # lazy: ~1s import + torch
                self._model = whisper.load_model(self.model_name)
                if status:
                    status.set_activity(None)
        return self._model

    def record_until_silence(self, status: "StatusBar | None" = None):
        """Block until an utterance is captured. Returns float32 numpy array
        at 16 kHz, or None if cancelled / nothing captured."""
        import numpy as np
        proc = subprocess.Popen(
            [
                "arecord",
                "-D", self.mic_device,
                "-r", str(self.SAMPLE_RATE),
                "-f", "S16_LE",
                "-c", "1",
                "-t", "raw",
                "-q",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        def read_frame() -> bytes | None:
            assert proc.stdout is not None
            buf = b""
            while len(buf) < self.FRAME_BYTES:
                chunk = proc.stdout.read(self.FRAME_BYTES - len(buf))
                if not chunk:
                    return None
                buf += chunk
            return buf

        def rms(frame_bytes: bytes) -> float:
            samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
            if samples.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(samples * samples)))

        captured: list[bytes] = []
        try:
            # Calibrate noise floor.
            if status:
                status.set_activity("calibrating mic")
            noise_levels = []
            for _ in range(self.CALIBRATE_FRAMES):
                f = read_frame()
                if f is None:
                    return None
                noise_levels.append(rms(f))
            noise_floor = sum(noise_levels) / len(noise_levels)
            threshold = max(noise_floor * 3.0, self.MIN_RMS_THRESHOLD)

            # Wait for speech, keeping a small pre-roll buffer.
            if status:
                status.set_activity("listening")
            preroll: list[bytes] = []
            consecutive_speech = 0
            max_pre_frames = int(self.PRE_TRIGGER_TIMEOUT_S * 1000 / self.FRAME_MS)
            triggered = False
            for _ in range(max_pre_frames):
                f = read_frame()
                if f is None:
                    return None
                preroll.append(f)
                if len(preroll) > self.PREROLL_FRAMES:
                    preroll.pop(0)
                if rms(f) > threshold:
                    consecutive_speech += 1
                    if consecutive_speech >= self.SPEECH_TRIGGER_FRAMES:
                        triggered = True
                        break
                else:
                    consecutive_speech = 0

            if not triggered:
                return None

            captured.extend(preroll)

            # Record until silence_hang frames are below threshold.
            if status:
                status.set_activity("recording")
            silent_run = 0
            max_frames = int(self.MAX_RECORD_SECONDS * 1000 / self.FRAME_MS)
            for _ in range(max_frames):
                f = read_frame()
                if f is None:
                    break
                captured.append(f)
                if rms(f) > threshold:
                    silent_run = 0
                else:
                    silent_run += 1
                    if silent_run >= self.SILENCE_HANG_FRAMES:
                        break
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if status:
                status.set_activity(None)

        if not captured:
            return None
        raw = b"".join(captured)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples

    def transcribe(self, samples, status: "StatusBar | None" = None) -> str:
        if samples is None or len(samples) == 0:
            return ""
        self._ensure_model(status)
        if status:
            status.set_activity("transcribing")
        try:
            result = self._model.transcribe(
                samples,
                language="en",
                fp16=False,           # CPU
                condition_on_previous_text=False,
            )
        finally:
            if status:
                status.set_activity(None)
        return (result.get("text") or "").strip()


class WakeListener:
    """Continuous wake-word listener.

    Owns a single arecord stream feeding a state machine that alternates
    between WAKE_LISTENING (every 80 ms frame fed to openwakeword) and
    CAPTURING (energy-VAD recording of the user's command after the wake
    word is heard). The captured audio is transcribed via the supplied
    Listener (so Whisper is loaded once and shared with push-to-talk).

    Detected commands are pushed onto `commands` for the REPL to consume.
    Detection is suppressed while the assistant is speaking, to avoid
    self-triggering on the speaker's audio.
    """

    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 1280              # openwakeword's expected chunk
    FRAME_BYTES = FRAME_SAMPLES * 2   # int16 mono
    FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE    # 80 ms
    SILENCE_HANG_FRAMES = 19          # ~1.5 s of silence ends utterance
    SPEECH_TRIGGER_FRAMES = 2         # ~160 ms above threshold to count as speech-started
    MAX_CAPTURE_FRAMES = 30 * 1000 // FRAME_MS        # 30 s hard cap
    POST_DETECT_COOLDOWN_S = 1.5      # ignore predictions briefly after detection
    MIN_RMS_THRESHOLD = 800.0         # floor for very quiet rooms

    def __init__(
        self,
        listener: Listener,
        speaker: "Speaker",
        status: "StatusBar | None" = None,
        mic_device: str = MIC_DEVICE,
        wake_model: str = WAKE_MODEL_NAME,
        threshold: float = WAKE_THRESHOLD,
    ):
        self.listener = listener
        self.speaker = speaker
        self.status = status
        self.mic_device = mic_device
        self.wake_model = wake_model
        self.threshold = float(threshold)
        self.commands: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._oww = None
        self._proc: subprocess.Popen | None = None

    def _load_oww(self):
        # Lazy and optional — keeps openwakeword off the import path until
        # the user actually opts in with --wake.
        if self._oww is not None:
            return self._oww
        if self.status:
            self.status.set_activity(f"loading wake model {self.wake_model}")
        else:
            print(f"[loading wake model {self.wake_model}...]", flush=True)
        from openwakeword.model import Model
        self._oww = Model(
            wakeword_models=[self.wake_model],
            inference_framework="onnx",
        )
        if self.status:
            self.status.set_activity(None)
        return self._oww

    def start(self) -> None:
        # Load model + Whisper up-front so the first wake doesn't pause for
        # a multi-second model load.
        self._load_oww()
        self.listener._ensure_model(self.status)
        self._thread = threading.Thread(target=self._run, daemon=True, name="wake-listener")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2)

    def _read_frame(self) -> bytes | None:
        assert self._proc is not None and self._proc.stdout is not None
        buf = b""
        while len(buf) < self.FRAME_BYTES:
            if self._stop.is_set():
                return None
            chunk = self._proc.stdout.read(self.FRAME_BYTES - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    @staticmethod
    def _rms(frame_bytes: bytes) -> float:
        import numpy as np
        s = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
        if s.size == 0:
            return 0.0
        return float((s * s).mean() ** 0.5)

    def _set_activity(self, label: str | None) -> None:
        if self.status:
            self.status.set_activity(label)

    def _run(self) -> None:
        import numpy as np
        try:
            self._proc = subprocess.Popen(
                [
                    "arecord",
                    "-D", self.mic_device,
                    "-r", str(self.SAMPLE_RATE),
                    "-f", "S16_LE",
                    "-c", "1",
                    "-t", "raw",
                    "-q",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            print(f"\n[wake mic error: {e}]")
            return

        self._set_activity("waiting for wake word")
        cooldown_until = 0.0
        try:
            while not self._stop.is_set():
                frame = self._read_frame()
                if frame is None:
                    break

                # Suppress detection while we're speaking — the speaker
                # bleeds into the mic and easily false-triggers.
                if self.speaker.is_playing() or self.speaker.pending() > 0:
                    # Reset OWW so post-playback predictions start clean.
                    self._oww.reset()
                    cooldown_until = time.monotonic() + 0.5
                    continue

                if time.monotonic() < cooldown_until:
                    continue

                samples = np.frombuffer(frame, dtype=np.int16)
                scores = self._oww.predict(samples)
                top = max(scores.values()) if scores else 0.0
                if top < self.threshold:
                    continue

                # --- WAKE WORD DETECTED ---
                self._set_activity(f"wake heard ({top:.2f})")
                self._oww.reset()

                # Capture the following utterance with a simple energy VAD
                # right off the same stream. Use a brief rolling baseline
                # to set a per-utterance threshold.
                self._set_activity("listening")
                captured: list[bytes] = []
                noise_levels: list[float] = []
                # Calibrate noise floor over ~240 ms.
                for _ in range(3):
                    f = self._read_frame()
                    if f is None:
                        break
                    noise_levels.append(self._rms(f))
                    captured.append(f)
                if not noise_levels:
                    break
                noise_floor = sum(noise_levels) / len(noise_levels)
                vad_threshold = max(noise_floor * 3.0, self.MIN_RMS_THRESHOLD)

                self._set_activity("recording")
                speech_started = False
                consecutive_speech = 0
                silent_streak = 0
                for _ in range(self.MAX_CAPTURE_FRAMES):
                    if self._stop.is_set():
                        break
                    f = self._read_frame()
                    if f is None:
                        break
                    captured.append(f)
                    level = self._rms(f)
                    if level > vad_threshold:
                        consecutive_speech += 1
                        if consecutive_speech >= self.SPEECH_TRIGGER_FRAMES:
                            speech_started = True
                        silent_streak = 0
                    else:
                        consecutive_speech = 0
                        if speech_started:
                            silent_streak += 1
                            if silent_streak >= self.SILENCE_HANG_FRAMES:
                                break

                if not speech_started:
                    self._set_activity("waiting for wake word")
                    cooldown_until = time.monotonic() + self.POST_DETECT_COOLDOWN_S
                    continue

                self._set_activity("transcribing")
                raw = b"".join(captured)
                samples_f = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                try:
                    text = self.listener.transcribe(samples_f, status=None)
                except Exception as e:  # noqa: BLE001
                    text = ""
                    print(f"\n[wake transcribe error: {e}]")

                text = text.strip()
                if text:
                    self.commands.put(text)

                self._set_activity("waiting for wake word")
                cooldown_until = time.monotonic() + self.POST_DETECT_COOLDOWN_S
        finally:
            self._set_activity(None)
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self._proc.kill()


class StatusBar:
    """Pinned bottom-row status with a braille throbber.

    Uses an ANSI scroll region so normal stdout writes stay above the bar.
    Falls back to a no-op when stdout is not a TTY.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    IDLE_GLYPH = "·"
    FPS = 10

    def __init__(self, speaker: Speaker):
        self.speaker = speaker
        self.enabled = sys.stdout.isatty()
        self._thinking = threading.Event()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._frame = 0
        self._rows = 0
        self._cols = 0
        self._write_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_painted: tuple[str, bool] | None = None
        self._activity: str | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._cols, self._rows = shutil.get_terminal_size((80, 24))
        if self._rows < 3:
            self.enabled = False
            return
        # Make room for the bar, define scroll region above it, park the
        # cursor at the bottom of the scroll region so prints feel natural.
        with self._write_lock:
            sys.stdout.write("\n")
            sys.stdout.write(f"\033[{self._rows};1H\033[2K")  # clear status row
            sys.stdout.write(f"\033[1;{self._rows - 1}r")     # set scroll region
            sys.stdout.write(f"\033[{self._rows - 1};1H")     # cursor in region
            sys.stdout.flush()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if not self.enabled:
            return
        with self._write_lock:
            # Reset scroll region, clear bar, leave cursor on a fresh line.
            sys.stdout.write("\033[r")
            sys.stdout.write(f"\033[{self._rows};1H\033[2K")
            sys.stdout.write(f"\033[{self._rows - 1};1H\n")
            sys.stdout.flush()

    def thinking(self, value: bool) -> None:
        if value:
            self._thinking.set()
        else:
            self._thinking.clear()

    def set_activity(self, label: str | None) -> None:
        """Override the auto-derived label (e.g. 'listening', 'recording')."""
        self._activity = label

    def pause(self) -> None:
        """Stop repainting and clear the bar so input() owns the terminal."""
        if not self.enabled:
            return
        self._paused.set()
        with self._write_lock:
            sys.stdout.write(f"\0337\033[{self._rows};1H\033[2K\0338")
            sys.stdout.flush()
        self._last_painted = None

    def resume(self) -> None:
        if not self.enabled:
            return
        self._paused.clear()

    def _state_label(self) -> tuple[str, bool]:
        """Return (label, animated)."""
        if self._activity is not None:
            return self._activity, True
        playing = self.speaker.is_playing()
        pending = self.speaker.pending()
        thinking = self._thinking.is_set()
        if playing or pending:
            extra = f" ({pending} queued)" if pending else ""
            prefix = "speaking" if playing else "synthesizing"
            return f"{prefix}{extra}", True
        if thinking:
            return "thinking", True
        return "idle", False

    def _render(self) -> None:
        if self._paused.is_set():
            return
        label, animated = self._state_label()
        # Skip the write entirely when nothing changed and there's no
        # animation to advance — keeps the terminal quiet so readline /
        # input() can echo the user's typing without interference.
        if not animated and self._last_painted == (label, animated):
            return
        if animated:
            glyph = self.FRAMES[self._frame % len(self.FRAMES)]
            self._frame += 1
        else:
            glyph = self.IDLE_GLYPH
        text = f" {glyph}  {label}"
        # Truncate to terminal width to avoid wrap.
        if len(text) > self._cols:
            text = text[: self._cols]
        with self._write_lock:
            # Save cursor, jump to status row, clear, paint, restore.
            sys.stdout.write(
                f"\0337\033[{self._rows};1H\033[2K\033[7m{text}\033[0m\0338"
            )
            sys.stdout.flush()
        self._last_painted = (label, animated)

    def _run(self) -> None:
        period = 1.0 / self.FPS
        while not self._stop.is_set():
            try:
                self._render()
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(period)


def stream_chat(messages: list[dict]):
    """Yield text chunks streamed from the Ollama chat endpoint."""
    payload = {"model": MODEL, "messages": messages, "stream": True}
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            if "message" in data and "content" in data["message"]:
                chunk = data["message"]["content"]
                if chunk:
                    yield chunk
            if data.get("done"):
                return


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a buffer into completed sentences + a leftover tail."""
    sentences: list[str] = []
    last_end = 0
    for m in SENTENCE_END.finditer(buffer):
        sentences.append(buffer[last_end:m.end()].strip())
        last_end = m.end()
    return [s for s in sentences if s], buffer[last_end:]


def parse_volume(s: str) -> float:
    """Accept '80', '80%', or '0.8' and return a Piper multiplier (0.0-2.0)."""
    s = s.strip().rstrip("%")
    v = float(s)
    # Treat values >2 as a percent scale (e.g. 80 -> 0.8, 150 -> 1.5).
    if v > 2:
        v /= 100.0
    if v < 0:
        v = 0.0
    if v > 2.0:
        v = 2.0
    return v


def run_repl(
    initial_volume: float = 1.0,
    mic_enabled: bool = True,
    mic_device: str = MIC_DEVICE,
    whisper_model: str = WHISPER_MODEL,
    wake_enabled: bool = False,
    wake_model: str = WAKE_MODEL_NAME,
    wake_threshold: float = WAKE_THRESHOLD,
) -> None:
    if not VOICE.exists():
        sys.exit(f"Voice model not found: {VOICE}")
    speaker = Speaker(VOICE, AUDIO_DEVICE, volume=initial_volume)
    status = StatusBar(speaker)
    # Wake mode needs the listener for transcription, even if push-to-talk
    # is logically a different feature.
    if wake_enabled and not mic_enabled:
        mic_enabled = True
    listener = Listener(mic_device=mic_device, model_name=whisper_model) if mic_enabled else None
    wake: WakeListener | None = None
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def handle_sigint(signum, frame):
        # Ctrl-C interrupts speech without exiting the program.
        speaker.interrupt()
        status.thinking(False)
        status.set_activity(None)
        print("\n[interrupted]")

    signal.signal(signal.SIGINT, handle_sigint)

    def capture_voice() -> str | None:
        """Record from mic, transcribe, return text (or None if nothing heard)."""
        if listener is None:
            print("[voice input disabled — start with --mic to enable]")
            return None
        try:
            samples = listener.record_until_silence(status=status)
        except FileNotFoundError as e:
            print(f"\n[mic error: {e}]")
            return None
        except Exception as e:  # noqa: BLE001
            print(f"\n[mic error: {e}]")
            status.set_activity(None)
            return None
        if samples is None:
            print("[no speech detected]")
            return None
        try:
            text = listener.transcribe(samples, status=status)
        except Exception as e:  # noqa: BLE001
            print(f"\n[transcribe error: {e}]")
            status.set_activity(None)
            return None
        if not text:
            print("[transcription was empty]")
            return None
        return text

    def read_user_turn() -> str | None:
        """Return the next user turn (typed or wake-driven). None on EOF."""
        if wake is None:
            # Plain blocking input — original path, keeps readline editing.
            status.pause()
            try:
                return input("\nyou> ").strip()
            except EOFError:
                return None
            finally:
                status.resume()
        # Wake mode: poll stdin (line-buffered, no readline editing) AND the
        # wake-listener queue. Whichever produces input first wins.
        sys.stdout.write("\nyou> ")
        sys.stdout.flush()
        while True:
            try:
                spoken = wake.commands.get_nowait()
                print(f"\nyou (wake)> {spoken}")
                return spoken.strip()
            except queue.Empty:
                pass
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.15)
            except (OSError, ValueError):
                return None
            if ready:
                line = sys.stdin.readline()
                if line == "":
                    return None  # EOF
                return line.strip()

    mic_hint = " '<Enter>' speak," if (mic_enabled and not wake_enabled) else ""
    wake_hint = f" (say '{wake_model.split('_')[0]} {wake_model.split('_')[1]}')" if wake_enabled else ""
    print(
        f"Voice chat with {MODEL}. Volume {int(speaker.volume * 100)}%."
        f"{wake_hint} Commands:{mic_hint} ':q' quit, ':reset' clear history, ':vol N' volume (0-200)."
    )
    status.start()
    if wake_enabled:
        try:
            wake = WakeListener(
                listener=listener, speaker=speaker, status=status,
                mic_device=mic_device, wake_model=wake_model, threshold=wake_threshold,
            )
            wake.start()
        except Exception as e:  # noqa: BLE001
            print(f"\n[wake disabled: {e}]")
            wake = None
    try:
        while True:
            user = read_user_turn()
            if user is None:
                print()
                break
            if not user:
                if not mic_enabled:
                    continue
                if wake is not None:
                    # In wake mode, blank line is a no-op (use the wake word
                    # or :say to record on demand).
                    continue
                # Push-to-talk: empty Enter starts a recording.
                spoken = capture_voice()
                if not spoken:
                    continue
                user = spoken
                print(f"you (spoken)> {user}")
            if user in (":q", ":quit", ":exit"):
                break
            if user == ":say":
                spoken = capture_voice()
                if not spoken:
                    continue
                user = spoken
                print(f"you (spoken)> {user}")
            if user == ":reset":
                history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("[history cleared]")
                continue
            if user == ":vol" or user.startswith(":vol "):
                parts = user.split(maxsplit=1)
                if len(parts) == 1:
                    print(f"[volume {int(speaker.volume * 100)}%]")
                else:
                    try:
                        new_vol = parse_volume(parts[1])
                    except ValueError:
                        print("[bad value, use e.g. ':vol 80' or ':vol 1.5']")
                        continue
                    speaker.set_volume(new_vol)
                    note = " (may clip)" if new_vol > 1.0 else ""
                    print(f"[volume set to {int(new_vol * 100)}%{note}]")
                continue

            history.append({"role": "user", "content": user})
            print("bot> ", end="", flush=True)
            full_reply = ""
            buffer = ""
            status.thinking(True)
            try:
                for chunk in stream_chat(history):
                    print(chunk, end="", flush=True)
                    full_reply += chunk
                    buffer += chunk
                    sentences, buffer = split_sentences(buffer)
                    for s in sentences:
                        speaker.say(s)
            except requests.RequestException as e:
                status.thinking(False)
                print(f"\n[ollama error] {e}", file=sys.stderr)
                history.pop()
                continue
            finally:
                status.thinking(False)

            # Speak any trailing text without a sentence terminator.
            tail = buffer.strip()
            if tail:
                speaker.say(tail)
            print()
            history.append({"role": "assistant", "content": full_reply})
            speaker.wait_until_idle()
    finally:
        if wake is not None:
            wake.stop()
        status.stop()
        speaker.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Type-to-voice chat with Ollama + Piper TTS.")
    parser.add_argument(
        "--volume", default="1.0",
        help="Initial speech volume. Accepts 0.0-2.0 multiplier or 0-200 percent (default: 1.0).",
    )
    parser.add_argument(
        "--no-mic", action="store_true",
        help="Disable speech input (default: enabled — press Enter or use ':say' to speak).",
    )
    parser.add_argument(
        "--mic-device", default=MIC_DEVICE,
        help=f"ALSA capture device for the microphone (default: {MIC_DEVICE}).",
    )
    parser.add_argument(
        "--whisper-model", default=WHISPER_MODEL,
        help=f"Whisper model name for transcription (default: {WHISPER_MODEL}).",
    )
    parser.add_argument(
        "--wake", action="store_true",
        help="Enable wake-word activation (default: off). Implies --mic.",
    )
    parser.add_argument(
        "--wake-model", default=WAKE_MODEL_NAME,
        help=f"openwakeword model to listen for (default: {WAKE_MODEL_NAME}).",
    )
    parser.add_argument(
        "--wake-threshold", type=float, default=WAKE_THRESHOLD,
        help=f"Detection score 0..1 (default: {WAKE_THRESHOLD}). Raise for fewer false positives.",
    )
    args = parser.parse_args()
    try:
        vol = parse_volume(args.volume)
    except ValueError:
        sys.exit(f"Invalid --volume: {args.volume!r}")
    run_repl(
        initial_volume=vol,
        mic_enabled=not args.no_mic,
        mic_device=args.mic_device,
        whisper_model=args.whisper_model,
        wake_enabled=args.wake,
        wake_model=args.wake_model,
        wake_threshold=args.wake_threshold,
    )
