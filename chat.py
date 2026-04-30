#!/usr/bin/env python3
"""Type a prompt -> Ollama (gemma4:e2b) -> Piper TTS -> USB speaker.

Streams the LLM response and speaks it sentence-by-sentence so audio starts
playing before generation finishes.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e2b"
VOICE = Path("/home/dingo/Programming/Piper/voices/en_US-amy-medium.onnx")
AUDIO_DEVICE = "plughw:1,0"   # USB speaker (card 1)
MIC_DEVICE = "plughw:0,0"     # C920 webcam mic (card 0)
WHISPER_MODEL = "tiny"        # tiny ~73MB, fast on Jetson CPU
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
) -> None:
    if not VOICE.exists():
        sys.exit(f"Voice model not found: {VOICE}")
    speaker = Speaker(VOICE, AUDIO_DEVICE, volume=initial_volume)
    status = StatusBar(speaker)
    listener = Listener(mic_device=mic_device, model_name=whisper_model) if mic_enabled else None
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

    mic_hint = " '<Enter>' speak," if mic_enabled else ""
    print(
        f"Voice chat with {MODEL}. Volume {int(speaker.volume * 100)}%. "
        f"Commands:{mic_hint} ':q' quit, ':reset' clear history, ':vol N' volume (0-200)."
    )
    status.start()
    try:
        while True:
            status.pause()
            try:
                user = input("\nyou> ").strip()
            except EOFError:
                print()
                break
            finally:
                status.resume()
            if not user:
                if not mic_enabled:
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
    )
