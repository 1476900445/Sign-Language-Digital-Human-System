# Capture microphone samples once so recognition retries reuse the same recording.
from pathlib import Path
import queue
import time
import wave

from ..core.errors import CancellationError


class MicrophoneRecorder:
    def record(self, destination, duration, device=None, cancel_event=None):
        import sounddevice as sd
        if duration <= 0:
            raise ValueError("录音时长必须大于 0")
        path = Path(destination)
        chunks = queue.Queue(maxsize=512)
        failures = []

        def callback(data, frames, timing, status):
            if status:
                failures.append(str(status))
            try:
                chunks.put_nowait(data.copy().tobytes())
            except queue.Full:
                failures.append("录音缓冲区已满")

        written = 0
        target_frames = max(1, round(duration * 16000))
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            with sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                                device=device, callback=callback):
                deadline = time.monotonic() + duration + 3.0
                while written < target_frames:
                    if cancel_event is not None and cancel_event.is_set():
                        raise CancellationError("录音已取消")
                    if failures:
                        raise RuntimeError("录音采集异常: " + failures[0])
                    if time.monotonic() > deadline:
                        raise RuntimeError("麦克风采集超时")
                    try:
                        data = chunks.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    frames = min(len(data) // 2, target_frames - written)
                    output.writeframesraw(data[:frames * 2])
                    written += frames
        if not written:
            raise RuntimeError("麦克风没有采集到有效数据")
        return path
