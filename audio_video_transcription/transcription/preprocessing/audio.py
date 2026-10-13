# Normalize each media input once and share the resulting PCM audio with all modules.
from pathlib import Path
import wave

from ..core.models import AudioAsset
from .media import MediaInspector


class AudioPreprocessor:
    def __init__(self, tools, runner, max_duration=0.0):
        self.tools = tools
        self.runner = runner
        self.inspector = MediaInspector(tools, runner, max_duration)

    def prepare(self, source, workspace, mode="auto"):
        info = self.inspector.inspect(source)
        destination = Path(workspace) / "normalized_audio.wav"
        self.runner.run([
            self.tools.resolve("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-y", "-i", info.source, "-map", f"0:{info.audio_stream_index}",
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", destination,
        ])
        with wave.open(str(destination), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or rate != 16000:
                raise ValueError("音频标准化结果不符合单声道 16 kHz PCM 要求")
            if frames <= 0:
                raise ValueError("音频标准化结果没有采样数据")
        return AudioAsset(destination, info.source.name, frames / rate, rate, frames,
                          "microphone" if mode == "microphone" else info.kind)


def read_pcm16(asset):
    with wave.open(str(asset.path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("VAD 输入必须是单声道 PCM16")
        return handle.readframes(handle.getnframes())


def read_float32(asset):
    import numpy as np
    return np.frombuffer(read_pcm16(asset), dtype="<i2").astype(np.float32) / 32768.0
