# Implement Silero, WebRTC, and FFmpeg detectors using the shared normalized audio.
from abc import ABC, abstractmethod
import re

from ..core.models import SpeechInterval
from ..preprocessing.audio import read_float32, read_pcm16
from .intervals import complement_intervals, normalize_intervals


class SpeechDetector(ABC):
    name = "unknown"
    signal_type = "speech"

    @abstractmethod
    def detect(self, asset):
        raise NotImplementedError


class SileroDetector(SpeechDetector):
    name = "silero"

    def __init__(self, config):
        self.config = config
        self._model = None

    def detect(self, asset):
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        if self._model is None:
            self._model = load_silero_vad()
        waveform = torch.from_numpy(read_float32(asset))
        timestamps = get_speech_timestamps(
            waveform, self._model, sampling_rate=16000,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_ms,
            min_silence_duration_ms=self.config.min_silence_ms,
            speech_pad_ms=0, return_seconds=False,
        )
        return [SpeechInterval(float(item["start"]) / asset.sample_rate,
                               float(item["end"]) / asset.sample_rate) for item in timestamps]


class WebRtcDetector(SpeechDetector):
    name = "webrtc"

    def __init__(self, config):
        self.config = config

    def detect(self, asset):
        import webrtcvad
        detector = webrtcvad.Vad(self.config.webrtc_aggressiveness)
        pcm = read_pcm16(asset)
        frame_ms = 30
        frame_bytes = asset.sample_rate * frame_ms // 1000 * 2
        result = []
        for index in range(len(pcm) // frame_bytes):
            frame = pcm[index * frame_bytes:(index + 1) * frame_bytes]
            if detector.is_speech(frame, asset.sample_rate):
                start = index * frame_ms / 1000.0
                result.append(SpeechInterval(max(0.0, start - 0.06), min(asset.duration, start + 0.12)))
        return normalize_intervals(result, asset.duration, self.config.merge_gap, 0.0)


class FfmpegDetector(SpeechDetector):
    name = "ffmpeg"
    signal_type = "non_silence"

    def __init__(self, config, tools, runner):
        self.config = config
        self.tools = tools
        self.runner = runner

    def detect(self, asset):
        result = self.runner.run([
            self.tools.resolve("ffmpeg"), "-nostdin", "-hide_banner", "-i", asset.path,
            "-af", f"silencedetect=noise={self.config.silence_noise_db}dB:d={self.config.silence_duration}",
            "-f", "null", "-",
        ])
        events = [(match.group(1), min(asset.duration, max(0.0, float(match.group(2)))))
                  for match in re.finditer(r"silence_(start|end):\s*([0-9.eE+-]+)", result.stderr)]
        silence = []
        active_start = None
        for kind, position in events:
            if kind == "start":
                if active_start is None:
                    active_start = position
            elif active_start is not None:
                silence.append(SpeechInterval(active_start, max(active_start, position)))
                active_start = None
        if active_start is not None:
            silence.append(SpeechInterval(active_start, asset.duration))
        return complement_intervals(silence, asset.duration)
