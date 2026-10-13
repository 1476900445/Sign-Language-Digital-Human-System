# Load and validate transcription settings without importing model libraries.
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
import json
import math


@dataclass(frozen=True)
class VadConfig:
    backends: tuple = ("silero", "webrtc", "ffmpeg")
    threshold: float = 0.65
    min_speech_ms: int = 120
    min_silence_ms: int = 180
    merge_gap: float = 0.16
    min_duration: float = 0.12
    webrtc_aggressiveness: int = 3
    silence_noise_db: float = -38.0
    silence_duration: float = 0.30

    def validate(self):
        if not self.backends or len(set(self.backends)) != len(self.backends):
            raise ValueError("VAD 后端列表不能为空或重复")
        if any(name not in {"silero", "webrtc", "ffmpeg"} for name in self.backends):
            raise ValueError("VAD 后端名称无效")
        if not 0 < self.threshold < 1:
            raise ValueError("VAD threshold 必须在 0 和 1 之间")
        if self.webrtc_aggressiveness not in range(4):
            raise ValueError("WebRTC aggressiveness 必须是 0 至 3")
        if self.min_speech_ms <= 0 or self.min_silence_ms <= 0:
            raise ValueError("VAD 时长参数必须大于 0")
        for value in (self.merge_gap, self.min_duration, self.silence_duration):
            if not math.isfinite(value) or value < 0:
                raise ValueError("VAD 时间参数必须是有限非负数")


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: str = "output"
    ffmpeg_dir: Optional[str] = None
    model_size: str = "medium"
    model_cache: Optional[str] = None
    engine: str = "whisper"
    device: str = "auto"
    language: Optional[str] = None
    task: str = "transcribe"
    simplified_chinese: bool = True
    word_gap: float = 0.28
    end_padding: float = 0.20
    process_timeout: float = 600.0
    max_duration: float = 0.0
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    retry_max_delay: float = 4.0
    recording_duration: float = 30.0
    microphone_device: Optional[int] = None
    vad: VadConfig = field(default_factory=VadConfig)

    def validate(self):
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device 必须是 auto、cpu 或 cuda")
        if self.task not in {"transcribe", "translate"}:
            raise ValueError("task 必须是 transcribe 或 translate")
        if not self.engine.strip() or not self.model_size.strip():
            raise ValueError("识别引擎和模型不能为空")
        if not str(self.output_dir).strip():
            raise ValueError("输出目录不能为空")
        for name in ("word_gap", "process_timeout", "recording_duration"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        for name in ("end_padding", "max_duration", "retry_delay", "retry_max_delay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限非负数")
        if not math.isfinite(self.retry_backoff) or self.retry_backoff < 1:
            raise ValueError("retry_backoff 必须大于或等于 1")
        self.vad.validate()
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError("未知配置项: " + ", ".join(sorted(unknown)))
        vad_values = dict(values.pop("vad", {}))
        if "backends" in vad_values:
            vad_values["backends"] = tuple(vad_values["backends"])
        values["vad"] = VadConfig(**vad_values)
        return cls(**values).validate()

    @classmethod
    def from_json(cls, path):
        with Path(path).open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("配置文件顶层必须是 JSON 对象")
        return cls.from_dict(data)
