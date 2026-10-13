# Define media assets, recognition specifications, and serializable task results.
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True, order=True)
class SpeechInterval:
    start: float
    end: float


@dataclass(frozen=True)
class MediaInfo:
    source: Path
    kind: str
    duration: float
    audio_stream_index: int
    sample_rate: int
    channels: int
    codec: str


@dataclass(frozen=True)
class AudioAsset:
    path: Path
    source_name: str
    duration: float
    sample_rate: int
    frame_count: int
    input_kind: str


@dataclass
class VadReport:
    intervals: list = field(default_factory=list)
    backend: str = "disabled"
    status: str = "unavailable"
    signal_type: str = "speech"
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return {"backend": self.backend, "status": self.status,
                "signal_type": self.signal_type,
                "intervals": [asdict(item) for item in self.intervals],
                "warnings": list(self.warnings)}


@dataclass(frozen=True)
class RecognitionSpec:
    engine: str
    model: str
    device: str

    def to_dict(self):
        return asdict(self)


@dataclass
class PipelineResult:
    srt: Path
    txt: Path
    json: Path
    original_srt: Optional[Path]
    attempts: int
    duration: float
    elapsed: float
    spec: RecognitionSpec
    vad_backend: str

    def to_dict(self):
        return {"srt": str(self.srt), "txt": str(self.txt), "json": str(self.json),
                "original_srt": str(self.original_srt) if self.original_srt else None,
                "attempts": self.attempts, "duration": self.duration,
                "elapsed": self.elapsed, "recognition": self.spec.to_dict(),
                "vad_backend": self.vad_backend}
