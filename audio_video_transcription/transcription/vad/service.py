# Distinguish valid silence from detector failures and apply ordered VAD fallback.
from ..core.models import VadReport
from .detectors import FfmpegDetector, SileroDetector, WebRtcDetector
from .intervals import normalize_intervals


class VadService:
    def __init__(self, config, tools, runner, detectors=None):
        self.config = config
        available = {"silero": lambda: SileroDetector(config),
                     "webrtc": lambda: WebRtcDetector(config),
                     "ffmpeg": lambda: FfmpegDetector(config, tools, runner)}
        self.detectors = list(detectors) if detectors is not None else [available[name]() for name in config.backends]

    def detect(self, asset, events=None, attempt=1):
        warnings = []
        for detector in self.detectors:
            if events:
                events.check_cancelled()
            try:
                raw_intervals = detector.detect(asset)
                intervals = normalize_intervals(raw_intervals, asset.duration,
                                                self.config.merge_gap, self.config.min_duration)
            except Exception as error:
                warnings.append({"backend": detector.name, "error": f"{type(error).__name__}: {error}"})
                if events:
                    events.emit("fallback", "vad", attempt, f"{detector.name} 不可用，尝试下一个检测器",
                                backend=detector.name, error_type=type(error).__name__)
                continue
            report = VadReport(intervals, detector.name,
                               "speech" if intervals else "silence", detector.signal_type, warnings)
            if events:
                events.emit("stage_completed", "vad", attempt,
                            f"VAD 检测完成，得到 {len(intervals)} 个区间",
                            backend=report.backend, status=report.status,
                            signal_type=report.signal_type)
            return report
        if events:
            events.emit("fallback", "vad", attempt, "VAD 均不可用，将保留识别引擎时间戳")
        return VadReport(warnings=warnings)
