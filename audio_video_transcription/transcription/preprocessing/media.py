# Inspect actual media streams instead of relying on filename extensions.
from pathlib import Path
import json
import math

from ..core.models import MediaInfo
from .validation import validate_input


class MediaInspector:
    def __init__(self, tools, runner, max_duration=0.0):
        self.tools = tools
        self.runner = runner
        self.max_duration = max_duration

    def inspect(self, source):
        path = validate_input(source)
        result = self.runner.run([
            self.tools.resolve("ffprobe"), "-v", "error", "-show_streams",
            "-show_format", "-of", "json", path,
        ], timeout=30)
        payload = json.loads(result.stdout)
        streams = payload.get("streams") or []
        audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if not audio:
            raise ValueError("输入媒体不包含音轨")
        selected = next((stream for stream in audio if stream.get("disposition", {}).get("default")), audio[0])
        duration = self._duration(payload, selected)
        if duration <= 0:
            raise ValueError("无法确认有效音频时长")
        if self.max_duration and duration > self.max_duration:
            raise ValueError(f"媒体时长超过限制 {self.max_duration:.1f} 秒")
        video = any(stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic") for stream in streams)
        return MediaInfo(path, "video" if video else "audio", duration,
                         int(selected["index"]), int(selected.get("sample_rate") or 0),
                         int(selected.get("channels") or 0), str(selected.get("codec_name") or "unknown"))

    @staticmethod
    def _duration(payload, selected):
        candidates = [selected.get("duration"), (payload.get("format") or {}).get("duration")]
        for value in candidates:
            try:
                number = float(value)
                if math.isfinite(number) and number > 0:
                    return number
            except (ValueError, TypeError):
                continue
        return 0.0
