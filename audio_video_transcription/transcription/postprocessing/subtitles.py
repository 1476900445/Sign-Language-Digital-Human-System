# Serialize and parse SRT subtitles with validated millisecond timestamps.
import math
import re


def format_timestamp(seconds):
    number = float(seconds)
    if not math.isfinite(number) or number < 0:
        raise ValueError("字幕时间必须是有限非负数")
    total = round(number * 1000)
    hours, remainder = divmod(total, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    second, millisecond = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{second:02d},{millisecond:03d}"


def parse_timestamp(value):
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})", value.strip())
    if not match:
        raise ValueError("SRT 时间码格式错误")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError("SRT 时间码范围错误")
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


def render_srt(result):
    blocks = []
    for segment in result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        if segment["end"] <= segment["start"]:
            raise ValueError("字幕结束时间不得小于或等于开始时间")
        if start == end:
            raise ValueError("字幕时间跨度小于可写入的毫秒精度")
        blocks.append(f"{len(blocks) + 1}\n{start} --> {end}\n{text}")
    if not blocks:
        raise ValueError("没有可写入的有效字幕")
    return "\n\n".join(blocks) + "\n"


def parse_srt(content):
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip())
    segments = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            raise ValueError("SRT 字幕块不完整")
        timing = lines[1].split("-->")
        if len(timing) != 2:
            raise ValueError("SRT 字幕时间行无效")
        start, end = map(parse_timestamp, timing)
        if end <= start:
            raise ValueError("SRT 字幕时间跨度无效")
        segments.append({"id": len(segments), "start": start, "end": end,
                         "text": "\n".join(lines[2:]).strip(), "words": []})
    return segments
