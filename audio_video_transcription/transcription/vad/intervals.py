# Clean, clip, and merge speech intervals while preserving meaningful pauses.
import math

from ..core.models import SpeechInterval


def normalize_intervals(intervals, duration, max_gap=0.16, min_duration=0.12):
    cleaned = []
    for item in intervals or []:
        try:
            if isinstance(item, SpeechInterval):
                start, end = item.start, item.end
            elif isinstance(item, dict):
                start, end = item["start"], item["end"]
            else:
                start, end = item[0], item[1]
            start, end = float(start), float(end)
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            start, end = max(0.0, start), min(duration, end)
            if end - start >= min_duration:
                cleaned.append(SpeechInterval(start, end))
        except (KeyError, IndexError, ValueError, TypeError):
            continue
    cleaned.sort()
    merged = []
    for item in cleaned:
        if merged and item.start - merged[-1].end <= max_gap:
            merged[-1] = SpeechInterval(merged[-1].start, max(merged[-1].end, item.end))
        else:
            merged.append(item)
    return merged


def overlapping_intervals(intervals, start, end):
    return [item for item in intervals if item.end > start and item.start < end]


def complement_intervals(silent_intervals, duration):
    silence = normalize_intervals(silent_intervals, duration, 0.0, 0.0)
    result = []
    cursor = 0.0
    for item in silence:
        if item.start > cursor:
            result.append(SpeechInterval(cursor, item.start))
        cursor = max(cursor, item.end)
    if cursor < duration:
        result.append(SpeechInterval(cursor, duration))
    return result
