# Refine word timings while preserving text and the earliest speech in unsplit segments.
from copy import deepcopy
import math

from ..vad.intervals import overlapping_intervals
from .text import has_spoken_text, join_texts, text_key


class SegmentAligner:
    def __init__(self, word_gap=0.28, end_padding=0.20):
        self.word_gap = word_gap
        self.end_padding = end_padding

    def _group_words(self, words):
        groups = []
        current = []
        last_spoken_end = None
        for word in words:
            spoken = has_spoken_text(word["word"])
            if spoken and current and last_spoken_end is not None and word["start"] - last_spoken_end >= self.word_gap:
                groups.append(current)
                current = []
            current.append(word)
            if spoken:
                last_spoken_end = word["end"]
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _word_text(words):
        return join_texts(word["word"] for word in words)

    def align(self, result, vad_report, duration):
        output = deepcopy(result)
        segments = []
        stats = {"input_segments": len(result["segments"]), "output_segments": 0,
                 "split_segments": 0, "vad_adjustments": 0, "overlap_adjustments": 0,
                 "retained_segments": 0, "warnings": []}
        original_keys = []
        output_keys = []
        for original_index, original in enumerate(result["segments"]):
            original_keys.append(text_key(original["text"]))
            words = original.get("words", [])
            groups = self._group_words(words)
            complete = bool(words) and text_key(self._word_text(words)) == text_key(original["text"])
            can_split = complete and len(groups) > 1 and all(any(has_spoken_text(word["word"]) for word in group) for group in groups)
            if can_split:
                proposed = []
                for group in groups:
                    spoken = [word for word in group if has_spoken_text(word["word"])]
                    start = max(original["start"], spoken[0]["start"])
                    end = min(original["end"], spoken[-1]["end"] + self.end_padding, duration)
                    proposed.append({"text": self._word_text(group), "start": start,
                                     "end": end, "words": deepcopy(group)})
                if any(item["end"] <= item["start"] for item in proposed):
                    proposed = [deepcopy(original)]
                    stats["warnings"].append(f"第 {original_index + 1} 段拆分时间异常，保留原句")
                else:
                    stats["split_segments"] += 1
            else:
                proposed = [deepcopy(original)]
                if complete:
                    spoken = [word for word in words if has_spoken_text(word["word"])]
                    if spoken:
                        start = max(original["start"], spoken[0]["start"])
                        end = min(original["end"], spoken[-1]["end"] + self.end_padding)
                        if end > start:
                            proposed[0]["start"], proposed[0]["end"] = start, end
            segment_outputs = []
            for item in proposed:
                raw_start, raw_end = item["start"], item["end"]
                candidates = overlapping_intervals(vad_report.intervals, raw_start, raw_end)
                if candidates:
                    earliest = min(candidates, key=lambda interval: interval.start)
                    adjusted = max(raw_start, earliest.start)
                    if adjusted < raw_end and adjusted > raw_start:
                        item["start"] = adjusted
                        stats["vad_adjustments"] += 1
                item["start"] = max(0.0, min(duration, item["start"]))
                item["end"] = max(0.0, min(duration, item["end"]))
                if not math.isfinite(item["start"]) or not math.isfinite(item["end"]) or item["end"] <= item["start"]:
                    segment_outputs = [deepcopy(original)]
                    stats["warnings"].append(f"第 {original_index + 1} 段校正失败，回退原时间")
                    break
                segment_outputs.append(item)
            if text_key(join_texts(item["text"] for item in segment_outputs)) != text_key(original["text"]):
                segment_outputs = [deepcopy(original)]
                stats["warnings"].append(f"第 {original_index + 1} 段触发文字完整性保护")
            if len(segment_outputs) == 1:
                stats["retained_segments"] += 1
            output_keys.append(text_key(join_texts(item["text"] for item in segment_outputs)))
            segments.extend(segment_outputs)
        if original_keys != output_keys:
            raise ValueError("后处理文字完整性检查失败")
        for index in range(len(segments) - 1):
            current, following = segments[index], segments[index + 1]
            if current["end"] > following["start"] > current["start"]:
                current["end"] = following["start"]
                stats["overlap_adjustments"] += 1
            elif current["end"] > following["start"]:
                stats["warnings"].append(f"第 {index + 1} 和 {index + 2} 段仍存在时间重叠，保留文字及原时间")
        for index, segment in enumerate(segments):
            segment["id"] = index
            segment["alignment_backend"] = vad_report.backend
        output["segments"] = segments
        output["text"] = join_texts(segment["text"] for segment in segments)
        stats["output_segments"] = len(segments)
        return output, stats
