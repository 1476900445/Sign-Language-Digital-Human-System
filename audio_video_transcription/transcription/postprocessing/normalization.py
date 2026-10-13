# Convert adapter results into validated segments without silently dropping recognized text.
from copy import deepcopy
import math

from .text import join_texts, text_key


def finite_number(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


class ResultNormalizer:
    def normalize(self, raw, duration):
        if not isinstance(raw, dict):
            raise TypeError("识别结果必须为字典")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("音频时长无效")
        result = {"text": "", "language": str(raw.get("language") or "unknown"), "segments": []}
        warnings = []
        segments = raw.get("segments") or []
        if not isinstance(segments, list):
            raise TypeError("识别结果 segments 必须为列表")
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise TypeError("识别分段必须为字典")
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            start = finite_number(segment.get("start"))
            end = finite_number(segment.get("end"))
            if start is not None and end is not None:
                start, end = max(0.0, start), min(duration, end)
            if start is None or end is None or end <= start:
                start, end = 0.0, duration
                warnings.append(f"第 {index + 1} 段时间无效，保留文字并回退完整音频范围")
            words = []
            for word in segment.get("words") or []:
                if not isinstance(word, dict):
                    continue
                word_text = str(word.get("word") or "")
                word_start = finite_number(word.get("start"))
                word_end = finite_number(word.get("end"))
                if word_start is None or word_end is None:
                    continue
                word_start, word_end = max(0.0, word_start), min(duration, word_end)
                if not word_text.strip() or word_end <= word_start:
                    continue
                record = {"word": word_text, "start": word_start, "end": word_end}
                probability = finite_number(word.get("probability"))
                if probability is not None:
                    record["probability"] = probability
                words.append(record)
            words.sort(key=lambda item: (item["start"], item["end"]))
            result["segments"].append({"id": index, "text": text, "start": start,
                                       "end": end, "words": words})
        full_text = str(raw.get("text") or "").strip()
        collected_text = join_texts(segment["text"] for segment in result["segments"])
        if full_text and not result["segments"]:
            result["segments"] = [{"id": 0, "text": full_text, "start": 0.0, "end": duration, "words": []}]
            warnings.append("识别结果缺少分段，使用完整音频时间保留全文")
        elif full_text and text_key(full_text) != text_key(collected_text):
            raise ValueError("识别结果全文与分段文字不一致，无法确认内容完整性")
        if not result["segments"]:
            raise ValueError("识别结果没有有效文字")
        result["text"] = join_texts(segment["text"] for segment in result["segments"])
        return deepcopy(result), warnings
