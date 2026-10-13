# Preserve spoken text and apply optional Traditional-to-Simplified Chinese conversion.
from copy import deepcopy
import unicodedata


def text_key(text):
    value = unicodedata.normalize("NFKC", str(text))
    return "".join(character for character in value if character.isalnum())


def has_spoken_text(text):
    return any(character.isalnum() for character in str(text))


def is_cjk(character):
    return bool(character) and ("\u3400" <= character <= "\u9fff" or "\u3040" <= character <= "\u30ff")


def join_texts(values):
    result = ""
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if result and result[-1].isalnum() and text[0].isalnum() and not is_cjk(result[-1]) and not is_cjk(text[0]):
            result += " "
        result += text
    return result


class ChineseConverter:
    def __init__(self):
        self._converter = None

    def convert(self, result):
        if self._converter is None:
            from opencc import OpenCC
            self._converter = OpenCC("t2s")
        converted = deepcopy(result)
        for segment in converted.get("segments", []):
            segment["text"] = self._converter.convert(str(segment.get("text", "")))
            for word in segment.get("words", []):
                word["word"] = self._converter.convert(str(word.get("word", "")))
        converted["text"] = join_texts(segment["text"] for segment in converted.get("segments", []))
        return converted
