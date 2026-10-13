# Verify timing regressions, text preservation, subtitle formatting, and output rollback.
from pathlib import Path
from unittest.mock import patch
import json
import os
import tempfile
import unittest

from transcription.core.models import AudioAsset, SpeechInterval, VadReport
from transcription.postprocessing.alignment import SegmentAligner
from transcription.postprocessing.normalization import ResultNormalizer
from transcription.postprocessing.subtitles import format_timestamp, parse_srt, render_srt
from transcription.postprocessing.text import join_texts, text_key
from transcription.postprocessing.writers import OutputWriter


class AlignmentTests(unittest.TestCase):
    def test_missing_words_preserve_earliest_speech(self):
        raw = {"text": "你好。世界。", "segments": [
            {"text": "你好。世界。", "start": 0, "end": 10, "words": []}]}
        normalized, warnings = ResultNormalizer().normalize(raw, 10)
        report = VadReport([SpeechInterval(0.2, 1), SpeechInterval(8, 9)], "test", "speech")
        output, stats = SegmentAligner().align(normalized, report, 10)
        self.assertEqual(output["segments"][0]["start"], 0.2)
        self.assertEqual(output["text"], "你好。世界。")

    def test_long_leading_silence_is_corrected(self):
        raw = {"text": "你好", "segments": [{"text": "你好", "start": 4.7, "end": 20.7, "words": []}]}
        normalized, _ = ResultNormalizer().normalize(raw, 25)
        report = VadReport([SpeechInterval(20, 20.6)], "silero", "speech")
        output, _ = SegmentAligner().align(normalized, report, 25)
        self.assertEqual(output["segments"][0]["start"], 20)
        self.assertEqual(output["segments"][0]["end"], 20.7)

    def test_complete_words_split_on_pause(self):
        raw = {"text": "你好世界", "segments": [{"text": "你好世界", "start": 0, "end": 4,
               "words": [{"word": "你好", "start": 0.2, "end": 1},
                         {"word": "世界", "start": 3, "end": 3.5}]}]}
        normalized, _ = ResultNormalizer().normalize(raw, 4)
        output, stats = SegmentAligner().align(normalized, VadReport(), 4)
        self.assertEqual(len(output["segments"]), 2)
        self.assertEqual(output["text"], "你好世界")
        self.assertEqual(stats["split_segments"], 1)

    def test_partial_words_do_not_drop_text(self):
        raw = {"text": "你好世界", "segments": [{"text": "你好世界", "start": 0, "end": 4,
               "words": [{"word": "你好", "start": 0.2, "end": 1}]}]}
        normalized, _ = ResultNormalizer().normalize(raw, 4)
        output, _ = SegmentAligner().align(normalized, VadReport(), 4)
        self.assertEqual(output["text"], "你好世界")
        self.assertEqual(len(output["segments"]), 1)

    def test_overlap_buffer_is_trimmed(self):
        raw = {"segments": [{"text": "甲", "start": 0, "end": 2},
                            {"text": "乙", "start": 1.5, "end": 3}]}
        normalized, _ = ResultNormalizer().normalize(raw, 3)
        output, stats = SegmentAligner().align(normalized, VadReport(), 3)
        self.assertEqual(output["segments"][0]["end"], 1.5)
        self.assertEqual(stats["overlap_adjustments"], 1)

    def test_invalid_time_preserves_text(self):
        raw = {"text": "保留句子", "segments": [{"text": "保留句子", "start": float("nan"), "end": 0}]}
        output, warnings = ResultNormalizer().normalize(raw, 8)
        self.assertEqual(output["text"], "保留句子")
        self.assertEqual(output["segments"][0]["end"], 8)
        self.assertTrue(warnings)

    def test_full_text_mismatch_is_reported(self):
        with self.assertRaises(ValueError):
            ResultNormalizer().normalize({"text": "你好世界", "segments": [{"text": "你好", "start": 0, "end": 1}]}, 2)

    def test_full_text_without_segments_is_retained(self):
        output, warnings = ResultNormalizer().normalize({"text": "完整文本"}, 5)
        self.assertEqual(output["segments"][0]["text"], "完整文本")
        self.assertEqual(output["segments"][0]["start"], 0)
        self.assertTrue(warnings)

    def test_unicode_and_english_spacing(self):
        self.assertEqual(join_texts(["Hello", "world"]), "Hello world")
        self.assertEqual(join_texts(["你好", "世界"]), "你好世界")
        self.assertTrue(text_key("café") == "café")


class SubtitleTests(unittest.TestCase):
    def test_timestamp_carry(self):
        self.assertEqual(format_timestamp(59.9996), "00:01:00,000")
        self.assertEqual(format_timestamp(3661.234), "01:01:01,234")

    def test_srt_roundtrip(self):
        raw = {"segments": [{"start": 0.2, "end": 1.2, "text": "你好\n世界"}]}
        parsed = parse_srt(render_srt(raw))
        self.assertEqual(parsed[0]["text"], "你好\n世界")
        self.assertEqual(parsed[0]["start"], 0.2)

    def test_submillisecond_segment_is_rejected(self):
        with self.assertRaises(ValueError):
            render_srt({"segments": [{"start": 1.00001, "end": 1.00002, "text": "甲"}]})

    def test_invalid_timestamp_is_rejected(self):
        with self.assertRaises(ValueError):
            format_timestamp(float("inf"))


class WriterTests(unittest.TestCase):
    def test_output_bundle_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            asset = AudioAsset(Path(folder) / "input.wav", "中文音频.wav", 2, 16000, 32000, "audio")
            raw = {"text": "你好", "language": "zh", "segments": [{"start": 0, "end": 1, "text": "你好"}]}
            paths = OutputWriter().write(raw, raw, asset, folder, {"attempts": 1}, True)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["processing"]["attempts"], 1)
            self.assertEqual(payload["text"], "你好")
            self.assertEqual(paths["srt"].name, "中文音频_simplified.srt")

    def test_partial_commit_restores_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "one.txt").write_text("old one", encoding="utf-8")
            (directory / "two.txt").write_text("old two", encoding="utf-8")
            real_replace = os.replace
            calls = []

            def failing_replace(source, destination):
                calls.append(str(destination))
                if len(calls) == 2:
                    raise OSError("synthetic output failure")
                return real_replace(source, destination)

            with patch("transcription.postprocessing.writers.os.replace", side_effect=failing_replace):
                with self.assertRaises(OSError):
                    OutputWriter()._commit(directory, {"one.txt": "new one", "two.txt": "new two"})
            self.assertEqual((directory / "one.txt").read_text(), "old one")
            self.assertEqual((directory / "two.txt").read_text(), "old two")


if __name__ == "__main__":
    unittest.main()
