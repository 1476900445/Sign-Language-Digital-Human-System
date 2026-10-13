# Verify detector fallback, valid silence, interval cleaning, and failed-process handling.
from pathlib import Path
import unittest

from transcription.core.config import VadConfig
from transcription.core.errors import ExternalProcessError
from transcription.core.models import AudioAsset, SpeechInterval
from transcription.vad.detectors import FfmpegDetector, SpeechDetector
from transcription.vad.intervals import normalize_intervals
from transcription.vad.service import VadService


class FakeDetector(SpeechDetector):
    def __init__(self, name, intervals=None, error=None):
        self.name, self.intervals, self.error = name, intervals or [], error
        self.calls = 0

    def detect(self, asset):
        self.calls += 1
        if self.error:
            raise self.error
        return self.intervals


class VadTests(unittest.TestCase):
    def setUp(self):
        self.asset = AudioAsset(Path("unused.wav"), "unused.wav", 10, 16000, 160000, "audio")

    def test_backend_exception_uses_next_detector(self):
        first = FakeDetector("silero", error=ImportError("missing"))
        second = FakeDetector("webrtc", [SpeechInterval(1, 2)])
        report = VadService(VadConfig(), None, None, [first, second]).detect(self.asset)
        self.assertEqual(report.backend, "webrtc")
        self.assertEqual(len(report.warnings), 1)

    def test_valid_silence_does_not_invoke_energy_fallback(self):
        first = FakeDetector("silero")
        second = FakeDetector("ffmpeg", [SpeechInterval(0, 10)])
        report = VadService(VadConfig(), None, None, [first, second]).detect(self.asset)
        self.assertEqual(report.status, "silence")
        self.assertEqual(second.calls, 0)

    def test_all_backends_unavailable(self):
        report = VadService(VadConfig(), None, None,
                            [FakeDetector("one", error=RuntimeError("failure"))]).detect(self.asset)
        self.assertEqual(report.status, "unavailable")
        self.assertEqual(report.intervals, [])

    def test_interval_clipping_merging_and_invalid_values(self):
        values = [(-1, 1), (1.1, 2), (9, 20), (float("nan"), 3), (4, 4.01)]
        result = normalize_intervals(values, 10)
        self.assertEqual(result, [SpeechInterval(0, 2), SpeechInterval(9, 10)])

    def test_ffmpeg_failure_is_not_entire_media_speech(self):
        class Tools:
            def resolve(self, name):
                return name
        class Runner:
            def run(self, command):
                raise ExternalProcessError("ffmpeg", 1, "failed")
        with self.assertRaises(ExternalProcessError):
            FfmpegDetector(VadConfig(), Tools(), Runner()).detect(self.asset)


if __name__ == "__main__":
    unittest.main()
