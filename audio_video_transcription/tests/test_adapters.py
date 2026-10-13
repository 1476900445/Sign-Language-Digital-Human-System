# Validate adapter registration and Whisper call parameters with isolated dependency doubles.
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from transcription.adapters.base import RecognitionAdapter
from transcription.adapters.registry import AdapterRegistry
from transcription.adapters.whisper_adapter import WhisperAdapter
from transcription.core.config import PipelineConfig, VadConfig
from transcription.core.models import AudioAsset, RecognitionSpec
from transcription.vad.detectors import SileroDetector


class AdapterTests(unittest.TestCase):
    def test_registry_requires_contract(self):
        registry = AdapterRegistry(include_defaults=False)
        registry.register("invalid", lambda: object())
        with self.assertRaises(TypeError):
            registry.create("invalid")

    def test_registry_rejects_duplicate_name(self):
        registry = AdapterRegistry()
        with self.assertRaises(ValueError):
            registry.register("whisper", WhisperAdapter)

    def test_whisper_options_and_model_reuse(self):
        model = SimpleNamespace(transcribe=Mock(return_value={"text": "你好", "segments": []}))
        whisper = SimpleNamespace(load_model=Mock(return_value=model))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, empty_cache=Mock()))
        asset = AudioAsset(Path("normalized.wav"), "source.mp4", 2, 16000, 32000, "video")
        config = PipelineConfig(language="zh", model_cache="models")
        adapter = WhisperAdapter()
        spec = RecognitionSpec("whisper", "medium", "cpu")
        waveform = object()
        with patch.dict("sys.modules", {"torch": torch, "whisper": whisper}):
            with patch("transcription.adapters.whisper_adapter.read_float32", return_value=waveform):
                adapter.recognize(asset, spec, config)
                adapter.recognize(asset, spec, config)
            self.assertEqual(whisper.load_model.call_count, 1)
            model.transcribe.assert_called_with(waveform, word_timestamps=True,
                                                verbose=False, task="transcribe", fp16=False,
                                                language="zh")
            adapter.close()

    def test_whisper_reloads_for_new_compute_device(self):
        model = SimpleNamespace(transcribe=Mock(return_value={"text": "你好", "segments": []}))
        whisper = SimpleNamespace(load_model=Mock(return_value=model))
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, empty_cache=Mock()))
        asset = AudioAsset(Path("normalized.wav"), "source.mp4", 2, 16000, 32000, "video")
        adapter = WhisperAdapter()
        with patch.dict("sys.modules", {"torch": torch, "whisper": whisper}):
            with patch("transcription.adapters.whisper_adapter.read_float32", return_value=object()):
                adapter.recognize(asset, RecognitionSpec("whisper", "medium", "cuda"), PipelineConfig())
                adapter.recognize(asset, RecognitionSpec("whisper", "medium", "cpu"), PipelineConfig())
            self.assertEqual(whisper.load_model.call_count, 2)

    def test_silero_sample_timestamps_preserve_precision(self):
        loader = Mock(return_value=object())
        timestamps = Mock(return_value=[{"start": 1234, "end": 16001}])
        silero = SimpleNamespace(load_silero_vad=loader, get_speech_timestamps=timestamps)
        torch = SimpleNamespace(from_numpy=lambda array: array)
        asset = AudioAsset(Path("normalized.wav"), "source.mp4", 2, 16000, 32000, "video")
        detector = SileroDetector(VadConfig())
        with patch.dict("sys.modules", {"torch": torch, "silero_vad": silero}):
            with patch("transcription.vad.detectors.read_float32", return_value=object()):
                result = detector.detect(asset)
                detector.detect(asset)
        self.assertEqual(result[0].start, 1234 / 16000)
        self.assertEqual(result[0].end, 16001 / 16000)
        self.assertEqual(loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
