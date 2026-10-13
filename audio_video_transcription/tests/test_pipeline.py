# Exercise the complete pipeline with real media preprocessing and a deterministic test adapter.
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import math
import shutil
import struct
import tempfile
import unittest
import wave

from transcription import PipelineConfig, TranscriptionPipeline
from transcription.adapters.base import RecognitionAdapter
from transcription.adapters.registry import AdapterRegistry
from transcription.core.errors import ProcessingFailed, StageFailure
from transcription.core.models import RecognitionSpec, SpeechInterval, VadReport
from transcription.degradation.policy import RecognitionFallbackPolicy
from transcription.preprocessing.audio import AudioPreprocessor
from transcription.preprocessing.process import MediaTools, ProcessRunner
from transcription.vad.detectors import FfmpegDetector


def create_wave(path, rate=48000, channels=2, duration=2, silence=False):
    values = bytearray()
    for index in range(round(rate * duration)):
        value = 0 if silence else round(12000 * math.sin(2 * math.pi * 220 * index / rate))
        values.extend(struct.pack("<h", value) * channels)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(values)


class TestAdapter(RecognitionAdapter):
    def __init__(self, failures=0, error="temporary engine failure"):
        self.failures = failures
        self.error = error
        self.calls = []

    def recognize(self, asset, spec, config):
        self.calls.append((asset, spec))
        if len(self.calls) <= self.failures:
            raise RuntimeError(self.error)
        return {"language": "zh", "text": "你好世界", "segments": [
            {"start": 0, "end": asset.duration, "text": "你好世界", "words": [
                {"word": "你好", "start": 0.2, "end": 0.7},
                {"word": "世界", "start": 1.2, "end": 1.7}]}]}


class FixedVad:
    def __init__(self):
        self.calls = 0

    def detect(self, asset, events, attempt):
        self.calls += 1
        return VadReport([SpeechInterval(0.2, 0.7), SpeechInterval(1.2, 1.7)], "test", "speech")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are required")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.source = self.folder / "中文样本.wav"
        create_wave(self.source)
        self.config = PipelineConfig(output_dir=str(self.folder / "output"), engine="test",
                                     simplified_chinese=False, retry_delay=0)

    def tearDown(self):
        self.temporary.cleanup()

    def make_pipeline(self, adapter, config=None, writer=None):
        registry = AdapterRegistry(include_defaults=False)
        registry.register((config or self.config).engine, lambda: adapter)
        vad = FixedVad()
        pipeline = TranscriptionPipeline(config or self.config, registry=registry,
                                         vad_service=vad, waiter=lambda delay: None, writer=writer)
        return pipeline, vad

    def test_preprocessor_produces_mono_16k(self):
        preprocessor = AudioPreprocessor(MediaTools(), ProcessRunner())
        asset = preprocessor.prepare(self.source, self.folder)
        self.assertEqual(asset.sample_rate, 16000)
        self.assertAlmostEqual(asset.duration, 2, places=3)
        with wave.open(str(asset.path), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getsampwidth(), 2)

    def test_pipeline_success_on_third_retry_reuses_audio(self):
        adapter = TestAdapter(failures=3)
        pipeline, vad = self.make_pipeline(adapter)
        with patch.object(pipeline.preprocessor, "prepare", wraps=pipeline.preprocessor.prepare) as prepare:
            result = pipeline.process(self.source)
        self.assertEqual(result.attempts, 4)
        self.assertEqual(len(adapter.calls), 4)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(vad.calls, 1)
        self.assertTrue(result.srt.is_file())
        self.assertEqual(result.txt.read_text(encoding="utf-8").strip(), "你好世界")
        payload = json.loads(result.json.read_text(encoding="utf-8"))
        self.assertEqual(payload["processing"]["attempts"], 4)
        self.assertEqual(len(payload["processing"]["previous_failures"]), 3)
        self.assertFalse(any(event.kind == "failed" for event in pipeline.last_events))

    def test_pipeline_failure_emits_one_terminal_event(self):
        adapter = TestAdapter(failures=10)
        pipeline, _ = self.make_pipeline(adapter)
        with self.assertRaises(ProcessingFailed):
            pipeline.process(self.source)
        self.assertEqual(len(adapter.calls), 4)
        self.assertEqual(sum(event.kind == "failed" for event in pipeline.last_events), 1)
        self.assertFalse((self.folder / "output" / "中文样本.srt").exists())

    def test_output_failure_reuses_recognition(self):
        from transcription.postprocessing.writers import OutputWriter
        class RecoveringWriter(OutputWriter):
            def __init__(self):
                self.calls = 0
            def write(self, *args, **kwargs):
                self.calls += 1
                if self.calls < 3:
                    raise OSError("temporary disk error")
                return super().write(*args, **kwargs)
        adapter = TestAdapter()
        writer = RecoveringWriter()
        pipeline, _ = self.make_pipeline(adapter, writer=writer)
        result = pipeline.process(self.source)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(writer.calls, 3)

    def test_cuda_failure_switches_to_cpu_within_retry_budget(self):
        adapter = TestAdapter(failures=1, error="CUDA out of memory")
        config = replace(self.config, engine="whisper", device="cuda")
        pipeline, _ = self.make_pipeline(adapter, config)
        result = pipeline.process(self.source)
        self.assertEqual(result.attempts, 2)
        self.assertEqual([spec.device for _, spec in adapter.calls], ["cuda", "cpu"])

    def test_ffmpeg_silence_detection_returns_empty_intervals(self):
        silent = self.folder / "silence.wav"
        create_wave(silent, silence=True)
        tools, runner = MediaTools(), ProcessRunner()
        asset = AudioPreprocessor(tools, runner).prepare(silent, self.folder)
        self.assertEqual(FfmpegDetector(self.config.vad, tools, runner).detect(asset), [])

    def test_missing_input_uses_same_retry_budget(self):
        pipeline, _ = self.make_pipeline(TestAdapter())
        with self.assertRaises(ProcessingFailed) as raised:
            pipeline.process(self.folder / "missing.wav")
        self.assertEqual(raised.exception.attempts, 4)

    def test_real_video_audio_track_is_extracted(self):
        video = self.folder / "video.mp4"
        runner = ProcessRunner()
        tools = MediaTools()
        runner.run([tools.resolve("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=10",
                    "-i", self.source, "-t", "2", "-c:v", "mpeg4", "-c:a", "aac", video])
        asset = AudioPreprocessor(tools, runner).prepare(video, self.folder)
        self.assertEqual(asset.input_kind, "video")
        self.assertEqual(asset.sample_rate, 16000)
        self.assertAlmostEqual(asset.duration, 2, delta=0.05)

    def test_video_without_audio_track_is_rejected(self):
        video = self.folder / "silent_video.mp4"
        runner = ProcessRunner()
        tools = MediaTools()
        runner.run([tools.resolve("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:r=10",
                    "-t", "0.3", "-c:v", "mpeg4", video])
        with self.assertRaisesRegex(ValueError, "不包含音轨"):
            AudioPreprocessor(tools, runner).prepare(video, self.folder)


class FallbackTests(unittest.TestCase):
    def test_cpu_memory_failure_selects_smaller_model(self):
        policy = RecognitionFallbackPolicy()
        current = RecognitionSpec("whisper", "medium", "cpu")
        next_spec = policy.next_spec(current, StageFailure("recognition", RuntimeError("out of memory")), 1)
        self.assertEqual(next_spec.model, "small")

    def test_output_failure_does_not_change_recognition(self):
        policy = RecognitionFallbackPolicy()
        current = RecognitionSpec("whisper", "medium", "cuda")
        self.assertEqual(policy.next_spec(current, StageFailure("output", RuntimeError("CUDA text")), 1), current)

    def test_unknown_config_key_is_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig.from_dict({"unknown_setting": True})


if __name__ == "__main__":
    unittest.main()
