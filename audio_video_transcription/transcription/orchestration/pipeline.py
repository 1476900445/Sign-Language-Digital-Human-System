# Run reusable preprocessing, VAD, recognition, postprocessing, and output stages.
from pathlib import Path
import tempfile
import threading
import time

from ..adapters.registry import AdapterRegistry
from ..core.config import PipelineConfig
from ..core.errors import CancellationError, StageFailure
from ..core.events import EventBus
from ..core.models import PipelineResult, RecognitionSpec
from ..degradation.device import resolve_device
from ..degradation.policy import RecognitionFallbackPolicy
from ..postprocessing.alignment import SegmentAligner
from ..postprocessing.normalization import ResultNormalizer
from ..postprocessing.text import ChineseConverter
from ..postprocessing.writers import OutputWriter
from ..preprocessing.audio import AudioPreprocessor
from ..preprocessing.microphone import MicrophoneRecorder
from ..preprocessing.process import MediaTools, ProcessRunner
from ..vad.service import VadService
from .retry import RetryExecutor, RetryPolicy


class TranscriptionPipeline:
    def __init__(self, config=None, registry=None, vad_service=None, waiter=None,
                 preprocessor=None, writer=None, converter=None):
        self.config = (config or PipelineConfig()).validate()
        self.registry = registry or AdapterRegistry()
        self.tools = MediaTools(self.config.ffmpeg_dir)
        self.runner = ProcessRunner(self.config.process_timeout)
        self.preprocessor = preprocessor or AudioPreprocessor(self.tools, self.runner, self.config.max_duration)
        self.vad = vad_service or VadService(self.config.vad, self.tools, self.runner)
        self.normalizer = ResultNormalizer()
        self.aligner = SegmentAligner(self.config.word_gap, self.config.end_padding)
        self.converter = converter or ChineseConverter()
        self.writer = writer or OutputWriter()
        self.waiter = waiter
        self._run_lock = threading.Lock()
        self.last_events = []
        self.last_attempts = []

    def process(self, source=None, mode="auto", callback=None, cancel_event=None):
        if mode not in {"auto", "video", "audio", "microphone"}:
            raise ValueError("输入模式无效")
        source_path = Path(source) if source is not None else None
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("当前管道正在处理任务，请创建独立实例处理并发任务")
        events = EventBus(callback=callback, cancel_event=cancel_event)
        retries = RetryExecutor(RetryPolicy(self.config.retry_delay, self.config.retry_backoff,
                                            self.config.retry_max_delay), events, self.waiter)
        started = time.perf_counter()
        self.last_events = events.history
        self.last_attempts = retries.records
        state = {"source": source_path,
                 "asset": None, "vad": None, "raw": None, "normalized": None,
                 "aligned": None, "final": None, "adapter": None,
                 "normalization_warnings": [], "alignment_stats": {}, "stages": []}
        fallback = RecognitionFallbackPolicy()
        spec = RecognitionSpec(self.config.engine, self.config.model_size,
                               resolve_device(self.config.device))

        def close_adapter():
            adapter = state["adapter"]
            state["adapter"] = None
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    events.logger.exception("Adapter cleanup failed")

        def failed(error, attempt):
            nonlocal spec
            next_spec = fallback.next_spec(spec, error, attempt)
            if next_spec != spec:
                events.emit("fallback", "recognition", attempt, "下一次重试将使用备用识别配置",
                            previous=spec.to_dict(), next=next_spec.to_dict())
                spec = next_spec
                close_adapter()
                state["raw"] = state["normalized"] = state["aligned"] = state["final"] = None
            elif isinstance(error, StageFailure) and error.stage == "recognition":
                close_adapter()

        def stage(name, attempt, action):
            events.check_cancelled()
            stage_started = time.perf_counter()
            events.emit("stage_started", name, attempt, f"正在执行 {name}")
            try:
                result = action()
                events.check_cancelled()
            except CancellationError:
                raise
            except Exception as error:
                raise StageFailure(name, error) from error
            state["stages"].append({"stage": name, "attempt": attempt,
                                    "elapsed": time.perf_counter() - stage_started})
            events.emit("stage_completed", name, attempt, f"{name} 已完成")
            return result

        try:
            with tempfile.TemporaryDirectory(prefix="av_transcription_") as temporary:
                workspace = Path(temporary)

                def operation(attempt):
                    if state["source"] is None:
                        if mode != "microphone":
                            raise StageFailure("preprocessing", ValueError("音频或视频模式需要输入文件"))
                        state["source"] = stage("microphone", attempt, lambda: MicrophoneRecorder().record(
                            workspace / "microphone_recording.wav", self.config.recording_duration,
                            self.config.microphone_device, events.cancel_event))
                    if state["asset"] is None:
                        state["asset"] = stage("preprocessing", attempt,
                                               lambda: self.preprocessor.prepare(state["source"], workspace, mode))
                    else:
                        events.emit("cache_reused", "preprocessing", attempt, "复用已标准化的音频")
                    asset = state["asset"]
                    if state["vad"] is None:
                        state["vad"] = stage("vad", attempt, lambda: self.vad.detect(asset, events, attempt))
                    if state["raw"] is None:
                        def recognize():
                            if state["adapter"] is None:
                                state["adapter"] = self.registry.create(spec.engine)
                            return state["adapter"].recognize(asset, spec, self.config)
                        state["raw"] = stage("recognition", attempt, recognize)
                    if state["normalized"] is None:
                        normalized, warnings = stage("normalization", attempt,
                                                     lambda: self.normalizer.normalize(state["raw"], asset.duration))
                        state["normalized"], state["normalization_warnings"] = normalized, warnings
                    if state["aligned"] is None:
                        aligned, stats = stage("alignment", attempt,
                                               lambda: self.aligner.align(state["normalized"], state["vad"], asset.duration))
                        state["aligned"], state["alignment_stats"] = aligned, stats
                    if state["final"] is None:
                        state["final"] = stage("text_postprocessing", attempt,
                                                lambda: self.converter.convert(state["aligned"]) if self.config.simplified_chinese else state["aligned"])
                    metadata = {"task_id": events.task_id, "attempts": attempt,
                                "max_retries": 3, "recognition": spec.to_dict(),
                                "vad": state["vad"].to_dict(), "fallbacks": fallback.history,
                                "previous_failures": list(retries.records),
                                "alignment": state["alignment_stats"],
                                "normalization_warnings": state["normalization_warnings"],
                                "stage_timings": list(state["stages"]),
                                "elapsed_before_output": time.perf_counter() - started}
                    paths = stage("output", attempt, lambda: self.writer.write(
                        state["aligned"], state["final"], asset, self.config.output_dir,
                        metadata, self.config.simplified_chinese))
                    return PipelineResult(**paths, attempts=attempt, duration=asset.duration,
                                          elapsed=time.perf_counter() - started, spec=spec,
                                          vad_backend=state["vad"].backend)

                return retries.execute(operation, failed)
        finally:
            close_adapter()
            self._run_lock.release()
