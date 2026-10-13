# Encapsulate Whisper loading, recognition, language results, and model resource cleanup.
import gc
import threading

from .base import RecognitionAdapter
from ..preprocessing.audio import read_float32


class WhisperAdapter(RecognitionAdapter):
    def __init__(self):
        self._model = None
        self._key = None
        self._lock = threading.RLock()

    def _load(self, spec, config):
        import torch
        import whisper
        key = (spec.model, spec.device, config.model_cache)
        if self._key != key or self._model is None:
            self.close()
            self._model = whisper.load_model(spec.model, device=spec.device,
                                             download_root=config.model_cache)
            self._key = key
        return self._model

    def recognize(self, asset, spec, config):
        with self._lock:
            model = self._load(spec, config)
            options = {"word_timestamps": True, "verbose": False,
                       "task": config.task, "fp16": spec.device == "cuda"}
            if config.language:
                options["language"] = config.language
            waveform = read_float32(asset)
            result = model.transcribe(waveform, **options)
            if not isinstance(result, dict):
                raise TypeError("Whisper 未返回结构化识别结果")
            return result

    def close(self):
        self._model = None
        self._key = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return None
