# Register adapter factories and replace recognition engines without changing the pipeline.
import threading

from .base import RecognitionAdapter


class AdapterRegistry:
    def __init__(self, include_defaults=True):
        self._factories = {}
        self._lock = threading.RLock()
        if include_defaults:
            from .whisper_adapter import WhisperAdapter
            self.register("whisper", WhisperAdapter)

    def register(self, name, factory, replace=False):
        normalized = str(name).strip().lower()
        if not normalized or not callable(factory):
            raise ValueError("适配器名称或工厂无效")
        with self._lock:
            if normalized in self._factories and not replace:
                raise ValueError(f"适配器已存在: {normalized}")
            self._factories[normalized] = factory

    def create(self, name):
        with self._lock:
            factory = self._factories.get(str(name).strip().lower())
        if factory is None:
            raise ValueError(f"未注册识别适配器: {name}")
        adapter = factory()
        if not isinstance(adapter, RecognitionAdapter):
            raise TypeError("适配器必须实现 RecognitionAdapter")
        return adapter

    def names(self):
        with self._lock:
            return tuple(sorted(self._factories))
