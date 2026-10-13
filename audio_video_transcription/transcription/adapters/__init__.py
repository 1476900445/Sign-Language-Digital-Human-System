# Expose a replaceable speech-recognition interface and adapter registry.
from .base import RecognitionAdapter
from .registry import AdapterRegistry

__all__ = ["RecognitionAdapter", "AdapterRegistry"]
