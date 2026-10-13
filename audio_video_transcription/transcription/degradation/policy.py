# Move failed recognition to CPU or smaller models within the shared retry budget.
import re

from ..core.errors import StageFailure
from ..core.models import RecognitionSpec


class RecognitionFallbackPolicy:
    def __init__(self):
        self.history = []

    @staticmethod
    def _resource_error(error):
        texts = []
        current = error
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            texts.append(str(current).lower())
            current = getattr(current, "cause", None) or current.__cause__
        message = " ".join(texts)
        return any(token in message for token in ("out of memory", "cuda", "cudnn", "cublas", "显存", "memoryerror", "cannot allocate memory")) or isinstance(error, MemoryError)

    def next_spec(self, current, error, attempt):
        if not isinstance(error, StageFailure) or error.stage != "recognition":
            return current
        if current.engine != "whisper" or not self._resource_error(error):
            return current
        if current.device == "cuda":
            candidate = RecognitionSpec(current.engine, current.model, "cpu")
        else:
            if current.model == "large" or re.fullmatch(r"large-v[0-9]+", current.model):
                model = "medium"
            else:
                choices = {"medium": "small", "small": "base", "base": "tiny"}
                model = choices.get(current.model)
                if model is None:
                    return current
            candidate = RecognitionSpec(current.engine, model, "cpu")
        self.history.append({"after_attempt": attempt, "from": current.to_dict(),
                             "to": candidate.to_dict(), "reason": str(error)})
        return candidate
