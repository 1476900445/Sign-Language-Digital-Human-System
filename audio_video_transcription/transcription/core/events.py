# Emit structured progress events without displaying dialogs during retries.
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
import threading
import uuid

from .errors import CancellationError


@dataclass(frozen=True)
class TaskEvent:
    task_id: str
    kind: str
    stage: str
    attempt: int
    message: str
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        return asdict(self)


class EventBus:
    def __init__(self, callback=None, task_id=None, cancel_event=None):
        self.callback = callback
        self.task_id = task_id or uuid.uuid4().hex
        self.cancel_event = cancel_event or threading.Event()
        self.history = []
        self.logger = logging.getLogger("transcription")
        self._lock = threading.RLock()

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise CancellationError("任务已取消")

    def emit(self, kind, stage, attempt, message, **details):
        event = TaskEvent(self.task_id, kind, stage, attempt, message, details)
        with self._lock:
            self.history.append(event)
        level = logging.ERROR if kind == "failed" else logging.INFO
        self.logger.log(level, "[%s] %s", kind, message)
        if self.callback is not None:
            try:
                self.callback(event)
            except Exception:
                self.logger.exception("Progress callback failed")
        return event
