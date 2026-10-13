# Retry failed processing three times and emit only one terminal failure event.
from dataclasses import dataclass
import time

from ..core.errors import CancellationError, ProcessingFailed, release_exception_frames


@dataclass(frozen=True)
class RetryPolicy:
    delay: float = 1.0
    backoff: float = 2.0
    maximum_delay: float = 4.0

    @property
    def max_retries(self):
        return 3

    @property
    def max_attempts(self):
        return self.max_retries + 1

    def delay_for(self, retry_number):
        return min(self.maximum_delay, self.delay * self.backoff ** (retry_number - 1))


class RetryExecutor:
    def __init__(self, policy, events, waiter=None):
        self.policy = policy
        self.events = events
        self.waiter = waiter
        self.records = []

    def execute(self, operation, on_failure=None):
        try:
            return self._execute(operation, on_failure)
        except CancellationError:
            attempt = self.records[-1]["attempt"] if self.records else 0
            self.events.emit("cancelled", "pipeline", attempt, "处理已取消")
            raise

    def _execute(self, operation, on_failure=None):
        failures = []
        for attempt in range(1, self.policy.max_attempts + 1):
            self.events.check_cancelled()
            started = time.perf_counter()
            self.events.emit("attempt_started", "pipeline", attempt,
                             f"开始第 {attempt}/{self.policy.max_attempts} 次处理")
            try:
                result = operation(attempt)
            except CancellationError:
                raise
            except Exception as error:
                failures.append(error)
                self.records.append({"attempt": attempt, "status": "failed",
                                     "stage": getattr(error, "stage", "pipeline"),
                                     "error": str(error), "elapsed": time.perf_counter() - started})
                if attempt == self.policy.max_attempts:
                    final_error = ProcessingFailed(failures)
                    self.events.emit("failed", "pipeline", attempt, str(final_error),
                                     retries=self.policy.max_retries,
                                     failures=list(self.records))
                    release_exception_frames(error)
                    raise final_error from error
                if on_failure:
                    on_failure(error, attempt)
                release_exception_frames(error)
                delay = self.policy.delay_for(attempt)
                self.events.emit("retry_scheduled", "pipeline", attempt,
                                 f"第 {attempt} 次处理失败，{delay:g} 秒后进行第 {attempt}/3 次重试",
                                 next_attempt=attempt + 1, delay=delay,
                                 failed_stage=getattr(error, "stage", "pipeline"))
                self._wait(delay)
                continue
            self.records.append({"attempt": attempt, "status": "completed",
                                 "elapsed": time.perf_counter() - started})
            self.events.emit("completed", "pipeline", attempt, "处理完成",
                             retries_used=attempt - 1)
            return result
        raise RuntimeError("重试控制器进入无效状态")

    def _wait(self, delay):
        if self.waiter is not None:
            self.waiter(delay)
        elif self.events.cancel_event.wait(delay):
            raise CancellationError("重试等待期间任务已取消")
        self.events.check_cancelled()
