# Verify retry counts, delayed recovery, terminal errors, and cancellation behavior.
import threading
import unittest

from transcription.core.errors import CancellationError, ProcessingFailed
from transcription.core.events import EventBus
from transcription.orchestration.retry import RetryExecutor, RetryPolicy


class RetryTests(unittest.TestCase):
    def test_success_on_first_attempt(self):
        calls = []
        bus = EventBus()
        executor = RetryExecutor(RetryPolicy(), bus, waiter=lambda delay: None)
        value = executor.execute(lambda attempt: calls.append(attempt) or "done")
        self.assertEqual(value, "done")
        self.assertEqual(calls, [1])
        self.assertFalse(any(event.kind == "retry_scheduled" for event in bus.history))

    def test_initial_failure_and_three_retries(self):
        calls, delays = [], []
        bus = EventBus()
        executor = RetryExecutor(RetryPolicy(), bus, waiter=delays.append)

        def operation(attempt):
            calls.append(attempt)
            raise RuntimeError("synthetic failure")

        with self.assertRaises(ProcessingFailed) as raised:
            executor.execute(operation)
        self.assertEqual(calls, [1, 2, 3, 4])
        self.assertEqual(delays, [1.0, 2.0, 4.0])
        self.assertEqual(raised.exception.attempts, 4)
        self.assertEqual(len(raised.exception.failures), 4)
        self.assertEqual(sum(event.kind == "failed" for event in bus.history), 1)
        self.assertEqual(sum(event.kind == "retry_scheduled" for event in bus.history), 3)

    def test_success_on_last_retry(self):
        bus = EventBus()
        executor = RetryExecutor(RetryPolicy(), bus, waiter=lambda delay: None)

        def operation(attempt):
            if attempt < 4:
                raise OSError("temporary")
            return attempt

        self.assertEqual(executor.execute(operation), 4)
        self.assertEqual([item["status"] for item in executor.records], ["failed", "failed", "failed", "completed"])
        self.assertFalse(any(event.kind == "failed" for event in bus.history))

    def test_cancel_before_processing(self):
        cancellation = threading.Event()
        cancellation.set()
        bus = EventBus(cancel_event=cancellation)
        executor = RetryExecutor(RetryPolicy(), bus)
        with self.assertRaises(CancellationError):
            executor.execute(lambda attempt: self.fail("operation should not start"))

    def test_cancel_during_retry_wait(self):
        cancellation = threading.Event()
        bus = EventBus(cancel_event=cancellation)
        executor = RetryExecutor(RetryPolicy(), bus, waiter=lambda delay: cancellation.set())
        calls = []

        def operation(attempt):
            calls.append(attempt)
            raise RuntimeError("temporary")

        with self.assertRaises(CancellationError):
            executor.execute(operation)
        self.assertEqual(calls, [1])
        self.assertFalse(any(event.kind == "failed" for event in bus.history))

    def test_callback_failure_does_not_abort_processing(self):
        def callback(event):
            raise RuntimeError("callback failure")
        bus = EventBus(callback)
        executor = RetryExecutor(RetryPolicy(), bus)
        with self.assertLogs("transcription", level="ERROR"):
            self.assertEqual(executor.execute(lambda attempt: "ok"), "ok")


if __name__ == "__main__":
    unittest.main()
