# Connect transcription to Qt while showing a single dialog only after terminal failure.
import threading

from transcription import TranscriptionPipeline
from transcription.core.errors import CancellationError, ProcessingFailed


def create_worker(config, input_path, parent=None):
    from PySide6.QtCore import QThread, Signal

    class TranscriptionWorker(QThread):
        progress = Signal(object)
        succeeded = Signal(object)
        failed = Signal(str)
        cancelled = Signal()

        def __init__(self):
            super().__init__(parent)
            self.cancel_event = threading.Event()

        def cancel(self):
            self.cancel_event.set()

        def run(self):
            try:
                pipeline = TranscriptionPipeline(config)
                result = pipeline.process(input_path, callback=self.progress.emit,
                                          cancel_event=self.cancel_event)
            except CancellationError:
                self.cancelled.emit()
            except ProcessingFailed as error:
                self.failed.emit(str(error))
            except Exception as error:
                self.failed.emit(f"配置或启动失败: {error}")
            else:
                self.succeeded.emit(result)

    return TranscriptionWorker()


def start_transcription(window, config, input_path, status_label, on_success):
    from PySide6.QtWidgets import QMessageBox
    worker = create_worker(config, input_path, window)

    def update_status(event):
        status_label.setText(event.message)

    def show_final_error(message):
        QMessageBox.critical(window, "处理失败", message)

    worker.progress.connect(update_status)
    worker.failed.connect(show_final_error)
    worker.succeeded.connect(on_success)
    worker.cancelled.connect(lambda: status_label.setText("任务已取消"))
    window.transcription_worker = worker
    worker.start()
    return worker
