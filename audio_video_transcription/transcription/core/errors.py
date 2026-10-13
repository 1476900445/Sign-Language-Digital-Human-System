# Preserve processing stages and report one final error after all retries fail.
class CancellationError(Exception):
    pass


class StageFailure(RuntimeError):
    def __init__(self, stage, cause):
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")


class ProcessingFailed(RuntimeError):
    def __init__(self, failures):
        self.failures = tuple(failures)
        self.attempts = len(failures)
        self.last_error = failures[-1] if failures else None
        super().__init__(f"处理失败：首次尝试及 3 次重试均未成功。最后错误：{self.last_error}")


class ExternalProcessError(RuntimeError):
    def __init__(self, tool, returncode, detail):
        self.tool = tool
        self.returncode = returncode
        self.detail = detail
        super().__init__(f"{tool} 执行失败 ({returncode}): {detail}")


def release_exception_frames(error):
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend((getattr(current, "cause", None), current.__cause__, current.__context__))
        current.__traceback__ = None
