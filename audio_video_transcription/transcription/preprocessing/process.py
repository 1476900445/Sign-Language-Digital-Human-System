# Resolve FFmpeg tools and execute bounded external media operations.
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess

from ..core.errors import ExternalProcessError


@dataclass(frozen=True)
class ProcessResult:
    stdout: str
    stderr: str
    returncode: int


class MediaTools:
    def __init__(self, directory=None):
        self.directory = Path(directory).expanduser() if directory else None

    def resolve(self, name):
        filename = name + (".exe" if os.name == "nt" else "")
        if self.directory is not None:
            candidate = self.directory / filename
            if not candidate.is_file():
                raise FileNotFoundError(f"指定目录中缺少 {filename}")
            return str(candidate.resolve())
        if os.name == "nt":
            candidate = Path(r"D:\FFmpeg\autosrt-1.4.9\autosrt-1.4.9\win") / filename
            if candidate.is_file():
                return str(candidate)
        resolved = shutil.which(name)
        if not resolved:
            raise FileNotFoundError(f"未找到 {filename}，请配置 ffmpeg_dir 或系统 PATH")
        return resolved

    def verify(self, timeout=15):
        runner = ProcessRunner(timeout)
        for name in ("ffmpeg", "ffprobe"):
            runner.run([self.resolve(name), "-version"])
        return True


class ProcessRunner:
    def __init__(self, timeout=600.0):
        self.timeout = timeout

    def run(self, command, timeout=None):
        args = [str(item) for item in command]
        try:
            result = subprocess.run(
                args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=timeout or self.timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as error:
            raise ExternalProcessError(Path(args[0]).name, "timeout", "媒体处理超时") from error
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        if result.returncode != 0:
            detail = "\n".join(stderr.strip().splitlines()[-12:]) or "没有错误详情"
            raise ExternalProcessError(Path(args[0]).name, result.returncode, detail)
        return ProcessResult(stdout, stderr, result.returncode)
