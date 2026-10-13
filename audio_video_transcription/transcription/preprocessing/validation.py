# Validate input files and construct safe output names across operating systems.
from pathlib import Path
import re


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v", ".ts"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff"}
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def validate_input(source):
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {path.name}")
    if path.stat().st_size <= 0:
        raise ValueError("输入文件为空")
    return path


def safe_basename(source_name):
    stem = Path(source_name).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    if not stem:
        stem = "transcription"
    if stem.upper() in RESERVED_NAMES:
        stem = "media_" + stem
    return stem[:160]


def ensure_output_directory(path):
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise NotADirectoryError("输出位置不是文件夹")
    return directory
