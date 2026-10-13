# Publish SRT, TXT, and JSON as a rollback-capable output bundle.
from pathlib import Path
import json
import os
import shutil
import tempfile
import threading

from ..preprocessing.validation import ensure_output_directory, safe_basename
from .subtitles import render_srt


class OutputWriter:
    _lock = threading.RLock()

    def write(self, original, final, asset, output_dir, metadata, simplified):
        directory = ensure_output_directory(output_dir)
        basename = safe_basename(asset.source_name)
        final_name = basename + ("_simplified" if simplified else "")
        files = {final_name + ".srt": render_srt(final),
                 final_name + ".txt": final["text"].strip() + "\n"}
        original_name = basename + ".srt" if simplified else None
        if original_name:
            files[original_name] = render_srt(original)
        payload = {"source_name": asset.source_name, "input_kind": asset.input_kind,
                   "duration": asset.duration, "sample_rate": asset.sample_rate,
                   "text": final["text"], "language": final["language"],
                   "segments": final["segments"], "processing": metadata}
        files[final_name + ".json"] = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with self._lock:
            self._commit(directory, files)
        return {"srt": directory / (final_name + ".srt"),
                "txt": directory / (final_name + ".txt"),
                "json": directory / (final_name + ".json"),
                "original_srt": directory / original_name if original_name else None}

    def _commit(self, directory, files):
        committed = []
        with tempfile.TemporaryDirectory(prefix=".transcription_", dir=directory) as temporary:
            workspace = Path(temporary)
            staging = workspace / "staging"
            backups = workspace / "backups"
            staging.mkdir()
            backups.mkdir()
            for name, content in files.items():
                staged = staging / name
                with staged.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                existing = directory / name
                if existing.exists():
                    if not existing.is_file():
                        raise IsADirectoryError(f"输出文件名已被目录占用: {name}")
                    shutil.copy2(existing, backups / name)
            try:
                for name in files:
                    os.replace(staging / name, directory / name)
                    committed.append(name)
            except Exception:
                for name in reversed(committed):
                    saved = backups / name
                    if saved.is_file():
                        os.replace(saved, directory / name)
                    else:
                        (directory / name).unlink(missing_ok=True)
                raise
