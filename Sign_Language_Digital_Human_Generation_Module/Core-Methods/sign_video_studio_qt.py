"""Qt workflow for transcription, pose extraction, skeleton rendering, and video compositing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CORE_DIR = BASE_DIR / "Core-Methods" if (BASE_DIR / "Core-Methods").is_dir() else BASE_DIR
TRANSCRIBER_DIR = BASE_DIR / "Speech-Methods" if (BASE_DIR / "Speech-Methods").is_dir() else BASE_DIR
SCRIPT_NAMES = {
    "transcribe": "audio_video_transcription.py",
    "convert": "mp4_to_avi.py",
    "openpose": "openpose_people_detection.py",
    "pose3d": "pose_2d_to_3d.py",
    "pipeline": "run_2d_to_3d_pipeline.py",
}
BODY_25_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24),
]
HAND_PAIRS = {
    "thumb": ((0, 1), (1, 2), (2, 3), (3, 4)),
    "index": ((0, 5), (5, 6), (6, 7), (7, 8)),
    "middle": ((0, 9), (9, 10), (10, 11), (11, 12)),
    "ring": ((0, 13), (13, 14), (14, 15), (15, 16)),
    "pinky": ((0, 17), (17, 18), (18, 19), (19, 20)),
}
LEFT_COLORS = {
    "thumb": (255, 0, 0), "index": (255, 127, 0),
    "middle": (255, 255, 0), "ring": (0, 255, 0), "pinky": (0, 0, 255),
}
RIGHT_COLORS = {
    "thumb": (0, 0, 255), "index": (0, 127, 255),
    "middle": (0, 255, 255), "ring": (0, 255, 0), "pinky": (255, 0, 0),
}
FRAME_NUMBER = re.compile(r"_(\d+)_keypoints\.json$", re.IGNORECASE)


@dataclass(frozen=True)
class JobConfig:
    source_mode: str
    source_path: Path | None
    transcriber_path: Path
    main_video: Path
    output_dir: Path
    openpose_dir: Path
    pipeline_dir: Path
    duration: float
    model: str
    person_mode: str
    scale: float


def dependency_status() -> list[tuple[str, bool, str]]:
    """Inspect prerequisites without loading heavy models or claiming task completion."""
    items = [
        ("FFmpeg media engine", shutil.which("ffmpeg") is not None, "ffmpeg -version"),
        ("Local conversion environment", (CORE_DIR / SCRIPT_NAMES["convert"]).is_file(), SCRIPT_NAMES["convert"]),
        ("Whisper transcription entry", (TRANSCRIBER_DIR / SCRIPT_NAMES["transcribe"]).is_file(), SCRIPT_NAMES["transcribe"]),
        ("OpenPose control entry", (CORE_DIR / SCRIPT_NAMES["openpose"]).is_file(), SCRIPT_NAMES["openpose"]),
        ("2D to 3D reconstruction entry", (CORE_DIR / SCRIPT_NAMES["pose3d"]).is_file(), SCRIPT_NAMES["pose3d"]),
        ("Five-stage pipeline entry", (CORE_DIR / SCRIPT_NAMES["pipeline"]).is_file(), SCRIPT_NAMES["pipeline"]),
        ("OpenCV rendering environment", importlib.util.find_spec("cv2") is not None, "opencv-python"),
        ("Numeric rendering environment", importlib.util.find_spec("numpy") is not None, "numpy"),
        ("HDF5 pipeline environment", importlib.util.find_spec("h5py") is not None, "h5py"),
    ]
    try:
        process = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, timeout=5)
        if process.returncode != 0:
            items[0] = (items[0][0], False, "FFmpeg executable failed its version check")
    except (OSError, subprocess.TimeoutExpired):
        items[0] = (items[0][0], False, "FFmpeg executable unavailable")
    return items


def frame_sort(path: Path):
    """Sort OpenPose JSON files by their frame index and then filename."""
    match = FRAME_NUMBER.search(path.name)
    return (int(match.group(1)) if match else 10**12, path.name)


def points_from_person(person: dict, name: str, np):
    """Decode a flattened OpenPose keypoint list when its shape is valid."""
    raw = person.get(name) or []
    if len(raw) % 3:
        raise ValueError(f"Malformed OpenPose keypoints in {name}: {len(raw)} values")
    return np.asarray(raw, dtype="float32").reshape(-1, 3)


def draw_skeleton_frame(person: dict, width: int, height: int, cv2, np):
    """Render the supplied BODY_25 and color-coded hand skeleton on white."""
    image = np.full((height, width, 3), 255, dtype="uint8")
    body = points_from_person(person, "pose_keypoints_2d", np)
    left = points_from_person(person, "hand_left_keypoints_2d", np)
    right = points_from_person(person, "hand_right_keypoints_2d", np)

    def draw_points(points, color, radius):
        for x, y, confidence in points:
            if confidence > 0.1 and np.isfinite(x) and np.isfinite(y):
                cv2.circle(image, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)

    def draw_lines(points, pairs, color, thickness):
        for a, b in pairs:
            if a < len(points) and b < len(points):
                x1, y1, c1 = points[a]
                x2, y2, c2 = points[b]
                if c1 > 0.2 and c2 > 0.2 and np.isfinite((x1, y1, x2, y2)).all():
                    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                             color, thickness, cv2.LINE_AA)

    draw_lines(body, BODY_25_PAIRS, (0, 255, 255), 2)
    draw_points(body, (0, 255, 255), 3)
    for points, colors, dot in ((left, LEFT_COLORS, (128, 0, 128)),
                                (right, RIGHT_COLORS, (0, 128, 128))):
        for finger, pairs in HAND_PAIRS.items():
            draw_lines(points, pairs, colors[finger], 3)
        draw_points(points, dot, 4)
    return image


def render_skeleton_images(json_dir: Path, image_dir: Path, main_video: Path, report):
    """Convert ordered OpenPose JSON frames into skeleton images."""
    import cv2
    import numpy as np

    files = sorted(json_dir.glob("*_keypoints.json"), key=frame_sort)
    if not files:
        raise ValueError("OpenPose produced no matching 2D JSON frames.")
    capture = cv2.VideoCapture(str(main_video))
    if not capture.isOpened():
        raise ValueError("Unable to read the main video for skeleton dimensions.")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise ValueError("Main video dimensions are invalid.")
    image_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    for index, path in enumerate(files):
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        people = data.get("people") or []
        person = people[0] if people else {}
        image = draw_skeleton_frame(person, width, height, cv2, np)
        destination = image_dir / f"frame_{index:06d}.jpg"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Unable to write image: {destination}")
        completed += 1
        if completed == 1 or completed % 50 == 0 or completed == len(files):
            report(f"Skeleton rendering: {completed}/{len(files)} frames", None)
    return files, width, height


def images_to_video(image_dir: Path, output: Path, fps: float, width: int,
                    height: int, report):
    """Encode ordered skeleton images as an MP4 video."""
    import cv2

    files = sorted(image_dir.glob("frame_*.jpg"))
    if not files:
        raise ValueError("No skeleton images were rendered.")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MP4 video writer.")
    try:
        for index, path in enumerate(files, 1):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Unable to read skeleton image: {path}")
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
            if index == 1 or index % 100 == 0 or index == len(files):
                report(f"Skeleton video encoding: {index}/{len(files)} frames", None)
    finally:
        writer.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Skeleton MP4 was not generated.")
    return output


def embed_video(main_video: Path, skeleton_video: Path, output: Path,
                scale: float, report):
    """Place the skeleton video at the bottom right of the source video."""
    import cv2

    primary = cv2.VideoCapture(str(main_video))
    inset = cv2.VideoCapture(str(skeleton_video))
    if not primary.isOpened() or not inset.isOpened():
        primary.release()
        inset.release()
        raise ValueError("Unable to open the main video or skeleton overlay.")

    fps = primary.get(cv2.CAP_PROP_FPS)
    width = int(primary.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(primary.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(primary.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        primary.release()
        inset.release()
        raise ValueError("Invalid video frame rate or dimensions.")
    inset_width = max(1, int(width * scale))
    inset_height = max(1, int(height * scale))
    x = width - inset_width - 10
    y = height - inset_height - 10
    if x < 0 or y < 0:
        primary.release()
        inset.release()
        raise ValueError("Inset size exceeds available video dimensions.")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        primary.release()
        inset.release()
        raise RuntimeError("Unable to initialize the composite video writer.")

    count = 0
    try:
        while True:
            ok, frame = primary.read()
            if not ok:
                break
            inset_ok, overlay = inset.read()
            if not inset_ok:
                inset.set(cv2.CAP_PROP_POS_FRAMES, 0)
                inset_ok, overlay = inset.read()
            if not inset_ok or overlay is None:
                raise RuntimeError("Skeleton video has no usable overlay frames.")
            frame[y:y + inset_height, x:x + inset_width] = cv2.resize(
                overlay, (inset_width, inset_height))
            writer.write(frame)
            count += 1
            if count == 1 or count % 100 == 0 or (total and count == total):
                report(f"Video compositing: {count}/{total or '?'} frames", None)
    finally:
        primary.release()
        inset.release()
        writer.release()
    if count == 0 or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Composite video was not generated.")
    return output


def video_fps(path: Path) -> float:
    """Read the main video frame rate used for skeleton encoding."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError("Unable to determine the main video frame rate.")
    return fps


def verify_environment(config: JobConfig, report) -> dict[str, Path]:
    """Check executable and file prerequisites before any processing starts."""
    paths = {
        "transcribe": config.transcriber_path,
        **{name: CORE_DIR / filename for name, filename in SCRIPT_NAMES.items()
           if name != "transcribe"},
    }
    for label, ok, detail in dependency_status():
        report(f"{'READY' if ok else 'MISSING'} | {label} | {detail}", None)
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required scripts are missing: " + "; ".join(missing))
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("FFmpeg is unavailable on the PATH.")
    for name in ("cv2", "numpy", "h5py"):
        if importlib.util.find_spec(name) is None:
            raise ImportError(f"Install {name} in the same Python environment as the Qt app.")
    if not config.main_video.is_file() or config.main_video.suffix.lower() != ".mp4":
        raise ValueError("Select an existing MP4 main video for final compositing.")
    if not config.openpose_dir.is_dir():
        raise FileNotFoundError("Select a valid OpenPose directory.")
    executable = config.openpose_dir / "bin" / (
        "OpenPoseDemo.exe" if os.name == "nt" else "openpose.bin"
    )
    if not executable.is_file():
        raise FileNotFoundError(f"OpenPose executable was not found: {executable}")
    if not config.pipeline_dir.is_dir():
        raise FileNotFoundError("Select a valid five-stage pipeline directory.")
    for filename in ("pipeline_demo_02_filter.py", "pipeline_demo_03_deltas.py",
                     "skeletalModel.py", "pose2D.py", "pose2Dto3D.py", "pose3D.py"):
        if not (config.pipeline_dir / filename).is_file():
            raise FileNotFoundError(f"Five-stage pipeline dependency is missing: {filename}")
    estimator = paths["pose3d"].parent.parent / "Text‑to‑Sign-Language" / "pose_3d_estimator"
    if not estimator.is_dir():
        raise FileNotFoundError(f"Single-frame 3D estimator directory is missing: {estimator}")
    if config.source_mode != "microphone" and (
        config.source_path is None or not config.source_path.is_file()
    ):
        raise FileNotFoundError("Select an existing input media file.")
    expected = ".mp4" if config.source_mode == "video" else ".mp3"
    if config.source_mode != "microphone" and config.source_path.suffix.lower() != expected:
        raise ValueError(f"The selected {config.source_mode} input must be a {expected} file.")
    if config.source_mode == "microphone" and importlib.util.find_spec("sounddevice") is None:
        raise ImportError("Microphone mode requires sounddevice.")
    report("READY | All required local components verified", 5)
    return paths


def run_command(command: list[str], report, label: str) -> None:
    """Stream logs and reject non-zero exit codes from external stages."""
    report(f"START | {label}", None)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        if process.stdout is not None:
            for line in process.stdout:
                if line.strip():
                    report(f"{label} | {line.strip()}", None)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"{label} failed with exit code {code}.")
    report(f"DONE | {label}", None)


def run_transcription(config: JobConfig, paths: dict, work: Path, report) -> Path:
    """Run the earlier Whisper application for all three media input modes."""
    speech_dir = work / "speech"
    command = [sys.executable, "-u", str(paths["transcribe"]),
               "--mode", config.source_mode, "--model", config.model,
               "--output-dir", str(speech_dir)]
    if config.source_mode == "microphone":
        command.extend(["--duration", str(config.duration), "--keep-recording"])
    else:
        command.append(str(config.source_path))
    report("Speech engine: extracting language and preparing audio", 10)
    run_command(command, report, "Whisper transcription")
    if not list(speech_dir.glob("*.srt")) or not list(speech_dir.glob("*.txt")):
        raise RuntimeError("Transcription completed without SRT and TXT outputs.")
    report("Language detected | transcription saved | SRT and TXT verified", 20)
    if config.source_mode == "microphone":
        recordings = list(speech_dir.glob("*.wav"))
        if not recordings:
            raise RuntimeError("Microphone audio was not retained.")
        return recordings[0]
    return config.source_path


def mux_audio(config: JobConfig, silent_video: Path, source_audio: Path,
              final_output: Path, report) -> Path:
    """Use FFmpeg to preserve video sound or attach imported microphone sound."""
    audio = config.main_video if config.source_mode == "video" else source_audio
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
               "-i", str(silent_video), "-i", str(audio),
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
               "-c:a", "aac", "-b:a", "192k", str(final_output)]
    report("FFmpeg: attaching source audio to final composite", 96)
    run_command(command, report, "Final audio mux")
    if not final_output.is_file() or final_output.stat().st_size == 0:
        raise RuntimeError("Final combined MP4 was not created.")
    return final_output


def run_workflow(config: JobConfig, report) -> Path:
    """Run all earlier components and the three supplied visualization stages."""
    paths = verify_environment(config, report)
    work = config.output_dir.resolve() / f"job_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    work.mkdir(parents=True, exist_ok=True)

    source_audio = run_transcription(config, paths, work, report)
    converted = work / "main_converted.avi"
    report("FFmpeg loaded | starting local MP4-to-AVI conversion", 25)
    run_command([sys.executable, "-u", str(paths["convert"]),
                 str(config.main_video), str(converted)], report, "MP4-to-AVI")
    if not converted.is_file() or converted.stat().st_size == 0:
        raise RuntimeError("AVI conversion reported success but produced no file.")
    report("AVI conversion verified", 35)

    json_dir = work / "openpose_2d_jsons"
    run_command([sys.executable, "-u", str(paths["openpose"]), str(converted),
                 "--openpose-dir", str(config.openpose_dir),
                 "--output-dir", str(json_dir), "--mode", config.person_mode],
                report, "OpenPose 2D extraction")
    json_files = sorted(json_dir.glob("*_keypoints.json"), key=frame_sort)
    if not json_files:
        raise RuntimeError("OpenPose did not save any 2D JSON frames.")
    report(f"2D JSON parsing ready | {len(json_files)} frame records", 48)

    single_3d = work / "single_frame_3d.json"
    run_command([sys.executable, "-u", str(paths["pose3d"]),
                 str(json_files[0]), str(single_3d)], report, "Single-frame 2D-to-3D")
    if not single_3d.is_file():
        raise RuntimeError("The initial 3D reconstruction produced no JSON file.")
    report("2D-to-3D preview verified | body and hand result saved", 56)

    pipeline_output = work / "five_stage"
    run_command([sys.executable, "-u", str(paths["pipeline"]), str(json_dir),
                 "--pipeline-dir", str(config.pipeline_dir),
                 "--output-dir", str(pipeline_output)], report, "Five-stage 2D-to-3D H5 pipeline")
    for name in ("keypoints-01-raw.h5", "keypoints-02-filter.h5",
                 "keypoints-03-deltas.h5", "keypoints-04-deltas-norm.h5", "mu.txt"):
        if not (pipeline_output / name).is_file():
            raise RuntimeError(f"The five-stage pipeline did not save {name}.")
    report("3D H5 saved | deltas normalized | mean pose verified", 68)

    image_dir = work / "skeleton_images"
    _, width, height = render_skeleton_images(json_dir, image_dir, config.main_video,
                                              report)
    report("2D skeleton images generated and verified", 76)
    skeleton_mp4 = images_to_video(image_dir, work / "skeleton.mp4",
                                   video_fps(config.main_video), width, height, report)
    report("Skeleton animation encoded", 84)
    silent = embed_video(config.main_video, skeleton_mp4, work / "combined_silent.mp4",
                         config.scale, report)
    report("Bottom-right picture-in-picture composition completed", 93)
    final = mux_audio(config, silent, source_audio, work / "combined.mp4", report)
    report(f"SUCCESS | Final combined video saved: {final}", 100)
    return final


def launch_qt() -> int:
    """Start a responsive Qt interface with threaded processing and real status events."""
    try:
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtWidgets import (
            QApplication, QComboBox, QFileDialog, QFormLayout, QGroupBox,
            QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
            QProgressBar, QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
        )
    except ImportError:
        print("Install PySide6 to launch the Qt interface.", file=sys.stderr)
        return 1

    class Worker(QThread):
        status = Signal(str, object)
        finished_path = Signal(str)
        failed = Signal(str)

        def __init__(self, config):
            super().__init__()
            self.config = config

        def run(self):
            try:
                result = run_workflow(self.config, self.status.emit)
                self.finished_path.emit(str(result))
            except Exception as error:
                self.status.emit(f"FAILED | {error}", None)
                self.status.emit(traceback.format_exc(), None)
                self.failed.emit(str(error))

    class Studio(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Sign Language Video Studio")
            self.resize(900, 720)
            self.worker = None
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            heading = QLabel("SIGN LANGUAGE VIDEO STUDIO  |  LOCAL WORKFLOW")
            heading.setStyleSheet("font-size: 20px; font-weight: 700; color: #DDEEFF;")
            layout.addWidget(heading)

            group = QGroupBox("Input and processing settings")
            form = QFormLayout(group)
            layout.addWidget(group)
            self.mode = QComboBox()
            self.mode.addItems(["Video file", "Audio file", "Microphone"])
            self.mode.currentIndexChanged.connect(self.refresh_mode)
            form.addRow("Input source", self.mode)

            def path_row(folder=False):
                box = QWidget()
                line = QLineEdit()
                row = QHBoxLayout(box)
                row.setContentsMargins(0, 0, 0, 0)
                button = QPushButton("Browse")
                row.addWidget(line)
                row.addWidget(button)
                if folder:
                    button.clicked.connect(lambda: line.setText(
                        QFileDialog.getExistingDirectory(self, "Select directory") or line.text()))
                else:
                    button.clicked.connect(lambda: line.setText(
                        QFileDialog.getOpenFileName(self, "Select media file")[0] or line.text()))
                return box, line

            box, self.source = path_row()
            self.source_widget = box
            form.addRow("Audio/video source", box)
            box, self.transcriber_script = path_row()
            self.transcriber_script.setText(str(TRANSCRIBER_DIR / SCRIPT_NAMES["transcribe"]))
            form.addRow("Whisper entry script", box)
            box, self.main_video = path_row()
            form.addRow("Main MP4 for composition", box)
            box, self.openpose_dir = path_row(folder=True)
            form.addRow("OpenPose directory", box)
            box, self.pipeline_dir = path_row(folder=True)
            self.pipeline_dir.setText(str(BASE_DIR.parent / "Text‑to‑Sign-Language" / "pose_pipeline"))
            form.addRow("Five-stage pipeline directory", box)
            box, self.output_dir = path_row(folder=True)
            self.output_dir.setText(str(BASE_DIR / "studio_output"))
            form.addRow("Output directory", box)
            self.model = QComboBox()
            self.model.addItems(["tiny", "base", "small", "medium", "large"])
            self.model.setCurrentText("medium")
            form.addRow("Whisper model", self.model)
            self.people = QComboBox()
            self.people.addItems(["single", "multi"])
            form.addRow("OpenPose people mode", self.people)
            self.duration = QSpinBox()
            self.duration.setRange(1, 3600)
            self.duration.setValue(30)
            form.addRow("Microphone duration (seconds)", self.duration)
            self.scale = QSpinBox()
            self.scale.setRange(10, 80)
            self.scale.setValue(30)
            form.addRow("Inset width and height (%)", self.scale)

            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            layout.addWidget(self.progress)
            self.log = QTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log)
            self.start = QPushButton("Start full workflow")
            self.start.clicked.connect(self.start_job)
            layout.addWidget(self.start)
            self.setStyleSheet(
                "QMainWindow, QWidget { background: #101827; color: #D9E7F8; }"
                "QGroupBox { border: 1px solid #36516B; margin-top: 12px; padding: 10px; }"
                "QLineEdit, QTextEdit, QComboBox, QSpinBox { background: #1A293A;"
                " border: 1px solid #49647B; padding: 5px; color: #F2F7FC; }"
                "QPushButton { background: #145C91; padding: 9px; border-radius: 5px; }"
                "QPushButton:hover { background: #2075AD; }"
            )
            self.refresh_mode()
            self.show_startup_status()

        def refresh_mode(self):
            is_mic = self.mode.currentIndex() == 2
            self.source_widget.setEnabled(not is_mic)
            self.duration.setEnabled(is_mic)
            if self.mode.currentIndex() == 0 and not self.main_video.text():
                self.main_video.setPlaceholderText("Select the same MP4 as your video input")

        def show_startup_status(self):
            for label, ready, detail in dependency_status():
                self.on_status(f"{'READY' if ready else 'MISSING'} | {label} | {detail}", None)
            self.on_status("Standby | No media processing has started", 0)

        def on_status(self, message, progress):
            from PySide6.QtCore import QDateTime
            stamp = QDateTime.currentDateTime().toString("HH:mm:ss")
            self.log.append(f"[{stamp}] {message}")
            if progress is not None:
                self.progress.setValue(int(progress))

        def start_job(self):
            modes = ("video", "audio", "microphone")
            mode = modes[self.mode.currentIndex()]
            source = Path(self.source.text()).expanduser().resolve() if mode != "microphone" else None
            main = self.main_video.text().strip() or (self.source.text().strip() if mode == "video" else "")
            if not main:
                QMessageBox.warning(self, "Main video required", "Select an MP4 main video to create the final composite.")
                return
            if self.worker and self.worker.isRunning():
                return
            config = JobConfig(
                mode, source, Path(self.transcriber_script.text()).expanduser().resolve(),
                Path(main).expanduser().resolve(),
                Path(self.output_dir.text()).expanduser().resolve(),
                Path(self.openpose_dir.text()).expanduser().resolve(),
                Path(self.pipeline_dir.text()).expanduser().resolve(),
                float(self.duration.value()), self.model.currentText(),
                self.people.currentText(), self.scale.value() / 100,
            )
            self.log.clear()
            self.progress.setValue(0)
            self.start.setEnabled(False)
            self.worker = Worker(config)
            self.worker.status.connect(self.on_status)
            self.worker.finished_path.connect(self.on_complete)
            self.worker.failed.connect(self.on_failure)
            self.worker.start()

        def on_complete(self, path):
            self.start.setEnabled(True)
            QMessageBox.information(self, "Processing complete", f"Combined video saved:\n{path}")

        def on_failure(self, reason):
            self.start.setEnabled(True)
            QMessageBox.critical(self, "Processing failed", reason)

        def closeEvent(self, event):
            if self.worker and self.worker.isRunning():
                QMessageBox.warning(self, "Processing active", "Wait for the current job to finish before closing.")
                event.ignore()
            else:
                event.accept()

    app = QApplication(sys.argv)
    window = Studio()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch_qt())
