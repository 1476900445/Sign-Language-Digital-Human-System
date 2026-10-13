"""Run OpenPose hand and body detection for one or multiple people."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_openpose(
    openpose_dir: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
    mode: str = "single",
    detect_hand: bool = True,
    detect_face: bool = False,
) -> int:
    """Run the OpenPose executable and return its process exit code."""
    if mode not in {"single", "multi"}:
        raise ValueError("Mode must be 'single' or 'multi'.")

    root = Path(openpose_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OpenPose directory does not exist: {root}")

    executable_name = "OpenPoseDemo.exe" if os.name == "nt" else "openpose.bin"
    executable = root / "bin" / executable_name
    if not executable.is_file():
        raise FileNotFoundError(f"OpenPose executable does not exist: {executable}")

    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video}")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    command = [
        str(executable),
        "--video", str(video),
        "--write_json", str(output),
    ]
    if detect_hand:
        command.append("--hand")
    if detect_face:
        command.append("--face")
    if mode == "single":
        command.extend(["--number_people_max", "1"])

    print(f"Running OpenPose in {mode}-person mode.", flush=True)
    print(f"Input video: {video}", flush=True)
    print(f"JSON output: {output}", flush=True)

    # Merge both output streams to avoid blocking on a full stderr pipe.
    with subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
        return_code = process.wait()

    if return_code == 0:
        print("OpenPose completed successfully.")
    else:
        print(f"OpenPose failed with exit code {return_code}.", file=sys.stderr)
    return return_code


def main() -> int:
    """Parse input paths and select the requested detection mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the input video.")
    parser.add_argument(
        "--openpose-dir",
        default=os.environ.get("OPENPOSE_DIR"),
        help="OpenPose root directory; alternatively set OPENPOSE_DIR.",
    )
    parser.add_argument(
        "--output-dir",
        default="output_json_folder",
        help="Directory for OpenPose JSON results.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Person detection mode. The default is single.",
    )
    parser.add_argument(
        "--no-hand", action="store_true", help="Disable hand keypoint detection."
    )
    parser.add_argument(
        "--face", action="store_true", help="Enable face keypoint detection."
    )
    args = parser.parse_args()

    if not args.openpose_dir:
        parser.error("Provide --openpose-dir or set OPENPOSE_DIR.")

    try:
        return run_openpose(
            openpose_dir=args.openpose_dir,
            video_path=args.video,
            output_dir=args.output_dir,
            mode=args.mode,
            detect_hand=not args.no_hand,
            detect_face=args.face,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("OpenPose was interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
