"""Convert a single MP4 file or a directory of MP4 files to AVI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> None:
    """Check that FFmpeg is available on the system PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not installed or is not on the system PATH.")


def convert_mp4_to_avi(
    input_path: str | Path,
    output_path: str | Path | None = None,
    codec: str = "libx264",
    crf: int = 23,
    verbose: bool = True,
    overwrite: bool = True,
) -> bool:
    """Convert one MP4 file to AVI and return whether conversion succeeded."""
    source = Path(input_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if source.suffix.lower() != ".mp4":
        raise ValueError(f"Input file must have an MP4 extension: {source}")
    if not 0 <= crf <= 51:
        raise ValueError("CRF must be between 0 and 51.")

    check_ffmpeg()
    destination = (
        Path(output_path).expanduser()
        if output_path is not None
        else source.with_suffix(".avi")
    )
    if destination.suffix.lower() != ".avi":
        raise ValueError(f"Output file must have an AVI extension: {destination}")
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must be different.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-y" if overwrite else "-n",
        "-i", str(source),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "libmp3lame",
        "-b:a", "128k",
        str(destination),
    ]

    if verbose:
        print(f"Converting: {source}")
        print(f"Output: {destination}")

    process = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        print(f"Conversion failed: {source}", file=sys.stderr)
        if process.stderr:
            print(process.stderr.strip(), file=sys.stderr)
        return False

    if verbose:
        print(f"Conversion completed: {destination}")
    return True


def batch_convert_mp4_to_avi(
    input_dir: str | Path,
    output_dir: str | Path,
    codec: str = "libx264",
    crf: int = 23,
    verbose: bool = True,
    overwrite: bool = True,
) -> int:
    """Convert all top-level MP4 files and return the success count."""
    source_dir = Path(input_dir).expanduser()
    if not source_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {source_dir}")
    destination_dir = Path(output_dir).expanduser()
    if source_dir.resolve() == destination_dir.resolve():
        raise ValueError("Input and output directories must be different.")
    files = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not files:
        raise ValueError(f"No MP4 files were found in: {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for index, source in enumerate(files, start=1):
        if verbose:
            print(f"[{index}/{len(files)}]")
        destination = destination_dir / f"{source.stem}.avi"
        if convert_mp4_to_avi(
            source, destination, codec=codec, crf=crf,
            verbose=verbose, overwrite=overwrite,
        ):
            success_count += 1

    print(f"Completed: {success_count}/{len(files)} files succeeded.")
    return success_count


def main() -> int:
    """Run single-file or batch conversion from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input MP4 file or input directory.")
    parser.add_argument(
        "output", nargs="?",
        help="Output AVI file or output directory. Optional for a single file.",
    )
    parser.add_argument("--batch", action="store_true", help="Convert all MP4 files in a directory.")
    parser.add_argument("--codec", default="libx264", help="FFmpeg video encoder name.")
    parser.add_argument("--crf", type=int, default=23, help="H.264 quality value, 0 to 51.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not replace existing AVI files.")
    args = parser.parse_args()

    try:
        if args.batch:
            if not args.output:
                parser.error("An output directory is required with --batch.")
            source = Path(args.input).expanduser()
            total = len([
                path for path in source.iterdir()
                if path.is_file() and path.suffix.lower() == ".mp4"
            ]) if source.is_dir() else 0
            converted = batch_convert_mp4_to_avi(
                args.input, args.output, codec=args.codec, crf=args.crf,
                verbose=not args.quiet, overwrite=not args.no_overwrite,
            )
            return 0 if converted == total else 1

        succeeded = convert_mp4_to_avi(
            args.input, args.output, codec=args.codec, crf=args.crf,
            verbose=not args.quiet, overwrite=not args.no_overwrite,
        )
        return 0 if succeeded else 1
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
