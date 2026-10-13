# Demonstrate calling the audio and video transcription module from Python.
from pathlib import Path
import argparse
import logging

from audio_video_transcription import AudioTranscriptionTool
from transcription.core.errors import ProcessingFailed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    def progress(event):
        if event.kind == "retry_scheduled":
            print(event.message)

    tool = AudioTranscriptionTool(output_dir=args.output_dir)
    try:
        path = tool.process_file(args.input_path, callback=progress)
    except ProcessingFailed:
        return 1
    print(Path(path).name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
