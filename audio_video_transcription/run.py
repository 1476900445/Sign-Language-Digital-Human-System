# Run video, audio, or microphone transcription and report one final command-line error.
from dataclasses import replace
import argparse
import json
import logging
import sys

from transcription import PipelineConfig, TranscriptionPipeline
from transcription.core.errors import CancellationError, ProcessingFailed


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone video/audio/microphone transcription")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("--mode", choices=("auto", "video", "audio", "microphone"), default="video")
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--model")
    parser.add_argument("--model-cache")
    parser.add_argument("--ffmpeg-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--language")
    parser.add_argument("--task", choices=("transcribe", "translate"))
    parser.add_argument("--duration", type=float)
    parser.add_argument("--microphone-device", type=int)
    parser.add_argument("--no-simplified-chinese", action="store_true")
    parser.add_argument("--log-file")
    parser.add_argument("--json-result", action="store_true")
    return parser


def configure_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        configure_logging(args.log_file)
        config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
        overrides = {}
        for name in ("output_dir", "model_cache", "ffmpeg_dir", "device", "language", "task", "microphone_device"):
            value = getattr(args, name)
            if value is not None:
                overrides[name] = value
        if args.model:
            overrides["model_size"] = args.model
        if args.duration is not None:
            overrides["recording_duration"] = args.duration
        if args.no_simplified_chinese:
            overrides["simplified_chinese"] = False
        config = replace(config, **overrides).validate()
        result = TranscriptionPipeline(config).process(args.input_path, mode=args.mode)
        if args.json_result:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"处理完成，共尝试 {result.attempts} 次，用时 {result.elapsed:.2f} 秒")
            print(f"SRT: {result.srt}\nTXT: {result.txt}\nJSON: {result.json}")
        return 0
    except ProcessingFailed:
        return 1
    except (KeyboardInterrupt, CancellationError):
        print("任务已取消", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"配置或启动失败: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
