# Provide public entry points for audio and video transcription.
from dataclasses import replace
from pathlib import Path

from transcription import PipelineConfig, TranscriptionPipeline
from transcription.postprocessing.subtitles import parse_srt, render_srt
from transcription.postprocessing.text import ChineseConverter, join_texts
from transcription.preprocessing.process import MediaTools


class AudioTranscriptionTool:
    def __init__(self, model_size="medium", output_dir="output", config=None):
        self.config = config or PipelineConfig(model_size=model_size, output_dir=output_dir)
        self.pipeline = TranscriptionPipeline(self.config)
        self.last_result = None

    def check_requirements(self):
        try:
            return MediaTools(self.config.ffmpeg_dir).verify()
        except Exception:
            return False

    def process_file(self, file_path, callback=None, cancel_event=None):
        self.last_result = self.pipeline.process(file_path, callback=callback, cancel_event=cancel_event)
        return str(self.last_result.srt)

    def transcribe_audio(self, file_path, callback=None, cancel_event=None):
        config = replace(self.config, simplified_chinese=False)
        self.last_result = TranscriptionPipeline(config).process(file_path, callback=callback,
                                                                 cancel_event=cancel_event)
        return str(self.last_result.srt)

    def convert_to_simplified(self, srt_file, output_file=None):
        source = Path(srt_file)
        destination = Path(output_file) if output_file else source.with_name(source.stem + "_simplified.srt")
        segments = parse_srt(source.read_text(encoding="utf-8-sig"))
        result = {"text": join_texts(item["text"] for item in segments),
                  "language": "zh", "segments": segments}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_srt(ChineseConverter().convert(result)), encoding="utf-8")
        return str(destination)


def process_video(file="./test2.mp4", output_dir="output", callback=None):
    return AudioTranscriptionTool(output_dir=output_dir).process_file(file, callback=callback)
