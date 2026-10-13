# Expose the standalone transcription pipeline and its configuration.
from .core.config import PipelineConfig
from .orchestration.pipeline import TranscriptionPipeline

__all__ = ["PipelineConfig", "TranscriptionPipeline"]
