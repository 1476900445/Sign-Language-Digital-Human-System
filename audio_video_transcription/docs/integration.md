# 应用接入方法

将 `transcription` 文件夹及 `audio_video_transcription.py` 放到应用主程序旁，或在工程根目录执行 `python -m pip install -e .` 安装到当前 Python 环境。

在应用中导入公共入口：

```python
from audio_video_transcription import AudioTranscriptionTool
```

创建转录工具并处理输入文件：

```python
transcriber = AudioTranscriptionTool(
    model_size="medium",
    output_dir="output",
)
generated_srt = transcriber.process_file("video1.mp4")
```

返回值是完整的字幕文件路径，可交给字幕解析、文本处理、播放器或其他下游模块。最终处理失败会抛出 `ProcessingFailed`，应用可在工作线程最外层统一捕获该异常。

应用的错误弹窗应连接最终失败信号，`retry_scheduled` 和 VAD 兜底事件只用于更新进度与日志。转录模块本身不会调用界面弹窗。

需要更细的状态信息时，可将 `process_file(..., callback=callback)` 的回调转换为工作线程的 Qt 信号，交由主线程更新界面。`examples/qt_integration.py` 提供后台线程、进度信号、成功信号、最终失败信号和取消接口。

同一 `TranscriptionPipeline` 实例一次处理一个任务；并发任务应使用独立实例。取消在步骤边界和重试等待期间生效；正在执行的 Whisper 推理需要完成当前调用才能响应取消。

转录任务的预处理、识别、后处理和输出共用首次尝试加三次重试的额度。后续应用模块自行执行的任务应独立管理重试次数。
