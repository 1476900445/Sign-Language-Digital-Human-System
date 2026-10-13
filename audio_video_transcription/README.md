# 音视频与麦克风语音转文字工程

本工程支持视频文件、音频文件和麦克风录音的语音转文字，采用本地 Whisper 识别引擎，通过预处理、VAD、调用编排与重试、结果后处理及降级模块完成统一处理。工程提供可插拔识别适配层、字幕时间校正、文字完整性保护、繁简转换、结果文件输出和 Qt 接入示例。

## 工程文件夹

| 文件夹 | 对应模块 | 主要文件 |
| --- | --- | --- |
| `transcription/preprocessing` | 预处理 | `validation.py`、`process.py`、`media.py`、`audio.py`、`microphone.py` |
| `transcription/vad` | VAD | `intervals.py`、`detectors.py`、`service.py` |
| `transcription/orchestration` | 调用编排与重试 | `retry.py`、`pipeline.py` |
| `transcription/postprocessing` | 结果后处理 | `text.py`、`normalization.py`、`alignment.py`、`subtitles.py`、`writers.py` |
| `transcription/degradation` | 降级 | `device.py`、`policy.py` |
| `transcription/adapters` | 可插拔识别适配层 | `base.py`、`registry.py`、`whisper_adapter.py` |
| `transcription/core` | 公共配置与任务数据 | `config.py`、`models.py`、`errors.py`、`events.py` |
| `config` | 配置文件 | `default.json`、`windows.json` |
| `examples` | Python 与 Qt 接入示例 | `python_integration.py`、`qt_integration.py` |
| `tests` | 自动化回归测试 | 重试、VAD、字幕、适配层、媒体处理和管道测试 |
| `docs` | 工程说明 | 模块职责、应用接入方法和验证范围 |

所有 Python 源码仅在文件顶部保留一行简短英文用途注释，没有函数说明字符串或行内注释。

## Whisper 开源说明与官方获取地址

Whisper 是 OpenAI 开源的语音识别项目，其代码和模型权重采用 MIT 许可证发布。本工程通过独立适配器调用 Whisper，在本地完成语音识别，并提供预处理、VAD 组织、调用重试、结果后处理和降级等工程功能。

| 官方资源 | 地址 |
| --- | --- |
| OpenAI 官方介绍 | [OpenAI Whisper](https://openai.com/index/whisper/) |
| 官方代码仓库与安装说明 | [openai/whisper](https://github.com/openai/whisper) |
| 官方源码压缩包下载 | [下载源码 ZIP](https://github.com/openai/whisper/archive/refs/heads/main.zip) |
| 官方许可证 | [MIT License](https://github.com/openai/whisper/blob/main/LICENSE) |

可打开官方仓库，选择 **Code → Download ZIP** 下载源码。源码压缩包不包含完整模型权重；使用官方 Whisper 模型加载接口时，会从缓存读取或获取指定模型权重。

官方 Python 包的安装命令为：

```bash
python -m pip install openai-whisper
```

本工程的基础依赖清单已经包含 `openai-whisper`，安装 `requirements.txt` 时会一并安装。已有本地模型权重可通过 `--model` 指定文件路径。

## 环境与安装

工程源码兼容 Python 3.9 及以上；实际可安装版本取决于本机 PyTorch、Whisper、Silero 和 Qt 版本。可优先使用现有语音识别环境，避免替换已经适配 GPU 的 PyTorch。

在工程根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_optional.txt
```

可选依赖中的 `silero-vad`、`webrtcvad-wheels` 用于前两级 VAD；`sounddevice` 用于麦克风；`PySide6` 用于 Qt 集成示例。主流程在前两级 VAD 不可用时自动尝试 FFmpeg 非静音检测。

FFmpeg 与 FFprobe 是外部可执行程序，需要由本机提供。配置 `ffmpeg_dir` 时会严格使用该目录。未配置时，Windows 优先检查预设目录 `D:\FFmpeg\autosrt-1.4.9\autosrt-1.4.9\win`，再查找系统 PATH。

Whisper 适配器直接接收标准化音频的浮点采样数组，不会再次调用系统 PATH 中的 FFmpeg 解码输入，避免自定义 FFmpeg 目录与识别引擎内部解码路径不一致。

识别使用本地 Whisper。压缩包包含工程源码、配置和测试，不包含 Whisper 权重或 FFmpeg 程序。已有本地权重可通过 `--model` 传入 `.pt` 文件；使用模型名称时，由 Whisper 读取缓存，缓存不存在时可能下载权重。

## 运行方法

默认视频模式：

```bash
python run.py "video1.mp4" --output-dir output
```

音频模式：

```bash
python run.py "audio1.mp3" --mode audio --output-dir output
```

麦克风录音 30 秒并转写：

```bash
python run.py --mode microphone --duration 30 --output-dir output
```

使用配置文件和固定 FFmpeg 目录：

```bash
python run.py "video1.mp4" --config config/windows.json --language zh
```

直接使用已有本地模型：

```bash
python run.py "video1.mp4" --model "models/medium.pt" --language zh
```

保存日志并输出机器可读结果：

```bash
python run.py "video1.mp4" --log-file "output/task.log" --json-result
```

关闭繁简转换：

```bash
python run.py "audio1.mp3" --mode audio --no-simplified-chinese
```

输入类型按真实媒体流判断，音频不会为了转录而重新封装成黑色背景视频。输入媒体必须包含有效音轨。

## 重试与最终报错

一次任务最多执行首次尝试加 3 次重试，共 4 次。首次失败后等待 1 秒，第 1 次重试失败后等待 2 秒，第 2 次重试失败后等待 4 秒。等待时间可在 JSON 配置中调整，重试次数固定为 3。

处理中发生的普通异常共用同一个任务重试额度，包含预处理、识别、后处理和文件输出错误。配置或启动参数无效时直接报错；用户取消及 Ctrl+C 不进行重试。

VAD 切换检测器属于同一次处理内部的兜底，不增加任务尝试次数。有效的空检测结果被记录为 `silence`，不会因为“没有人声”而继续用能量检测覆盖该结果。VAD 全部不可用时仍执行全量识别并保留识别时间戳。

成功完成的预处理和 VAD 会在本次任务的后续重试中复用。识别成功但输出失败时会复用识别结果。麦克风识别失败不会要求用户重新说一遍。

重试过程中只产生 `retry_scheduled`、`fallback` 和进度日志。最终 4 次均失败时，产生一次 `failed` 事件并抛出 `ProcessingFailed`，命令行返回退出码 1。Qt 示例只连接工作线程的最终 `failed` 信号显示一次错误弹窗，普通进度事件只更新状态文字。

CUDA 或显存相关识别错误会在下一次重试中切换为 CPU；CPU 内存不足时可继续切换较小的官方模型。降级和普通重试共用 3 次额度。自定义 `.pt` 权重只进行设备降级，不擅自更换模型文件。

## 结果文件

默认繁简转换开启时输出：

| 文件 | 内容 |
| --- | --- |
| `名称.srt` | 校正时间后的原语言字幕 |
| `名称_simplified.srt` | 繁简转换后的字幕，也是公共入口返回的结果 |
| `名称_simplified.txt` | 最终纯文本 |
| `名称_simplified.json` | 最终全文、分段、词时间及处理记录 |

关闭繁简转换时输出 `名称.srt`、`名称.txt`、`名称.json`。JSON 记录实际使用的引擎、设备、模型、VAD 后端、时间校正统计、既往失败和降级历史。标准化音频属于临时文件，任务结束后清理。部分测试结果文件存放在路径\transcription\output下。

同名任务输出会覆盖同名结果文件；文件写入失败时会尝试恢复原有文件，避免半写入结果。

## Python 调用与应用接入

公共入口提供 `AudioTranscriptionTool`、`process_file()`、`transcribe_audio()`、`convert_to_simplified()` 和 `process_video()`。正常返回字幕路径，最终失败抛出异常。应用工作线程可统一捕获最终异常。

```python
from audio_video_transcription import AudioTranscriptionTool

tool = AudioTranscriptionTool(model_size="medium", output_dir="output")
srt_path = tool.process_file("video1.mp4")
```

从工程根目录运行完整 Python 示例：

```bash
python -m examples.python_integration "video1.mp4"
```

Qt 示例位于 `examples/qt_integration.py`，用于接入现有界面。更详细的接入步骤见 `docs/integration.md`。

## 可插拔适配接口

自定义引擎继承 `RecognitionAdapter` 并实现 `recognize(asset, spec, config)`，返回包含 `text`、`language`、`segments` 的字典。每段包含 `start`、`end`、`text`，可选 `words` 包含 `start`、`end`、`word`。通过 `AdapterRegistry.register()` 注册工厂，再向 `TranscriptionPipeline` 传入该注册表和匹配的引擎名称。

全部时间必须基于原音频时间轴，单位为秒。当前主流程保留完整音频，没有将静音片段删除后重新拼接，因此不会引入切片拼接造成的整体时间偏移。

