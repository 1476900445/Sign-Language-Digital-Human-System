# 模块职责与接口

| 工程模块 | 主要输入 | 主要输出与处理职责 |
| --- | --- | --- |
| 预处理 | 视频、音频或麦克风录音 | 媒体真实流检查、有效音轨选择、统一 PCM16 WAV、采样参数校验、麦克风缓冲采集、超时与临时文件管理 |
| VAD | 标准化音频 | 语音或非静音区间、检测状态和后端记录；执行区间裁剪、合并、采样级时间转换及后端兜底 |
| 调用编排与重试 | 输入参数与任务配置 | 分阶段编排、统一三次重试额度、指数等待、任务取消、阶段缓存、结构化事件及单次最终失败通知 |
| 结果后处理 | 全文、分段、词时间及 VAD 区间 | 结果结构校验、全文与分段一致性检查、词间断句、起点校正、文字完整性保护、繁简转换及 SRT/TXT/JSON 输出 |
| 降级 | 当前识别配置及异常信息 | CUDA 失败后选择 CPU，CPU 内存不足后选择较小模型；记录具体配置变化 |
| 适配层 | 标准化音频与识别选项 | 统一识别接口、适配器注册、独立 Whisper 调用、模型复用和资源清理 |

## 时间校正策略

未拆分的字幕使用最早重叠人声起点，保留段落前部的语音对应时间。只有词级文字完整覆盖原句时才按停顿拆分；拆分或校正出现文字不一致时回退原句。

FFmpeg 非静音检测通过统一进程执行器检查返回码，失败时作为检测器不可用处理。检测结果在处理记录中标为 `non_silence`，基于语音模型的结果标为 `speech`。处理流程保留完整音频，检测区间用于时间校正。

## 第三方组件边界

| 第三方组件 | 提供的能力 | 本工程组织与处理代码 |
| --- | --- | --- |
| Whisper、PyTorch | 识别模型与计算能力 | 模型适配、配置选择、统一调用、重试、降级和识别结果结构处理 |
| Silero VAD | 语音活动检测 | 共享音频接入、时间换算、结果规范化与后端兜底 |
| WebRTC VAD | PCM 帧语音判断 | 30 毫秒分帧、缓冲区间和区间合并 |
| FFmpeg、FFprobe | 媒体探测、解码及非静音检测 | 进程封装、返回码检查、超时、流选择、标准化校验和日志解析 |
| OpenCC | 中文繁简转换 | 全文、分段和词文本同步转换及结果文件输出 |
| sounddevice | 麦克风设备输入 | 录音队列、采样数量控制、异常检查和录音复用 |

Whisper 的代码和模型权重以 MIT 许可证开源发布，官方资源见：

- [OpenAI 官方介绍](https://openai.com/index/whisper/)
- [Whisper 官方代码仓库](https://github.com/openai/whisper)
- [Whisper 官方源码下载](https://github.com/openai/whisper/archive/refs/heads/main.zip)
- [Whisper 官方许可证](https://github.com/openai/whisper/blob/main/LICENSE)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [WebRTC VAD Python wrapper](https://github.com/wiseman/py-webrtcvad)
- [FFmpeg](https://ffmpeg.org/documentation.html)
