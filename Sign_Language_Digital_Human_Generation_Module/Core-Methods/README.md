# Core-Methods：手语数字人生成模块核心算法使用模块

本模块是手语数字人生成系统中核心算法的使用与调用模块，负责媒体处理、外部程序调用、算法流程组织、骨架预览和视频合成。本文为第一份说明（MD1），对应 `Core-Methods/README.md`；其中的单帧 2D 到 3D 重建和连续帧姿态处理，需要使用第二份说明（MD2）[Text‑to‑Sign-Language：手语数字人生成模块核心算法](../Text‑to‑Sign-Language/README.md)中介绍的算法代码。

本模块提供 MP4 转 AVI、OpenPose 身体与双手关键点提取、单帧 3D 姿态估计、连续帧姿态流水线，以及 Qt 骨架画中画预览。命令行脚本可以分别运行，也可以由 `sign_video_studio_qt.py` 串联为本地处理流程；姿态处理入口仍需配置 MD2 对应的算法目录和运行环境。

本目录只包含流程控制与媒体处理代码，不包含 OpenPose 程序、FFmpeg、TensorFlow/Keras 环境、Whisper 转录模块、模型权重或 `audio_video_transcription.py`。这些外部组件需要按本文说明另行安装或提供。

## 与第二份说明（MD2）的调用关系

`Core-Methods` 负责调用和组织流程，`Text‑to‑Sign-Language` 提供被调用的核心算法。运行本模块的姿态处理功能前，应先按 MD2 准备 `pose_3d_estimator` 和 `pose_pipeline`。两个项目应放在同一个父目录下，完整姿态流程需要同时具备这两个项目的代码。具体调用关系如下。

| MD1 中的调用入口 | MD2 中的算法目录 | 调用用途 |
| --- | --- | --- |
| `pose_2d_to_3d.py` | `pose_3d_estimator` | 导入 `pose2Dto3D.py` 与 `skeletalModel.py`，完成单帧 3D 估计 |
| `run_2d_to_3d_pipeline.py` | `pose_pipeline` | 动态加载滤波与差分模块，生成四阶段 HDF5、均值姿态和逐帧 3D 数据 |
| `sign_video_studio_qt.py` | `pose_3d_estimator`、`pose_pipeline` | 在 Qt 完整流程中依次执行单帧预览和连续帧处理 |
| 当前 Core 流程 | `sign_synthesis` | 当前未直接调用；该目录用于骨架序列模型训练与评估 |

### 配置与使用顺序

1. 按 MD2 准备算法代码，并确认三个算法文件夹的名称和位置。
2. 按本文安装 OpenPose、FFmpeg、Whisper 等外部组件；按 MD2 配置姿态算法环境，并完成相应示例检查。
3. 在本模块的脚本或 Qt 配置中，将算法路径指向 `Text‑to‑Sign-Language/pose_3d_estimator` 和 `Text‑to‑Sign-Language/pose_pipeline`。
4. 确认 Core 与算法环境的调用方式兼容后，再运行单帧重建、连续帧流水线或 Qt 处理流程。

MD2 说明算法本身的环境、数据和运行方法；本文说明如何在 `Core-Methods` 中使用这些算法。

## 工程文件

| 文件 | 主要功能 |
| --- | --- |
| `mp4_to_avi.py` | 调用 FFmpeg 转换单个 MP4，或批量转换目录顶层的 MP4 |
| `openpose_people_detection.py` | 调用 OpenPose 输出身体、双手及可选面部关键点 JSON |
| `pose_2d_to_3d.py` | 读取 OpenPose JSON，并调用 `pose_3d_estimator` 保存单帧 3D JSON |
| `run_2d_to_3d_pipeline.py` | 调用 `pose_pipeline` 完成 JSON 打包、3D 滤波、差分、归一化与均值计算 |
| `sign_video_studio_qt.py` | 串联转录、格式转换、OpenPose、姿态处理、骨架渲染、画中画与音轨合成 |

## 项目目录

两个模块采用以下目录结构，两份说明分别放在对应目录的 `README.md` 中：

```text
项目根目录/
├── Core-Methods/
│   └── README.md                  # MD1：核心算法使用模块说明
└── Text‑to‑Sign-Language/
    ├── README.md                  # MD2：核心算法说明
    ├── pose_3d_estimator/
    ├── pose_pipeline/
    └── sign_synthesis/
```

`Text‑to‑Sign-Language` 使用特殊连接符“‑”，目录名必须与源码中的路径配置一致。复制或重新命名项目时，应同步检查并修改 `pose_2d_to_3d.py`、`run_2d_to_3d_pipeline.py` 和 `sign_video_studio_qt.py` 中的默认路径。三个算法文件夹的原名称与新名称对应关系见 MD2。

## Core Python 环境

Core 源码使用 `str | Path` 等语法，建议使用 Python 3.10 或以上：

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install numpy h5py opencv-python PySide6
```

## 外部组件安装与获取

### FFmpeg

FFmpeg 用于 MP4/AVI 转换以及最终音轨合成。

- 官方下载页：[FFmpeg Download](https://ffmpeg.org/download.html)
- Windows 可在官方下载页的 `Windows EXE Files` 中选择预编译版本。

安装方法：

1. 下载并解压 Windows 构建。
2. 将包含 `ffmpeg.exe` 的 `bin` 目录加入系统 `PATH`。
3. 重新打开 PowerShell 后验证：

```powershell
ffmpeg -version
```

### OpenPose

OpenPose 用于从视频中提取 Body25 和左右手关键点。

- 官方仓库：[CMU OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- 官方发行版：[OpenPose Releases](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases)
- 官方安装说明：[OpenPose Installation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md)

Windows 推荐使用官方便携版：

1. 从 Releases 下载 Windows 压缩包并解压。
2. 按压缩包内 `Instructions.txt` 下载模型文件。
3. 确认根目录下存在 `bin\OpenPoseDemo.exe`。
4. 在 Qt 界面选择该根目录，或在命令行传入 `--openpose-dir`。

验证示例：

```powershell
OPENPOSE_ROOT\bin\OpenPoseDemo.exe --video input.avi --hand
```

### Whisper 与转录入口

Whisper 用于音视频或麦克风语音转文字。

- 官方仓库与安装说明：[openai/whisper](https://github.com/openai/whisper)
- GPU 环境如需手动选择 PyTorch 版本，可参考：[PyTorch Get Started](https://pytorch.org/get-started/locally/)

在 Core 的 Python 3.10+ 环境中安装：

```powershell
python -m pip install -U openai-whisper
python -c "import whisper; print('Whisper READY')"
```

Whisper 还依赖 FFmpeg。使用模型名称时，首次运行通常会下载模型权重；离线机器应提前准备权重缓存。

`openai-whisper` 只提供识别库和命令行，不会生成本项目要求的 `audio_video_transcription.py`。Qt 完整流程还需要单独提供该入口脚本，并确保它支持 `--mode`、`--model`、`--output-dir`、`--duration` 和 `--keep-recording` 参数。

### TensorFlow 1.x 与旧版 Keras

`Text‑to‑Sign-Language/pose_3d_estimator` 和 `pose_pipeline` 的原始优化代码使用 `tf.Session`、`tf.placeholder`、独立 Keras 及 `keras.legacy` 接口，因此不能直接安装最新版 TensorFlow/Keras。

- 历史 TensorFlow 包：[TensorFlow 1.15.0](https://pypi.org/project/tensorflow/1.15.0/)
- 历史 Keras 包：[Keras 2.2.4](https://pypi.org/project/Keras/2.2.4/)
- 当前 TensorFlow 官方安装页：[Install TensorFlow](https://www.tensorflow.org/install/pip)

复现旧模块时建议使用独立的 Python 3.6 环境：

```powershell
conda create -n sign-pose-legacy python=3.6
conda activate sign-pose-legacy
python -m pip install "pip<22"
python -m pip install numpy==1.19.5 h5py==2.10.0
python -m pip install tensorflow==1.15.0 keras==2.2.4
python -c "import tensorflow as tf; import keras; print(tf.__version__, keras.__version__)"
```

Core 使用 Python 3.10+，旧姿态环境使用较早的 Python 与深度学习包。若 Core 入口直接导入旧姿态算法，两个独立环境不会自动实现兼容；需要先完成算法代码与依赖版本的适配。另一种集成方式是修改 Core 调用入口，通过旧环境的 Python 解释器运行算法，并交换输入输出文件。该方式需要相应的调用适配代码，不能仅通过创建两个环境实现。

## 单独运行

### MP4 转 AVI

```powershell
python mp4_to_avi.py input.mp4 output.avi
python mp4_to_avi.py input_dir output_dir --batch
```

可用 `--codec` 设置编码器，`--crf` 设置 0 至 51 的质量值，`--no-overwrite` 禁止覆盖已有输出。

### OpenPose 关键点提取

```powershell
python openpose_people_detection.py input.avi `
  --openpose-dir OPENPOSE_ROOT `
  --output-dir openpose_2d_jsons `
  --mode single
```

`--mode multi` 允许 OpenPose 返回多人；`--no-hand` 关闭手部关键点；`--face` 开启面部关键点。也可以设置 `OPENPOSE_DIR` 环境变量。

### 单帧或批量 2D 到 3D

```powershell
python pose_2d_to_3d.py frame_keypoints.json frame_3d.json --verbose
python pose_2d_to_3d.py openpose_2d_jsons output_3d --batch
```

脚本读取 `people[0]`，按 25 个身体关节和左右手各 21 个关节组织数据，输出共 67 个 3D 关键点。

### 连续帧姿态流水线

```powershell
python run_2d_to_3d_pipeline.py openpose_2d_jsons `
  --pipeline-dir ..\Text‑to‑Sign-Language\pose_pipeline `
  --output-dir pipeline_output
```

该流程选取上半身 8 个关节及左右手各 21 个关节，共 50 个关节。

## 姿态流水线结果

| 阶段 | 输出 | 内容 |
| --- | --- | --- |
| 1 | `keypoints-01-raw.h5` | 50 关节的 2D 坐标与置信度，150 值/帧 |
| 2 | `keypoints-02-filter.h5` | 插值和姿态优化后的 3D 坐标，150 值/帧 |
| 3 | `keypoints-03-deltas.h5` | 49 条平面骨架边的差分，98 值/帧 |
| 4 | `keypoints-04-deltas-norm.h5` | 按全局标准差归一化的差分特征 |
| 5 | `mu.txt` | 所有归一化差分帧的均值 |
| 附加 | `frame_data` | 阶段 2 的 3D 坐标逐帧文本 |

## Qt 骨架预览

```powershell
python sign_video_studio_qt.py
```

界面支持视频、音频和麦克风三种转录来源。视频模式可直接把输入 MP4 作为主画面；音频和麦克风模式仍需另选一段 MP4 作为姿态提取和画中画的主视频。

运行前应确认：

- Whisper 入口指向实际的 `audio_video_transcription.py`。
- OpenPose 目录包含有效程序和模型。
- 姿态流水线目录指向 `Text‑to‑Sign-Language/pose_pipeline`。
- `pose_3d_estimator` 位于第二项目根目录。
- 主视频可解码，输出目录可写。

每次任务会创建 `job_时间戳` 子目录，保存转录结果、转换视频、OpenPose JSON、单帧 3D JSON、姿态流水线结果、骨架图像、`skeleton.mp4` 和最终 `combined.mp4`。

## 数据与算法说明

- OpenPose 可检测多人，但后续 3D 处理默认使用 `people[0]`。
- 单帧 3D 深度来自骨长与方向估计，不等同于深度相机或多视角测量。
- 差分 HDF5 不是完整关节坐标，反解或渲染时应保留阶段 2 数据及归一化参数。
- 面部关键点即使由 OpenPose 输出，也不会被当前骨架渲染流程使用。
- 外部进程的退出码和结果文件应同时核验，不能只根据日志文字判断成功。
