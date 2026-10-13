# Sign_Language_Digital_Human_Generation_Module：手语数字人生成模块

本目录整合手语数字人生成系统的核心算法及其使用模块，提供媒体格式转换、OpenPose 关键点提取、2D 到 3D 姿态估计、连续帧姿态处理、骨架预览和视频合成，并保留文本驱动手语骨架序列模型的训练与评估代码。

本 README 是模块的总使用说明，应放在 `Sign_Language_Digital_Human_Generation_Module` 根目录，与 `Core-Methods`、`Text-to-Sign-Language` 两个文件夹同级。

两个子模块的职责如下：

| 子模块 | 定位 | 主要职责 | 详细说明 |
| --- | --- | --- | --- |
| `Core-Methods` | 手语数字人生成模块核心算法使用模块 | 提供命令行与 Qt 入口，组织媒体处理、外部程序调用、算法执行、骨架预览和视频合成 | [核心算法使用模块 README](./Core-Methods/README.md) |
| `Text-to-Sign-Language` | 手语数字人生成模块核心算法 | 提供 3D 姿态估计、连续帧特征处理以及文本驱动骨架序列的训练与评估算法 | [核心算法 README](./Text-to-Sign-Language/README.md) |

`Core-Methods` 中的姿态处理入口需要调用 `Text-to-Sign-Language` 的算法代码。完整姿态处理流程需要同时具备这两个子模块，并配置相应的算法路径与运行环境。

## 1. 目录结构与代码组成

以下路径均相对于 `Sign_Language_Digital_Human_Generation_Module` 根目录。

| 路径 | 内容或用途 |
| --- | --- |
| `README.md` | 本总说明：模块关系、安装配置、运行顺序及结果说明 |
| `Core-Methods/README.md` | 核心算法使用模块的详细说明 |
| `Core-Methods/mp4_to_avi.py` | 调用 FFmpeg 转换单个 MP4，或批量转换目录顶层的 MP4 |
| `Core-Methods/openpose_people_detection.py` | 调用 OpenPose 输出身体、双手及可选面部关键点 JSON |
| `Core-Methods/pose_2d_to_3d.py` | 读取 OpenPose JSON，调用姿态估计算法并保存单帧 3D JSON |
| `Core-Methods/run_2d_to_3d_pipeline.py` | 组织 JSON 打包、3D 滤波、差分、归一化和均值计算 |
| `Core-Methods/sign_video_studio_qt.py` | Qt 入口，组织转录、姿态处理、骨架渲染、画中画和音轨合成 |
| `Text-to-Sign-Language/README.md` | 核心算法的详细说明 |
| `Text-to-Sign-Language/pose_3d_estimator/` | 2D 坐标归一化、低质量帧筛选、缺失点插值、3D 初始化和姿态优化 |
| `Text-to-Sign-Language/pose_3d_estimator/data/` | 单帧及序列姿态估计所用的示例数据与阶段输出 |
| `Text-to-Sign-Language/pose_pipeline/` | 连续帧姿态预处理、骨架差分、归一化及模型演示脚本 |
| `Text-to-Sign-Language/pose_pipeline/data/demo/` | OpenPose JSON、HDF5、文本、说话人信息及训练/测试列表 |
| `Text-to-Sign-Language/pose_pipeline/visualization/` | MATLAB 差分反解和骨架播放脚本 |
| `Text-to-Sign-Language/sign_synthesis/` | 文本驱动骨架序列模型、DTW、损失函数、梯度累积优化器及训练/评估入口 |
| `Text-to-Sign-Language/sign_synthesis/data/` | 文本、说话人/日期信息、训练/停止列表和姿态 HDF5 |

### 算法文件夹命名

三个算法文件夹统一使用以下名称：

| 原名称 | 统一名称 | 功能 |
| --- | --- | --- |
| `wacv2020` | `pose_pipeline` | 连续帧姿态预处理与流水线算法 |
| `3DposeEstimator` | `pose_3d_estimator` | 2D 到 3D 姿态估计算法 |
| `SLSynthesis` | `sign_synthesis` | 文本驱动的手语骨架序列合成算法 |

本总说明的目录路径统一使用普通连字符 `-`。若已有目录或源码使用 `Text‑to‑Sign-Language` 中的特殊连接符 `‑`，应以实际目录名为准，并同步调整命令、README 链接和源码配置；这两种字符在路径中不能混用。

更改文件夹名称时，需要同时检查导入语句、模块搜索路径和配置路径，尤其是 `pose_2d_to_3d.py`、`run_2d_to_3d_pipeline.py` 和 `sign_video_studio_qt.py` 的算法目录配置。修改 README 中的名称不会自动修改源码。

## 2. 子模块调用关系

`Core-Methods` 是使用入口，`Text-to-Sign-Language` 提供被调用的算法。具体对应关系如下。

| Core 调用入口 | 使用的算法目录 | 功能 |
| --- | --- | --- |
| `pose_2d_to_3d.py` | `pose_3d_estimator` | 导入 `pose2Dto3D.py`、`skeletalModel.py` 等代码，完成单帧 3D 姿态估计 |
| `run_2d_to_3d_pipeline.py` | `pose_pipeline` | 使用滤波、优化及差分算法，生成 HDF5、均值姿态和逐帧 3D 数据 |
| `sign_video_studio_qt.py` | `pose_3d_estimator`、`pose_pipeline` | 组织单帧预览和连续帧处理，再进行骨架渲染与结果合成 |
| 当前 Core 流程 | `sign_synthesis` | 尚未直接调用；该目录提供独立训练与评估入口 |

### 视频姿态处理流程

1. 使用 FFmpeg 将视频转换为处理所需的格式。
2. 使用 OpenPose 从视频提取身体与双手 2D 关键点。
3. 由 Core 入口调用算法模块，进行单帧 3D 估计或连续帧姿态处理。
4. 使用处理结果生成骨架图像及骨架视频，并完成画中画和音轨合成。

Qt 入口还可组织视频、音频或麦克风的语音转文字任务。音频和麦克风模式仍需要另选一段 MP4，作为姿态提取与画中画的主视频。

### 文本骨架序列模型

`sign_synthesis` 使用文本、说话人等条件以及目标姿态序列进行训练，并通过 DTW 对齐预测骨架与目标动作。该模块输出骨架序列或模型相关结果，保留独立的训练与评估方法。

当前 Core 流程的姿态数据来自输入视频的 OpenPose 结果。运行文本模型的训练与评估脚本，不会自动将文本生成能力接入 Qt 流程；语音转文字结果也不会自动转换为新的手语动作序列。

## 3. 外部依赖与运行环境

本目录不包含 OpenPose 程序、FFmpeg 可执行程序、已经配置完成的 TensorFlow/Keras 环境、Whisper 转录模块或 `audio_video_transcription.py` 入口脚本。外部程序、依赖和模型权重需要按实际功能另行安装或提供。

| 依赖 | 用途 | 使用位置 |
| --- | --- | --- |
| FFmpeg | 视频格式转换与音轨合成 | Core 媒体处理与 Qt 流程 |
| OpenPose | 身体及双手 2D 关键点提取 | Core 外部程序调用 |
| NumPy、h5py | 数值计算与 HDF5 数据处理 | Core 与算法模块 |
| OpenCV、PySide6 | 图像处理与 Qt 界面 | Core 预览与合成入口 |
| Whisper | 音视频及麦克风语音转文字 | 转录入口 |
| TensorFlow 1.x、旧版独立 Keras | 旧版姿态优化、模型训练与评估 | 算法模块 |

### 3.1 Core 环境

Core 代码使用 `str | Path` 等语法，采用 Python 3.10 或以上环境。以下命令从模块根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy h5py opencv-python PySide6
```

### 3.2 FFmpeg

官方下载页：[FFmpeg Download](https://ffmpeg.org/download.html)。Windows 构建可从该页面的 `Windows EXE Files` 入口获取。

下载并解压后，将包含 `ffmpeg.exe` 的 `bin` 目录加入系统 `PATH`，重新打开 PowerShell，然后验证：

```powershell
ffmpeg -version
```

### 3.3 OpenPose

- 官方仓库：[CMU OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- 官方发行版：[OpenPose Releases](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases)
- 官方安装说明：[OpenPose Installation](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md)

使用 Windows 便携版本时，下载并解压程序包，按包内 `Instructions.txt` 准备模型，并确认根目录下存在 `bin\OpenPoseDemo.exe`。源码安装则按官方安装说明配置依赖并编译。

配置 Core 时，在 Qt 界面选择 OpenPose 根目录，或在命令行传入 `--openpose-dir`。以下示例中的 `OPENPOSE_ROOT` 应替换为实际安装目录：

```powershell
OPENPOSE_ROOT\bin\OpenPoseDemo.exe --video input.avi --hand
```

### 3.4 Whisper 与转录入口

- 官方仓库与安装说明：[openai/whisper](https://github.com/openai/whisper)
- PyTorch 安装配置：[PyTorch Get Started](https://pytorch.org/get-started/locally/)

在 Core 环境中安装并检查：

```powershell
python -m pip install -U openai-whisper
python -c "import whisper; print('Whisper READY')"
```

Whisper 还依赖 FFmpeg。使用模型名称时，首次运行通常需要下载权重；离线运行前应准备相应模型缓存。

安装 `openai-whisper` 不会生成本项目需要的 `audio_video_transcription.py`。Qt 转录流程需要单独提供该脚本，并确认支持 `--mode`、`--model`、`--output-dir`、`--duration` 和 `--keep-recording` 参数。脚本路径应在 Qt 配置中指向实际文件。

### 3.5 旧版姿态算法环境

姿态优化、训练与评估代码使用 `tf.Session`、`tf.placeholder`、独立 Keras 和 `keras.legacy` 等旧接口。可按原有算法环境配置独立的 Python 3.6 环境：

```powershell
conda create -n sign-pose-legacy python=3.6
conda activate sign-pose-legacy
python -m pip install "pip<22"
python -m pip install numpy==1.19.5 h5py==2.10.0
python -m pip install tensorflow==1.15.0 keras==2.2.4
python -c "import tensorflow as tf; import keras; print(tf.__version__, keras.__version__)"
```

相关地址：

- [TensorFlow 1.15.0](https://pypi.org/project/tensorflow/1.15.0/)
- [Keras 2.2.4](https://pypi.org/project/Keras/2.2.4/)
- [TensorFlow 官方安装说明](https://www.tensorflow.org/install/pip)

上述命令用于准备旧算法环境，安装成功后仍需通过算法示例检查代码与依赖的实际兼容性。迁移到 TensorFlow 2.x 或新版 Keras 时，需要改写会话、占位符、优化器和自定义层等接口。

### 3.6 Core 与算法环境的集成条件

Core 的现代 Python 环境与旧版算法环境分开安装后，不会自动打通调用关系。若 Core 脚本在当前解释器中直接导入算法模块，必须先完成算法代码和依赖版本的兼容适配。

也可以修改调用入口，让 Core 通过旧环境的 Python 解释器运行算法，并用 JSON、HDF5 或文本文件交换结果。这种方式需要相应的调用适配代码，不能只创建两个环境便视为完成集成。

运行时，应确认当前 Python 解释器对应所需环境，并确认源码中的算法目录与本地目录一致。

## 4. 使用顺序

1. 将两个子模块放在本根目录下，确认文件夹名称和调用路径。
2. 安装 FFmpeg、OpenPose；使用转录功能时，再准备 Whisper 和转录入口脚本。
3. 配置 Core 与算法环境，先运行算法示例检查姿态估计功能。
4. 按实际调用方式解决环境兼容问题，再运行 Core 的单项处理入口。
5. 单项功能正常后，配置并运行 Qt 流程。
6. 需要文本骨架模型时，另行准备训练数据与权重，并使用独立训练和评估入口。

下面第 5 节的命令均以模块根目录为当前工作目录，并要求对应入口的环境与调用条件已经满足。示例输入文件需自行准备。

## 5. Core 命令行与 Qt 入口

### 5.1 MP4 转 AVI

```powershell
python .\Core-Methods\mp4_to_avi.py .\input.mp4 .\output.avi
python .\Core-Methods\mp4_to_avi.py .\input_dir .\output_dir --batch
```

可使用 `--codec` 设置编码器，`--crf` 设置 0 至 51 的质量值，`--no-overwrite` 禁止覆盖已有输出。

### 5.2 OpenPose 关键点提取

将 `OPENPOSE_ROOT` 替换为实际 OpenPose 安装目录：

```powershell
python .\Core-Methods\openpose_people_detection.py .\output.avi `
  --openpose-dir OPENPOSE_ROOT `
  --output-dir .\openpose_2d_jsons `
  --mode single
```

`--mode multi` 允许返回多人，`--no-hand` 关闭手部关键点，`--face` 开启面部关键点。也可通过 `OPENPOSE_DIR` 环境变量配置程序目录。

### 5.3 单帧或批量 2D 到 3D

```powershell
python .\Core-Methods\pose_2d_to_3d.py .\frame_keypoints.json .\frame_3d.json --verbose
python .\Core-Methods\pose_2d_to_3d.py .\openpose_2d_jsons .\output_3d --batch
```

该入口调用 `pose_3d_estimator`，默认读取 `people[0]`，按 25 个身体关节和左右手各 21 个关节组织数据，输出共 67 个 3D 关键点。

### 5.4 连续帧姿态流水线

```powershell
python .\Core-Methods\run_2d_to_3d_pipeline.py .\openpose_2d_jsons `
  --pipeline-dir .\Text-to-Sign-Language\pose_pipeline `
  --output-dir .\pipeline_output
```

连续帧入口选取上半身 8 个关节及左右手各 21 个关节，共 50 个关节。它与单帧入口的 67 关节输出采用不同的数据组织方式，后续读取和渲染应使用相应格式。

### 5.5 Qt 骨架预览与视频合成

```powershell
python .\Core-Methods\sign_video_studio_qt.py
```

在界面中配置转录入口、OpenPose 根目录、`pose_pipeline` 算法目录、主视频和输出目录，并确认 `pose_3d_estimator` 位于算法项目目录内。视频模式可使用输入 MP4 作为主画面；音频和麦克风模式需要另选主视频。

每次任务会创建 `job_时间戳` 子目录，保存转录结果、转换视频、OpenPose JSON、单帧 3D JSON、连续帧处理结果、骨架图像、`skeleton.mp4` 和最终 `combined.mp4`。

## 6. 算法示例与训练评估

本节使用算法环境。命令中的 `Push-Location` 从模块根目录进入相应算法文件夹，`Pop-Location` 返回根目录。

### 6.1 2D 到 3D 姿态估计示例

```powershell
Push-Location .\Text-to-Sign-Language\pose_3d_estimator
python demo.py
Pop-Location
```

示例读取 `data/demo-sequence.h5` 中的 `20161025_pocasi`，依次生成或覆盖以下文件：

| 文件 | 内容 |
| --- | --- |
| `data/demo1.txt` | 归一化后的 2D 坐标与置信度 |
| `data/demo2.txt` | 剔除低质量帧后的数据 |
| `data/demo3.txt` | 缺失点插值后的数据 |
| `data/demo4.txt` | 初始化得到的 3D 姿态 |
| `data/demo5.txt` | 反向传播滤波后的 3D 姿态 |

`pose3D.py` 默认执行较多优化循环，CPU 环境下运行时间可能较长。

### 6.2 连续帧流水线示例

可通过第 5.4 节的 Core 统一入口处理自定义 OpenPose JSON，也可在 `pose_pipeline` 中，按 `pipeline_demo_01` 至 `pipeline_demo_05` 的顺序单独运行各阶段脚本。

处理随项目提供的示例 JSON 时，可从模块根目录执行：

```powershell
python .\Core-Methods\run_2d_to_3d_pipeline.py `
  .\Text-to-Sign-Language\pose_pipeline\data\demo\keypoints\json `
  --pipeline-dir .\Text-to-Sign-Language\pose_pipeline `
  --output-dir .\pipeline_output
```

该命令使用 Core 入口，因此仍需满足第 3.6 节的环境集成条件。

### 6.3 文本驱动骨架序列模型

```powershell
Push-Location .\Text-to-Sign-Language\sign_synthesis
python demo_training.py
python demo_evaluation.py
Pop-Location
```

`sign_synthesis` 使用自身的 `data` 和 `temp`。训练前应准备文本、说话人信息、训练/停止列表及目标姿态 HDF5；评估前应确认相应模型权重已经生成。

### 6.4 姿态流水线模型演示

```powershell
Push-Location .\Text-to-Sign-Language\pose_pipeline
python pipeline_demo_06-train-model_FF_05.py
python pipeline_demo_07-evaluation-model_FF_05.py
Pop-Location
```

该模型使用 `data/demo` 和 `temp/demo`。运行前应检查列表键是否同时存在于文本、说话人信息和目标 HDF5 中，并确认 `temp`、`models` 等输出目录可写。

## 7. 连续帧算法流程与输出

连续帧预处理依次读取 OpenPose JSON，选取 50 个关节，归一化 2D 坐标，剔除缺失过多的帧并插值修复缺失点，再初始化骨长、根节点和肢体方向，通过优化得到 3D 姿态。随后计算骨架差分、进行全局标准差归一化，并计算均值姿态。

| 阶段 | 输出文件或目录 | 数据内容 |
| --- | --- | --- |
| 1：原始数据打包 | `keypoints-01-raw.h5` | 50 关节的 2D 坐标与置信度，150 值/帧 |
| 2：姿态滤波与优化 | `keypoints-02-filter.h5` | 50 关节的估计 3D 坐标，150 值/帧 |
| 3：骨架差分 | `keypoints-03-deltas.h5` | 49 条平面骨架边的差分，98 值/帧 |
| 4：差分归一化 | `keypoints-04-deltas-norm.h5` | 按全局标准差归一化后的差分特征 |
| 5：均值计算 | `mu.txt` | 全部归一化差分帧的均值 |
| 附加：逐帧导出 | `frame_data/` | 阶段 2 的 3D 坐标逐帧文本 |

差分 HDF5 保存平面 `x/y` 特征，不是可直接显示的完整 3D 坐标。读取、反解或渲染时，应保留所需的阶段 2 数据和归一化参数。

## 8. 数据约定与结果检查

- HDF5 顶层键表示一段动作序列，值为“帧数 × 特征宽度”的矩阵。
- 文本文件每行由“序列键 + 分词结果”组成；列表文件每行保存一个序列键。
- 缺失关键点以 `x=0`、`y=0`、`confidence=0` 表示。
- OpenPose 可检测多人，但后续姿态处理默认使用 `people[0]`；多人视频应在前置阶段确定目标人物。
- 单帧 3D 深度来自骨长与方向估计，不等同于深度相机或多视角测量。
- OpenPose 输出的面部关键点不会被当前骨架渲染流程使用。
- 独立训练与评估入口依赖其工作目录；运行时应进入对应算法目录，并备份需要保留的固定路径输出。
- 外部进程运行后，应同时检查退出码与结果文件，确认 JSON、HDF5 或视频产物有效。

## 9. 子模块详细说明

本总说明用于统一配置和运行整个模块。进一步查看调用参数、算法示例和数据说明时，可阅读：

- [Core-Methods：手语数字人生成模块核心算法使用模块](./Core-Methods/README.md)
- [Text-to-Sign-Language：手语数字人生成模块核心算法](./Text-to-Sign-Language/README.md)
