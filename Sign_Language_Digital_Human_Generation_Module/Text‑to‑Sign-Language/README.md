# Text‑to‑Sign-Language：手语数字人生成模块核心算法

本模块提供手语数字人生成系统的核心算法代码，供第一份说明（MD1）[Core-Methods：：手语数字人生成模块核心算法使用模块](../Core-Methods/README.md)所对应的 `Core-Methods` 程序调用。本文为第二份说明（MD2），对应 `Text‑to‑Sign-Language/README.md`，介绍算法目录、运行环境、数据格式、处理流程及示例方法。

核心算法包括 2D 姿态到 3D 骨架估计、连续帧姿态预处理，以及文本到手语骨架序列的训练与评估代码。当前 `Core-Methods` 调用前两类姿态处理算法；文本驱动的 `sign_synthesis` 模块保留独立训练与评估入口。算法输出骨架数据、特征数据或模型权重，由调用模块继续进行骨架预览和视频合成。

## 供第一份说明（MD1）调用的方式

本模块作为同级 `Core-Methods` 的算法依赖，由 MD1 中的调用入口使用。`Core-Methods` 负责媒体转换、OpenPose 调用、Qt 界面、流程组织和结果合成；本模块提供 3D 姿态估计与连续帧特征处理的具体算法。调用对应关系如下。

| MD2 中的算法目录 | MD1 中的调用入口 | 提供的算法功能 |
| --- | --- | --- |
| `pose_3d_estimator` | `Core-Methods/pose_2d_to_3d.py` | 提供单帧 2D 到 3D 初始化和骨架结构 |
| `pose_pipeline` | `Core-Methods/run_2d_to_3d_pipeline.py` | 提供连续帧滤波、3D 优化和骨架差分 |
| `pose_3d_estimator`、`pose_pipeline` | `Core-Methods/sign_video_studio_qt.py` | 用于 Qt 完整流程中的首帧预览与连续帧处理 |
| `sign_synthesis` | 当前未被 Core 直接调用 | 独立进行文本到骨架序列模型训练与评估 |

使用本模块时，可先运行算法示例检查环境，再按 MD1 通过 `pose_2d_to_3d.py`、`run_2d_to_3d_pipeline.py` 或 Qt 界面调用。独立运行 `sign_synthesis` 的训练与评估脚本，不会自动将文本生成能力接入当前 Core 流程。

两个项目应保持以下同级结构，两份说明分别放在对应目录的 `README.md` 中：

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

## 算法文件夹命名

本文统一采用按功能命名的文件夹名称：

| 原文件夹名称 | 本文采用的名称 | 功能定位 |
| --- | --- | --- |
| `wacv2020` | `pose_pipeline` | 连续帧姿态预处理与流水线算法 |
| `3DposeEstimator` | `pose_3d_estimator` | 2D 到 3D 姿态估计算法 |
| `SLSynthesis` | `sign_synthesis` | 文本驱动的手语骨架序列合成算法 |

如果本地源码仍使用原名称，应同步修改文件夹名、导入语句、模块搜索路径及脚本配置。仅修改 README 中的名称不会自动修改源码；完成更名后，还需检查 MD1 中各调用入口的算法路径。

## 工程文件夹

| 文件夹 | 主要内容 |
| --- | --- |
| `pose_3d_estimator` | 2D 坐标归一化、低质量帧筛选、缺失点插值、3D 初始化和基于反向传播的姿态优化 |
| `pose_3d_estimator/data` | `demo-sequence.h5` 与各阶段示例输出 |
| `pose_pipeline` | JSON 打包、连续帧 3D 滤波、骨架差分、归一化、均值姿态、训练和评估脚本 |
| `pose_pipeline/data/demo` | OpenPose JSON、HDF5 中间结果、文本、说话人信息和训练/测试列表 |
| `pose_pipeline/visualization` | MATLAB 差分反解与骨架播放脚本 |
| `sign_synthesis` | 文本驱动骨架序列模型、DTW、损失函数、梯度累积优化器、训练和评估入口 |
| `sign_synthesis/data` | 文本、说话人/日期信息、训练/停止列表和姿态 HDF5 |

## 数据处理流程

1. 从 OpenPose JSON 的 `people[0]` 读取关键点。
2. 选取上半身 8 个关节及左右手各 21 个关节，共 50 个关节；原始帧宽度为 150。
3. 归一化 2D 坐标，剔除缺失过多的帧，并插值修复缺失点。
4. 初始化骨长、根节点和肢体方向，通过优化生成 50 个关节的 3D 坐标。
5. 将相邻关节转换为 49 条平面差分向量，每帧得到 98 个数值。
6. 使用全局标准差归一化差分数据并计算均值姿态。
7. 使用文本和说话人等条件训练序列模型，通过 DTW 对齐预测骨架与目标动作。

## 环境说明

基础数据处理依赖：

```powershell
python -m pip install numpy h5py
```

3D 优化、训练和评估代码使用 TensorFlow 1.x 图模式与独立 Keras 旧接口。建议使用独立的 Python 3.6 旧版环境：

```powershell
conda create -n sign-pose-legacy python=3.6
conda activate sign-pose-legacy
python -m pip install "pip<22"
python -m pip install numpy==1.19.5 h5py==2.10.0
python -m pip install tensorflow==1.15.0 keras==2.2.4
```

本目录不包含 OpenPose 程序、FFmpeg、Whisper 转录模块或已经配置完成的 TensorFlow/Keras 环境。相关下载地址与安装方法见 MD1 的[外部组件安装与获取](../Core-Methods/README.md#外部组件安装与获取)。本目录未提供统一的 `requirements.txt`；如迁移到 TensorFlow 2.x 或新版 Keras，需要改写会话、占位符、优化器和自定义层接口。MD1 的 Core 环境与本模块的旧版算法环境还需按实际调用方式完成兼容适配。

## 运行 2D 到 3D 示例

```powershell
cd pose_3d_estimator
python demo.py
```

脚本读取 `data/demo-sequence.h5` 中的 `20161025_pocasi`，并依次覆盖生成：

| 文件 | 内容 |
| --- | --- |
| `data/demo1.txt` | 归一化后的 2D 坐标与置信度 |
| `data/demo2.txt` | 剔除低质量帧后的数据 |
| `data/demo3.txt` | 缺失点插值后的数据 |
| `data/demo4.txt` | 初始化得到的 3D 姿态 |
| `data/demo5.txt` | 反向传播滤波后的 3D 姿态 |

`pose3D.py` 默认执行较多优化循环，CPU 环境下运行时间可能较长。

## 运行连续帧姿态流水线

`pose_pipeline/pipeline_demo_01_*.py` 至 `pipeline_demo_05_*.py` 已改为基于脚本所在目录解析 `data/demo` 和 `temp/demo`，不再依赖开发机盘符。处理自定义 OpenPose 数据时，推荐从本项目根目录通过 `Core-Methods` 的统一入口运行，并显式传入 `pose_pipeline`：

```powershell
python ..\Core-Methods\run_2d_to_3d_pipeline.py `
  .\pose_pipeline\data\demo\keypoints\json `
  --pipeline-dir .\pose_pipeline `
  --output-dir .\pipeline_output
```

统一入口生成：

| 结果 | 形状或用途 |
| --- | --- |
| `keypoints-01-raw.h5` | 50 关节的 2D 坐标与置信度，150 值/帧 |
| `keypoints-02-filter.h5` | 50 关节的估计 3D 坐标，150 值/帧 |
| `keypoints-03-deltas.h5` | 49 条平面骨架边，98 值/帧 |
| `keypoints-04-deltas-norm.h5` | 全局标准差归一化后的差分特征 |
| `mu.txt` | 全部归一化差分帧的均值姿态 |
| `frame_data` | 滤波后逐帧导出的 3D 坐标文本 |

如需使用随项目提供的演示数据，也可以进入 `pose_pipeline` 后按 `pipeline_demo_01` 至 `pipeline_demo_05` 的顺序单独运行各阶段脚本。

## 训练与评估

`sign_synthesis` 使用自身的 `data` 和 `temp`；`pose_pipeline` 使用 `data/demo` 和 `temp/demo`。所有相对路径都按当前工作目录解析。

文本驱动骨架合成模型：

```powershell
cd sign_synthesis
python demo_training.py
python demo_evaluation.py
```

姿态流水线演示模型：

```powershell
cd pose_pipeline
python pipeline_demo_06-train-model_FF_05.py
python pipeline_demo_07-evaluation-model_FF_05.py
```

训练前应检查列表键是否同时存在于文本、说话人信息和目标 HDF5 中，并确认 `temp`、`models` 目录可写。评估入口要求相应模型权重已经生成。

## 数据约定

- HDF5 顶层键表示一段动作序列，值为“帧数 × 特征宽度”的矩阵。
- 文本文件每行由“序列键 + 分词结果”组成；列表文件每行保存一个序列键。
- 缺失关键点以 `x=0`、`y=0`、`confidence=0` 表示。
- 预处理默认选择 OpenPose JSON 中的第一个人物；多人视频应在前置阶段确定目标人物。
- 差分特征只保留骨架边的平面 `x/y` 变化，不等同于可直接显示的完整 3D 坐标。
- 训练和评估脚本会读取或覆盖固定相对路径下的数据、权重和示例输出，运行前应备份需要保留的结果。
