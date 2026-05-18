# AI 视频超分批处理工具需求规格说明书

## 1. 背景

当前使用 `registry.cn-qingdao.aliyuncs.com/wod/video2x:6.4.0` 处理真人视频时，`realesrgan-plus` 只有 `x4` ncnn/Vulkan 模型。720p 视频会被 AI 超分到 2880p，再得到名为 `_1080p.mp4` 的输出，导致单帧处理速度约 `0.53 fps`，单个长视频预计耗时几十到一百多小时，不能满足批量生产需求。

在 RTX 4090 上，旧方案还暴露出明确的性能释放问题：GPU 使用率约 `60%`，显存占用约 `1.2GB`，单个长视频预计耗时约 `170` 小时。这说明瓶颈不只是模型效果，而是整条处理链路没有充分利用 4090。

本项目目标不是普通 ffmpeg 拉伸，也不是继续封装 Video2X 6.4.0。项目使用适合真人/通用视频的 AI 模型，将低于 1080p 的视频批量处理为最终 1080p 成品。

0.3.0 已发布的实现采用 TensorRT/ZeroCopy 主线，而不是直接运行官方 PyTorch 视频脚本：

```text
realesr-general-x4v3 TensorRT FP16
NVDEC CUDA/P010 input
CUDA/NV12 AVFrame output
NVENC HEVC
```

## 2. 目标

- 速度优先，充分释放单张 RTX 4090 的性能。
- 对目录内视频进行批量扫描和处理。
- 1080p 及以上视频直接跳过。
- 720p 真人视频使用 AI 超分到最终 1080p，避免完整 x4 大图回到 CPU 往返处理。
- 420p/480p 等更低分辨率视频也输出最终 1080p。
- 使用 NVIDIA GPU 加速，目标是在 RTX 4090 上显著快于 Video2X `realesrgan-plus-x4` 方案。
- 解决旧方案 GPU 使用率约 `60%`、显存仅约 `1.2GB`、长视频预计约 `170` 小时的问题。
- 每个任务开始前打印清晰任务清单，包括输入分辨率、帧数、模型、目标倍率、输出路径。
- 默认直接开始正式处理，运行中持续打印 fps、百分比、预计剩余时间和 GPU 状态。
- 输出文件命名为原文件同目录下的 `*_1080p.mp4`。
- 人工质检用输出必须生成在输入文件同目录，不能只落在临时测试目录。
- 0.3.0 已验证 720p 性能线 `77.106fps`，720p 质量线 `45.124fps`，420p `142.033fps`。

## 3. 非目标

- 不做普通 ffmpeg-only 拉伸作为默认方案。
- 不继续依赖 Video2X 6.4.0 的 ncnn/Vulkan Real-ESRGAN 处理链路。
- 不把 720p 真人视频默认做 `x4` AI 超分到 2880p。
- 不承诺所有长视频都能稳定在 2 小时内完成，必须以实测 fps 和视频帧数估算。
- 不做 GUI。

## 4. 输入输出

输入：

- `DATA_DIR`：待处理目录，默认 `/data`。
- 支持递归扫描 `.mp4` 文件。
- 已经是 `*_1080p.mp4` 的文件跳过，避免重复处理输出文件。

输出：

- 输出路径：`原文件名_1080p.mp4`。
- 输出目录：输入文件所在目录。
- 输出高度：默认 `1080`。
- 输出宽度：按原始宽高比自动计算，必须为偶数。
- 音频：默认复制原音频流。
- 可拖拽播放：输出 MP4 必须支持快速 seek，默认 `VIDEO_GOP_SIZE=60`，并启用 `+faststart`。

## 5. 模型要求

0.3.0 默认速度主线模型：

- `realesr-general-x4v3`

原因：

- 它是当前已验证的轻量通用 Real-ESRGAN v3 模型。
- TensorRT FP16 + ZeroCopy 后，420p/720p 均已超过 30fps。
- `RealESRGAN_x2plus` 纯推理仍太慢，只保留为质量参考，不作为速度主线。

禁止作为真人默认模型：

- `realesr-animevideov3`：动画视频模型，不适合作为写真/真人默认方案。
- `realesrgan-plus-x4` ncnn 模型：720p 会先 AI 到 2880p，速度不可接受。
- 不存在官方 `realesr-general-x2v3` 速度捷径，不能把推理参数伪装成 x2 当作结构优化。

## 6. 分辨率策略

设输入高度为 `H`，目标高度为 `T`，默认 `T=1080`。

- `H >= T`：跳过。
- `H < T`：按已有 TensorRT engine 和标准 profile 选择处理路线。
- 720p 性能线：源视频 `1280x720`，GPU 预处理到 `960x540`，使用 `realesr-general-x4v3-960x540-conv48-fp16.engine`，最终输出 1080p。
- 720p 质量线：源视频 `1280x720`，使用 `realesr-general-x4v3-1280x720-conv48-fp16.engine`，最终输出 1080p。
- 480p：`1080 / 480 = 2.25`，优先评估 `RealESRGAN_x4plus` 或分阶段策略；不能盲目套用 x2 模型导致质量或尺寸异常。
- 420p：当前样本兼容 profile `720x420` 已验证，正式镜像达到 `142.033fps`。

实现必须允许按高度区间配置模型与 outscale 策略。默认建议：

| 输入高度  | 默认模型            | outscale | 说明             |
| --------- | ------------------- | -------- | ---------------- |
| 720p      | `realesr-general-x4v3` TensorRT conv48 | `960x540` 性能线或 `1280x720` 质量线 | 当前主路径 |
| 540p-719p | `realesr-general-x4v3` TensorRT        | 标准 profile 选择                    | 先实测质量 |
| 360p-539p | `realesr-general-x4v3` TensorRT        | 标准 profile 选择                    | 420p 已验证 |
| 1080p+    | 不处理              | 不处理   | 直接跳过         |

## 7. 性能要求

- 启动时必须打印每个视频的预计总帧数。
- 处理时必须打印实时 fps、已处理帧数、百分比、预计剩余时间。
- 必须提供可选小样本 benchmark 模式，用于估算整片耗时和对比旧方案；默认不启用 benchmark，默认直接正式处理。
- RTX 4090 上应优先使用 TensorRT/CUDA，不使用 CPU 推理。
- 必须记录或显示 GPU 利用率、显存占用，便于判断 4090 是否被充分释放。
- 不允许长期因为 CPU 编码、解码、磁盘 IO 或 Python 调度导致 GPU 空闲而不记录原因。
- 编码、解码和写盘策略必须服务于吞吐，默认实现如果成为瓶颈，必须提供替代路径或调优参数。
- 支持 `tile` 参数，避免显存不足。
- 支持多容器/多进程按 GPU 分片运行，每个进程绑定一张 GPU。
- 支持 `VIDEO_INPUT_MODE=cuda-p010` 和 `VIDEO_OUTPUT_MODE=cuda-nvenc` 作为高性能 ZeroCopy 主线。
- 支持 `VIDEO_POSTPROCESS_MODE=srvgg-conv48-tail` 和 `TRT_ENGINE_SIZE=960x540` 作为 720p 性能线。

## 8. 配置项

必须支持环境变量：

- `DATA_DIR`：输入目录，默认 `/data`。
- `TARGET_HEIGHT`：目标高度，默认 `1080`。
- `MODEL_NAME`：构建镜像可用于选择待构建模型；runtime 主线默认使用已有 TensorRT engine。
- `TRT_ENGINE_PATH`：手工指定 TensorRT engine，默认 `auto`。
- `TRT_ENGINE_SIZE`：强制 engine/decode 规格，例如 `960x540`。
- `VIDEO_INPUT_MODE`：默认兼容模式，性能线使用 `cuda-p010`。
- `VIDEO_OUTPUT_MODE`：默认兼容模式，性能线使用 `cuda-nvenc`。
- `VIDEO_POSTPROCESS_MODE`：默认 `engine-output`，conv48 性能线使用 `srvgg-conv48-tail`。
- `OUTSCALE`：手动覆盖倍率，默认按 `TARGET_HEIGHT / input_height` 自动计算。
- `TILE`：默认 `0`，显存不足时可设置 `256` 或 `400`。
- `GPU_ID`：默认 `0`。
- `OUTPUT_SUFFIX`：默认 `_1080p`。
- `BENCHMARK_FRAMES`：只处理前 N 帧，默认空，表示全量处理；仅在用户显式设置时启用。
- `SKIP_EXISTING`：默认 `true`。

## 9. 容器要求

- runtime 镜像应包含 CUDA runtime、ffmpeg、TensorRT Python runtime 和本项目 worker。
- build 镜像包含 PyTorch、Real-ESRGAN、ONNX、TensorRT 构建工具和 `trtexec`。
- Real-ESRGAN Python 源码必须放在本项目内，不能在 Docker build 时临时 clone 外部项目。
- 模型权重不打包进镜像。
- runtime 不自动下载模型，不构建 engine；模型权重和 engine 优先从 `/models` 挂载目录读取。
- 运行时使用 NVIDIA 容器运行时或 CDI 设备，例如 `--device nvidia.com/gpu=3`。

## 10. 验收标准

功能验收：

- 1080p 原片被跳过。
- 720p 输入输出为 1080p。
- 480p 输入输出为 1080p。
- 输出视频有音频。
- 输出视频可被 `ffprobe` 正常读取。
- 已存在输出时默认跳过。

质量验收：

- 720p 真人样本不得出现明显动画化、皮肤蜡像化、边缘强烈振铃。
- 与 Video2X `realesrgan-plus-x4` 相比，1080p 成品主观质量应接近或可接受。

性能验收：

- 必须提供同一样本下的 fps 对比：
  - Video2X `realesrgan-plus-x4`
  - 早期官方 Real-ESRGAN `RealESRGAN_x2plus -s 1.5`
  - 0.3.0 TensorRT/ZeroCopy 正式镜像
- 必须记录正式处理过程中的实时 fps、GPU 利用率、显存占用和预计剩余时间。
- `RealESRGAN_x2plus` 已记录为过重的质量参考，不再作为速度主线。
- 如果 GPU 利用率长期偏低，不能只接受结果，必须定位并修正瓶颈。

## 11. 风险

- TensorRT engine 强依赖 GPU 架构、CUDA/TensorRT 版本和输入 shape，版本变化后需要重建。
- `RealESRGAN_x2plus` 虽适合通用图像，但当前速度不足，只能作为质量参考。
- 低清输入如 420p/480p 到 1080p 的放大倍率较高，质量未必稳定。
- 长视频帧数极大，需要依赖运行中实时 fps 和预计剩余时间做决策；benchmark 是可选预估工具，不是默认流程。
