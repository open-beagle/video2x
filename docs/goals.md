# 项目目标

## 1. 核心目标

本项目的目标是构建一个比旧 `video2x` 方案更快、更适合真人视频批处理的 AI 视频超分容器。

旧 Video2X 项目地址：

```text
https://github.com/k4yt3x/video2x
```

新 video2x 项目地址：

```text
https://github.com/open-beagle/video2x
```

旧方案的问题：

- Video2X 6.4.0 的 `realesrgan-plus` 路线基于 ncnn/Vulkan 模型。
- 当前可用的 `realesrgan-plus` 默认走 x4 模型。
- 720p 视频会被 AI 先计算到 2880p，再得到名为 1080p 的输出。
- 这会产生大量不必要的计算，导致单帧速度很慢。
- 对 2 小时、720p、30fps 的长视频来说，旧路线不适合批量生产。

旧方案在 RTX 4090 上的实测问题：

- GPU 使用率约 `60%`。
- 显存占用约 `1.2GB`。
- 单个长视频预计耗时约 `170` 小时。

这些数据说明旧方案没有充分释放 RTX 4090 性能。新项目必须围绕这个问题重做处理链路，而不是只改容器封装。

新方案必须避免这条低效路径。早期探索过官方 Real-ESRGAN Python/CUDA 与 `RealESRGAN_x2plus + outscale`，但实测后已经确认它不能作为速度主线。

0.3.0 实测后，主线已经从早期 PyTorch/outscale 方案升级为 TensorRT/ZeroCopy 方案：

```text
realesr-general-x4v3
-> TensorRT FP16 engine
-> NVDEC CUDA/P010 input
-> CUDA/NV12 AVFrame output
-> NVENC HEVC
```

`RealESRGAN_x2plus` 仍保留为通用真实场景质量参考，但它在 TensorRT FP16 下仍然太慢，不再作为 720p 速度主线。

## 2. 性能目标

在单张 RTX 4090 上，项目应尽可能充分释放 GPU 性能，让 2 小时、720p 或 480p、30fps 的视频具备可批量生产的处理速度。

速度是本项目的基石。功能、质量和易用性都必须服务于这个目标：不能因为错误模型、错误倍率、CPU 编码、磁盘 IO、串行流程或过度保守的默认参数，让 RTX 4090 长时间空闲。

第一阶段目标：

- 明显快于 Video2X 6.4.0 `realesrgan-plus-x4` 方案。
- 720p 到 1080p 的性能线必须避免完整 x4 大图进入 CPU 往返。
- 480p 视频也必须输出最终 1080p。
- 单卡 RTX 4090 上运行时必须优先让瓶颈接近 GPU 推理，而不是 Python、ffmpeg、磁盘 IO 或错误的模型倍率。
- 默认直接处理并持续显示实时 fps、百分比和预计剩余时间。
- 提供可选 benchmark 模式，用短样本估算完整 2 小时视频的耗时，但 benchmark 不能成为默认阻塞步骤。

参考换算：

- 2 小时 30fps 视频约为 216000 帧。
- 如果希望 2 小时 30fps 视频在 2 小时内处理完成，实际处理速度需要达到约 30fps。
- 本项目第一阶段按 30fps 输入视频设计和验收，不以 60fps 视频作为默认目标。
- 0.3.0 已经把 720p 性能线推进到 `60fps+`。
- 720p 到 1080p 的最终倍率为 `1.5`。
- 480p 到 1080p 的最终倍率为 `2.25`，需要单独验证模型选择、质量和速度。

0.3.0 正式镜像回归：

| 路线 | 完整 5 分钟样本 fps | 2 小时 30fps 估算 |
| ---- | ------------------- | ----------------- |
| 420p ZeroCopy | `142.033` | 约 `25.3` 分钟 |
| 720p `960x540 conv48` 性能线 | `77.106` | 约 `46.7` 分钟 |
| 720p `1280x720 conv48` 质量线 | `45.124` | 约 `79.8` 分钟 |

因此，本项目不能只声称“更快”，必须记录实际处理 fps、GPU 利用率、显存占用和估算整片耗时。benchmark 样本可以用于对比测试，但正式使用路径必须默认直接干活。

## 3. 验收方式

每次性能验证至少记录：

- GPU 型号：例如 RTX 4090。
- 输入视频分辨率、帧率、总帧数。
- 模型名称和 engine：当前速度主线为 `realesr-general-x4v3` TensorRT FP16。
- 路线：`performance` 使用 `960x540 conv48 ZeroCopy`，`quality` 使用 `1280x720 conv48 ZeroCopy`。
- `tile` 参数。
- 实测 fps。
- GPU 利用率和显存占用。
- 估算完整 2 小时视频耗时。
- 如启用 benchmark，记录 benchmark 帧数。

验收标准：

- 输出必须是真 AI 超分结果，不允许用 ffmpeg-only 拉伸冒充。
- 720p 和 480p 输入的最终输出高度必须是 1080。
- 输出文件必须能被 `ffprobe` 正常读取。
- 输出应保留音频。
- 性能报告必须和旧 Video2X x4 路线对比。
- 如果 GPU 利用率长期偏低，必须定位是解码、编码、磁盘 IO、Python 调度还是模型策略造成的瓶颈。

## 4. 非目标

- 不追求继续兼容 Video2X 6.4.0 的 ncnn/Vulkan 模型路线。
- 不把 `realesr-animevideov3` 作为真人视频默认模型。
- 不在没有 benchmark 数据的情况下承诺固定完成时间。
- 不用普通转码或 resize 替代 AI 超分。
