# AI 视频超分批处理工具技术方案

## 1. 总体方案

本项目实现一个面向批量真人视频的 AI 超分容器。核心处理链路采用官方 Real-ESRGAN Python/CUDA，而不是 Video2X 6.4.0 的 ncnn/Vulkan Real-ESRGAN。

技术设计的第一优先级是速度。实现必须围绕单张 RTX 4090 的吞吐设计，避免 GPU 等 CPU、等 ffmpeg、等磁盘、等 Python 串行调度。只要 GPU 利用率长期偏低，就必须把它视为性能缺陷，而不是正常现象。

核心原因：

- Video2X 6.4.0 的 `realesrgan-plus` 只有 x4 ncnn 模型。
- 720p 输入会被 AI 处理成 2880p，计算量过大。
- RTX 4090 上旧方案 GPU 使用率约 `60%`，显存占用约 `1.2GB`，单个长视频预计耗时约 `170` 小时，没有充分释放硬件性能。
- 官方 Real-ESRGAN 提供 `RealESRGAN_x2plus.pth`，并支持 `outscale` 浮点倍率。
- 720p 到 1080p 可走 `RealESRGAN_x2plus` + `outscale=1.5`。

## 2. 推荐技术栈

- Python 3
- PyTorch CUDA 版本
- 官方 Real-ESRGAN
- ffmpeg / ffprobe
- Bash entrypoint
- Docker

可选：

- `nvidia-smi` 用于运行时诊断。
- `opencv-python-headless` 用于视频帧处理依赖。

## 3. 镜像设计

基础镜像跟随 Jasna 的 CUDA runtime 路线：

```dockerfile
FROM nvidia/cuda:<cuda-runtime-tag>-runtime-ubuntu24.04
```

构建步骤：

1. 安装系统依赖：`ffmpeg`、`git`、`libgl1`、`libglib2.0-0`。
2. 安装 PyTorch CUDA 版本和 Real-ESRGAN Python 依赖。
3. 克隆或复制官方 Real-ESRGAN 源码。
4. 下载默认模型权重 `RealESRGAN_x2plus.pth`。
5. 复制本项目 `entrypoint.sh`。

模型目录建议：

```text
/models/RealESRGAN_x2plus.pth
/models/RealESRGAN_x4plus.pth
```

## 4. 入口流程

入口脚本执行流程：

1. 校验 `ffmpeg`、`ffprobe`、`python`、CUDA 可用性。
2. 扫描 `DATA_DIR` 下 `.mp4` 文件。
3. 跳过：
   - `*_1080p.mp4`
   - 输出文件已存在且 `SKIP_EXISTING=true`
   - 高度大于等于 `TARGET_HEIGHT` 的视频
4. 读取输入视频：
   - 宽度
   - 高度
   - 帧率
   - 总帧数
   - 时长
5. 为每个视频生成任务：
   - 输出路径
   - 模型
   - `outscale`
   - `tile`
   - 预计处理方式
6. 打印完整任务清单。
7. 默认直接执行正式处理。
8. 运行中持续打印帧进度、实时 fps、百分比、预计剩余时间、GPU 利用率和显存占用。
9. 处理完成后用 `ffprobe` 验证输出高度和可读性。

## 5. 模型与倍率决策

默认决策函数：

```text
target_height = 1080
outscale = target_height / input_height
```

720p：

```text
input_height = 720
outscale = 1.5
model = RealESRGAN_x2plus
```

命令：

```bash
python inference_realesrgan_video.py \
  -i input.mp4 \
  -o output_dir \
  -n RealESRGAN_x2plus \
  -s 1.5 \
  --suffix 1080p \
  --tile 0
```

注意：官方 `inference_realesrgan_video.py` 的 `-o` 是输出目录，不是输出文件路径；GPU 绑定通过容器可见设备或 `CUDA_VISIBLE_DEVICES` 控制，不使用不存在的 `--gpu-id` 参数。

`RealESRGAN_x2plus` 模型内部是 x2 网络，`outscale=1.5` 表示最终输出缩放倍率。这样避免 Video2X x4 路线把 720p 先计算到 2880p。

低分辨率策略需要实测：

- 540p-719p：优先 `RealESRGAN_x2plus`。
- 480p：输出 1080p，`outscale=2.25`，必须单独评估 `RealESRGAN_x2plus` 与 `RealESRGAN_x4plus` 的速度、质量和 GPU 利用率。
- 360p-539p：评估 `RealESRGAN_x2plus` 与 `RealESRGAN_x4plus`。
- 低于 360p：默认不自动处理，除非用户显式允许。

## 6. 速度与监控

默认路径必须直接处理正式视频，并持续输出足够让用户决策的进度日志。

运行中必须记录：

- 当前帧和总帧数。
- 实时 fps。
- 百分比。
- 预计剩余时间。
- GPU 利用率。
- 显存占用。
- 当前模型、`outscale`、`tile`。

如果 GPU 利用率长期偏低，需要在日志中暴露可能瓶颈：

- 解码速度不足。
- 编码速度不足。
- 磁盘读写不足。
- Python 单进程调度不足。
- `tile` 设置过小导致切块开销过大。
- 模型或倍率选择不合理。

## 7. Benchmark 模式

必须实现可选 benchmark 模式，用来让用户在正式处理前做决策。benchmark 不是默认流程。

环境变量：

```bash
BENCHMARK_FRAMES=300
```

处理方式：

1. 先用 ffmpeg 截取前 N 帧或前 N/帧率 秒为临时样本。
2. 对临时样本运行完整 AI 超分。
3. 记录耗时、fps、显存、输出尺寸。
4. 按总帧数估算整片耗时。

输出示例：

```text
Benchmark:
  input: INU-047-U.mp4
  frames: 300 / 303298
  model: RealESRGAN_x2plus
  outscale: 1.5
  speed: 8.2 fps
  estimated full time: 10h 16m
```

## 8. 多 GPU 策略

单个视频默认绑定一张 GPU。不要假设一个视频能自动拆到多张 GPU。

推荐运行方式：

```bash
docker run --name realesrgan-gpu3 \
  --device nvidia.com/gpu=3 \
  -d --rm \
  -e DATA_DIR=/data \
  -e GPU_ID=0 \
  -v /data/jasna/ais0:/data \
  open-beagle/video2x:latest
```

因为容器里只暴露了一张 GPU，`GPU_ID=0` 通常就是容器内可见的第一张卡。

多目录并行：

```text
GPU 3 -> /data/jasna/ais0
GPU 4 -> /data/jasna/ais1
```

## 9. 输出编码

输出编码必须服从速度目标。官方脚本默认使用 `libx264`，如果它导致 RTX 4090 等待 CPU 编码，就不能把它视为最终最优路径。

- 不允许用 ffmpeg 替代 AI 超分。
- 允许用 ffmpeg 做封装、音频复制、元数据修复、faststart。
- 输出必须可被 `ffprobe` 读取。
- 如果 CPU 编码成为瓶颈，应评估 NVENC、预设参数或封装流程优化。
- 如果磁盘 IO 成为瓶颈，应评估临时目录、输出目录和并发策略。

如需重新封装：

```bash
ffmpeg -hide_banner -y \
  -i ai_output.mp4 \
  -c copy \
  -movflags +faststart \
  final_1080p.mp4
```

## 10. 日志要求

启动日志必须包含：

```text
DATA_DIR=/data
TARGET_HEIGHT=1080
MODEL_NAME=RealESRGAN_x2plus
GPU_ID=0
TILE=0
```

任务清单必须包含：

```text
1. /data/INU-047-U.mp4
   input: 1280x720, 303298 frames
   output: /data/INU-047-U_1080p.mp4
   model: RealESRGAN_x2plus
   outscale: 1.5
```

运行中必须包含：

- 当前帧
- 总帧数
- fps
- 百分比
- 预计剩余时间
- GPU 利用率
- 显存占用

## 11. 错误处理

必须失败退出的情况：

- 输入目录不存在。
- GPU 不可用。
- 模型文件不存在且无法下载。
- 输出文件生成失败。
- 输出不是目标高度。

可以跳过的情况：

- 输入文件不可读。
- 已存在输出文件。
- 输入高度已经大于等于目标高度。

## 12. 与旧 Video2X 方案的区别

旧方案：

```text
Video2X 6.4.0
realesrgan-plus-x4
720p -> 2880p
速度约 0.53 fps
RTX 4090 GPU 使用率约 60%
RTX 4090 显存占用约 1.2GB
单个长视频预计约 170 小时
```

新方案：

```text
官方 Real-ESRGAN Python/CUDA
RealESRGAN_x2plus
720p -> 1080p final outscale
默认直接处理，运行中实时显示 fps 和预计剩余时间
```

核心改进：

- 不再把 720p 真人视频默认 x4 到 2880p。
- 不再受 ncnn 模型目录缺少 `realesrgan-plus-x2` 的限制。
- 使用官方 `.pth` 模型和 Python 推理脚本。

## 13. 第一阶段交付

第一阶段只做最小可用版本：

- Dockerfile
- entrypoint.sh
- 官方 Real-ESRGAN 集成
- 默认模型 `RealESRGAN_x2plus`
- 递归扫描 `/data`
- 720p 到 1080p
- 480p 到 1080p
- 1080p 跳过
- 运行中 fps、预计剩余时间、GPU 利用率和显存日志
- 可选 benchmark 模式
- README 运行示例

第一阶段不做：

- GUI
- Web API
- 分布式调度
- 自动画质评分
- 复杂队列系统
