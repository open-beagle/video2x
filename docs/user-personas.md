# 用户画像

## 1. 核心用户

本项目的核心用户不是 AI 工程师，也不需要理解 Real-ESRGAN、模型倍率、CUDA、PyTorch 或视频编码细节。

用户的真实需求很简单：

- 有一批 `.mp4` 视频。
- 视频可能是 720p，也可能是 480p。
- 希望统一输出最终 1080p 成品。
- 希望单张 RTX 4090 尽量跑满性能。
- 希望用 Docker 启动后自动干活。

因此，本项目默认使用方式必须尽量简单。用户只需要挂载 `data` 和 `models` 目录，不需要传递复杂参数。

## 2. 默认使用方式

用户期望的操作方式：

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  video2x
```

默认行为：

- 扫描 `/data` 目录。
- 读取 `/models` 中已有模型。
- 自动查找所有 `.mp4` 文件。
- 自动判断每个视频的分辨率、帧率、总帧数。
- 自动决定是否需要处理。
- 自动选择处理方案。
- 自动输出到原视频同目录。

用户不应该必须知道：

- 该选哪个模型。
- `outscale` 应该是多少。
- 720p 到 1080p 为什么是 `1.5`。
- 480p 到 1080p 为什么是 `2.25`。
- `tile` 是什么。
- ffmpeg 如何保留音频。

这些都应该由容器默认策略处理。

## 3. 自动化识别

容器启动后必须先扫描任务，而不是立刻盲跑。

扫描阶段需要识别：

- 文件路径。
- 输入宽度和高度。
- 帧率。
- 总帧数。
- 是否已有输出文件。
- 是否已经达到 1080p。
- 预计输出路径。
- 推荐模型。
- 推荐 `outscale`。
- 推荐 `tile`。
- 预计处理耗时。如果没有历史速度数据，显示为运行后估算。

示例日志：

```text
Scan result:
1. /data/a.mp4
   input: 1280x720, 30fps, 216000 frames
   output: /data/a_1080p.mp4
   action: upscale
   model: RealESRGAN_x2plus
   outscale: 1.5
   estimated time: after start

2. /data/b.mp4
   input: 854x480, 30fps, 216000 frames
   output: /data/b_1080p.mp4
   action: upscale
   model: auto
   outscale: 2.25
   estimated time: after start

3. /data/c_1080p.mp4
   input: 1920x1080, 30fps, 216000 frames
   action: skip
   reason: already 1080p or higher
```

开始处理前，容器必须打印完整任务清单，让用户知道即将处理什么、跳过什么、为什么跳过。

## 4. 进度与耗时

用户不懂 AI，也不应该靠猜测判断一个视频要跑多久。

默认行为应该是直接开始正式处理，并在日志中持续显示进度和预计剩余时间：

- 当前任务。
- 已处理帧数。
- 总帧数。
- 实时 fps。
- 完成百分比。
- 预计剩余时间。
- GPU 利用率。
- 显存占用。

示例日志：

```text
Progress:
  input: /data/a.mp4
  frames: 2400 / 216000
  progress: 1.11%
  speed: 8.2 fps
  estimated remaining: 7h 14m
  gpu: 92%, memory: 11.2GB / 24GB
```

如果用户想先判断是否值得完整处理，可以显式开启 benchmark 模式。benchmark 不是默认流程，而是一个可选决策工具。

## 5. 性能期望

项目必须围绕单张 RTX 4090 做优化。

优化目标：

- 尽量减少错误倍率带来的无效计算。
- 720p 到 1080p 不允许先 AI 计算到 2880p。
- 480p 到 1080p 必须单独验证模型、质量和速度。
- 默认使用 CUDA/PyTorch 推理。
- 避免因为 Python、ffmpeg、磁盘 IO 或串行流程让 GPU 长时间空闲。
- 日志中应能看出实际 fps、预计剩余时间、GPU 利用率和显存占用。

性能目标不靠口头承诺，必须靠正式处理日志和可选 benchmark 数据验证。

## 6. Docker 交付

最终交付形式是 Docker 镜像。

用户不需要本机安装 Python、PyTorch、Real-ESRGAN 或 ffmpeg。用户只需要：

- 安装 Docker。
- 配好 NVIDIA GPU 容器运行环境。
- 准备输入视频目录。
- 准备或挂载模型目录。
- 运行容器。

容器内部负责：

- 检查 GPU 是否可用。
- 检查模型是否存在。
- 检查输入目录是否存在。
- 扫描视频。
- 打印任务计划。
- 执行正式处理。
- 持续输出进度和性能日志。
- 在用户显式开启时执行 benchmark。
- 校验输出文件。

如果失败，错误信息必须面向普通用户，不能只输出 Python 堆栈。
