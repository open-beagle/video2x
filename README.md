# video2x

面向真人视频批处理的 AI 超分容器。

本项目聚焦一个具体目标：把低于 1080p 的视频，尤其是 720p、30fps 的长视频，使用 NVIDIA GPU 批量处理为最终 1080p 成品，并尽量避免无意义的高倍率中间计算。

## 特点

- 基于官方 Real-ESRGAN Python/CUDA 推理路线。
- 默认使用 `RealESRGAN_x2plus` 处理真人/通用视频。
- 720p 到 1080p 默认使用 `outscale=1.5`。
- 递归扫描输入目录中的 `.mp4` 文件。
- 自动跳过已是 1080p 及以上的视频。
- 输出文件默认命名为 `*_1080p.mp4`。
- 支持 benchmark 模式，用短样本估算完整视频耗时。
- 支持通过 `tile` 参数缓解显存压力。

## 目标场景

- 输入：720p、30fps 真人视频。
- 输出：最终 1080p 视频。
- 硬件：单张 RTX 4090。
- 目标：比旧 Video2X x4 路线更快，并尽量让瓶颈接近 GPU 推理本身。

## 运行

默认运行方式：

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  video2x
```

模型目录说明：

- 模型不打包进镜像。
- 默认从 `/models` 读取权重。
- 如果 `/models` 中缺少默认权重，容器运行时按需下载到 `/models`。
- Real-ESRGAN Python 代码已经放在本项目 `vendor/realesrgan`，便于后续针对性能和日志做调整。
- 本项目自己的扫描、规划、模型、GPU 监控和运行编排代码放在 `src/`。

只扫描和打印任务计划，不执行 AI 推理：

```bash
docker run --rm \
  -e DRY_RUN=true \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  video2x
```

本地构建：

```bash
bash .beagle/build.sh
```

默认本地构建标签为 `video2x:0.3.0`。如需覆盖：

```bash
IMAGE_TAG=latest bash .beagle/build.sh
```

## 文档

- [项目目标](docs/goals.md)
- [用户画像](docs/user-personas.md)
- [技术架构图](docs/architecture.md)
- [项目计划](docs/plans.md)
- [需求规格说明书](docs/requirements-specification.md)
- [技术方案](docs/technical-design.md)

## 状态

项目处于第一阶段实现中，当前重点是容器化、批处理入口、benchmark 和 720p 到 1080p 的性能验证。
