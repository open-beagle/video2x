# video2x

面向真人视频批处理的 AI 超分容器。

本项目聚焦一个具体目标：把低于 1080p 的视频，尤其是 720p、30fps 的长视频，使用 NVIDIA GPU 批量处理为最终 1080p 成品，并尽量避免无意义的高倍率中间计算。

## 特点

- 默认使用 `RUNNER=trt-cuda` 高速路线。
- 当前主路径为 `realesr-general-x4v3 + TensorRT FP16 + CUDA 后处理`。
- 420p 到 1080p 已在 RTX 4090 上验证超过 30fps。
- 递归扫描输入目录中的 `.mp4` 文件。
- 自动跳过已是 1080p 及以上的视频。
- 输出文件默认命名为 `*_1080p.mp4`。
- 支持 benchmark 模式，用短样本估算完整视频耗时。
- 运行入口只支持当前 TRT-CUDA 主线，避免误走旧 PyTorch 链路。

## 目标场景

- 输入：低于 1080p、30fps 真人视频。
- 输出：最终 1080p 视频。
- 硬件：单张 RTX 4090。
- 目标：2 小时视频在单张 RTX 4090 上 2 小时内完成，优先接近或超过实时。

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
- 高速默认路径需要 `/models/realesr-general-x4v3-420x720-fp16.engine`。
- Real-ESRGAN Python 代码仅用于本地模型导出和对比实验，不进入默认运行链路。
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
- [模型准备](docs/models.md)
- [需求规格说明书](docs/requirements-specification.md)
- [技术方案](docs/technical-design.md)

## 状态

当前 `RUNNER=trt-cuda` 已接入主入口。420p 样本到 1080p 在 RTX 4090 上已验证 `48fps+`，下一步重点是流水线镜像验证、真正 5 分钟样本回归和 NVENC 编码修复。
