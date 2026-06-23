# video2x

面向真人视频批处理的 AI 超分容器。

本项目聚焦一个具体目标：把低于 1080p 的视频，尤其是 720p、30fps 的长视频，使用 NVIDIA GPU 批量处理为最终 1080p 成品，并尽量避免无意义的高倍率中间计算。

## 特点

- 默认使用 `RUNNER=trt-cuda` 高速路线。
- 当前主路径为 `realesr-general-x4v3 + TensorRT FP16 + CUDA NV12 + HEVC NVENC`。
- 标准 480p/720p 路线按固定输入规格 engine 管理，当前 420p/720p 样本到 1080p 已在 RTX 4090 上验证超过 30fps。
- 递归扫描输入目录中的 `.mp4`、`.mkv` 文件。
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
docker run --rm \
  --device nvidia.com/gpu=0 \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  video2x
```

默认编码为 `hevc_nvenc`，镜像内置 NVENC hook。运行时需要使用 NVIDIA CDI GPU 设备，并保持 `NVENC_GPU_INDEX` 与 GPU 编号一致；默认值为 `0`。

模型目录说明：

- 模型不打包进镜像。
- 默认从 `/models` 读取权重。
- 高速路线需要 `/models` 中存在匹配输入规格的 `*-fp16.engine`，文件名使用 `宽x高`。
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

模型构建镜像用于 `.pth -> .onnx -> .engine`，不参与默认视频处理：

```text
video2x:0.3.0-build
```

使用构建镜像生成 TensorRT Engine：

```bash
docker run --rm \
  --device nvidia.com/gpu=0 \
  -v /path/to/models:/models \
  video2x:0.3.0-build
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

当前 `CUDA NV12 + HEVC NVENC` 完整 5 分钟样本回归已通过：420p `72fps+`，720p `46fps+`。人工画质评审结论为“尚可”，下一步重点是真正 Zero-Copy 管线。

## CI

```powershell
# build
git switch build ;`
  git merge main --ff-only ;`
  git push origin build ;`
  git switch main

# release
git switch release ;`
  git merge main --ff-only ;`
  git push origin release ;`
  git switch main
```
