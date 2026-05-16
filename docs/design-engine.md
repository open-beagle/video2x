# TensorRT Engine 设计

## 1. 目标

本文件只记录 TensorRT engine 的生成、命名、选择和分辨率策略。

Zero-Copy、NVENC、NVDEC 和显存闭环设计不放在这里，见 `docs/design-zerocopy.md`。

## 2. 基本原则

当前运行主线只读取已经构建好的 TensorRT engine：

```text
*.engine
```

运行镜像不负责：

- 从 `.pth` 导出 ONNX。
- 从 ONNX 构建 TensorRT engine。
- 自动下载模型。
- 自动为未知分辨率生成 engine。

模型构建由 build 镜像负责：

```text
video2x:0.3.0-build
```

运行镜像负责：

```text
扫描视频 -> 选择已有 engine -> TensorRT 推理 -> 输出视频
```

## 3. 命名规则

engine 文件名使用人类可读的 `宽x高`：

```text
{model}-{width}x{height}-fp16.engine
```

示例：

```text
realesr-general-x4v3-640x360-fp16.engine
realesr-general-x4v3-854x480-fp16.engine
realesr-general-x4v3-960x540-fp16.engine
realesr-general-x4v3-1280x720-fp16.engine
```

注意：Tensor shape 仍然是 `NCHW`，即：

```text
1x3xheightxwidth
```

例如：

```text
文件名：realesr-general-x4v3-720x480-fp16.engine
Tensor shape：1x3x480x720
```

文件名不能写成 `480x720`，否则人类会误读。

## 4. 标准 Profile

真实视频常见输入不是测试样本的 `720x420`，而是：

- `360p`
- `480p`
- `720p`
- `1080p`

因此推荐准备这些标准 engine：

```text
realesr-general-x4v3-640x360-fp16.engine
realesr-general-x4v3-720x480-fp16.engine
realesr-general-x4v3-854x480-fp16.engine
realesr-general-x4v3-960x540-fp16.engine
realesr-general-x4v3-1280x720-fp16.engine
```

其中：

- `640x360`：标准 16:9 360p。
- `720x480`：常见 3:2 / NTSC-ish 480p。
- `854x480`：常见 16:9 480p。
- `960x540`：720p 预缩性能 profile，也可覆盖接近 16:9 的 480p。
- `1280x720`：标准 720p，质量更直接，但中间帧更大。

`1920x1080` 及以上输入默认跳过，不做超分。

## 5. 样本兼容 Profile

当前测试样本里有一个非标准 `720x420` 输入。它不是标准分辨率，但可以作为兼容 profile 保留：

```text
realesr-general-x4v3-720x420-fp16.engine
```

该 profile 只用于精确匹配 `720x420` 输入，不作为标准 480p 或 16:9 输入的替代。

## 6. 选择规则

程序扫描 `/models` 下已有 engine，并按 `宽x高` 解析可用规格。

选择顺序：

1. 如果存在与输入分辨率完全一致的 engine，优先使用它。
2. 如果没有完全一致的 engine，只在标准 profile 中选择已有 engine。
3. 宽高比误差超过 `2%` 时跳过，不强行变形处理。
4. 预缩/预放比例超过 `25%` 时跳过，不为了能跑牺牲画质。
5. engine 的 x4 输出必须覆盖最终 `1920x1080`，否则跳过。
6. `1080p` 及以上输入默认跳过。

典型结果：

| 输入        | 选择                                                              |
| ----------- | ----------------------------------------------------------------- |
| `640x360`   | `640x360` engine                                                  |
| `854x480`   | `854x480` engine；如果没有但有 `960x540` engine，可使用 `960x540` |
| `720x480`   | `720x480` engine；不套 16:9 engine                                |
| `1280x720`  | `1280x720` engine，如果存在                                       |
| `720x420`   | 只有存在 `720x420` engine 时才处理                                |
| `1920x1080` | 跳过                                                              |

跳过不是失败，而是保护画质和避免错误 engine。

## 7. Build 镜像

build 镜像用于构建 engine：

```text
video2x:0.3.0-build
```

它包含：

- PyTorch / torchvision。
- Real-ESRGAN 模型结构。
- ONNX 导出依赖。
- TensorRT 构建工具。
- `trtexec`。

运行镜像不包含这些构建依赖。

## 8. 构建流程

推荐流程：

```text
.pth -> .onnx -> .engine
```

build 镜像默认自动完成导出 ONNX 和构建 engine。手工流程等价于：

```bash
python /app/tools/export_realesrgan_onnx.py \
  --model realesr-general-x4v3 \
  --weights /models/realesr-general-x4v3.pth \
  --output /models/realesr-general-x4v3-854x480.onnx \
  --input-shape 1,3,480,854
```

trtexec \
  --onnx=/models/realesr-general-x4v3-854x480.onnx \
  --saveEngine=/models/realesr-general-x4v3-854x480-fp16.engine \
  --fp16
```

不要默认使用 INT8。超分是像素级回归任务，INT8 可能带来断层、网格和细节异常，只能作为单独画质验证分支。

## 9. 兼容性

TensorRT engine 强依赖：

- GPU 架构。
- TensorRT 版本。
- CUDA 运行库。
- 输入 shape。
- 构建参数。

当前目标硬件是 RTX 4090。建议在 RTX 4090 或同架构环境构建 engine，并在 CUDA / TensorRT 版本变更后重新构建。

## 10. Review Checkpoint

- [x] engine 文件名使用 `宽x高`。
- [x] 运行镜像只读取 `.engine`。
- [x] build 镜像负责 `.pth -> .onnx -> .engine`。
- [x] `720x420` 被降级为样本兼容 profile。
- [ ] 标准 profile engine 全部构建完成。
- [ ] 使用标准 profile 回归 360p、480p、720p 样本。
- [ ] 明确是否保留 `1280x720` 质量 profile，或默认只走 `960x540` 性能 profile。

## 11. 运行命令

build 镜像运行时不需要输入任何业务参数。准备好本地 `models/` 目录，并确保里面至少有：

```text
realesr-general-x4v3.pth
```

然后直接运行：

```bash
docker run --rm \
  --device nvidia.com/gpu=0 \
  -v /path/to/models:/models \
  video2x:0.3.0-build
```

容器会自动构建这些 profile：

```text
640x360
720x480
854x480
960x540
1280x720
720x420
```

每个 profile 会自动生成：

```text
{model}-{width}x{height}.onnx
{model}-{width}x{height}.onnx.data
{model}-{width}x{height}-fp16.engine
```

如果 `.onnx` 或 `.engine` 已经存在，容器会跳过对应步骤，不重复构建。
