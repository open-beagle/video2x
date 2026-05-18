# AI 视频超分批处理工具技术方案

## 0. 0.3.0 当前技术方案

0.3.0 已经从早期 PyTorch/outscale 方案升级为 TensorRT/ZeroCopy 方案。当前 runtime 主线是：

```text
NVDEC CUDA/P010
-> CUDA P010 to NCHW FP16
-> TensorRT FP16
-> CUDA/NV12 AVFrame
-> NVENC HEVC
-> faststart MP4 + audio copy
```

正式镜像回归：

| 路线 | 完整 5 分钟样本 fps | 输出 |
| ---- | ------------------- | ---- |
| 420p ZeroCopy | `142.033` | `1920x1080 / 30fps / 9103 frames / HEVC / AAC` |
| 720p `960x540 conv48` 性能线 | `77.106` | `1920x1080 / 30fps / 9103 frames / HEVC / AAC` |
| 720p `1280x720 conv48` 质量线 | `45.124` | `1920x1080 / 30fps / 9103 frames / HEVC / AAC` |

因此 0.3.0 可以作为当前发布版本。后续阶段不再围绕“能不能超过 30fps”打转，而是继续推进默认策略、C++ runner、TensorRT Plugin 和 FlashVSR 竞品验证。

## 1. 总体方案

本项目实现一个面向批量真人视频的 AI 超分容器。核心目标是把低于 1080p 的 `.mp4` 视频批量输出为标准 `1920x1080` 成品，并充分释放单张 RTX 4090 的性能。

旧 Video2X 6.4.0 的 `realesrgan-plus-x4` 路线会把 720p 先 AI 放大到 2880p，再得到名为 1080p 的输出，导致速度和资源利用都不可接受。0.3.0 不再使用该路线，也不再把官方 PyTorch 视频脚本作为性能主线。

## 2. 推荐技术栈

runtime 镜像：

- CUDA runtime。
- FFmpeg / ffprobe。
- TensorRT Python runtime。
- `cuda-python` / `numpy`。
- 本项目 `src/` worker。
- 原生 CUDA bridge：`libffmpeg_cuda_chw_bridge.so`。
- 预编译后处理 PTX：`postprocess.ptx`。
- NVENC hook：`nvenc_ioctl_hook.so`。

build 镜像：

- PyTorch / torchvision。
- Real-ESRGAN 模型结构。
- ONNX / onnxscript。
- TensorRT 构建工具。
- `trtexec`。

运行镜像不包含 PyTorch、ONNX 导出依赖和 engine 构建工具。`.pth -> .onnx -> .engine` 全部由 build 镜像负责。

## 3. 镜像职责

runtime 镜像只做业务处理：

```text
扫描 /data -> 选择 /models 下已有 engine -> TensorRT 推理 -> 输出 MP4
```

runtime 不负责：

- 自动下载模型。
- 从 `.pth` 导出 ONNX。
- 从 ONNX 构建 TensorRT engine。
- 为未知分辨率动态生成 engine。

build 镜像负责：

```text
.pth -> FP16 .onnx -> FP16 .engine
```

对 `realesr-general-x4v3` 和 `realesr-general-wdn-x4v3`，build 镜像还会生成 `-conv48-fp16.engine` 和对应 tail 参数，供 `VIDEO_POSTPROCESS_MODE=srvgg-conv48-tail` 使用。

## 4. 入口流程

入口脚本执行流程：

1. 校验 `/data`、`/models`、GPU 和 FFmpeg。
2. 扫描 `DATA_DIR` 下 `.mp4` 文件。
3. 跳过 `*_1080p.mp4`、已存在输出、1080p 及以上视频。
4. 读取输入宽高、帧率、总帧数、时长。
5. 选择已有 TensorRT engine。
6. 打印任务清单。
7. 默认直接执行正式处理。
8. 运行中持续打印帧进度、实时 fps、百分比、预计剩余时间和 GPU 状态。
9. 处理完成后验证输出宽高、帧数、音频、seek 和 `ffprobe` 可读性。

输出文件必须生成在输入文件同目录，方便人工质检从挂载目录直接取回。

## 5. 模型与分辨率决策

当前速度主线：

```text
realesr-general-x4v3 TensorRT FP16
```

`RealESRGAN_x2plus` 是通用真实场景质量参考，但在当前基线中速度太慢，不作为 720p 主线。

720p 有两条路线：

- 性能线：`1280x720 -> 960x540 conv48 engine -> 1920x1080`，正式镜像 `77.106fps`。
- 质量线：`1280x720 -> 1280x720 conv48 engine -> 1920x1080`，正式镜像 `45.124fps`。

420p 样本兼容路线：

- `720x420` engine，正式镜像 `142.033fps`。

1080p 及以上输入默认跳过，不做超分。

## 6. ZeroCopy 视频链路

高性能模式使用：

```text
VIDEO_INPUT_MODE=cuda-p010
VIDEO_OUTPUT_MODE=cuda-nvenc
```

输入侧：

- FFmpeg C API / NVDEC 读取 CUDA/P010 surface。
- CUDA kernel 直接把 P010 转为 NCHW FP16。
- 不再走 CPU raw RGB，也不再做 raw RGB H2D。

输出侧：

- CUDA 后处理直接写入 FFmpeg 分配的 CUDA/NV12 AVFrame。
- `hevc_nvenc` 从 CUDA surface 编码。
- 不再每帧 D2H rawvideo，也不再通过 ffmpeg stdin 喂帧。

MP4 封装：

- 音频复制。
- `VIDEO_GOP_SIZE=60`。
- `+faststart`，确保输出可快速拖拽播放。

## 7. 配置项

常用环境变量：

```text
DATA_DIR=/data
TARGET_HEIGHT=1080
OUTPUT_SUFFIX=_1080p
SKIP_EXISTING=true
TRT_ENGINE_PATH=auto
TRT_ENGINE_SIZE=auto
VIDEO_INPUT_MODE=cuda-p010
VIDEO_OUTPUT_MODE=cuda-nvenc
VIDEO_POSTPROCESS_MODE=engine-output
VIDEO_GOP_SIZE=60
GPU_ID=0
```

720p 性能线通常使用：

```text
TRT_ENGINE_SIZE=960x540
VIDEO_POSTPROCESS_MODE=srvgg-conv48-tail
```

## 8. 多 GPU 策略

单个容器默认绑定一张 GPU。多卡并发建议按目录或任务分片运行多个容器：

```bash
docker run --rm \
  --device nvidia.com/gpu=1 \
  -e GPU_ID=0 \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  video2x:0.3.0
```

容器内通常只看见一张 GPU，因此 `GPU_ID=0` 表示容器内可见设备。NVENC 多卡场景依赖 CDI 设备、CUDA primary context 和 NVENC hook 保持一致。

## 9. 错误处理

必须失败退出的情况：

- 输入目录不存在。
- GPU 不可用。
- `/models` 缺少可用 TensorRT engine。
- 输出文件生成失败。
- 输出不是目标高度或 `ffprobe` 不可读。

可以跳过的情况：

- 输入文件不可读。
- 已存在输出文件。
- 输入高度已经大于等于 1080p。
- 没有匹配 engine 且选择规则判定不应强行变形。

## 10. 下一阶段

0.3.0 已完成发布主线。下一阶段重点：

- 默认策略收敛：`performance` / `quality` profile 自动化。
- TensorRT Plugin：融合 PixelShuffle + Downsample，减少 720p direct 的 x4 中间 tensor。
- C++ runner：统一 FFmpeg、TensorRT、CUDA stream/event 和 NVENC，降低 Python/ctypes 调度开销。
- FlashVSR：作为 P0 竞品验证线，用真实 420p/720p 样本对比速度、画质和时序一致性。
