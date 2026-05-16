# GPU Zero-Copy 与 NVENC 技术设计

## 1. 背景与目标

本项目的核心目标是把 420p/720p、30fps 视频 AI 超分到标准 `1920x1080`，并在 RTX 4090 上尽量达到或超过实时处理速度。

当前已经证明，单纯依赖 Python/PyTorch 逐帧推理无法满足目标。性能主线必须从“只优化模型”升级为“模型推理 + 视频管线”整体优化：

```text
输入视频 -> 解码 -> 预处理 -> TensorRT 推理 -> 后处理 -> 编码 -> 合并音频 -> 输出 MP4
```

本设计文档记录 GPU 直通、NVENC hook、CUDA 后处理、NV12 输出和后续真正 Zero-Copy 的技术路线。

## 2. 当前结论

已验证的高性能主线：

```text
ffmpeg 解码
-> CUDA 预处理
-> TensorRT FP16 推理
-> CUDA 后处理到 1920x1080
-> CUDA 直接生成 NV12
-> ffmpeg stdin
-> hevc_nvenc / libx265 编码
-> 合并音频
```

它还不是真正的 GPU surface 到 NVENC Zero-Copy，因为当前 NV12 帧仍会从显存复制回主机内存，再写入 ffmpeg stdin。

但相比 RGB 输出，它已经消除了两个关键浪费：

- 不再把 `1920x1080 RGB24` 大帧写给 ffmpeg。
- 不再让 ffmpeg/CPU 做 RGB 到 NV12 的色彩转换。

## 3. 已验证数据

测试条件：

- 单卡 RTX 4090。
- 输出统一为 `1920x1080`、`30fps`。
- TensorRT FP16 engine。
- 模型主线为 `realesr-general-x4v3`。
- 编码目标为 HEVC 约 `5Mbps`。

### 3.1 稳定镜像基线

| 输入 | 编码       | 端到端速度   | 5 分钟样本耗时 | 输出                 |
| ---- | ---------- | ------------ | -------------- | -------------------- |
| 420p | libx265 5M | `52.623 fps` | `173.099s`     | HEVC / 1080p / 30fps |
| 720p | libx265 5M | `37.221 fps` | `244.726s`     | HEVC / 1080p / 30fps |

当前基线已经超过 30fps，但仍有两个问题：

- x265 是 CPU 编码，会争抢 CPU 资源。
- 输出帧数曾出现 2 帧偏差，需要继续修复音画同步与帧数严格一致性。

### 3.2 NVENC Hook 验证

容器内直接使用 NVENC 曾出现 session 打不开的问题。引入 Desktop 项目的 `nvenc_ioctl_hook.so` 后，验证结果如下：

- `hevc_nvenc` session 可以打开。
- `testsrc2 -> hevc_nvenc` 成功。
- `rawvideo stdin -> hevc_nvenc` 成功。
- video2x 可以输出 HEVC / 1080p / 5M 文件。

结论：NVENC session 问题不是模型问题，而是容器内 GPU/NVENC 设备访问问题。Hook 路线有效。

### 3.3 RGB + NVENC 问题

最初 video2x 使用 RGB rawvideo 写入 ffmpeg stdin，再由 ffmpeg 转 NV12 后交给 NVENC。

| 输入 | 路线        | 300 帧速度   | encode/write |
| ---- | ----------- | ------------ | ------------ |
| 420p | RGB + NVENC | `28.829 fps` | `3.688s`     |
| 720p | RGB + NVENC | `27.399 fps` | `3.865s`     |

结论：NVENC session 已经可用，但 RGB 大帧传输和 CPU 色彩转换把速度吃掉了。

### 3.4 CUDA NV12 + NVENC

改造为 CUDA kernel 直接输出 NV12 后：

```text
TRT output CHW FP32 -> CUDA kernel -> 1920x1080 NV12 -> ffmpeg rawvideo nv12 -> hevc_nvenc
```

| 输入 | 路线         | 300 帧速度   | encode/write |
| ---- | ------------ | ------------ | ------------ |
| 420p | NV12 + NVENC | `57.931 fps` | `1.405s`     |
| 720p | NV12 + NVENC | `39.906 fps` | `1.324s`     |

结论：

- NV12 输出让 NVENC 路线重新超过 30fps。
- `encode/write` 从约 `3.7-3.9s / 300 frames` 降到约 `1.3-1.4s / 300 frames`。
- 这一步不是最终 Zero-Copy，但已经证明“大图拷贝”和“CPU 色彩转换”是必须消除的核心瓶颈。

### 3.5 完整样本回归

完整 5 分钟样本回归结果：

| 输入 | 路线         | 端到端速度   | 输出帧数 | 输出码率   | 输出                 |
| ---- | ------------ | ------------ | -------- | ---------- | -------------------- |
| 420p | NV12 + NVENC | `72.039 fps` | `9103`   | `5.01Mbps` | HEVC / 1080p / 30fps |
| 720p | NV12 + NVENC | `46.431 fps` | `9103`   | `5.01Mbps` | HEVC / 1080p / 30fps |

阶段判断：

- NV12 + NVENC 已通过 420p/720p 两个完整 5 分钟样本。
- 速度已经超过当前 `libx265 + 5M` 稳定路线。
- 最终 MP4 的 ffprobe 校验结果符合 `1920x1080`、`30fps`、HEVC、约 `5Mbps`。
- 人工画质评审结论为“尚可”，可以进入工程化阶段继续推进。
- runner 已按源视频探测帧数停止，日志帧数与最终封装帧数对齐。

## 4. GPU 直通设计

容器内需要稳定获得三类能力：

- `compute`：TensorRT / CUDA 推理。
- `video`：NVENC / NVDEC。
- `utility`：`nvidia-smi`、诊断和基础运行时能力。

GPU 直通目标：

```text
容器 -> NVIDIA runtime / CDI -> GPU compute + video capability -> CUDA / TensorRT / NVENC
```

Review Checkpoint：

- [x] 容器能看到目标 GPU。
- [x] 容器内能运行 CUDA / TensorRT。
- [x] 容器内能看到 `hevc_nvenc` 和 `h264_nvenc`。
- [x] `hevc_nvenc` 可以成功打开 session。
- [x] NVENC hook 路线能在目标服务器稳定复现。
- [ ] 不混用多套 GPU 暴露方式，避免设备 index 和权限错乱。

## 5. 当前视频管线

当前高性能工程路线：

```text
MP4 input
  |
  v
ffmpeg decode to raw frame
  |
  v
CUDA preprocess
  |
  v
TensorRT FP16 engine
  |
  v
CUDA postprocess + resize/pad to 1920x1080
  |
  v
CUDA RGB/NV12 output kernel
  |
  v
ffmpeg encoder
  |
  v
audio merge
  |
  v
MP4 output
```

其中 720p 路线采用预缩策略：

```text
1280x720 -> 960x540 -> x4 TensorRT -> 3840x2160 -> CUDA resize to 1920x1080
```

该策略避免直接 x4 到 `5120x2880` 的巨大中间帧，换取端到端速度超过 30fps。

## 6. 真正 Zero-Copy 目标架构

最终目标不是把 NV12 从显存复制回 CPU 后再喂给 ffmpeg，而是让帧尽可能留在 GPU 内部：

```text
NVDEC
  |
  v
CUDA surface / GPU memory
  |
  v
NPP / CUDA preprocess
  |
  v
TensorRT FP16
  |
  v
CUDA / NPP postprocess to NV12
  |
  v
NVENC
  |
  v
MP4 mux
```

关键变化：

- 解码使用 NVDEC，减少 CPU 解码压力。
- 解码后的帧以 GPU surface 或 CUDA device memory 形式进入预处理。
- TensorRT 输入输出绑定 GPU buffer。
- 后处理直接生成 NVENC 可接受的 NV12/P010 surface。
- 编码使用 NVENC，避免 CPU x265。
- CPU 只负责调度、日志、音频、mux，不参与大图像素级搬运。

## 7. 分阶段推进

### 7.1 阶段 A：稳定 NV12 + NVENC

目标：把当前短样本成功路线扩展到完整 5 分钟样本。

Review Checkpoint：

- [x] 420p 完整 5 分钟样本 NV12 + NVENC 跑通。
- [x] 720p 完整 5 分钟样本 NV12 + NVENC 跑通。
- [x] 输出 `1920x1080`、`30fps`、HEVC、约 `5Mbps`。
- [x] 最终 MP4 帧数、时长、音频通过 ffprobe 校验。
- [x] 人工画质评审通过，结论为“尚可”。
- [x] 和 `libx265 + 5M` 做速度、体积、基础规格对比。
- [ ] 补充更细的抽帧画质记录，包括人脸、字幕边缘、暗部噪声和运动闪烁。
- [x] 确认完整样本没有沿用 300 帧短测结论。
- [x] 解释并修复 runner raw frame 计数与最终封装帧数的差异。

成功标准：

- 420p 稳定超过 `50fps`。
- 720p 稳定超过 `35fps`。
- 不出现明显色彩偏移、字幕振铃、人脸蜡像化或音画不同步。
- 输出帧数必须与输入一致，除非有明确、可解释、可复现的封装原因。

当前状态：

- 420p 完整样本 `72.039 fps`，速度达标。
- 720p 完整样本 `46.431 fps`，速度达标。
- 人工画质评审结论为“尚可”，允许进入 hook 工程化。
- 下一步进入启动自检验证和真正 Zero-Copy 方案推进。

### 7.2 阶段 B：NVENC Hook 工程化

目标：将 hook 作为镜像内主线资产，并在启动时完成显式自检。

Review Checkpoint：

- [x] 明确 hook 文件来源和版本。
- [x] hook 已进入镜像构建上下文。
- [ ] 使用流水线镜像验证 hook 与当前基础镜像 glibc / driver / ffmpeg 兼容。
- [ ] 验证不同 GPU index 下行为一致。
- [x] 失败时显式报错，并输出 NVENC/hook/GPU 直通诊断信息。
- [x] 文档说明 hook 是高性能主线依赖，需要明确安装和验证方式。

成功标准：

- hook 可重复启用。
- NVENC session 错误可诊断。
- 不启用 hook 时直接失败并提示缺失条件，不做静默降级。

### 7.3 阶段 C：减少 Host 往返

目标：进一步降低当前 NV12 D2H + stdin 的残余成本。

候选路线：

- 使用 Video Codec SDK 直接调用 NVENC。
- 使用 GStreamer CUDA/NVDEC/NVENC 元件构建 GPU 内存管线。
- 使用 FFmpeg CUDA hwframes 管线验证是否能接入自定义 CUDA buffer。
- 使用 NPP 完成 resize、colorspace、layout 转换。

Review Checkpoint：

- [ ] 明确当前每帧 D2H 数据量。
- [ ] 建立纯编码吞吐基准。
- [ ] 建立 NVDEC 解码吞吐基准。
- [ ] 建立 GPU surface 到 NVENC 的最小 demo。
- [ ] 比较 GStreamer 与 Video Codec SDK 的工程复杂度。

成功标准：

- 端到端速度比 NV12 + stdin 路线继续提升。
- CPU 占用下降。
- 帧数和音画同步不倒退。

## 8. 风险

### 8.1 画质风险

720p 预缩到 540p 再 x4 虽然速度达标，但可能损失细节。

必须对比：

- 人脸皮肤纹理。
- 字幕边缘。
- 高频纹理。
- 暗部噪声。
- 运动画面时序闪烁。

### 8.2 同步风险

最终验收必须严格保证：

- 帧率一致。
- 时长一致。
- 音频存在。
- 音画同步。
- 输出帧数不能无故缺帧。

### 8.3 Hook 风险

NVENC hook 属于容器 GPU 访问修正手段，必须防止它变成不可诊断的黑盒依赖。

需要保留：

- 最小 NVENC session 测试。
- rawvideo stdin 编码测试。
- video2x 短样本测试。
- 失败诊断方案。

### 8.4 真 Zero-Copy 工程风险

真正 Zero-Copy 可能需要从当前 Python 主导流程转向 C++/CUDA/GStreamer/Video Codec SDK 混合架构。

风险包括：

- 开发复杂度显著上升。
- 调试难度上升。
- 容器驱动兼容要求更严格。
- 音频 mux 和异常恢复需要重新设计。

## 9. 当前判断

短期主线：

```text
realesr-general-x4v3 + TensorRT FP16 + CUDA NV12 + HEVC NVENC
```

这是当前最值得继续推进的速度路线。

长期极限路线：

```text
NVDEC -> CUDA/NPP -> TensorRT -> CUDA/NPP NV12 -> NVENC
```

这才是真正意义上的 Zero-Copy 视频超分架构，也是后续突破更高吞吐和更低 CPU 占用的方向。
