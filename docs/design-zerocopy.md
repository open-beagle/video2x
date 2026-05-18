# GPU Zero-Copy 与 NVENC 技术设计

## 1. 背景与目标

本项目的核心目标是把 420p/720p、30fps 视频 AI 超分到标准 `1920x1080`，并在 RTX 4090 上尽量达到或超过实时处理速度。

当前已经证明，单纯依赖 Python/PyTorch 逐帧推理无法满足目标。性能主线必须从“只优化模型”升级为“模型推理 + 视频管线”整体优化：

```text
输入视频 -> 解码 -> 预处理 -> TensorRT 推理 -> 后处理 -> 编码 -> 合并音频 -> 输出 MP4
```

本设计文档记录 GPU 直通、NVENC hook、CUDA 后处理、NV12 输出和 0.3.0 ZeroCopy 发布验收。

## 2. 当前结论

截至 0.3.0，**视频像素链路的 Zero-Copy 已经打通，并已固化到正式 runtime 镜像**。

当前已验证的业务路线是：

```text
FFmpeg C API / NVDEC
-> AV_PIX_FMT_CUDA / P010 surface
-> CUDA p010_to_chw_half
-> TensorRT FP16 推理
-> CUDA 后处理到 FFmpeg CUDA/NV12 AVFrame
-> hevc_nvenc 编码
-> faststart MP4 + 音频
```

代码证据见 `src/worker.py`：

- `VIDEO_INPUT_MODE=cuda-p010` 时，解码不再走 `ffmpeg stdout RGB`，而是通过 `libffmpeg_cuda_chw_bridge.so` 读取 CUDA/P010 surface。
- 输入侧不再有 CPU raw RGB、raw RGB H2D，也不再调用 `rgb8_to_chw_*` 预处理。
- `VIDEO_OUTPUT_MODE=cuda-nvenc` 时，后处理 kernel 直接写入 FFmpeg 分配的 CUDA/NV12 AVFrame。
- 输出侧不再执行每帧 `cudaMemcpyDeviceToHost`，也不再通过 `ffmpeg stdin` 喂 rawvideo。
- `PIPELINE_DEPTH=2` 双缓冲已启用，减少逐段同步等待。
- `VIDEO_GOP_SIZE=60` 与 `+faststart` 已启用，输出 MP4 可快速 seek。

因此当前视频像素主线已经是：

```text
NVDEC CUDA surface -> CUDA/TRT -> NVENC CUDA surface
-> 合并音频
```

仍需明确边界：

- 音频仍通过最终 `ffmpeg -c copy` 合并；这不涉及视频像素搬运。
- TensorRT full x4 engine 仍会产生完整 x4 输出 tensor，720p direct 仍有 `5120x2880` 中间图；PixelShuffle + Downsample Plugin 尚未实现。
- 0.3.0 正式镜像已固化 ZeroCopy input/output；未完成的是 TensorRT engine 内部的算子级融合。

当前阶段可以命名为：

```text
NVDEC CUDA/P010 -> TensorRT FP16 -> CUDA/NV12 AVFrame -> NVENC
```

正式 runtime 镜像验收结果：

| 输入 | 模式 | fps | 输出 |
| ---- | ---- | --- | ---- |
| 420p | ZeroCopy input + output surface | `142.033` | `1920x1080 / 9103 frames / HEVC / AAC` |
| 720p 性能线 | `960x540 conv48 ZeroCopy` | `77.106` | `1920x1080 / 9103 frames / HEVC / AAC` |
| 720p 质量线 | `1280x720 conv48 ZeroCopy` | `45.124` | `1920x1080 / 9103 frames / HEVC / AAC` |

阶段判断：

- 视频像素 Zero-Copy 已完成。
- 720p direct 当前尚未达到 `60fps`；已验证完整样本最好结果约 `45fps`。
- `960x540` 性能线已达到 `60fps+`，但这是 720p 预缩后的速度路线，不是 720p direct 质量路线。
- `srvgg-conv48-tail` 已成为 720p 性能线/质量线的当前发布路线。
- `conv48-tail` 的价值是避免最终 RGB x4 大图输出 binding，并为后续 TensorRT Plugin 提供更好的工程边界；真正根治仍需要算子级融合。

当前剩余性能空间：

- TensorRT Plugin：融合 `PixelShuffle + Downsample`，避免完整 `5120x2880` x4 中间图写回。
- 更低开销的 C++ runner：减少 Python 调度与 ctypes 边界。
- 多 GPU CDI 场景下继续确认 CUDA primary context、NVENC hook 和 `/dev/nvidia0` alias 的一致性。

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

### 3.6 当前正式镜像复核

正式 `0.3.0` 瘦身镜像已验证：

```text
registry.cn-qingdao.aliyuncs.com/wod/video2x:0.3.0
Size: 3.66GB
TensorRT: 10.16.1.11
```

GPU1 回归确认：

```text
--device nvidia.com/gpu=1
GPU_ID=0
VIDEO_ENCODER=hevc_nvenc
VIDEO_PIXEL_FORMAT=nv12

420p 完整样本：
frames=9103
elapsed=151.654s
fps=60.025
```

420p + 720p 并发处理复核，优化前旧数据：

| 输入 | GPU | Engine            | 耗时       | fps      | 输出                      |
| ---- | --- | ----------------- | ---------- | -------- | ------------------------- |
| 420p | 0   | `720x420`         | `150.937s` | `60.310` | `1920x1080 / 9103 frames` |
| 720p | 1   | `1280x720` direct | `394.293s` | `23.087` | `1920x1080 / 9103 frames` |

历史判断：

- 720p 并发复核走的是 `1280x720` direct engine，会产生 `5120x2880` x4 中间帧，因此低于 30fps。
- 之前 `46.431 fps` 的 720p 性能数据来自 `960x540` 性能 profile，不应和 direct engine 混用结论。
- 该结论已被后续 pipeline2 优化更新。当前 720p direct 并发完整样本为 `45.140 fps`，可以作为质量主线。

### 3.7 CUDA 预处理热更新验证

为释放 4090 性能，已将 CPU NumPy 预处理：

```text
np.frombuffer -> astype(float32) -> /255 -> HWC to CHW transpose
```

替换为：

```text
H2D raw RGB -> CUDA rgb8_to_chw_float / rgb8_to_chw_half
```

720p direct 300 帧热更新验证：

```text
input=1280x720
engine=realesr-general-x4v3-1280x720-fp16.engine
encoder=hevc_nvenc
pix_fmt=nv12
frames=300
elapsed=9.224s
fps=32.525
decode=0.810s
preprocess=0.030s
h2d=0.145s
infer=6.515s
kernel=0.049s
d2h_frame=0.184s
encode_write=1.257s
merge_audio=0.079s
```

结论：

- CPU NumPy 预处理瓶颈已经被移出主线。
- 720p direct 300 帧已经超过实时 30fps。
- 当前最大耗时重新回到 TensorRT x4 推理本身，下一刀应推进 PixelShuffle + Downsample 融合或异步流水。

### 3.8 Pinned Host Buffer 验证

在 CUDA 预处理基础上，继续把 host 往返缓冲改成 page-locked pinned memory：

```text
decoder stdout -> readinto(pinned raw RGB)
pinned raw RGB -> H2D
TRT/CUDA
D2H -> pinned NV12
encoder stdin.write(memoryview)
```

这一步仍不是真 Zero-Copy，因为数据仍经过 host；但它减少了 Python 每帧 `bytes` / `tobytes()` 副本，并让 H2D/D2H 具备真正异步流水的前置条件。

720p direct 300 帧热更新验证：

```text
frames=300
elapsed=8.112s
fps=36.983
decode=0.497s
preprocess=0.029s
h2d=0.053s
infer=6.524s
kernel=0.048s
d2h_frame=0.047s
encode_write=0.726s
merge_audio=0.068s
```

对比 CUDA 预处理但未使用 pinned buffer 的短测：

```text
fps:          32.525 -> 36.983
h2d:          0.145s -> 0.053s
d2h_frame:    0.184s -> 0.047s
encode_write: 1.257s -> 0.726s
```

结论：

- Pinned host buffer 对当前 stdin/rawvideo 路线仍有明显收益。
- H2D/D2H 已经从可见瓶颈变成小项。
- 下一阶段不应继续在 Python host copy 上小修小补，应进入双缓冲/三缓冲异步流水或真正 surface 接入。

### 3.9 双缓冲异步流水验证

在 pinned buffer 基础上，继续把单帧串行链路：

```text
read -> H2D -> preprocess -> TRT -> postprocess -> D2H -> write
```

改成按 slot 轮转的异步流水：

```text
slot0: H2D/preprocess/TRT/postprocess/D2H
slot1: read next frame / enqueue next GPU work
finish oldest slot -> write encoder
```

实现约束：

- `PIPELINE_DEPTH=2` 默认启用。
- 每个 slot 独立拥有 pinned input/output、CUDA stream、TensorRT execution context、device buffers。
- 输出按 slot 轮转等待，保持帧顺序。
- 日志新增 `sync_wait`。异步后 `infer` 只代表 enqueue 时间，不能再当作 GPU 实际推理耗时；端到端 `fps` 和 `sync_wait` 才是主要观察项。

720p direct 300 帧热更新验证：

```text
frames=300
elapsed=7.012s
fps=42.785
decode=0.585s
preprocess=0.009s
h2d=0.017s
infer=0.271s
kernel=0.004s
d2h_frame=0.007s
sync_wait=5.140s
encode_write=0.787s
merge_audio=0.076s
```

对比 pinned buffer 但未双缓冲的短测：

```text
fps: 36.983 -> 42.785
```

结论：

- 双缓冲让 decode/read、GPU enqueue、D2H 和 encode write 形成了初步重叠。
- 720p direct 短测已经明显超过实时 30fps。
- 当前真正大头仍是 TensorRT x4 graph 本身，下一阶段应验证完整 5 分钟 direct 样本，并推进 PixelShuffle + Downsample 融合或 TensorRT Plugin。

完整 5 分钟 720p direct 验证：

```text
frames=9103
elapsed=206.125s
fps=44.162
decode=13.252s
preprocess=0.221s
h2d=0.448s
infer=6.693s
kernel=0.105s
d2h_frame=0.178s
sync_wait=164.934s
encode_write=18.913s
merge_audio=0.258s
```

输出校验：

```text
codec=hevc
width=1920
height=1080
avg_frame_rate=30/1
nb_frames=9103
video_bit_rate=5008477
audio=aac / 44100Hz / stereo
duration=303.433333s
size=195130386
```

该结果说明 `PIPELINE_DEPTH=2` 不只是短测有效，完整 5 分钟 direct 样本也通过了帧数、时长、音频和封装校验。

并发完整样本验证：

```text
420p: GPU0, /data/jasna/420p/SDMT-506-U-420p_1080p_pipeline2_parallel.mp4
720p: GPU1, /data/jasna/720p/INU-047-U-720p_1080p_pipeline2_parallel.mp4
```

结果：

| 输入 | GPU | Engine            | 耗时       | fps       | 输出                                   |
| ---- | --- | ----------------- | ---------- | --------- | -------------------------------------- |
| 420p | 0   | `720x420`         | `68.709s`  | `132.487` | `1920x1080 / 9103 frames / HEVC / AAC` |
| 720p | 1   | `1280x720` direct | `201.663s` | `45.140`  | `1920x1080 / 9103 frames / HEVC / AAC` |

注意：

- 人工质检输出必须落在业务挂载目录，也就是容器 `/data` 对应的宿主目录。
- 临时 `/tmp/video2x-*` 目录只用于隔离实验，不作为质检交付位置。

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
- [x] 不混用多套 GPU 暴露方式，使用 CDI `--device nvidia.com/gpu=N`。
- [x] 多 GPU CDI 下不再固定 `NVENC_GPU_INDEX=0`。
- [x] 启动时自动为 CDI 单卡创建 `/dev/nvidia0 -> /dev/nvidia<N>` alias，让 hook 推导物理 GPU。

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

720p 有两条路线，必须区分：

```text
性能路线：
1280x720 -> 960x540 -> x4 TensorRT -> 3840x2160 -> CUDA resize to 1920x1080

质量/direct 路线：
1280x720 -> x4 TensorRT -> 5120x2880 -> CUDA resize to 1920x1080
```

历史上 direct 路线因为 `5120x2880` x4 中间帧只有 `23.087 fps`。经过 CUDA 预处理、pinned host buffer 和 `PIPELINE_DEPTH=2` 双缓冲后，720p direct 并发完整样本已提升到 `45.140 fps`。因此 `960x540` 性能路线只作为兼容/降载路线，720p 质量主线可以回到 `1280x720` direct engine。

TensorRT engine 的命名、标准分辨率 profile、非标准输入选择、build 镜像和重建策略独立记录在 `docs/design-engine.md`。Zero-Copy 设计只关心选定 engine 之后的视频管线。

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

- 420p pipeline2 并发完整样本 `132.487 fps`，速度达标。
- 720p direct pipeline2 并发完整样本 `45.140 fps`，速度达标。
- 人工画质评审结论为“尚可”，当前输出已具备继续做正式镜像回归的条件。
- hook、CUDA preprocess、pinned buffer、双缓冲、GOP/faststart 均已进入当前主线。
- 下一步是用正式流水线镜像固化当前热更新结果，再推进真正 Zero-Copy / Plugin 方案。

### 7.2 阶段 B：NVENC Hook 工程化

目标：将 hook 作为镜像内主线资产，并在启动时完成显式自检。

Review Checkpoint：

- [x] 明确 hook 文件来源和版本。
- [x] hook 已进入镜像构建上下文。
- [x] 使用流水线镜像验证 hook 与当前基础镜像 glibc / driver / ffmpeg 兼容。
- [x] 验证 GPU0 / GPU1 CDI 场景可用。
- [x] 失败时显式报错，并输出 NVENC/hook/GPU 直通诊断信息。
- [x] 文档说明 hook 是高性能主线依赖，需要明确安装和验证方式。

成功标准：

- hook 可重复启用。
- NVENC session 错误可诊断。
- 不启用 hook 时直接失败并提示缺失条件，不做静默降级。

当前实现细节：

- Dockerfile 不再固定 `NVENC_GPU_INDEX=0`。
- entrypoint 在未显式传 `NVENC_GPU_INDEX` 时扫描 `/dev/nvidia<N>`。
- 当 CDI 只注入 `/dev/nvidia1`、`/dev/nvidia2` 等非 0 节点时，entrypoint 会创建 `/dev/nvidia0 -> /dev/nvidia<N>`。
- hook 通过 `/dev/nvidia0` 的真实目标推导物理 GPU。
- GPU 状态探测调用 `nvidia-smi` 时会移除 `LD_PRELOAD`，避免 hook 影响监控。

### 7.3 阶段 C：FFmpeg CUDA Surface 探针

目标：先确认当前 runtime 镜像是否具备纯视频 GPU surface 闭环能力，再决定 TensorRT 如何接入。

已新增探针：

```text
tools/debug/probe_zerocopy_ffmpeg.sh
```

该探针验证：

- `ffmpeg -hwaccels` 是否支持 `cuda`。
- `ffmpeg -filters` 是否存在 `scale_cuda`、`hwupload_cuda`、`hwdownload` 等 CUDA/hwframes 组件。
- `ffmpeg -encoders` 是否存在 `hevc_nvenc` / `h264_nvenc`。
- `ffmpeg -decoders` 是否存在 `hevc_cuvid` / `h264_cuvid`。
- 最小闭环是否能跑通：

```text
NVDEC / CUDA hwframe -> scale_cuda -> hevc_nvenc
```

当前正式 `0.3.0` 镜像探针结果：

```text
cuda hwaccel: yes
scale_cuda: yes
colorspace_cuda: yes
hevc_nvenc: yes
h264_nvenc: yes
hevc_cuvid: yes
h264_cuvid: yes
```

720p 样本 300 帧 smoke test：

```text
ffmpeg -hwaccel cuda -hwaccel_output_format cuda
-> scale_cuda=1920:1080:format=nv12
-> hevc_nvenc

frames=300
speed=1.23x
约 37 fps
输出=1920x1080 / 30fps / hevc / 300 frames
```

FFmpeg 输出流显示为：

```text
Video: hevc, cuda(...), 1920x1080
```

结论：

- 当前镜像的 FFmpeg 具备 CUDA hwframes 基础能力。
- `NVDEC -> CUDA filter -> NVENC` 这条纯视频 GPU surface 路线已经最小跑通。
- 这不是业务 Zero-Copy，因为它没有经过 TensorRT 超分，只是证明 FFmpeg 侧不是死路。

补充探针：

```text
src/native/ffmpeg_cuda_chw_bridge.cu
.tmp/probes/probe_ffmpeg_cuda_hwframe.c
.tmp/probes/probe_ffmpeg_cuda_p010_to_rgb.cu
.tmp/probes/probe_ffmpeg_cuda_p010_to_chw.cu
.tmp/probes/probe_trt_with_ffmpeg_cuda_input.py
.tmp/probes/probe_trt_ffmpeg_cuda_loop.py
```

其中 `src/native/ffmpeg_cuda_chw_bridge.cu` 已从探针升级为 runtime native 代码；`.tmp/probes/` 下文件只保留实验记录，不进入镜像和发布主线。

这些探针验证了 FFmpeg C API 下的 CUDA hwframe 能被业务 CUDA 代码直接访问：

```text
AV_PIX_FMT_CUDA / p010le
data0 = Y plane CUDA device pointer
data1 = UV plane CUDA device pointer
```

关键发现：

- 720p 样本解码输出为 `format=cuda`、`sw_format=p010le`、`linesize=2560/2560`。
- 420p 样本解码输出为 `format=cuda`、`sw_format=p010le`、`linesize=1536/1536`。
- 直接 `cuMemcpyDtoH(frame->data[0])` 会失败，因为 CUDA current context 不是 FFmpeg 创建的 context。
- 从 `AVHWFramesContext -> AVCUDADeviceContext` 取出 `cuda_ctx` 并 `cuCtxPushCurrent` 后，可以直接读取 `frame->data[0]`。
- 自定义 CUDA kernel 已能直接读取 P010 CUDA surface，完成 `P010 -> RGB` 小块采样。
- 自定义 CUDA kernel 已能直接完成 `P010 -> NCHW FP16`，输出大小符合 TensorRT input：

```text
720p: 1280 * 720 * 3 * 2 = 5,529,600 bytes
420p: 720 * 420 * 3 * 2 = 1,814,400 bytes
```

runtime `0.3.0` 探针结果：

```text
720p:
frame=0 format=cuda sw_format=p010le width=1280 height=720 linesize=2560/2560
chw_probe dtype=fp16 bytes=5529600 checksum=3569448592

420p:
frame=0 format=cuda sw_format=p010le width=720 height=420 linesize=1536/1536
chw_probe dtype=fp16 bytes=1814400 checksum=429226993
```

历史探针结论：

- Zero-Copy 输入半边的关键风险已经降低：`NVDEC CUDA surface -> CUDA preprocess -> TensorRT input buffer` 有可行路径。
- Python TensorRT binding 已经能接收该 bridge 写入的 device buffer 并完成 `execute_async_v3`。
- 当时下一步不是继续从 ffmpeg stdout 读 RGB，而是把 bridge 从“第一帧探针”扩展成持续帧循环，并接入现有 CUDA postprocess。
- 这段记录的是输入侧探针阶段；0.3.0 正式 runtime 已经完成输出侧 `TRT/CUDA -> NVENC surface` 接入。

TensorRT 输入半边最小闭环：

```text
FFmpeg C API NVDEC
-> AV_PIX_FMT_CUDA / P010 surface
-> CUDA kernel P010 to NCHW FP16
-> TensorRT input binding
-> TensorRT execute_async_v3
```

runtime `0.3.0` 结果：

```text
720p:
input_shape=(1, 3, 720, 1280) input_dtype=HALF input_bytes=5529600
output_shape=(1, 3, 2880, 5120) output_dtype=HALF output_bytes=88473600
decoded_cuda_frames=1
trt_execute=ok output_sample_bytes=4096 checksum=2787802402

420p:
input_shape=(1, 3, 420, 720) input_dtype=HALF input_bytes=1814400
output_shape=(1, 3, 1680, 2880) output_dtype=HALF output_bytes=29030400
decoded_cuda_frames=1
trt_execute=ok output_sample_bytes=4096 checksum=2769816846
```

这证明输入侧已经不需要：

```text
ffmpeg stdout RGB
CPU raw frame
H2D raw RGB
rgb8_to_chw_* preprocess
```

后续工程化时，输入侧应收敛为：

```text
NVDEC CUDA hwframe -> p010_to_chw_half -> TRT input binding
```

连续帧输入侧闭环：

```text
FFmpeg C API NVDEC
-> AV_PIX_FMT_CUDA / P010 surface
-> p010_to_chw_half
-> TensorRT execute
-> CUDA NV12 postprocess
-> 小样本 D2H checksum
```

这条 probe 仍未接 NVENC surface，也没有生成视频；它用于验证输入侧替换后可以连续跑多帧。

300 帧结果：

```text
720p direct:
frames=300
elapsed=8.389s
fps=35.761
decode_to_trt_input=0.090s
infer_enqueue=0.172s
postprocess_sync=6.346s
sample_d2h=0.005s

420p:
frames=300
elapsed=2.424s
fps=123.754
decode_to_trt_input=0.077s
infer_enqueue=0.172s
postprocess_sync=1.823s
sample_d2h=0.005s
```

解释：

- `decode_to_trt_input` 已经包括 NVDEC 解码、P010 surface 读取和 `P010 -> NCHW FP16` kernel。
- 该项在 300 帧中只有 `0.09s` / `0.077s`，说明输入侧 host 往返消除后成本很低。
- 当前 probe 每帧同步 postprocess，没有做双缓冲，所以 720p 的 `35.761 fps` 不是最终上限。
- 该 bridge 后续已经接入业务 pipeline slot；0.3.0 正式主线为 NVDEC、TRT、CUDA/NV12 postprocess、NVENC 的异步流水。

### 7.3.1 业务 Runner 接入验证

已将输入侧 bridge 接入 `src/worker.py`，新增可切换输入模式：

```text
VIDEO_INPUT_MODE=rgb24      # 默认，原 ffmpeg stdout RGB 路线
VIDEO_INPUT_MODE=cuda-p010  # NVDEC CUDA hwframe 输入路线
```

相关参数：

```text
CUDA_P010_BRIDGE=/app/src/libffmpeg_cuda_chw_bridge.so
```

当前 `cuda-p010` 业务路线：

```text
FFmpeg C API NVDEC
-> AV_PIX_FMT_CUDA / P010 surface
-> CUDA p010_to_chw_half
-> TensorRT FP16
-> CUDA NV12 postprocess
-> D2H pinned NV12
-> ffmpeg stdin / hevc_nvenc
-> faststart MP4 + audio
```

这已经不再使用：

```text
ffmpeg stdout RGB
CPU raw RGB frame
H2D raw RGB
rgb8_to_chw_* preprocess
```

但它仍然保留：

```text
D2H pinned NV12
ffmpeg stdin
NVENC surface 未直连
```

这是输入半边接入时的历史状态；0.3.0 正式主线已经实现完整视频像素 ZeroCopy，剩余问题转为 TensorRT 内部 x4 中间 tensor 的算子级融合。

300 帧业务 MP4 验证：

```text
720p direct:
frames=300
elapsed=7.024s
fps=42.713
decode=0.184s
preprocess=0.000s
h2d=0.000s
infer=0.232s
kernel=0.008s
d2h_frame=0.011s
sync_wait=5.675s
encode_write=0.727s

420p:
frames=300
elapsed=2.491s
fps=120.441
decode=0.105s
preprocess=0.000s
h2d=0.000s
infer=0.177s
kernel=0.006s
d2h_frame=0.008s
sync_wait=1.295s
encode_write=0.720s
```

完整 5 分钟业务 MP4 验证：

```text
720p direct:
/data/jasna/720p/INU-047-U-720p_1080p_zcinput_full.mp4
frames=9103
elapsed=208.723s
fps=43.613
decode=4.720s
preprocess=0.000s
h2d=0.000s
infer=6.583s
kernel=0.250s
d2h_frame=0.359s
sync_wait=176.933s
encode_write=19.074s

420p:
/data/jasna/420p/SDMT-506-U-420p_1080p_zcinput_full.mp4
frames=9103
elapsed=67.646s
fps=134.569
decode=2.619s
preprocess=0.000s
h2d=0.000s
infer=5.114s
kernel=0.177s
d2h_frame=0.251s
sync_wait=39.785s
encode_write=18.191s
```

输出校验：

```text
codec=hevc
width=1920
height=1080
avg_frame_rate=30/1
nb_frames=9103
audio=aac / 44100Hz / stereo
```

阶段结论：

- 输入侧 ZeroCopy 已经进入业务 runner，并生成真实 MP4。
- 720p direct 保持实时以上：`43.613 fps`。
- 420p 达到 `134.569 fps`。
- 当前最大剩余 host 往返是 `D2H pinned NV12 -> ffmpeg stdin`。
- 下一阶段应推进输出侧：`CUDA NV12/P010 -> NVENC surface`，或者先用 C/C++ runner 把 NVDEC/TRT/NVENC 放入同一进程统一调度。

### 7.3.2 输出侧 NVENC Surface 接入验证

已将输出侧 bridge 接入 `src/worker.py`，新增可切换输出模式：

```text
VIDEO_OUTPUT_MODE=stdin       # 原 D2H pinned NV12 -> ffmpeg stdin 路线
VIDEO_OUTPUT_MODE=cuda-nvenc  # CUDA/NV12 AVFrame -> hevc_nvenc 路线
```

相关参数：

```text
CUDA_NVENC_BRIDGE=/app/src/libffmpeg_cuda_chw_bridge.so
```

当前完整视频像素路线：

```text
FFmpeg C API NVDEC
-> AV_PIX_FMT_CUDA / P010 surface
-> CUDA p010_to_chw_half
-> TensorRT FP16
-> CUDA postprocess 写入 FFmpeg CUDA/NV12 AVFrame
-> hevc_nvenc
-> faststart MP4
-> 音频 copy merge
```

这一步已经不再使用：

```text
D2H pinned NV12
ffmpeg stdin rawvideo
```

300 帧业务 MP4 验证：

```text
720p direct:
/data/jasna/720p/INU-047-U-720p_1080p_zcfull_benchmark.mp4
frames=300
elapsed=6.779s
fps=44.253
decode=0.163s
preprocess=0.000s
h2d=0.000s
infer=0.288s
kernel=0.010s
d2h_frame=0.000s
sync_wait=6.066s
encode_write=0.104s

420p:
/data/jasna/420p/SDMT-506-U-420p_1080p_zcfull_benchmark.mp4
frames=300
elapsed=2.195s
fps=136.698
decode=0.085s
preprocess=0.000s
h2d=0.000s
infer=0.166s
kernel=0.004s
d2h_frame=0.000s
sync_wait=1.764s
encode_write=0.053s
```

完整 5 分钟业务 MP4 验证：

```text
720p direct:
/data/jasna/720p/INU-047-U-720p_1080p_zcfull_full.mp4
frames=9103
elapsed=205.728s
fps=44.248
decode=3.746s
preprocess=0.000s
h2d=0.000s
infer=5.732s
kernel=0.190s
d2h_frame=0.000s
sync_wait=193.224s
encode_write=1.924s
merge_audio=0.415s

420p:
/data/jasna/420p/SDMT-506-U-420p_1080p_zcfull_full.mp4
frames=9103
elapsed=62.801s
fps=144.949
decode=2.145s
preprocess=0.000s
h2d=0.000s
infer=5.645s
kernel=0.165s
d2h_frame=0.000s
sync_wait=52.342s
encode_write=1.673s
merge_audio=0.403s
```

输出校验：

```text
codec=hevc
width=1920
height=1080
avg_frame_rate=30/1
nb_frames=9103
audio=aac / 44100Hz / stereo
```

阶段结论：

- 视频像素链路已经实现 `NVDEC CUDA surface -> TensorRT -> NVENC CUDA surface`。
- 720p direct 速度仍约 `44 fps`，说明当前瓶颈主要是 x4 TensorRT graph，而不是视频 I/O。
- 420p 从输入半边 ZeroCopy 的 `134.569 fps` 提升到输出 surface 路线的 `144.949 fps`。
- `encode_write` 从输入半边 ZeroCopy 的约 `18-19s / 9103 frames` 降到约 `1.7-1.9s / 9103 frames`。
- `src/ffprobe.py` 已改为默认不使用 `-count_frames`，避免扫描和验证阶段无意义地扫完整片。
- 该热修验证已经进入正式 `0.3.0` runtime 镜像；发布判断以后以正式镜像回归数据为准。

### 7.4 阶段 D：业务链路减少 Host 往返

当前未完成项：

```text
ffmpeg decode stdout -> CPU RGB
H2D raw RGB
CUDA preprocess
TRT/CUDA
D2H NV12 frame
ffmpeg stdin
NVENC
```

需要消除的 Host 往返：

- 解码后的 RGB 帧从 ffmpeg stdout 进入 CPU。
- 每帧 raw RGB 仍需要 `H2D` 输入拷贝。
- 每帧 `D2H` NV12 输出拷贝。
- ffmpeg stdin 再把主机内存帧交给 NVENC。

候选路线：

- **P0：CUDA preprocess 替换 NumPy**  
  已完成热更新验证。720p direct 300 帧 `preprocess=0.030s`，CPU `astype + transpose` 不再是主瓶颈。

- **P0：Pinned host buffer 减少 Python 拷贝**  
  已完成热更新验证。720p direct 300 帧 `fps=36.983`，H2D/D2H 和 encode/write 明显下降。它不是 Zero-Copy，但给异步流水打了基础。

- **P0：减少同步点，改成异步流水**  
  已完成双缓冲热更新验证。720p direct 300 帧 `fps=42.785`。当前默认 `PIPELINE_DEPTH=2`，可在环境变量中降为 `1` 做对照。

- **P0：FFmpeg CUDA hwframes 接入 TensorRT**  
  已完成。通过 `src/native/ffmpeg_cuda_chw_bridge.cu` 读取 `AV_PIX_FMT_CUDA / P010` surface，并直接写入 TensorRT input binding。

- **P0：CUDA/NV12 AVFrame 接入 NVENC**  
  已完成热修验证。通过同一个 bridge 分配 FFmpeg CUDA/NV12 frame，CUDA postprocess kernel 直接写入 frame plane，再交给 `hevc_nvenc`。

- **P1：GStreamer CUDA/NVDEC/NVENC 管线**  
  Desktop 项目已有 GStreamer/NVENC/hook 经验。可用 GStreamer 负责 NVDEC/NVENC 和 mux，业务侧用自定义 CUDA/TensorRT element 处理 GPU memory。

- **P1：Video Codec SDK C++ runner**  
  当前 Python + C bridge 已经打通视频像素 Zero-Copy。C++ runner 的价值转为减少 Python 调度、ctypes 边界和更精细的 CUDA stream / event 管理，而不是证明路线可行性。

- 使用 NPP 完成 resize、colorspace、layout 转换。

Review Checkpoint：

- [x] 明确当前每帧 D2H 数据量。
- [x] 确认 runtime FFmpeg 支持 CUDA hwaccel / CUDA filters / NVENC / CUVID。
- [x] 建立 `NVDEC/CUDA hwframe -> scale_cuda -> NVENC` 最小 demo。
- [x] 建立 CUDA/NV12 surface -> NVENC 最小 demo。
- [ ] 建立更完整的纯编码吞吐基准。
- [ ] 建立更完整的 NVDEC 解码吞吐基准。
- [x] 将 CUDA preprocess 接入当前 `src/worker.py`，替换 NumPy 预处理。
- [x] 使用 pinned host buffer 降低 H2D/D2H 和 stdin 写入成本。
- [x] 将当前逐段同步改成双缓冲异步流水。
- [x] 使用完整 5 分钟样本验证 `PIPELINE_DEPTH=2` 的帧数、音画同步和稳定性。
- [x] 建立 `CUDA hwframe -> P010 -> NCHW FP16` 最小 demo。
- [x] 建立 `CUDA hwframe -> TensorRT binding -> TensorRT execute` 最小 demo。
- [x] 建立 `CUDA hwframe -> TensorRT -> CUDA postprocess -> host NV12 checksum` 连续帧 demo。
- [x] 将连续帧 bridge 接入业务 runner，输出实际 MP4。
- [x] 将 CUDA/NV12 AVFrame writer 接入业务 runner，消除 D2H 和 stdin rawvideo。
- [ ] 比较 GStreamer 与 Video Codec SDK 的工程复杂度。

当前每帧 D2H 数据量：

```text
NV12 1920x1080 = 1920 * 1080 * 1.5 = 3,110,400 bytes/frame
约 2.97 MiB/frame
30fps 时约 93.3 MB/s
60fps 时约 186.6 MB/s
```

如果输出 RGB24：

```text
RGB24 1920x1080 = 1920 * 1080 * 3 = 6,220,800 bytes/frame
约 5.93 MiB/frame
```

当前 NV12 路线已经把 D2H 输出带宽减半，但没有消除 D2H。

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
- MP4 必须可快速拖拽播放。NVENC 输出不能只有首帧一个关键帧，默认使用 `VIDEO_GOP_SIZE=60`，并在编码和最终音频合并阶段启用 `-movflags +faststart`。

### 8.2.1 Seek 风险复盘

问题：

```text
原始视频拖拽进度条可以立即播放。
生成的 1080p 拖拽后无法立即播放。
```

诊断结果：

```text
原始 720p：37 个关键帧，约 8.33s 一个。
旧 1080p：1 个关键帧，全片只有开头 I 帧。
```

旧输出全片只有一个关键帧时，播放器拖到任意位置都只能从开头关键帧开始解码，表现就是拖动后长时间不播或像卡死。

修复：

```text
VIDEO_GOP_SIZE=60
hevc_nvenc -g 60 -forced-idr 1
编码阶段 -movflags +faststart
最终 merge audio 阶段 -movflags +faststart
```

验证文件：

```text
/data/jasna/720p/INU-047-U-720p_1080p_seekfix_full_faststart.mp4
```

验证结果：

```text
width=1920
height=1080
avg_frame_rate=30/1
nb_frames=9103
关键帧数量=152
关键帧间隔=2s
moov=32
mdat=260912
```

结论：

- GOP 修复解决 seek 关键帧问题。
- faststart 修复 MP4 索引前置问题。
- 后续所有质检输出都必须用该编码参数。

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

但必须明确：

```text
CUDA NV12 + HEVC NVENC != Zero-Copy
```

当前仍有 CPU 解码、H2D raw RGB、D2H、ffmpeg stdin 等 Host 往返。它是高性能过渡路线，不是最终显存闭环。

长期极限路线：

```text
NVDEC -> CUDA/NPP -> TensorRT -> CUDA/NPP NV12 -> NVENC
```

这才是真正意义上的 Zero-Copy 视频超分架构，也是后续突破更高吞吐和更低 CPU 占用的方向。

## 10. Review 结论

本轮 review 结论：

- [x] GPU CDI 与 NVENC hook 工程化已基本完成。
- [x] CUDA 后处理直接输出 NV12 已实现。
- [x] HEVC NVENC 输出已实现。
- [x] GPU0/GPU1 CDI 场景已验证。
- [x] 输出帧数、时长、分辨率已通过 ffprobe。
- [x] CPU NumPy 预处理已迁移到 CUDA kernel。
- [x] 当前 rawvideo stdin 路线已使用 pinned host buffer 降低拷贝成本。
- [x] 当前 rawvideo stdin 路线已引入双缓冲异步流水。
- [x] 720p direct 完整样本已恢复到实时以上。
- [x] MP4 seek 问题已通过 GOP 和 faststart 修复。
- [x] NVDEC CUDA hwframe 指针可由自定义 CUDA kernel 读取。
- [x] `P010 CUDA surface -> NCHW FP16` 输入预处理探针已跑通。
- [x] `P010 CUDA surface -> NCHW FP16 -> TensorRT execute` 单帧探针已跑通。
- [x] `NVDEC -> TRT -> CUDA NV12 postprocess` 连续 300 帧探针已跑通。
- [x] `NVDEC -> TRT -> CUDA NV12 -> ffmpeg stdin/NVENC` 业务 MP4 已跑通。
- [x] 业务 runner 可选 NVDEC CUDA hwframe 输入已实现。
- [x] 业务 runner 可选 CUDA/NVENC surface 输出已实现。
- [x] 视频像素链路已消除 CPU raw RGB、H2D raw RGB、D2H rawvideo 和 ffmpeg stdin。
- [x] 720p direct 完整样本 `44.248fps`，420p 完整样本 `144.949fps`。
- [ ] TensorRT x4 输出仍会产生完整 `5120x2880` 中间 tensor。
- [ ] PixelShuffle + Downsample TensorRT Plugin 尚未实现。

因此当前工程状态应定义为：

```text
已实现：NVDEC CUDA/P010 -> CUDA preprocess -> TensorRT FP16 -> CUDA/NV12 AVFrame -> NVENC
未实现：TensorRT 内部 PixelShuffle + Downsample 融合，仍会暴露 x4 大图 tensor
```

## 11. 下一阶段空间

当前视频像素 Zero-Copy 主线已经能打，继续优化要分清两类问题：

- 视频管线层面：NVDEC、TRT、NVENC 已经在 GPU surface / CUDA memory 内闭环。
- 算子层面：`realesr-general-x4v3` 仍然会在 TensorRT engine 输出完整 x4 tensor，这是 720p direct 的主要剩余瓶颈。

后续真正值得投入的方向如下：

| 优先级 | 方向                               | 目标                                                                   | 判断                                    |
| ------ | ---------------------------------- | ---------------------------------------------------------------------- | --------------------------------------- |
| P0     | 正式镜像回归                       | build/runtime 镜像完成后复跑 420p/720p、seek、GPU0/GPU1                | 0.3.0 已完成                            |
| P0     | 画质抽帧评审                       | 固定抽帧点，对比 direct 720p、性能线、原片                             | 性能线已人工接受，质量线保留            |
| P1     | SRVGG conv48 tail                  | TRT 保留 final conv，CUDA 融合 `DepthToSpace + residual + resize/NV12` | 完整样本已过 45fps，当前首选折中路线    |
| P1     | SRVGG full tail fuse               | `prelu_32 + final conv + PixelShuffle + resize/NV12` 一体化             | 数学可行但当前 CUDA 标量卷积低于 30fps  |
| P1     | TensorRT Plugin 融合               | 把 tail fuse 放进 TRT Plugin，减少 Python/CUDA launch 和大图 binding   | 这是 x4v3 的核心性能上限突破点          |
| P1     | C++ runner                         | 统一管理 FFmpeg、TensorRT、CUDA stream/event 和 NVENC                  | 减少 Python/ctypes 调度开销             |
| P2     | GStreamer / Video Codec SDK runner | C++/GStreamer 管线统一管理 NVDEC/TRT/NVENC                             | 工程量大，但长期最干净                  |

短期建议：

```text
0.3.0 已完成 ZeroCopy input/output 正式回归。
随后进入默认 profile、SRVGG tail fuse / TensorRT Plugin 和 C++ runner 方案设计。
```

### 11.1 SRVGG Tail Fuse 实验

`realesr-general-x4v3-1280x720.onnx` 的末端结构已经确认：

```text
prelu_32 [1,64,720,1280]
-> Conv(64 -> 48, 3x3)
-> DepthToSpace(blocksize=4, mode=CRD)
-> Add(ResizeNearest(input, x4))
-> output [1,3,2880,5120]
```

只融合 `DepthToSpace + Downsample` 不够，因为 `Conv(64 -> 48)` 的输出元素数量与 x4 RGB 大图相同。真正有意义的融合边界必须前移到 `prelu_32`：

```text
prelu_32 + input + final_conv_weight/bias
-> fused final conv + PixelShuffle + residual + resize/NV12
```

已完成的实验：

| kernel          | 300 帧耗时 | fps      | 画质对比                       | 判断             |
| --------------- | ---------- | -------- | ------------------------------ | ---------------- |
| `pixel`         | `43.656s`  | `6.872`  | PSNR `47.371430` / SSIM `0.993079` | 数学正确但太慢   |
| `2x2`           | `14.244s`  | `21.061` | PSNR `47.371178` / SSIM `0.993061` | 减少 UV 重复采样 |
| `2x2-rgb`       | `12.249s`  | `24.492` | PSNR `47.370826` / SSIM `0.993064` | 当前最佳实验路径 |
| `2x2-rgb-cell`  | `15.671s`  | `19.143` | 未采用                         | 权重加权组合增加算术量和寄存器压力 |

结论：

- `2x2-rgb` 保留为默认 tail fuse 实验 kernel。
- `2x2-rgb-cell` 只保留为实验选项，不作为默认路径。
- 当前 tail fuse 仍低于 30fps，不能替代完整 x4 TensorRT 输出路线。
- 下一步应把 tail fuse 做成真正 TensorRT Plugin 或 tiled CUDA kernel，减少重复 final conv、权重读取和 Python launch 边界。

### 11.2 SRVGG Conv48 Tail 折中路线

完整 fused tail 把 final conv 从 TensorRT 中拿出来，虽然可以从理论上避免 `conv2d_33 [1,48,720,1280]` 这个大 tensor，但当前手写 CUDA 标量卷积太慢。

因此增加一条折中路线：

```text
TensorRT:
input[1,3,720,1280] -> conv2d_33[1,48,720,1280]

CUDA:
conv2d_33 + input -> DepthToSpace(x4) + nearest residual + resize -> 1920x1080 NV12
```

它仍会暴露 `conv2d_33` binding，但不再暴露最终 `5120x2880x3` RGB x4 输出 binding。final conv 仍由 TensorRT tactic 负责，CUDA 只做轻量重排、残差和采样。

已验证结果：

| 输入 | 模式 | 输出 | 耗时 | fps |
| ---- | ---- | ---- | ---- | --- |
| 720p 300 帧 | `srvgg-conv48-tail` | `/data/jasna/720p/INU-047-U-720p_1080p_conv48tail_benchmark.mp4` | `6.595s` | `45.490` |
| 720p 5 分钟 | `srvgg-conv48-tail` | `/data/jasna/720p/INU-047-U-720p_1080p_conv48tail_full.mp4` | `201.466s` | `45.184` |

完整样本封装验证：

```text
video: hevc / 1920x1080 / 30fps / 9103 frames
audio: aac / 44100Hz / stereo
keyframes: 152
moov offset: 32
```

画质对比当前完整 x4 ZeroCopy 300 帧输出：

```text
PSNR average=48.909172 dB
SSIM All=0.993279
```

判断：

- 这条路线速度接近完整 x4 ZeroCopy，但显存输出 binding 从 RGB x4 大图降为 conv48 feature。
- 它比 `prelu_32` full tail fuse 更适合作为下一阶段工程化路线。
- build 镜像会为 SRVGG general v3 模型生成 `-conv48-fp16.engine`。
- runtime 启用 `VIDEO_POSTPROCESS_MODE=srvgg-conv48-tail` 时，会自动选择 `-conv48-fp16.engine`，无需手写 `TRT_ENGINE_PATH`。
- 后续如要继续减少 `conv2d_33` binding，应该进入 TensorRT Plugin，而不是在 Python runner 内继续手写 final conv。

运行示例：

```bash
docker run --rm \
  --device nvidia.com/gpu=0 \
  -e VIDEO_INPUT_MODE=cuda-p010 \
  -e VIDEO_OUTPUT_MODE=cuda-nvenc \
  -e VIDEO_POSTPROCESS_MODE=srvgg-conv48-tail \
  -v /data/jasna/720p:/data \
  -v /data/jasna/models:/models \
  video2x:0.3.0
```

### 11.3 60fps 性能线

720p direct 的 TensorRT engine 本身约 `47qps`，因此仅靠 Zero-Copy、pipeline depth、CUDA Graph 或多 inference stream 不能把 direct 路线推到 `60fps`。

trtexec 对照：

```text
1280x720 full x4 engine:      45.9197 qps
1280x720 conv48 engine:       46.7459 qps
1280x720 feature engine:      45.9851 qps
960x540 full x4 engine:       79.8147 qps
```

多 inference stream / CUDA Graph 对 `1280x720 conv48` 没有提升，吞吐仍约 `47qps`。

因此 `60fps+` 必须走性能 profile：

```text
1280x720 NVDEC CUDA/P010
-> GPU resize/preprocess to 960x540 NCHW FP16
-> 960x540 conv48 TensorRT
-> CUDA DepthToSpace + residual + resize/NV12
-> NVENC
```

已验证完整样本：

```text
输出：
/data/jasna/720p/INU-047-U-720p_1080p_960conv48_zc_full.mp4

frames=9103
elapsed=115.957s
fps=78.503
decode=3.179s
preprocess=0.000s
h2d=0.000s
infer=4.769s
kernel=0.127s
d2h_frame=0.000s
sync_wait=105.541s
encode_write=1.594s
merge_audio=0.370s
```

封装验证：

```text
video=hevc 1920x1080 30/1 nb_frames=9103 duration=303.433333
audio=aac 44100Hz stereo duration=303.414966
keyframes=152
moov_offset_first_4k=32
```

相对 720p direct conv48 输出的画质指标：

```text
PSNR average=41.039066 dB
SSIM All=0.986860
```

判断：

- `960x540 conv48 Zero-Copy` 是当前可用的 `60fps+` 性能线。
- 720p direct 质量线仍约 `45fps`。
- 人工画质评审已接受 `960x540` 性能线。
- 当前策略：需要 `60fps+` 时使用 `960x540 conv48 Zero-Copy`；需要最大输入细节时保留 `720p direct conv48 Zero-Copy` 质量线。
- 收口结论：本阶段不再继续硬冲 `1280x720 direct` 到 `60fps`。`60fps+` 里程碑按 `960x540 conv48 Zero-Copy` 记录，后续优化转向默认策略、发布回归和更底层 Plugin/C++ runner。

### 11.4 0.3.0 正式发布验收

build 镜像：

```text
registry.cn-qingdao.aliyuncs.com/wod/video2x:0.3.0-build
image_id=sha256:17b8158ed4279286af79fd103aaf296a317a3e698c605d7065b9604af7b80597
created=2026-05-18T09:37:50Z
```

runtime 镜像：

```text
registry.cn-qingdao.aliyuncs.com/wod/video2x:0.3.0
image_id=sha256:7a479268842a70e4f2c465891ee11afdcf1214abcddc0dd3d79792311a5f0f4d
digest=sha256:7d3a297ec1f01a723af7c5acabe231a3bd424566956f7213009d44af10580277
created=2026-05-18T14:49:45Z
size=3662710760
```

正式样本：

| 路线 | 输出 | fps |
| ---- | ---- | --- |
| 420p ZeroCopy | `/data/jasna/420p/SDMT-506-U-420p_1080p_release_zc_420p.mp4` | `142.033` |
| 720p 性能线 | `/data/jasna/720p/INU-047-U-720p_1080p_release_960conv48_zc.mp4` | `77.106` |
| 720p 质量线 | `/data/jasna/720p/INU-047-U-720p_1080p_release_direct_conv48_zc.mp4` | `45.124` |

```text
ffprobe:
1920x1080 / 30fps / 9103 frames / HEVC / AAC
keyframes=152
moov_offset_first_4k=32
```
