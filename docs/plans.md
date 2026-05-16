# 项目计划

## 1. 计划写法

本文件不是 TODO List。

本项目的计划会随着实测速度、画质、GPU 利用率和瓶颈定位反复调整，因此采用章节化计划，而不是一次性勾选清单。

文中的 checkbox 不是简单 TODO，也不表示当前已经完成。它们是每个阶段的 Review Checkpoint，用来在推进过程中逐项确认事实、数据和风险。

每个阶段都应该回答三个问题：

- 这一阶段要验证什么。
- 成功标准是什么。
- 如果不达标，下一步分支是什么。

## 2. 第一阶段：最小可运行容器

第一阶段目标是让用户只挂载 `/data` 和 `/models` 后，容器可以自动扫描并正式处理视频。

本阶段重点不是追求最终极限速度，而是先建立完整闭环：

- [ ] Docker 镜像可构建。
- [x] Real-ESRGAN Python 源码在本项目内，可调整。
- [x] 本项目自己的业务代码放在 `src/`。
- [x] 模型不打包进镜像，运行时从 `/models` 读取或下载。
- [ ] 容器能识别 NVIDIA GPU。
- [x] 能扫描 `/data` 下所有 `.mp4`。
- [x] 能读取分辨率、帧率、总帧数。
- [x] 能跳过 1080p 及以上视频。
- [x] 能为 720p 视频规划 `outscale=1.5`。
- [x] 能为 480p 或更低清视频规划到 1080p 的倍率。
- [ ] 能输出最终 1080p 文件。
- [ ] 能保留音频。
- [ ] 能用 `ffprobe` 校验输出。

成功标准：

容器可以处理一个 720p/30fps 样本视频，并输出可播放、可探测的 1080p 文件。

如果不达标：

- 如果 GPU 不可用，优先修容器运行环境和 CUDA/PyTorch 安装。
- 如果模型不可用，优先修 `/models` 挂载和模型查找逻辑。
- 如果输出不可读，优先修 ffmpeg 写出和封装流程。

## 3. 第二阶段：速度日志闭环

第二阶段目标是让用户在正式处理过程中看得见速度。

必须输出：

- [ ] 当前处理文件。
- [ ] 当前帧数和总帧数。
- [ ] 实时 fps。
- [ ] 完成百分比。
- [ ] 预计剩余时间。
- [ ] GPU 利用率。
- [ ] 显存占用。

这一阶段的核心不是优化，而是让问题可见。

成功标准：

处理 720p/30fps 样本时，日志能持续显示吞吐和 4090 状态。用户能在运行中判断是否有希望接近 2 小时内完成。

如果不达标：

- 如果帧数无法可靠获取，优先修 ffprobe 探测。
- 如果 fps 不稳定，先确认统计窗口和日志刷新周期。
- 如果 GPU 指标缺失，优先接入 `nvidia-smi` 或 NVML。

## 4. 第三阶段：720p 到 1080p 速度验证

第三阶段目标是验证主路径：2 小时、720p、30fps 视频 AI 超分到 1080p。

关键目标：

```text
216000 frames / 2 hours = 30 fps
```

因此实测吞吐需要尽量接近或超过 `30fps`。

测试方式：

- [ ] 使用真实 720p/30fps 样本。
- [ ] 默认模型为 `RealESRGAN_x2plus`。
- [ ] 默认倍率为 `outscale=1.5`。
- [ ] 默认不启用 benchmark，直接正式处理。
- [ ] 记录正式处理过程中的 fps、ETA、GPU 利用率和显存。

成功标准：

如果稳定接近或超过 `30fps`，则主路径进入可用状态。

如果不达标：

- GPU 利用率高但 fps 低，说明瓶颈可能是 AI 推理本身，需要评估模型、fp16、tile 和批处理策略。
- GPU 利用率低且 CPU 高，说明瓶颈可能是解码、编码或 Python 调度。
- GPU 利用率低且磁盘繁忙，说明瓶颈可能是 IO。
- 编码拖慢时，需要评估 NVENC 或更快的封装策略。

## 5. 第四阶段：低清视频到 1080p 策略验证

第四阶段目标是验证低清输入到最终 1080p 的处理策略。当前真实样本不是标准 480p，而是 420p，因此后续速度和画质判断以 420p 样本为准。

标准 480p 到 1080p 的倍率是：

```text
1080 / 480 = 2.25
```

420p 到 1080p 的倍率更高：

```text
1080 / 420 = 2.5714...
```

这个倍率高于 720p 到 1080p，因此不能简单假设和 720p 一样稳定。所有输出最终仍必须规范到 `1920x1080`，不能留下奇数宽度或非标准画幅。

需要比较：

- [ ] `RealESRGAN_x2plus` + 自动倍率
- [ ] `RealESRGAN_x4plus` + 自动倍率
- [x] `realesr-general-x4v3` + 自动倍率，当前 420p PyTorch 基线为 `4.583 fps`
- [ ] `realesr-general-x4v3` + TensorRT FP16 视频闭环
- [ ] 必要时评估分阶段或其他模型策略

成功标准：

输出高度必须是 1080p，画质不能明显动画化、蜡像化或强烈振铃，速度不能退化到无法批量生产。

如果不达标：

- 如果质量不达标，优先调整模型策略。
- 如果速度不达标，优先看 GPU 利用率和编码/解码瓶颈。
- 如果显存不足，调整 tile，但要关注 tile 过小带来的吞吐损失。

## 6. 第五阶段：性能主线重构

第五阶段不是固定顺序，而是根据日志反复进入的性能重构阶段。

当前实测已经证明 Real-ESRGAN Python 逐帧脚本远低于目标吞吐。第五阶段升级为性能主线阶段，目标不再是局部调参，而是把 `realesr-general-x4v3 + TensorRT FP16` 接入视频处理闭环，并用端到端 fps 判断是否必须继续推进 Zero-Copy。

已完成的关键事实：

- [x] `video2x:trt` 性能实验镜像已在服务器构建。
- [x] `realesr-general-x4v3` 已导出 ONNX。
- [x] TensorRT FP16 engine 已构建。
- [x] 420p 输入 `1x3x420x720` 的 TensorRT FP16 纯推理达到 `149.447 qps`。
- [x] 420p 真实视频样本 TensorRT 视频闭环已跑通，端到端 `6.918 fps`。
- [x] 420p 真实视频样本 TensorRT + CUDA 后处理 + libx264 跑通，端到端 `47.798 fps`。
- [x] 正式 CLI 已接入 `RUNNER=trt-cuda`，当前样本端到端 `53.377 fps`。
- [x] 模型侧吞吐已远超 30fps，当前主瓶颈转移到视频管线。

可能的瓶颈方向：

### 6.1 P0：TensorRT Engine 视频推理闭环

这是当前第一推进项。

目标不是一步到位 Zero-Copy，而是先做一个能端到端落盘的 TRT 视频版本：

```text
ffmpeg 解码帧 -> Python/CUDA/TRT 推理 -> 后处理到 1920x1080 -> NVENC 编码 -> 合并音频
```

这一步允许存在 CPU/GPU 往返拷贝，因为它的目的不是最终架构，而是测真实端到端损耗。

本阶段的定位是“工程闭环探针”：先把 TensorRT engine 放进真实视频链路，测出解码、预处理、推理、后处理、编码和音频合并分别消耗多少时间。只有拿到端到端数据后，才决定 Zero-Copy 应该从哪里下刀。

Review Checkpoint：

- [x] 新增 TRT engine 加载与推理代码。
- [x] 读取 `/models/realesr-general-x4v3-420x720-fp16.engine`。
- [x] 对 420p 样本逐帧推理。
- [x] 分段记录解码、预处理、TRT 推理、后处理、编码、音频合并耗时。
- [x] 输出标准 1920x1080。
- [ ] 使用 NVENC 编码视频。
- [x] 保留或重新合并原始音频。
- [x] 校验输出宽高、帧率、时长、音频。
- [x] 记录端到端 fps、GPU 利用率、显存、CPU 占用。
- [x] 与当前 PyTorch 视频链路 `4.583 fps` 对比。

成功标准：

端到端处理速度至少超过 15fps；如果超过 30fps，则 420p 主路径进入候选可用状态。

当前状态：

- [x] 420p 主路径已经达到 `47.798 fps`，进入候选可用状态。
- [x] `RUNNER=trt-cuda` 主入口验证通过，端到端 `53.377 fps`。
- [x] 用当前完整 420p 样本回归，不再限制帧数，速度 `47.527 fps`。
- [x] 用用户指定 420p 样本目录回归，速度 `48.497 fps`。
- [x] 使用流水线正式镜像 `0.3.0` 回归真正 5 分钟 420p 样本，速度 `69.026 fps`。
- [x] 正式镜像输出校验通过：`1920x1080`、`30fps`、`9103` 帧、音频保留。
- [x] 构建并验证 720p 输入匹配的 TensorRT engine，固定 x4 路线端到端 `24.774 fps`。
- [x] 验证 `RealESRGAN_x2plus` TensorRT FP16，纯推理 `10.475 fps`，模型太重，淘汰为速度主线。
- [x] 设计并验证 720p 预缩 540p + `realesr-general-x4v3` 路线，CLI 端到端 `39.758 fps`。
- [x] 将成品编码改为 `libx265 + 5M`，300 帧短样本 `34.554 fps`，输出约 `5.23Mbps`。
- [ ] 抽帧对比 720p 预缩 540p 路线与 720p x4 路线，确认画质是否可接受。
- [ ] 抽帧对比 PyTorch 输出，确认人脸、字幕、边缘和噪声没有明显劣化。
- [ ] 修复 NVENC 后重新测试端到端速度。

如果不达标：

- 如果 TRT 推理很快但端到端低于 15fps，瓶颈在解码、拷贝、后处理或编码，立即进入 6.3 Zero-Copy 管线。历史慢路径实测为 `6.918 fps`，最大瓶颈是 CPU 后处理 `14.410s`，已通过 CUDA 后处理修复。
- 如果输出画质和 PyTorch 明显不一致，先做 TRT/PyTorch 单帧输出对比。
- 如果音画不同步，优先修封装、帧率和音频合并流程。

### 6.2 P0：realesr-general-x4v3 + TensorRT FP16 模型侧验证

模型侧验证已经完成，但保留为回归检查项。

Review Checkpoint：

- [x] 新建性能实验镜像，包含 TensorRT、ONNX、trtexec。
- [x] 导出 `realesr-general-x4v3` 到 ONNX。
- [x] 构建 TensorRT FP16 engine。
- [x] 记录 TensorRT FP16 纯推理速度：`149.447 qps`。
- [ ] 对比 TensorRT FP16 与 PyTorch 输出画质。
- [x] 明确禁止默认 INT8；INT8 只作为画质无明显损伤时的实验分支。

成功标准：

TensorRT FP16 在 420p 样本上显著快于 PyTorch，并且单帧输出与 PyTorch 版本没有明显视觉差异。

如果不达标：

- 如果导出失败，优先处理模型算子兼容。
- 如果 TRT 加速不足，回到 ONNX 导出和 engine 构建参数。
- 如果画质劣化，回退 FP16 配置并禁止 INT8。

### 6.3 P0：Zero-Copy NVDEC/TRT/NPP/NVENC Pipeline

只有当 6.1 的端到端数据证明瓶颈主要来自解码、CPU/GPU 拷贝、后处理或编码时，才进入这一阶段。

目标链路：

```text
NVDEC -> GPU surface / CUDA memory -> TRT FP16 inference -> CUDA/NPP postprocess -> NVENC
```

Review Checkpoint：

- [ ] 确认 FFmpeg 镜像具备 `cuda` hwaccel。
- [ ] 确认具备 `h264_cuvid` / `hevc_cuvid`。
- [ ] 确认具备 `h264_nvenc` / `hevc_nvenc`，并修复当前 `OpenEncodeSessionEx failed: unsupported device (2)`。
- [x] 验证 Desktop GStreamer 1.28.2 包可提供 `cudaupload`、`cudascale`、`cudadownload`。
- [x] 验证 `2880x1680 -> 1920x1080` CUDA 后处理 153 帧约 `3s`，明显快于 Python/OpenCV `14.410s`。
- [x] 验证 GStreamer `appsrc` 按完整帧接入 TRT 输出，端到端提升到 `8.163 fps`。
- [ ] 验证 720p 输入可以硬解码、硬编码。
- [ ] 验证 420p 输入可用 NPP/CUDA 补边或缩放到 1920x1080。
- [x] 将 TRT 输出接入 GStreamer `appsrc`，按完整帧 push GstBuffer，避免裸 `fdsrc` 的 65536 字节分块问题。
- [ ] 消除 `trt_to_appsrc_push=11.378s` 的大图 CPU 拷贝，改为 CUDA device memory/NPP/自定义 kernel 路线。
- [ ] 统计 CPU/GPU 往返拷贝次数。

成功标准：

视频帧尽量常驻显存，CPU 不再承担逐帧解码、逐帧编码和频繁内存拷贝。

硬标准是端到端速度明显接近或超过 30fps；如果只是在纯推理 benchmark 中很快，但落盘仍慢，本阶段不能算完成。

如果不达标：

- 如果 FFmpeg CUDA filter 无法满足显存闭环，评估 GStreamer/NVIDIA Video Codec SDK/C++ 管线。
- 如果 420p 尺寸对齐画质劣化，比较 NPP、硬件双线性、Lanczos 的损耗。

#### 6.3.1 消除大图 CPU 拷贝计划

当前瓶颈已经定位：

```text
TRT 输出 2880x1680 RGB -> Python numpy -> Gst.Buffer/appsrc -> cudascale
```

`cudascale` 本身不慢，153 帧文件源实验约 `3s`；慢的是 Python 将 `2880x1680` 大图复制进 `Gst.Buffer`，实测 `trt_to_appsrc_push=11.378s`。因此下一步目标不是继续调 GStreamer，而是让 TRT 输出尽量留在 CUDA device memory。

推荐推进顺序：

1. Python device pointer 基准。

先绕过 GStreamer，只用 TensorRT Python API 或 CUDA Python 显式分配 input/output device buffer，确认：

- [x] TRT 输出留在 device buffer，不立刻 `numpy` 化。
- [x] 只做 `cudaMemcpyDtoH` 计时，量化单纯 D2H 拷贝成本。
- [x] 对比 Polygraphy `runner.infer()` 的隐式输出拷贝成本。
- [x] 记录 output buffer 大小：`1x3x1680x2880 FP32`，每帧 `58,060,800 bytes`。

成功标准：

能证明当前 `runner.infer()` 或 `Gst.Buffer` 封装到底拷贝了多少数据、耗时多少。没有这个数据，不进入 C++/CUDA 改造。

当前结论：

- 不拷输出时 H2D + TRT：`148.750 fps`。
- 每帧显式 D2H 拷完整输出：`78.939 fps`。
- 单纯 PCIe D2H 不是主要瓶颈；主要瓶颈是 Python/numpy/Gst.Buffer 大图格式转换和对象封装。

2. CUDA 后处理内核原型。

用 CUDA/NPP 或自定义 kernel 直接在 device 上完成：

```text
TRT output CHW float/half -> resize/pad -> RGB/NV12 1920x1080
```

Review Checkpoint：

- [x] 支持 `2880x1680 -> 1920x1080`。
- [x] 支持居中补边，输出标准 `1920x1080`。
- [x] 输出可以选择 RGB，用于先回 CPU 验证画质。
- [ ] 输出可以选择 NV12，用于后续直接接 NVENC。
- [x] 单独记录 kernel/NPP 耗时。

成功标准：

153 帧后处理耗时接近或低于 GStreamer CUDA scale 文件源实验的 `3s`，并且画面尺寸、补边位置正确。

当前结论：

- 自定义 CUDA kernel：`0.037ms/frame`。
- 最终 `1920x1080 RGB` D2H：`0.451ms/frame`。
- H2D + TRT + CUDA postprocess + RGB D2H：`139.209 fps`。
- 算力层面已经证明 30fps 可达，下一步应把该 CUDA 后处理接入真实视频 runner。
- 已接入真实视频 runner，420p 样本达到 `47.798 fps`，输出标准 `1920x1080`、`30fps`、`153` 帧。

3. 最小 C++/CUDA/TRT worker。

如果 Python 无法拿到足够低的拷贝成本，进入 C++ worker：

```text
decode CPU/raw frame -> cudaMemcpyHtoD -> TRT enqueueV3 -> CUDA/NPP postprocess -> cudaMemcpyDtoH 或 NVENC
```

Review Checkpoint：

- [ ] 使用 TensorRT C++ API 管理 input/output binding。
- [ ] 使用 CUDA stream 串联 H2D、TRT、postprocess。
- [ ] 输出先回 CPU 写 raw，验证画质和速度。
- [ ] 再接 NVENC 或 GStreamer CUDA memory。

成功标准：

端到端不含编码至少超过 `15fps`，否则说明 x4 中间分辨率路线本身仍过重。

4. 消除 x4 巨大中间帧。

如果 device memory 路线仍然不够快，必须重新评估模型倍率，而不是继续搬 `2880x1680` 大图：

- [ ] 导出/寻找 x2 或 x1.5 的轻量模型。
- [ ] 对 420p 比较先传统 GPU scale 到 540p/720p，再 AI x2 到 1080p。
- [ ] 比较 x4 输出再缩回 1080p 与直接目标倍率输出的画质差异。

成功标准：

在画质不明显低于当前 `realesr-general-x4v3` 的前提下，减少中间帧尺寸和显存带宽压力。

### 6.4 P0：RTX Video SDK / Maxine VSR

这是最接近 DLSS/RTX VSR 的黑盒路线。

Review Checkpoint：

- [ ] 确认可否在服务器 Linux 环境调用 SDK。
- [ ] 验证 720p 到 1080p 的速度和画质。
- [ ] 验证 420p 非整数倍率路线：先硬件/NPP 对齐，再 VSR，或 VSR 后补边。
- [ ] 比较 420p 二次缩放是否糊边、振铃、字幕变形。
- [ ] 确认可离线落盘，且音频、帧率、时长同步。

成功标准：

能接近或超过 30fps，且主观画质不明显低于 Real-ESRGAN 基线。

如果不达标：

- 如果 SDK 无法离线落盘，降级为研究参考。
- 如果 420p 画质损耗明显，只保留 720p 路线。

### 6.5 P0：RealBasicVSR 长视频稳定性

RealBasicVSR 在公开 VideoLQ 基准中有质量和速度优势，但视频模型存在长时序风险。

Review Checkpoint：

- [ ] 跑通 420p/720p 样本输出 1920x1080。
- [ ] 记录 fps、显存、GPU 利用率。
- [ ] 设计 chunk / overlap / flush 策略。
- [ ] 连续运行长样本，观察显存是否持续上涨。
- [ ] 检查切片边界是否闪烁或断裂。

成功标准：

画质不低于 Real-ESRGAN 基线，速度明显高于当前基线，并且长视频显存稳定。

如果不达标：

- 如果显存上涨，必须修 chunk 生命周期和缓存释放。
- 如果切片边界有视觉断裂，增加 overlap 或放弃长视频主线。

### 6.6 传统专项瓶颈

以下检查保留，但优先级低于 P0 性能主线。

#### 6.6.1 GPU 推理瓶颈

当 GPU 利用率高、显存合理、fps 仍低时，说明瓶颈更可能在模型推理。

优化方向：

- [ ] 确认使用 fp16。
- [ ] 调整 tile。
- [ ] 评估模型选择。
- [ ] 评估是否需要更直接的视频帧管线。

#### 6.6.2 解码瓶颈

当 GPU 利用率低、CPU 或 ffmpeg 解码占用高时，说明输入帧供给不足。

优化方向：

- [ ] 减少不必要的中间转码。
- [ ] 检查 ffmpeg 参数。
- [ ] 评估硬件解码。

#### 6.6.3 编码瓶颈

当 AI 推理完成后写出速度拖慢，说明编码成为瓶颈。

优化方向：

- [ ] 评估 NVENC。
- [ ] 调整编码预设。
- [ ] 区分“正式成品质量”和“速度验证模式”的编码策略。

#### 6.6.4 IO 瓶颈

当磁盘读写成为限制时，说明挂载目录、临时目录或输出目录需要优化。

优化方向：

- [ ] 检查输入输出盘性能。
- [ ] 减少临时文件。
- [ ] 避免跨慢盘读写。

## 7. 第六阶段：默认用户体验收敛

这一阶段目标是让普通用户不懂 AI 也能稳定使用。

默认体验应该是：

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  video2x
```

容器应该自动：

- [ ] 扫描视频。
- [ ] 打印任务清单。
- [ ] 选择模型和倍率。
- [ ] 开始正式处理。
- [ ] 显示速度和预计剩余时间。
- [ ] 校验输出。
- [ ] 给出普通用户能理解的错误信息。

成功标准：

用户不需要知道 `RealESRGAN_x2plus`、`outscale`、`tile` 或 ffmpeg 参数，也能完成批处理。

如果不达标：

- 如果参数太多，收敛默认值。
- 如果日志太吵，分层显示关键日志和调试日志。
- 如果错误信息太技术化，增加面向用户的解释。

## 8. 第七阶段：回归与发布

发布前必须回归：

- [ ] 720p/30fps 到 1080p。
- [ ] 420p/30fps 到 1080p。
- [ ] 标准 480p/30fps 到 1080p。
- [ ] 1080p 及以上跳过。
- [ ] 已存在输出跳过。
- [ ] 模型缺失时的错误提示。
- [ ] GPU 不可用时的错误提示。
- [ ] 输出文件可被 `ffprobe` 读取。
- [ ] 日志包含 fps、ETA、GPU 利用率和显存。

发布判断：

只有当速度日志能证明 4090 被充分释放，并且主路径明显优于旧 Video2X x4 路线，才进入可发布状态。
