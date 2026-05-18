# FlashVSR 引擎分析

## 1. 结论

FlashVSR 值得作为 P0 实验候选，但不能直接替换当前 `realesr-general-x4v3 + TensorRT` 主线。

0.3.0 发布后，FlashVSR 的对标基线已经提高。它不再是和早期 PyTorch/CPU 视频链路比较，而必须面对正式镜像结果：

| 路线 | 完整 5 分钟样本 fps | 说明 |
| ---- | ------------------- | ---- |
| 420p ZeroCopy | `142.033` | 已远超实时 |
| 720p `960x540 conv48` 性能线 | `77.106` | 当前 60fps+ 主线 |
| 720p `1280x720 conv48` 质量线 | `45.124` | 当前质量线 |

三条输出均已通过 `1920x1080 / 30fps / 9103 frames / HEVC / AAC` 与 faststart seek 验证。因此 FlashVSR 进入主线的条件是：在 RTX 4090 上至少接近上述速度，并且画质或时序一致性明显值得引入新模型复杂度。

原因很简单：

- 它的方向非常对：one-step diffusion、streaming VSR、sparse attention、tiny decoder，目标就是高质量视频超分和实时化。
- 官方公开指标很强：论文和 README 都写到单 A100 上 `768x1408` 视频约 `17 FPS`，并宣称相对 prior one-step diffusion VSR 有最高约 `12x` 加速。
- 但工程风险也很硬：官方说明主要为 `4x` 超分优化；Block-Sparse Attention 在 A100/A800 上验证较好，RTX 40/50 系兼容和性能未知。
- 当前业务目标是 RTX 4090、5 分钟到 2 小时长视频、最终 `1920x1080`、速度优先、Docker 离线运行。FlashVSR 必须先通过本地样本实测。

阶段判断：

```text
FlashVSR = P0 竞品验证线
不是当前 runtime 主线
不是 TensorRT engine build 流程的一部分
```

## 2. 来源信息

官方仓库：

```text
https://github.com/OpenImagingLab/FlashVSR
```

论文：

```text
https://arxiv.org/abs/2510.12747
```

官方 README 关键信息：

- FlashVSR 是 diffusion-based one-step streaming VSR framework。
- v1.1 已发布，官方推荐使用 v1.1。
- 主要设计和优化目标是 `4x` video super-resolution。
- 依赖 Block-Sparse Attention。
- Block-Sparse Attention 官方说明中，A100/A800 表现理想，H200 可运行但加速受限，RTX 40/50 系兼容和性能未知。
- 官方提醒第三方实现如果缺少 Locality-Constrained Sparse Attention，可能在高分辨率下出现明显质量下降。

论文摘要关键信息：

- 单 A100 上 `768x1408` 视频约 `17 FPS`。
- 通过三阶段蒸馏、locality-constrained sparse attention、tiny conditional decoder 提速。
- 面向 real-world video super-resolution。

## 3. 为什么它看起来很强

当前 Real-ESRGAN 路线的问题是逐帧图像模型，本质上不理解视频时序：

```text
frame -> SR -> frame
```

FlashVSR 是视频模型，目标链路更接近：

```text
frames window -> streaming VSR -> temporally consistent output
```

潜在优势：

- 时序一致性可能更好，减少逐帧超分导致的闪烁。
- one-step diffusion 比传统多步 diffusion 快很多。
- Locality-Constrained Sparse Attention 能减少高分辨率注意力冗余。
- Tiny Conditional Decoder 直接面向解码加速。

这正好击中当前项目的痛点：速度、视频一致性、高清重建。

## 4. 关键风险

### 4.1 RTX 4090 不确定

官方明确说 Block-Sparse Attention 在 A100/A800 上验证较好，RTX 40/50 系兼容和性能未知。

我们的目标硬件是 RTX 4090。不能拿 A100 的 `17 FPS` 直接推导 4090 性能。

必须实测：

- 是否能编译 Block-Sparse Attention。
- 是否能在 4090 上正常运行。
- 是否真的比当前 `realesr-general-x4v3 TensorRT` 快。
- 显存峰值是否能接受。

### 4.2 官方脚本不是生产视频流水线

官方 v1.1 long video 脚本仍然有明显研究代码特征：

- Python 入口固定输入文件。
- 使用 `imageio` 读写。
- 输入会按 `4x` 放大并裁剪到 `128` 倍数。
- 输出音频、帧数、时长同步不是主流程重点。

这和当前项目要求不同：

```text
扫描 /data -> 自动决策 -> 保持音频/帧率/时长 -> NVENC 5M HEVC -> Docker 离线运行
```

所以 FlashVSR 第一阶段只做实验容器，不直接并入 runtime。

### 4.3 4x 与 1080p 目标冲突

官方推荐 `4x`，但我们的目标是统一落到 `1920x1080`：

```text
720p -> 1080p = 1.5x
480p -> 1080p = 2.25x
420p -> 1080p ~= 2.57x
360p -> 1080p = 3x
```

如果 FlashVSR 强制 `4x`，仍会出现超目标输出再缩回 1080p 的问题。它是否能稳定接受非 4x，或者能否先 4x 后 GPU downsample，需要实测。

### 4.4 第三方实现不可直接信

官方 README 提醒部分第三方实现缺少 LCSA，可能导致高分辨率质量下降。

因此验证必须优先使用官方仓库和官方 v1.1 权重，不以 ComfyUI 插件结果作为主结论。

## 5. 与当前主线对比

| 路线                                           | 优势                                     | 风险                            | 当前定位    |
| ---------------------------------------------- | ---------------------------------------- | ------------------------------- | ----------- |
| `realesr-general-x4v3 + TRT FP16 + ZeroCopy`   | 已发布，速度达标，工程可控               | 720p direct 有 x4 中间张量压力  | 当前主线    |
| TensorRT Plugin 融合 PixelShuffle + Downsample | 保留当前质量路线，直接砍中间张量浪费     | 需要写 TRT Plugin，工程难度高   | P0 主线优化 |
| FlashVSR v1.1                                  | 视频模型、时序一致性潜力强、论文指标亮眼 | RTX 4090 未知，生产流水线未打通 | P0 竞品验证 |
| RealESRGAN_x2plus                              | 通用真实场景质量参考                     | 太慢                            | P2 质量对照 |

## 6. 验证计划

### 6.1 源码准备

下载到 `.tmp`，不进入主仓库：

```bash
git clone https://github.com/OpenImagingLab/FlashVSR .tmp/FlashVSR
```

权重放到本地模型目录，不打进镜像：

```text
models/FlashVSR-v1.1/
```

预期权重文件：

```text
LQ_proj_in.ckpt
TCDecoder.ckpt
Wan2.1_VAE.pth
diffusion_pytorch_model_streaming_dmd.safetensors
```

### 6.2 第一阶段：可运行性

目标只验证能不能在 RTX 4090 上跑起来。

检查项：

- [ ] Python 3.11 环境可创建。
- [ ] PyTorch CUDA 可用。
- [ ] Block-Sparse Attention 可编译。
- [ ] 官方 v1.1 tiny long video 脚本能处理短视频。
- [ ] 不出现 CUDA OOM。

### 6.3 第二阶段：业务样本

统一使用当前固定样本：

```text
/data/jasna/720p/INU-047-U-720p.mp4
/data/jasna/420p/SDMT-506-U-420p.mp4
```

记录：

- 输入分辨率。
- 输出分辨率。
- 是否强制 4x。
- 是否裁剪到 `128` 倍数。
- 总耗时。
- 推理 fps。
- 显存峰值。
- 是否保留音频。
- 帧数是否等于 `9103`。
- 时长是否与源视频一致。

### 6.4 第三阶段：画质评审

抽帧对比：

- 源视频。
- 当前 `realesr-general-x4v3 + TRT FP16` 输出。
- FlashVSR v1.1 输出。

重点看：

- 人脸是否蜡像化。
- 皮肤纹理是否被过度生成。
- 字幕边缘是否振铃。
- 快速运动是否拖影。
- 时序是否闪烁。
- 暗部噪声是否被错误增强。

## 7. 进入主线条件

FlashVSR 只有同时满足这些条件，才允许进入主线设计：

- RTX 4090 上可稳定运行。
- 720p 和 420p 样本都能输出标准 `1920x1080`。
- 5 分钟样本端到端速度至少接近当前主线。
- 主观画质明显优于或不低于当前主线。
- 长视频处理不会一次性把所有帧塞进显存或内存。
- 能保留音频、帧率、帧数、时长。
- 能 Docker 化，模型外置，不下载 GitHub。

如果只满足画质但速度不够，它只能作为高质量离线模式。

如果速度够但输出尺寸、音频、时长不可控，它只能作为研究分支。

## 8. 初步判断

FlashVSR 很值得测，而且它的方向比继续堆传统逐帧 SR 更有想象力。

但当前最现实的判断是：

```text
短期：作为 P0 竞品验证线
中期：如果 4090 实测稳定，再设计 FlashVSR Docker 实验镜像
长期：如果速度和质量都赢，才考虑并入自动 planner
```

当前 0.3.0 主线保持：

```text
realesr-general-x4v3 + TensorRT FP16 + ZeroCopy NVDEC/NVENC + conv48 性能线
```

同时继续推进：

```text
TensorRT Plugin 融合 PixelShuffle + Downsample
```

FlashVSR 是一条可能很强的新战线，但必须用真实 720p/420p 样本打出来，不能只看论文指标。
