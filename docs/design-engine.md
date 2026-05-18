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
.pth -> FP16 .onnx -> FP16 .engine
```

build 镜像默认自动完成导出 ONNX 和构建 engine。ONNX 导出必须带 `--fp16`，否则 720p direct 的 x4v3 输出会以 FP32 大图存在，显存峰值和后处理压力都会被放大。

手工流程等价于：

```bash
python /app/tools/export_realesrgan_onnx.py \
  --model realesr-general-x4v3 \
  --weights /models/realesr-general-x4v3.pth \
  --output /models/realesr-general-x4v3-854x480.onnx \
  --input-shape 1,3,480,854 \
  --fp16
```

```bash
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

## 10. 720p 质量路线问题

当前 720p 性能线是：

```text
1280x720 -> 960x540 -> x4 -> 3840x2160 -> 1920x1080
```

这条路线在 0.3.0 正式镜像中已经达到 `77.106fps`，人工画质评审可接受，因此作为当前发布性能线保留。它仍是性能折中；问题根源不是 720p 本身，而是 `realesr-general-x4v3` 固定 x4 输出：

```text
1280x720 -> x4 -> 5120x2880
```

这个中间张量太大，会显著增加显存、D2H、后处理和编码前数据搬运压力。把 720p 压到 540p 是为了避开这个中间张量，但画质上不够理想。

当前修正方向：

```text
1280x720 -> x4v3 FP16 TensorRT -> FP16 CUDA fused resize/NV12 -> 1920x1080
```

这条路线保留 720p 输入细节，不再先缩到 540p。关键点是 engine 输出也必须是 FP16，否则单帧 `5120x2880x3` 输出约 `177MB`，而 FP16 可降到约 `88.5MB`。它不能消除 x4 中间图，但能把最直接的显存浪费砍掉一半，并为后续真正的显存闭环打基础。

理想路线应该是 720p 直接输入，输出阶段只生成最终需要的 1080p：

```text
1280x720 -> AI feature / x4 internal -> CUDA fused downsample -> 1920x1080
```

可选解法：

- **P0：TensorRT Plugin 融合 PixelShuffle + Downsample**  
  不接受 `5120x2880` 完整张量写回 Global Memory 后再二次采样。应审查 ONNX 图，把最后的 x4 重构/PixelShuffle 与 `1920x1080` 降采样合并为 TensorRT Plugin，让冗余像素在算子内部直接消失，最终只向后续链路输出 `1920x1080` 的 FP16/RGB 或 NV12 数据。

- **P0：CUDA fused 后处理减内存峰值**  
  这是 Plugin 前的过渡方案。保留 x4v3，允许 TensorRT 输出完整 x4 FP16 张量，但后处理必须直接在 GPU 上采样到 `1920x1080 NV12`，不能回到 CPU 再处理。

- **P1：RealESRGANv2-animevideo-xsx2 真实样本验证**  
  它是原生 x2 轻量候选，理论上能从结构上避免 x4 中间张量。但它面向 Anime 视频，真实人物、皮肤、毛发和自然纹理存在蜡像化风险。只能作为对照实验，不能替代主线。

- **P2：RealESRGAN_x2plus 通用 x2 对照**  
  它是通用真实场景 x2 模型，但现有基线已经证明太重。即使 TensorRT FP16 加速，也不应作为 30fps 主线，只保留为质量参考。

- **P1：TensorRT dynamic / profile 优化**  
  仍然不能改变 x4 输出本质，但可以改善 engine 选择和 tactic，不解决根因。

- **P2：分块 streaming 推理**  
  用 tile/overlap 控制显存峰值，但要处理接缝、时序一致性和吞吐下降。

阶段判断：

- `960x540 conv48` profile 是 0.3.0 当前可发布的 60fps+ 性能线。
- 720p 质量线保留 `1280x720 conv48` direct engine，0.3.0 正式镜像为 `45.124fps`。
- 不寻找不存在的 `realesr-general-x2v3` 捷径。general-v3 主线只有 x4 权重，推理时伪装 x2 不能从网络结构上消除 PixelShuffle 膨胀。
- 下一阶段应优先验证 720p direct FP16 engine 的 GPU fused postprocess 和显存峰值，随后推进 TensorRT Plugin 融合 PixelShuffle + Downsample，目标是不牺牲输入细节。

## 11. TensorRT Plugin 融合路线

### 11.1 为什么必须做

`realesr-general-x4v3` 当前问题不在输入 `1280x720`，而在最后 x4 重构会产生：

```text
5120x2880x3
```

即使是 FP16，这个输出张量仍约 `88.5MB/frame`。如果每帧都完整写入 Global Memory，再由后处理 kernel 读回采样到 `1920x1080`，显存带宽和缓存都会被无意义像素消耗。

目标是把：

```text
Conv/Feature -> PixelShuffle(x4) -> Resize(1920x1080)
```

改成：

```text
Conv/Feature -> FusedPixelShuffleResizePlugin(1920x1080)
```

Plugin 内部只计算最终输出需要采样的位置。中间 `5120x2880` 不作为完整 TensorRT binding 暴露，不作为整帧张量写回 Global Memory。

### 11.2 技术边界

Plugin 必须满足：

- 输入保持 `realesr-general-x4v3` 最后一层重构前的 feature tensor。
- 输出固定为最终目标尺寸，默认 `1920x1080`。
- 插值策略与当前 CUDA fused resize 保持一致，先追求画质一致，再优化速度。
- 支持 FP16，严禁默认 INT8。
- 输出优先 FP16/RGB；NV12 可作为下一阶段继续融合。

Plugin 不解决：

- NVDEC 到 TRT 输入的 CPU 拷贝。
- NVENC 写入路径。
- 未知分辨率 engine 自动生成。

这些仍归 `docs/design-zerocopy.md` 和 build 镜像流程处理。

### 11.3 推进顺序

1. 导出 `realesr-general-x4v3-1280x720.onnx`，用 ONNX 工具确认末端节点结构。
2. 找到最后的 x4 重构边界，确认是否是 PixelShuffle / DepthToSpace / Reshape-Transpose-Reshape 组合。
3. 先做等价性实验：fused tail 输出 NV12，对比当前 `x4 TensorRT -> CUDA resize` 的抽帧差异。
4. 再做性能实验：统计显存峰值、TRT latency、总 fps。
5. 最后考虑把 fused tail 从 Python + CUDA kernel 原型推进为 TensorRT Plugin，减少 Python/ctypes 调度边界。

当前 ONNX 末端已经确认：

```text
Conv(64 -> 48, 3x3)
-> DepthToSpace(blocksize=4, mode=CRD)
-> Add(ResizeNearest(input, x4))
-> output[1,3,2880,5120]
```

对于 `1280x720` 输入：

```text
prelu_32:     [1,64,720,1280] FP16
conv2d_33:   [1,48,720,1280] FP16
pixel_shuffle/output: [1,3,2880,5120] FP16
```

关键判断：

- 只融合 `DepthToSpace + Downsample` 不够，因为 `Conv(64 -> 48)` 的输出元素数与 x4 RGB 输出相同，仍然是约 `88.5MB/frame`。
- 真正的融合边界必须前移到最后一个 Conv 之前，也就是以 `prelu_32` 和原始 `input` 为输入，直接生成 `1920x1080 NV12`。
- SRVGG 还有 `nearest x4 input` 残差分支，Plugin 必须同时实现 `Conv tail + PixelShuffle + Add nearest base + Downsample`。

已完成 P0 原型：

```text
feature engine:
input[1,3,720,1280] -> prelu_32[1,64,720,1280]

CUDA fused tail:
prelu_32 + input + final_conv_weight/bias
-> implicit PixelShuffle + nearest residual + resize
-> FFmpeg CUDA/NV12 AVFrame
```

原型验证：

```text
输出：
/data/jasna/720p/INU-047-U-720p_1080p_tailfuse_benchmark.mp4

frames=300
elapsed=43.656s
fps=6.872
d2h_frame=0.000s
encode_write=0.208s
```

画质量化，对比当前稳定 `zcfull_benchmark`：

```text
PSNR average=47.371430 dB
SSIM All=0.993079
```

2x2 NV12 block 原型：

```text
输出：
/data/jasna/720p/INU-047-U-720p_1080p_tailfuse2x2_benchmark.mp4

frames=300
elapsed=14.244s
fps=21.061
d2h_frame=0.000s
encode_write=0.097s
```

画质量化，对比当前稳定 `zcfull_benchmark`：

```text
PSNR average=47.371178 dB
SSIM All=0.993061
```

2x2 版本把每个 CUDA thread 从单像素改成处理一个 NV12 2x2 block，避免 UV 分支再次重复采样四个像素。速度从 `6.872 fps` 提升到 `21.061 fps`，证明重复采样是主要开销之一，但仍未达到 30fps。

阶段结论：

- 原型证明融合边界和数学路径可行。
- 朴素 fused tail kernel 按目标像素反算 `3x3x64` 最后一层卷积，计算量过大，不可作为主线。
- 2x2 原型证明减少重复采样有效，但仍未解决根因。
- 下一版必须把 tail conv 从“按 HR 输出采样点算”改为“按 LR 像素算 48 个子像素一次，然后供 1080p resize 复用”，或者直接写 TensorRT Plugin / CUDA tile kernel，让同一块 LR feature 被复用，而不是每个输出采样点重复卷积。

### 11.4 验收标准

- 720p 输入不能预缩到 `960x540`。
- 输出必须是 `1920x1080`。
- 画质不得低于当前 `1280x720 -> x4v3 -> CUDA resize`。
- 显存峰值必须明显低于完整 x4 输出路线。
- 5 分钟 720p 样本必须保持音频、帧率、帧数、时长同步。

## 12. Review Checkpoint

- [x] engine 文件名使用 `宽x高`。
- [x] 运行镜像只读取 `.engine`。
- [x] build 镜像负责 `.pth -> .onnx -> .engine`。
- [x] `720x420` 被降级为样本兼容 profile。
- [x] 标准 profile engine 全部构建完成或由 build 镜像确认可复用。
- [x] 使用正式镜像回归 420p 样本和 720p 样本。
- [x] 明确 `960x540 conv48` 是 720p 当前 60fps+ 性能线。
- [x] build 镜像默认导出 FP16 ONNX。
- [x] runtime CUDA 后处理支持 FP16 TRT 输出。
- [x] 验证 720p direct conv48 ZeroCopy 质量线，正式镜像 `45.124fps`。
- [x] 审查 `1280x720` ONNX 末端图结构，定位 PixelShuffle / DepthToSpace 边界。
- [x] 完成 fused tail CUDA 原型：融合最后 Conv、PixelShuffle、nearest residual、Downsample、NV12 输出。
- [x] 验证 fused tail 原型输出与当前 CUDA resize 路线的画质一致性。
- [ ] 优化 fused tail：避免按目标像素重复执行 `3x3x64` tail conv。
- [ ] 设计 TensorRT Plugin：融合最后 Conv + PixelShuffle + Downsample。

## 13. 0.3.0 发布验收

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

正式样本回归：

| 路线 | engine | fps | 输出 |
| ---- | ------ | --- | ---- |
| 420p ZeroCopy | `realesr-general-x4v3-720x420-fp16.engine` | `142.033` | `1920x1080 / 30fps / 9103 frames / HEVC / AAC` |
| 720p 性能线 | `realesr-general-x4v3-960x540-conv48-fp16.engine` | `77.106` | `1920x1080 / 30fps / 9103 frames / HEVC / AAC` |
| 720p 质量线 | `realesr-general-x4v3-1280x720-conv48-fp16.engine` | `45.124` | `1920x1080 / 30fps / 9103 frames / HEVC / AAC` |

三条输出均验证 `keyframes=152`、`moov_offset_first_4k=32`，可拖拽播放。

## 14. 运行命令

build 镜像运行时不需要输入任何业务参数。准备好本地 `models/` 目录，容器会扫描并构建所有已存在且受支持的 `.pth`。

```text
realesr-general-x4v3.pth
realesr-general-wdn-x4v3.pth
RealESRGAN_x2plus.pth
RealESRGAN_x4plus.pth
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

对 `realesr-general-x4v3` 和 `realesr-general-wdn-x4v3`，build 镜像还会额外生成 conv48 tail engine：

```text
{model}-{width}x{height}-conv48.onnx
{model}-{width}x{height}-conv48.onnx.data
{model}-{width}x{height}-conv48-fp16.engine
{model}-tail-{width}x{height}-conv48.npz
```

该 engine 输出 SRVGG 的 `conv2d_33 [1,48,H,W]`，供 runtime 的 `VIDEO_POSTPROCESS_MODE=srvgg-conv48-tail` 使用。runtime 会自动选择 `-conv48-fp16.engine`，不需要手工指定 `TRT_ENGINE_PATH`。

如果 `.onnx` 或 `.engine` 已经存在，容器会跳过对应步骤，不重复构建。

如果只想临时构建单个模型，可显式指定 `MODEL_NAME`；默认不要指定，让 build 镜像处理全部已上传模型。
