# 性能竞品分析

## 当前基线

实测硬件：RTX 4090，单容器单 GPU，输出统一为 1920x1080。

| 输入 | 模型                 | 实测 fps | 5 分钟 30fps | 2 小时 30fps | 判断         |
| ---- | -------------------- | -------- | ------------ | ------------ | ------------ |
| 720p | RealESRGAN_x2plus    | 2.798    | 约 53.6 分钟 | 约 21.4 小时 | 太慢         |
| 420p | RealESRGAN_x4plus    | 2.042    | 约 73.5 分钟 | 约 29.4 小时 | 淘汰         |
| 420p | realesr-general-x4v3 | 4.583    | 约 32.7 分钟 | 约 13.1 小时 | 当前最快基线 |
| 420p | realesr-general-x4v3 TensorRT FP16 纯推理 | 149.447 | 约 1.0 分钟 | 约 24.1 分钟 | 模型算力充足，需接视频管线 |
| 420p | realesr-general-x4v3 TensorRT FP16 视频闭环 | 6.918 | 约 21.7 分钟 | 约 8.7 小时 | 已跑通但仍太慢 |
| 420p | realesr-general-x4v3 TensorRT FP16 + GStreamer CUDA scale | 8.163 | 约 18.4 分钟 | 约 7.4 小时 | 后处理改善，但仍有大图拷贝瓶颈 |
| 420p | realesr-general-x4v3 TensorRT FP16 + CUDA device postprocess | 139.209 | 约 1.1 分钟 | 约 25.9 分钟 | 算力层面达标，需接真实视频链路 |
| 420p | realesr-general-x4v3 TensorRT FP16 + CUDA postprocess + libx264 实视频链路 | 47.798 | 约 3.1 分钟 | 约 1.26 小时 | 已达到 2 小时内完成目标 |
| 420p | RUNNER=trt-cuda 正式 CLI | 53.377 | 约 2.8 分钟 | 约 1.12 小时 | 主入口已达标 |

目标是 2 小时 30fps 视频在 2 小时内完成，至少需要 30 fps。当前最快 `realesr-general-x4v3` 只有 4.583 fps，缺口约 6.5 倍。

TensorRT FP16 纯推理已经达到 `149.447 qps`，说明模型本身不再是 30fps 目标的主要瓶颈。接入真实 420p 视频闭环后，端到端速度为 `6.918 fps`，分段耗时显示最大瓶颈是 CPU 后处理：TRT 输出 `2880x1680` 后，再由 Python/OpenCV resize/pad 到 `1920x1080`，该段耗时 `14.410s`。改用 Desktop GStreamer `appsrc + cudascale` 后速度提升到 `8.163 fps`，但 `trt_to_appsrc_push=11.378s` 成为新瓶颈，说明问题已经收敛到 Python/Gst.Buffer 大图拷贝。后续主风险转移到视频管线：解码、预处理、GPU/CPU 拷贝、后处理、编码和音画同步。

消除大图拷贝的计划分四步推进：

- 先做 TensorRT device pointer 基准，量化 `runner.infer()` 隐式 D2H 与显式 `cudaMemcpyDtoH` 成本。
- 再做 CUDA/NPP 后处理原型，把 `CHW float/half -> 1920x1080 RGB/NV12` 放在 device 上完成。
- 如果 Python 仍然无法控制拷贝，进入最小 C++/CUDA/TRT worker，用 CUDA stream 串联 H2D、TRT、postprocess。
- 如果仍不达标，停止搬运 `2880x1680` 大图，重新评估模型倍率和中间分辨率路线。

最新 device buffer 基准显示：不拷输出时 H2D + TRT 为 `148.750 fps`；每帧显式 D2H 拷回完整 `58,060,800 bytes` 输出时仍有 `78.939 fps`。因此 appsrc 路线的 `11.378s` 主要不是 PCIe 物理拷贝，而是 Python/numpy/Gst.Buffer 对 `2880x1680` 大图的格式转换和对象封装开销。

进一步的 CUDA device 后处理原型显示：`2880x1680 CHW FP32 -> 1920x1080 RGB uint8` 的自定义 CUDA kernel 仅 `0.037ms/frame`，最终 RGB D2H 仅 `0.451ms/frame`，H2D + TRT + CUDA 后处理 + RGB D2H 合计达到 `139.209 fps`。这证明 30fps 目标在算力层面已经成立，下一步瓶颈只会来自真实视频解码、编码、音频封装和调度。

接入真实 420p 视频链路后，使用 `libx264` 编码已经达到 `47.798 fps`，输出校验为 `1920x1080`、`30fps`、`153` 帧。按该速度估算，2 小时 30fps 视频约 `1.26` 小时完成，已经满足核心速度目标。后续优化重点转为 NVENC、完整样本回归和画质一致性验证。

接入正式 CLI 后，`RUNNER=trt-cuda` 在当前 420p 样本上达到 `53.377 fps`，按该速度估算 2 小时 30fps 视频约 `1.12` 小时完成。该结果仍使用 `libx264`，尚未依赖 NVENC。

## 最高推进指令

当前目标不是继续压榨 Python/PyTorch 脚本，而是重构吞吐架构。主线必须同时解决模型推理、视频 I/O、显存拷贝三类瓶颈。

### Zero-Copy Pipeline

必须切断传统链路：

```text
CPU 解码 -> 内存帧 -> 显存 -> PyTorch 推理 -> 内存帧 -> CPU 编码
```

目标链路：

```text
NVDEC -> GPU surface / CUDA memory -> TRT FP16 inference -> CUDA/NPP postprocess -> NVENC
```

硬性要求：

- 解码优先使用 NVDEC / CUVID。
- 编码优先使用 NVENC。
- 中间帧尽量常驻显存，禁止无意义的 CPU round-trip。
- 420p 非标准比例到 1920x1080 时，使用 CUDA/NPP 或硬件缩放/补边完成尺寸对齐。
- 音频、帧率、时长必须保持同步。

当前镜像能力检查：

| 能力                | 当前状态 | 说明                                         |
| ------------------- | -------- | -------------------------------------------- |
| FFmpeg CUDA hwaccel | 已具备   | `ffmpeg -hwaccels` 可见 `cuda`               |
| NVDEC / CUVID       | 已具备   | 可见 `h264_cuvid`、`hevc_cuvid` 等           |
| NVENC               | 已具备   | 可见 `h264_nvenc`、`hevc_nvenc`、`av1_nvenc` |
| TensorRT Python     | 未具备   | `import tensorrt` 不存在                     |
| ONNX / ONNX Runtime | 未具备   | 当前镜像未安装                               |
| trtexec             | 未具备   | 当前镜像未发现                               |

结论：现有 `0.3.0` 镜像可验证 NVDEC/NVENC，但 TensorRT 主线需要新的性能实验镜像。

当前服务器已经构建 `video2x:trt` 实验镜像，并完成 `realesr-general-x4v3` 的 ONNX 与 TensorRT FP16 engine 验证：

| 项目 | 结果 |
| ---- | ---- |
| ONNX 输入 | `1x3x420x720` |
| ONNX 输出 | `1x3x1680x2880` |
| TensorRT engine | `realesr-general-x4v3-420x720-fp16.engine` |
| Engine size | 2.684 MiB |
| Throughput | 149.447 qps |
| GPU Compute Time mean | 6.6874 ms |
| H2D / D2H | 0 ms，使用 `--noDataTransfers` 测纯推理 |

结论：`realesr-general-x4v3 + TensorRT FP16` 已证明模型侧具备充足吞吐，视频闭环也已跑通，但当前 CPU 后处理吞掉主要时间。下一步必须把 resize/pad/postprocess 移到 GPU 侧，优先 CUDA/NPP 或真正 Zero-Copy 管线。

### TRT FP16 主线

`realesr-general-x4v3 + TensorRT FP16` 升为 P0。

约束：

- 禁止盲目 INT8。超分是像素级回归任务，INT8 容易引入网格、色带、纹理断层。
- 优先 FP16 探顶 RTX 4090 Tensor Core 吞吐。
- 必须记录 PyTorch baseline、ONNX Runtime、TensorRT FP16 三组速度和画质差异。
- 如果 TensorRT FP16 仍无法接近 30fps，再切换模型路线，而不是继续堆参数。

### RTX Video SDK / Maxine

RTX Video SDK / Maxine VSR 仍是 P0，但按黑盒路线管理。

技术陷阱：

- 720p 到 1080p 是 1.5 倍，理论上更容易匹配 VSR/硬件缩放组合。
- 420p 到 1080p 是非整数比例，不能假设黑盒模型直接支持。
- 对 420p 必须验证 NPP / 硬件双线性 / Lanczos 等前后处理组合，监控二次缩放导致的糊边、振铃和字幕变形。

### RealBasicVSR

RealBasicVSR 仍是 P0 竞品，但必须优先验证长视频稳定性。

风险：

- 它是视频模型，存在时序缓存、滑窗、双向传播或未来帧依赖。
- 2 小时视频不能整段塞入显存，必须设计 chunk / overlap / flush 策略。
- 必须测试连续运行时显存是否稳定释放，不能出现显存缓慢上涨。

## 公开基准信息

### Real-ESRGAN / BSRGAN

Real-ESRGAN 官方定位是通用图像/视频修复，但官方仓库仍以 Python/PyTorch 推理脚本为主，安装链路依赖 BasicSR、facexlib、gfpgan 等。  
来源：https://github.com/xinntao/Real-ESRGAN

在 RealBasicVSR 论文的 VideoLQ 真实视频基准中：

| 方法         | 参数量 | Runtime | NRQM ↑ | NIQE ↓ | PI ↓   | BRISQUE ↓ |
| ------------ | ------ | ------- | ------ | ------ | ------ | --------- |
| BSRGAN       | 16.7M  | 149ms   | 5.7172 | 4.2460 | 4.2644 | 30.213    |
| Real-ESRGAN  | 16.7M  | 149ms   | 5.7108 | 4.2091 | 4.2492 | 32.103    |
| RealBasicVSR | 6.3M   | 63ms    | 6.0477 | 3.7662 | 3.8593 | 29.030    |

来源：RealBasicVSR / CVPR 2022，表 3：  
https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.pdf

结论：在该公开基准里，RealBasicVSR 同时比 Real-ESRGAN 更快、更小，并且无参考质量指标更好。它是第一优先竞品。

### BasicVSR++

BasicVSR++ 在合成退化视频超分上 PSNR/SSIM 很强，论文表格中模型参数约 7.3M，Runtime 77ms。  
来源：https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.pdf

但在 RealBasicVSR 的真实退化 VideoLQ 表中，BasicVSR++ 的无参考质量指标明显弱于 Real-ESRGAN 和 RealBasicVSR。  
结论：它适合做对照，不应作为第一质量候选。

### RTX Video SDK / NVIDIA VSR

NVIDIA RTX Video SDK 明确支持 Super Resolution、Artifact Reduction、SDR to HDR，使用 GeForce RTX Tensor Cores 实时增强视频。SDK 1.1 支持 CUDA API、10-bit super resolution，并声明改进了更快的超分模型。  
来源：https://developer.nvidia.com/rtx-video-sdk/getting-started

NVIDIA VFX SDK 文档说明 VSR 是深度学习视频超分，支持 denoise/deblur 相关模式，Linux 驱动要求也已列出。  
来源：https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html

结论：这是最像 DLSS/RTX VSR 的路线，速度希望最大，但模型封闭，质量和离线落盘能力必须本地验证。

### TensorRT

TensorRT 是 NVIDIA 官方高性能推理 SDK，支持从 PyTorch/TensorFlow/ONNX 模型优化部署到 GPU，并支持 FP32/FP16/BF16/FP8/INT8、dynamic shapes 等。  
来源：https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html

结论：TensorRT 是必要但不是充分条件。它能减少 Python/PyTorch overhead、启用 FP16/INT8/engine 优化，但如果模型本身太重，不能保证单独拿到 10 倍。

## 候选优先级

| 优先级 | 候选                                   | 目的                                | 预期收益                           | 风险                                                 |
| ------ | -------------------------------------- | ----------------------------------- | ---------------------------------- | ---------------------------------------------------- |
| P0     | realesr-general-x4v3 TensorRT FP16 + GPU 后处理 | 当前最快模型主线加速                | 纯推理已远超 30fps，视频闭环已定位瓶颈 | 必须消除 CPU 后处理和大图内存拷贝                    |
| P0     | Zero-Copy NVDEC/TRT/NPP/NVENC Pipeline | 消除 I/O 和 CPU round-trip          | 解决吞吐架构瓶颈                   | 工程复杂，需要 C++/CUDA 或 GStreamer/FFmpeg 深度集成 |
| P0     | RTX Video SDK / Maxine VSR             | 验证是否能接近实时                  | 最有 30fps 希望                    | 黑盒模型，420p 非整数倍率是盲区                      |
| P0     | RealBasicVSR                           | 找到质量不输 Real-ESRGAN 的视频模型 | 公开基准显示质量与速度都强         | 视频模型接入复杂，长视频显存稳定性要验证             |
| P1     | Real-ESRGAN TensorRT FP16              | 保持当前质量路线，优化工程栈        | 给现有质量基线一个公平上限         | 重模型可能仍然慢                                     |
| P1     | BasicVSR++                             | 合成退化视频对照                    | 时序一致性好                       | 真实退化指标不如 RealBasicVSR                        |
| P2     | SwinIR / Swin2SR                       | 高质量图片超分对照                  | 可能质量好                         | Transformer 通常不够快，视频一致性要验证             |

## 下一步验证标准

每个候选必须在同一组样本上输出标准 1920x1080，并记录：

- 单卡 RTX 4090 fps。
- 5 分钟、2 小时估算耗时。
- GPU 利用率和显存。
- 是否支持 720p、420p 两类输入。
- 是否能保持音频、帧率、时长。
- 是否做到 NVDEC/NVENC。
- 是否存在 CPU/GPU 往返拷贝。
- TRT FP16 与 PyTorch 输出的画质差异。
- 画质观察：人脸、皮肤纹理、字幕、边缘振铃、噪声、时序闪烁。

最低进入主线标准：

- 速度：720p/420p 至少 15 fps，优先 30 fps。
- 质量：主观观察不能明显低于 Real-ESRGAN，尤其不能糊脸、蜡像化、强振铃。
- 工程：必须 Docker 化，可离线处理本地视频，可写出最终 mp4。
- 同步：音频、帧率、时长必须绝对同步。
- 精度：默认 FP16；INT8 只允许在画质无明显损伤时作为实验分支。
