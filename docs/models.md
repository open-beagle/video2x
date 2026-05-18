# 模型准备

## 1. 本地目录

模型不打包进 Docker 镜像。

服务器无法访问 GitHub，因此不要依赖容器在服务器上运行时下载模型。请先把模型文件下载或转换到当前项目的本地目录：

```text
models
```

部署时再把 `models/` 同步到目标机器的模型目录。运行容器时把宿主机模型目录挂载到容器内：

```bash
-v /path/to/models:/models
```

容器内默认读取：

```text
/models
```

## 2. 当前运行必备文件

当前默认运行路线是：

```text
realesr-general-x4v3 -> ONNX -> TensorRT FP16 engine -> ZeroCopy TRT-CUDA 视频处理
```

运行 `0.3.0` 建议准备标准输入规格 engine：

```text
models/realesr-general-x4v3-640x360-fp16.engine
models/realesr-general-x4v3-720x480-fp16.engine
models/realesr-general-x4v3-854x480-fp16.engine
models/realesr-general-x4v3-960x540-fp16.engine
models/realesr-general-x4v3-1280x720-fp16.engine
```

如果启用 `VIDEO_POSTPROCESS_MODE=srvgg-conv48-tail`，还需要对应 conv48 engine：

```text
models/realesr-general-x4v3-960x540-conv48-fp16.engine
models/realesr-general-x4v3-1280x720-conv48-fp16.engine
```

0.3.0 正式回归中的两条 720p 路线：

```text
性能线：1280x720 源视频 -> 960x540 conv48 engine -> 1920x1080，77.106fps
质量线：1280x720 源视频 -> 1280x720 conv48 engine -> 1920x1080，45.124fps
```

其中 `720x420` 不是标准分辨率，只是当前样本兼容规格；如果仍要处理该类样本，可以额外保留：

```text
models/realesr-general-x4v3-720x420-fp16.engine
```

注意：当前 engine 是固定输入尺寸，文件名使用 `宽x高`，而 Tensor shape 仍是 `NCHW`。例如 `720x480` engine 对应输入 shape `1x3x480x720`。

## 3. 源权重与中间产物

为了重新导出或复现实验，建议本地 `models/` 保留：

```text
realesr-general-x4v3.pth
realesr-general-wdn-x4v3.pth
RealESRGAN_x2plus.pth
RealESRGAN_x4plus.pth
```

其中运行默认只读取 `.engine`。`.pth`、`.onnx` 和 `.onnx.data` 用于模型转换、重建 engine、画质对比和后续实验。

build 镜像会为所有已存在且受支持的 `.pth` 扫描构建：

```text
640x360
720x480
854x480
960x540
1280x720
720x420
```

对 `realesr-general-x4v3` 和 `realesr-general-wdn-x4v3`，还会额外生成：

```text
{model}-{width}x{height}-conv48.onnx
{model}-{width}x{height}-conv48.onnx.data
{model}-{width}x{height}-conv48-fp16.engine
{model}-tail-{width}x{height}-conv48.npz
```

## 4. 本地下载源权重

```bash
mkdir -p models

curl -L -o models/realesr-general-x4v3.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth

curl -L -o models/realesr-general-wdn-x4v3.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth

curl -L -o models/RealESRGAN_x2plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

curl -L -o models/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

## 5. 导出 ONNX

本项目提供导出脚本：

```bash
python tools/export_realesrgan_onnx.py \
  --model realesr-general-x4v3 \
  --weights models/realesr-general-x4v3.pth \
  --output models/realesr-general-x4v3-720x420.onnx \
  --height 420 \
  --width 720
```

导出脚本会使用本项目内的 Real-ESRGAN 代码作为模型结构来源。它是模型制作工具，不是默认视频推理链路。

## 6. 构建 TensorRT Engine

在具备 TensorRT 的机器上构建 FP16 engine。推荐使用 build 镜像：

```text
video2x:0.3.0-build
```

build 镜像默认会自动扫描 `/models` 并构建标准 profile：

```bash
docker run --rm \
  --device nvidia.com/gpu=0 \
  -v /path/to/models:/models \
  video2x:0.3.0-build
```

0.3.0 正式 build 镜像已在 RTX 4090 服务器通过零参数回归，确认全部支持模型 engine 可生成或复用，general v3 / wdn general v3 的 conv48 profile 已补齐。

不要默认使用 INT8。超分是像素级回归任务，INT8 可能带来断层、网格和细节异常，只能作为单独画质验证分支。

## 7. 同步到部署机器

```bash
rsync -av models/ user@host:/path/to/models/
```

Windows PowerShell 可使用：

```powershell
scp -r models\* user@host:/path/to/models/
```

公开文档只记录相对目录和占位路径，不记录真实部署路径。

## 8. 容器挂载示例

```bash
docker run --rm \
  --device nvidia.com/gpu=0 \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  registry.cn-qingdao.aliyuncs.com/wod/video2x:0.3.0
```
