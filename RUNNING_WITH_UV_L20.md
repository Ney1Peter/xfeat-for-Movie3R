# XFeat L20 服务器运行参考

本文档记录当前 L20 服务器上已经验证通过的 XFeat 环境配置、CUDA/PyTorch 版本选择、最小推理测试、样例图匹配测试和常用运行方式。

## 与 H800 版本的区别

仓库中的 `RUNNING_WITH_UV_H800.md` 是之前的 H800 服务器参考环境：

```text
GPU: NVIDIA H800
Driver: 560.35.03
CUDA: 12.6
PyTorch: 2.11.0+cu126
```

当前 L20 服务器环境不同：

```text
GPU: NVIDIA L20
Driver: 550.127.08
NVIDIA-SMI CUDA: 12.4
PyTorch: 2.6.0+cu124
```

因此在当前 L20 服务器上不要照搬 H800 文档中的 `cu126` 安装命令。当前驱动显示最高支持 CUDA 12.4，应安装 PyTorch 的 `cu124` wheel。

## 当前已验证环境

```text
OS: Ubuntu 20.04, Linux 5.15.0-130-generic, x86_64
GPU: 8 x NVIDIA L20, 46068 MiB each
Driver: 550.127.08
NVIDIA-SMI CUDA: 12.4
uv: 0.9.18
Python in .venv: 3.10.19
PyTorch: 2.6.0+cu124
Torch CUDA runtime: 12.4
```

本次验证使用的 GPU：

```bash
CUDA_VISIBLE_DEVICES=1
```

当前机器上 GPU 3、5、7 曾经有较高显存占用，建议运行测试时优先使用空闲 GPU，例如 1、2、4、6。实际使用前以 `nvidia-smi` 为准。

## 1. 基础检查

在项目根目录执行：

```bash
nvidia-smi
uv --version
python3 --version
uv python list --only-installed
```

当前关键输出应类似：

```text
Driver Version: 550.127.08
CUDA Version: 12.4
uv 0.9.18
Python 3.8.10
cpython-3.10.19 ...
```

系统 Python 是 `3.8.10` 不影响本项目，因为后面会用 `uv` 创建 Python 3.10 虚拟环境。

## 2. 确认项目关键文件

项目根目录应包含：

```text
README.md
RUNNING_WITH_UV_H800.md
RUNNING_WITH_UV_L20.md
requirements.txt
minimal_example.py
realtime_demo.py
modules/xfeat.py
weights/xfeat.pt
weights/xfeat-lighterglue.pt
data/aabb_ref_22010708_00000304.png
data/aabb_cur_22010710_00000305.png
scripts/
```

权重文件是默认推理所需文件：

```text
weights/xfeat.pt
weights/xfeat-lighterglue.pt
```

如果 `weights/xfeat.pt` 不存在，`XFeat()` 默认加载权重时会失败。

## 3. 创建 uv 虚拟环境

在项目根目录执行：

```bash
uv venv .venv --python 3.10
```

本次实际输出：

```text
Using CPython 3.10.19
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

激活环境：

```bash
source .venv/bin/activate
```

也可以不激活，后续命令直接使用 `.venv/bin/python`。

## 4. 安装 PyTorch CUDA 12.4 版本

当前 L20 服务器驱动为 `550.127.08`，`nvidia-smi` 显示 CUDA `12.4`，因此安装 PyTorch `cu124`。

推荐使用本次已验证版本：

```bash
uv pip install --python .venv/bin/python torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

如果不指定版本，当前环境解析到的也是：

```text
torch==2.6.0+cu124
```

本次实际安装得到的关键包包括：

```text
torch==2.6.0+cu124
triton==3.2.0
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cublas-cu12==12.4.5.8
```

不要在当前 L20 服务器上使用 H800 文档中的 `cu126` 命令，除非驱动已经升级到支持 CUDA 12.6 的版本。

## 5. 安装项目依赖

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

当前 `requirements.txt` 内容：

```text
opencv-contrib-python-headless==4.10.0.84
poselib
kornia==0.7.2
tqdm
gdown
```

本次实际安装的关键版本：

```text
numpy==2.2.6
opencv-contrib-python-headless==4.10.0.84
poselib==2.0.5
kornia==0.7.2
tqdm==4.67.3
gdown==6.0.0
```

## 6. 验证 PyTorch 和 GPU

建议指定一张空闲 GPU，例如物理 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

当前验证输出：

```text
2.6.0+cu124
True
12.4
NVIDIA L20
```

如果 `torch.cuda.is_available()` 是 `False`，优先检查：

```text
1. nvidia-smi 是否正常
2. 是否安装了 CPU 版本 PyTorch
3. 是否误装了 cu126 等高于当前驱动能力的 wheel
4. CUDA_VISIBLE_DEVICES 是否设置为空
```

## 7. 验证 XFeat 权重加载和单图推理

运行：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python - <<'PY'
import torch
from modules.xfeat import XFeat

print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')

xfeat = XFeat(top_k=128)
x = torch.randn(1, 3, 480, 640)
out = xfeat.detectAndCompute(x, top_k=128)[0]
print('keypoints:', tuple(out['keypoints'].shape))
print('descriptors:', tuple(out['descriptors'].shape))
print('scores:', tuple(out['scores'].shape))
PY
```

当前验证输出：

```text
torch: 2.6.0+cu124
cuda: True
device: NVIDIA L20
loading weights from: /data/wangzheng/iJCV-CODE/xfeat-for-Movie3R/modules/../weights/xfeat.pt
keypoints: (128, 2)
descriptors: (128, 64)
scores: (128,)
```

说明：

```text
keypoints    -> torch.Tensor(N, 2), 关键点坐标，格式为 x, y
descriptors  -> torch.Tensor(N, 64), 每个关键点的 64 维描述子
scores       -> torch.Tensor(N), 关键点分数
```

## 8. 验证两张样例图匹配

样例图：

```text
data/aabb_ref_22010708_00000304.png
data/aabb_cur_22010710_00000305.png
```

运行 sparse 和 semi-dense 两种匹配，并统计 homography RANSAC 内点：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python - <<'PY'
import cv2
import torch
from modules.xfeat import XFeat

img0_path = 'data/aabb_ref_22010708_00000304.png'
img1_path = 'data/aabb_cur_22010710_00000305.png'
img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
if img0 is None:
    raise RuntimeError(f'Failed to read {img0_path}')
if img1 is None:
    raise RuntimeError(f'Failed to read {img1_path}')

print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('image0:', img0.shape)
print('image1:', img1.shape)

xfeat = XFeat(top_k=4096)

mkpts0, mkpts1 = xfeat.match_xfeat(img0, img1, top_k=4096, min_cossim=0.82)
print('sparse_matches:', len(mkpts0))
if len(mkpts0) >= 4:
    try:
        _, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
    except Exception:
        _, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.999)
    print('sparse_inliers:', int(inliers.reshape(-1).sum()) if inliers is not None else 0)
else:
    print('sparse_inliers:', 0)

mkpts0, mkpts1 = xfeat.match_xfeat_star(img0, img1, top_k=4096)
print('semi_dense_matches:', len(mkpts0))
if len(mkpts0) >= 4:
    try:
        _, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
    except Exception:
        _, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.999)
    print('semi_dense_inliers:', int(inliers.reshape(-1).sum()) if inliers is not None else 0)
else:
    print('semi_dense_inliers:', 0)
PY
```

当前 L20 环境验证输出：

```text
torch: 2.6.0+cu124
cuda: True
device: NVIDIA L20
image0: (2048, 1500, 3)
image1: (2048, 1500, 3)
loading weights from: /data/wangzheng/iJCV-CODE/xfeat-for-Movie3R/modules/../weights/xfeat.pt
sparse_matches: 1270
sparse_inliers: 312
semi_dense_matches: 1022
semi_dense_inliers: 426
```

由于 RANSAC 具有随机性，内点数量可能有小幅波动。只要匹配数量明显大于 0，且内点数量接近上述结果，就说明环境和推理流程正常。

## 9. 生成匹配可视化

如果需要输出可视化图片，运行：

```bash
mkdir -p outputs
CUDA_VISIBLE_DEVICES=1 .venv/bin/python - <<'PY'
import cv2
import numpy as np
from modules.xfeat import XFeat

img0_path = 'data/aabb_ref_22010708_00000304.png'
img1_path = 'data/aabb_cur_22010710_00000305.png'
raw_out_path = 'outputs/l20_xfeat_matches_raw.png'
inlier_out_path = 'outputs/l20_xfeat_matches_inliers.png'

img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
if img0 is None:
    raise RuntimeError(f'Failed to read {img0_path}')
if img1 is None:
    raise RuntimeError(f'Failed to read {img1_path}')

xfeat = XFeat(top_k=4096)
mkpts0, mkpts1 = xfeat.match_xfeat(img0, img1, top_k=4096, min_cossim=0.82)
print('raw_matches:', len(mkpts0))
if len(mkpts0) == 0:
    raise RuntimeError('No matches found. Try lowering min_cossim.')

kp0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts0]
kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts1]
matches = [cv2.DMatch(i, i, 0.0) for i in range(len(mkpts0))]

raw_vis = cv2.drawMatches(
    img0, kp0, img1, kp1, matches[:500], None,
    matchColor=(0, 255, 255), singlePointColor=(255, 0, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.imwrite(raw_out_path, raw_vis)

inlier_mask = None
if len(mkpts0) >= 4:
    try:
        _, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
    except Exception:
        _, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.999)
    if inliers is not None:
        inlier_mask = inliers.reshape(-1).astype(bool)

if inlier_mask is not None and int(inlier_mask.sum()) > 0:
    draw_indices = np.flatnonzero(inlier_mask)[:500]
    title = f'XFeat L20 sparse: raw={len(matches)}, inliers={int(inlier_mask.sum())}'
    print('ransac_inliers:', int(inlier_mask.sum()))
else:
    draw_indices = np.arange(min(len(matches), 500))
    title = f'XFeat L20 sparse: raw={len(matches)}, no homography inliers'
    print('ransac_inliers:', 0)

draw_matches = [matches[i] for i in draw_indices]
vis = cv2.drawMatches(
    img0, kp0, img1, kp1, draw_matches, None,
    matchColor=(0, 220, 0), singlePointColor=(255, 0, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.rectangle(vis, (0, 0), (min(vis.shape[1] - 1, 1000), 44), (0, 0, 0), -1)
cv2.putText(vis, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imwrite(inlier_out_path, vis)
print('saved:', raw_out_path)
print('saved:', inlier_out_path)
PY
```

输出文件：

```text
outputs/l20_xfeat_matches_raw.png
outputs/l20_xfeat_matches_inliers.png
```

## 10. 运行 minimal_example.py

可以运行官方最小示例：

```bash
.venv/bin/python minimal_example.py
```

注意：`minimal_example.py` 内部有这一行：

```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

它会强制使用 CPU。因此该脚本只能作为 CPU 路径 smoke test，不能用于验证 GPU 是否可用。如果要用 GPU 跑，需要注释掉这行，或使用本文第 7、8 节的 GPU 验证命令。

## 11. 在其他脚本中运行

通用格式：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python your_script.py
```

例如 AvatarReX 样例诊断：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/test_avatarrex_xfeat_geometry.py
```

例如 RICH mesh 几何诊断：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/test_rich_aabb_xfeat_mesh_geometry.py \
  --rich_root /workspace/data/RICH \
  --data_root /workspace/data/RICH/RICH_4Human3R/Training \
  --source_sequence BBQ_001_juggle \
  --cam_a 3 \
  --cam_b 4 \
  --start_frame 100 \
  --match_mode semidense
```

如果使用 RICH 相关脚本，需要确认本机存在对应数据路径。

## 12. 常见问题

### 误装 cu126

现象可能包括：

```text
CUDA driver version is insufficient for CUDA runtime version
torch.cuda.is_available() == False
```

处理方式：

```bash
uv pip uninstall --python .venv/bin/python torch
uv pip install --python .venv/bin/python torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 找不到权重

检查：

```bash
git ls-files weights/xfeat.pt weights/xfeat-lighterglue.pt
```

默认权重路径：

```text
weights/xfeat.pt
```

### GPU 被占满

先查看：

```bash
nvidia-smi
```

然后选择空闲卡运行：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python your_script.py
```

### Kornia FutureWarning

当前环境运行时可能出现：

```text
FutureWarning: torch.cuda.amp.custom_fwd(args...) is deprecated
```

这是 Kornia 内部 API 弃用提示，不影响 XFeat 当前推理和匹配运行。

## 13. 当前 L20 环境验收标准

环境配置完成后，至少满足：

```text
torch.__version__ == 2.6.0+cu124
torch.cuda.is_available() == True
torch.version.cuda == 12.4
torch.cuda.get_device_name(0) == NVIDIA L20
XFeat() 能加载 weights/xfeat.pt
detectAndCompute() 能输出 keypoints/descriptors/scores
样例图 sparse/semi-dense 匹配数量大于 0
```

当前完整验证结果：

```text
sparse_matches: 1270
sparse_inliers: 312
semi_dense_matches: 1022
semi_dense_inliers: 426
```
