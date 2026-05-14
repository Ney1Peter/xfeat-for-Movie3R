# XFeat H800 服务器运行参考

本文档记录 H800 服务器上如何恢复项目、用 `uv` 配置环境、运行最小测试、执行两张图片的特征匹配可视化，以及各脚本的输入输出。可作为 H800 环境接入 XFeat 的参考。

注意：本文档记录的是 H800 / Driver 560 / CUDA 12.6 / PyTorch cu126 参考环境。当前 L20 / Driver 550 / CUDA 12.4 服务器请使用 `RUNNING_WITH_UV_L20.md`。

## 新设备快速恢复

在新设备上从零恢复项目，优先执行这一节：

```bash
git clone https://github.com/Ney1Peter/xfeat-for-Movie3R.git
cd xfeat-for-Movie3R
git checkout main
git pull --ff-only origin main
```

确认关键文件已经随 Git 拉取下来：

```bash
git ls-files RUNNING_WITH_UV_H800.md RUNNING_WITH_UV_L20.md requirements.txt minimal_example.py realtime_demo.py weights data scripts
```

应该至少包含这些文件：

```text
RUNNING_WITH_UV_H800.md
RUNNING_WITH_UV_L20.md
requirements.txt
minimal_example.py
realtime_demo.py
weights/xfeat.pt
weights/xfeat-lighterglue.pt
data/aabb_ref_22010708_00000304.png
data/aabb_cur_22010710_00000305.png
scripts/test_avatarrex_xfeat_geometry.py
scripts/test_rich_aabb_xfeat_geometry.py
scripts/test_rich_aabb_xfeat_mesh_geometry.py
scripts/compute_rich_camera_overlap.py
scripts/visualize_rich_gt_projection.py
scripts/visualize_rich_mesh_projection.py
scripts/visualize_rich_mesh_correspondences.py
```

然后继续执行第 1 节到第 4 节完成环境配置和最小测试。如果只想验证两张样例图的匹配，继续执行第 5 节或第 6 节。

当前仓库里的 `outputs/` 是运行生成结果，不需要随 Git 迁移；在新设备上重新运行脚本即可生成。

## 1. 环境检查

先检查 GPU 和 `uv` 是否可用：

```bash
nvidia-smi
uv --version
```

本项目当前验证环境：

```text
GPU: NVIDIA H800
Driver: 560.35.03
CUDA: 12.6
uv: 0.11.7
Python: 3.10.20
PyTorch: 2.11.0+cu126
```

## 2. 使用 uv 创建环境

在项目根目录执行：

```bash
uv venv .venv --python 3.10
```

安装 CUDA 12.6 版本 PyTorch：

```bash
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu126
```

安装项目依赖：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

激活环境：

```bash
source .venv/bin/activate
```

验证 PyTorch 是否识别 GPU：

```bash
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

如果只需要 CPU，也可以安装 CPU 版 PyTorch，参考 PyTorch 官网选择合适 wheel。

## 3. 权重文件

默认推理权重路径：

```text
weights/xfeat.pt
```

LighterGlue 权重路径：

```text
weights/xfeat-lighterglue.pt
```

`XFeat()` 默认会从 `weights/xfeat.pt` 加载权重。如果缺少该文件，需要先下载或复制到上述路径。

## 4. 最小测试脚本

运行：

```bash
.venv/bin/python minimal_example.py
```

输入：

```python
torch.Tensor(B, C, H, W)
```

示例中使用随机输入：

```python
torch.randn(1, 3, 480, 640)
```

输出：

```python
output = xfeat.detectAndCompute(image, top_k=4096)[0]
```

字段含义：

```text
output["keypoints"]    -> torch.Tensor(N, 2), 关键点坐标，格式为 x, y
output["descriptors"]  -> torch.Tensor(N, 64), 每个关键点的 64 维描述子
output["scores"]       -> torch.Tensor(N), 关键点分数
```

注意：`minimal_example.py` 内部设置了：

```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

这会强制使用 CPU。如果要用 GPU 跑该脚本，需要注释掉这一行。

## 5. 两张图片匹配并输出可视化

本项目中示例输入图片放在：

```text
data/aabb_ref_22010708_00000304.png
data/aabb_cur_22010710_00000305.png
```

运行下面命令会执行 XFeat 匹配，使用 RANSAC 过滤几何一致内点，并输出可视化图片：

```bash
mkdir -p outputs
.venv/bin/python - <<'PY'
import cv2
import numpy as np
import torch
from modules.xfeat import XFeat

img0_path = 'data/aabb_ref_22010708_00000304.png'
img1_path = 'data/aabb_cur_22010710_00000305.png'
out_path = 'outputs/xfeat_matches_inliers.png'
raw_out_path = 'outputs/xfeat_matches_raw.png'

img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
if img0 is None:
    raise RuntimeError(f'Failed to read {img0_path}')
if img1 is None:
    raise RuntimeError(f'Failed to read {img1_path}')

print('image0:', img0_path, img0.shape)
print('image1:', img1_path, img1.shape)
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())

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
    inlier_indices = np.flatnonzero(inlier_mask)
    draw_matches = [matches[i] for i in inlier_indices[:500]]
    title = f'XFeat matches: raw={len(matches)}, inliers={int(inlier_mask.sum())}'
    print('ransac_inliers:', int(inlier_mask.sum()))
else:
    draw_matches = matches[:500]
    title = f'XFeat matches: raw={len(matches)}, no homography inliers'
    print('ransac_inliers:', 0)

vis = cv2.drawMatches(
    img0, kp0, img1, kp1, draw_matches, None,
    matchColor=(0, 220, 0), singlePointColor=(255, 0, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.rectangle(vis, (0, 0), (min(vis.shape[1] - 1, 900), 44), (0, 0, 0), -1)
cv2.putText(vis, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imwrite(out_path, vis)
print('saved:', out_path)
print('saved:', raw_out_path)
PY
```

输入：

```text
两张图片，OpenCV 可读取格式均可，例如 png/jpg/jpeg/bmp/tif。
默认读取为 BGR 格式的 numpy.ndarray，形状为 H, W, 3。
```

核心调用：

```python
mkpts0, mkpts1 = xfeat.match_xfeat(img0, img1, top_k=4096, min_cossim=0.82)
```

输出：

```text
mkpts0 -> np.ndarray(N, 2), 第一张图中的匹配点，格式 x, y
mkpts1 -> np.ndarray(N, 2), 第二张图中的匹配点，格式 x, y
```

可视化输出：

```text
outputs/xfeat_matches_inliers.png  # RANSAC 内点匹配结果
outputs/xfeat_matches_raw.png      # 原始匹配结果预览
```

本项目当前两张示例图的运行结果：

```text
raw_matches: 1272
ransac_inliers: 316
```

## 6. 使用 semi-dense 匹配

XFeat 还提供 semi-dense 匹配，代码入口是：

```python
mkpts0, mkpts1 = xfeat.match_xfeat_star(img0, img1, top_k=4096)
```

它和 `match_xfeat()` 不同：

```text
match_xfeat()      -> sparse matching，先检测关键点，再匹配关键点描述子。
match_xfeat_star() -> semi-dense matching，提取更密集的 coarse features，再用 fine matcher 做匹配细化。
```

运行下面命令可以对同一组图片执行 semi-dense 匹配并输出可视化：

```bash
mkdir -p outputs
.venv/bin/python - <<'PY'
import cv2
import numpy as np
import torch
from modules.xfeat import XFeat

img0_path = 'data/aabb_ref_22010708_00000304.png'
img1_path = 'data/aabb_cur_22010710_00000305.png'
out_path = 'outputs/xfeat_semidense_matches.png'

img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
if img0 is None:
    raise RuntimeError(f'Failed to read {img0_path}')
if img1 is None:
    raise RuntimeError(f'Failed to read {img1_path}')

print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())

xfeat = XFeat(top_k=4096)
mkpts0, mkpts1 = xfeat.match_xfeat_star(img0, img1, top_k=4096)
print('semi_dense_matches:', len(mkpts0))

if len(mkpts0) == 0:
    raise RuntimeError('No matches found.')

inlier_mask = None
if len(mkpts0) >= 4:
    try:
        _, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 4.0, maxIters=5000, confidence=0.999)
    except Exception:
        _, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.999)
    if inliers is not None:
        inlier_mask = inliers.reshape(-1).astype(bool)

if inlier_mask is not None and int(inlier_mask.sum()) > 0:
    draw_ids = np.flatnonzero(inlier_mask)[:500]
    title = f'XFeat semi-dense: raw={len(mkpts0)}, inliers={int(inlier_mask.sum())}'
    print('ransac_inliers:', int(inlier_mask.sum()))
else:
    draw_ids = np.arange(min(len(mkpts0), 500))
    title = f'XFeat semi-dense: raw={len(mkpts0)}, no homography inliers'
    print('ransac_inliers:', 0)

kp0 = [cv2.KeyPoint(float(mkpts0[i][0]), float(mkpts0[i][1]), 5) for i in draw_ids]
kp1 = [cv2.KeyPoint(float(mkpts1[i][0]), float(mkpts1[i][1]), 5) for i in draw_ids]
matches = [cv2.DMatch(i, i, 0.0) for i in range(len(draw_ids))]

vis = cv2.drawMatches(
    img0, kp0, img1, kp1, matches, None,
    matchColor=(0, 220, 0), singlePointColor=(255, 0, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
cv2.rectangle(vis, (0, 0), (min(vis.shape[1] - 1, 1100), 44), (0, 0, 0), -1)
cv2.putText(vis, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imwrite(out_path, vis)
print('saved:', out_path)
PY
```

输入：

```text
两张图片，OpenCV 可读取格式均可，例如 png/jpg/jpeg/bmp/tif。
默认读取为 BGR 格式的 numpy.ndarray，形状为 H, W, 3。
```

输出：

```text
mkpts0 -> np.ndarray(N, 2), 第一张图中的匹配点，格式 x, y
mkpts1 -> np.ndarray(N, 2), 第二张图中的匹配点，格式 x, y
```

可视化输出：

```text
outputs/xfeat_semidense_matches.png
```

本项目当前两张示例图的对比结果：

```text
sparse_matches: 1272
sparse_inliers: 316
semi_dense_matches: 1019
semi_dense_inliers: 424
```

这组图上 semi-dense 的原始匹配更少，但几何一致内点更多，内点率更高。对于配准、姿态估计、几何验证这类任务，建议优先尝试 `match_xfeat_star()`；如果更看重简单和低延迟，可以继续使用 `match_xfeat()`。

## 7. 生成缩小预览图

如果完整可视化图太大，可以生成压缩预览：

```bash
.venv/bin/python - <<'PY'
import cv2

src = 'outputs/xfeat_matches_inliers.png'
dst = 'outputs/xfeat_matches_inliers_preview.jpg'
img = cv2.imread(src, cv2.IMREAD_COLOR)
if img is None:
    raise RuntimeError(f'Failed to read {src}')

max_w = 900
scale = min(1.0, max_w / img.shape[1])
preview = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
cv2.imwrite(dst, preview, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
print('saved:', dst, preview.shape)
PY
```

输出：

```text
outputs/xfeat_matches_inliers_preview.jpg
```

## 8. 实时 Demo

查看参数：

```bash
.venv/bin/python realtime_demo.py -h
```

使用 XFeat 启动摄像头实时匹配：

```bash
.venv/bin/python realtime_demo.py --method XFeat
```

也可以对比 SIFT 或 ORB：

```bash
.venv/bin/python realtime_demo.py --method SIFT
.venv/bin/python realtime_demo.py --method ORB
```

输入：

```text
摄像头视频流。
默认摄像头编号为 0，默认分辨率为 640x480。
```

常用参数：

```text
--width      视频宽度，默认 640
--height     视频高度，默认 480
--max_kpts   最大关键点数量，默认 3000
--method     ORB / SIFT / XFeat，默认 XFeat
--cam        摄像头编号，默认 0
```

输出：

```text
OpenCV 窗口显示参考帧、当前帧、匹配线和估计的单应性注册结果。
```

使用方式：

```text
按 s 设置参考图像。
该 demo 使用 homography 模型，适合平面场景或纯旋转运动。
```

## 9. 在其他项目中接入 XFeat

最小代码：

```python
import cv2
from modules.xfeat import XFeat

xfeat = XFeat(top_k=4096)

img0 = cv2.imread('image0.png', cv2.IMREAD_COLOR)
img1 = cv2.imread('image1.png', cv2.IMREAD_COLOR)

mkpts0, mkpts1 = xfeat.match_xfeat(img0, img1, top_k=4096, min_cossim=0.82)
```

如果只提取单张图的局部特征：

```python
import torch
from modules.xfeat import XFeat

xfeat = XFeat(top_k=4096)
image = torch.randn(1, 3, 480, 640)
features = xfeat.detectAndCompute(image, top_k=4096)[0]

keypoints = features['keypoints']
descriptors = features['descriptors']
scores = features['scores']
```

## 10. 诊断脚本

本仓库还包含一些面向 Movie3R/RICH/AvatarReX 数据的诊断脚本。它们不是运行 XFeat 的必要步骤，但已经提交到 Git，可在新设备上直接使用。

### AvatarReX 样例几何诊断

脚本：

```text
scripts/test_avatarrex_xfeat_geometry.py
```

默认输入使用仓库中的两张样例图：

```text
data/aabb_ref_22010708_00000304.png
data/aabb_cur_22010710_00000305.png
```

运行：

```bash
.venv/bin/python scripts/test_avatarrex_xfeat_geometry.py
```

常用参数：

```text
--img0             第一张图片路径
--img1             第二张图片路径
--top_k            最大关键点数量，默认 4096
--min_cossim       sparse 匹配余弦阈值，默认 0.82
--reproj_thresh    重投影误差阈值，默认 32.0
--depth_rel_thresh 深度相对误差阈值，默认 0.20
--ransac_thresh    RANSAC 阈值，默认 4.0
--out_dir          输出目录，默认 outputs/avatarrex_xfeat_a5b5
```

输出：

```text
JSON 统计结果
匹配/几何内点可视化图片
```

### RICH 生成深度几何诊断

脚本：

```text
scripts/test_rich_aabb_xfeat_geometry.py
```

该脚本依赖本地 RICH 数据，默认路径为：

```text
/workspace/data/RICH/RICH_4Human3R/Training
```

运行示例：

```bash
.venv/bin/python scripts/test_rich_aabb_xfeat_geometry.py \
  --data_root /workspace/data/RICH/RICH_4Human3R/Training \
  --source_sequence BBQ_001_guitar \
  --cam_a 0 \
  --cam_b 1 \
  --start_frame 100
```

输出：

```text
RICH 图像对的 XFeat 匹配统计
RANSAC/深度几何一致性统计
outputs/ 下的可视化结果
```

### RICH mesh 几何诊断

脚本：

```text
scripts/test_rich_aabb_xfeat_mesh_geometry.py
```

该脚本使用 RICH 官方静态 scan mesh 和 XML 标定，比生成深度 `.npy` 更适合评估静态背景几何。默认路径：

```text
--rich_root /workspace/data/RICH
--data_root /workspace/data/RICH/RICH_4Human3R/Training
```

运行 sparse 模式：

```bash
.venv/bin/python scripts/test_rich_aabb_xfeat_mesh_geometry.py \
  --rich_root /workspace/data/RICH \
  --data_root /workspace/data/RICH/RICH_4Human3R/Training \
  --source_sequence BBQ_001_juggle \
  --cam_a 3 \
  --cam_b 4 \
  --start_frame 100 \
  --match_mode sparse
```

运行 semi-dense 模式：

```bash
.venv/bin/python scripts/test_rich_aabb_xfeat_mesh_geometry.py \
  --rich_root /workspace/data/RICH \
  --data_root /workspace/data/RICH/RICH_4Human3R/Training \
  --source_sequence BBQ_001_juggle \
  --cam_a 3 \
  --cam_b 4 \
  --start_frame 100 \
  --match_mode semidense
```

常用参数：

```text
--top_k                 最大特征数量，默认 8192
--min_cossim            sparse 匹配余弦阈值，默认 0.9
--match_mode            sparse 或 semidense，默认 sparse
--max_dim               匹配前图片最大边，默认 1200
--mesh_max_dim          mesh 投影最大边，默认 1400
--mesh_lookup_radius    mesh 顶点查找半径，默认 4
--mesh_z_tol            mesh z-buffer 容差，默认 0.03
--reproj_thresh         mesh 重投影阈值，默认 24.0
--out_dir               输出目录，默认按参数自动生成
```

输出：

```text
匹配数量
homography/fundamental RANSAC 内点数量
mesh geometry 内点数量
mesh visible overlap 统计
可视化图片和 JSON 报告
```

### RICH 相机重叠度计算

脚本：

```text
scripts/compute_rich_camera_overlap.py
```

用途：基于 RICH 静态 scan mesh 计算不同相机之间的可见顶点重叠度，并默认排除投影到 human mask 上的顶点，使指标更接近静态背景 anchor 的可用量。

运行：

```bash
.venv/bin/python scripts/compute_rich_camera_overlap.py \
  --rich_root /workspace/data/RICH \
  --data_root /workspace/data/RICH/RICH_4Human3R/Training \
  --source_sequence BBQ_001_juggle
```

只计算指定相机：

```bash
.venv/bin/python scripts/compute_rich_camera_overlap.py \
  --rich_root /workspace/data/RICH \
  --data_root /workspace/data/RICH/RICH_4Human3R/Training \
  --source_sequence BBQ_001_juggle \
  --cams 3,4,5
```

输出：

```text
overlap_summary.json
overlap_pairs_sorted.csv
training_pairs_overlap_sorted.csv
overlap_min_matrix.csv
jaccard_matrix.csv
overlap_min_matrix.jpg
jaccard_matrix.jpg
```

输出目录默认类似：

```text
outputs/rich_camera_overlap_<source_sequence>_<frame>_<mask_mode>
```

### RICH 可视化辅助脚本

脚本：

```text
scripts/visualize_rich_gt_projection.py
scripts/visualize_rich_mesh_projection.py
scripts/visualize_rich_mesh_correspondences.py
```

用途：检查 RICH 生成深度、静态 mesh 投影、跨相机 mesh 对应点是否合理。

运行示例：

```bash
.venv/bin/python scripts/visualize_rich_mesh_projection.py \
  --rich_root /workspace/data/RICH \
  --data_root /workspace/data/RICH/RICH_4Human3R/Training \
  --source_sequence BBQ_001_juggle \
  --cam 3 \
  --frame 101
```

```bash
.venv/bin/python scripts/visualize_rich_mesh_correspondences.py \
  --rich_root /workspace/data/RICH \
  --data_root /workspace/data/RICH/RICH_4Human3R/Training \
  --source_sequence BBQ_001_juggle \
  --cam_a 3 \
  --cam_b 4 \
  --frame_a 101 \
  --frame_b 102
```

## 11. 常见问题

如果提示找不到权重：

```text
检查 weights/xfeat.pt 是否存在。
```

如果 `torch.cuda.is_available()` 为 `False`：

```text
检查 nvidia-smi 是否正常。
检查安装的 PyTorch wheel 是否和 CUDA/驱动匹配。
如果只安装了 CPU 版 PyTorch，需要重新安装 CUDA 版本。
```

如果匹配数量太少：

```text
适当降低 min_cossim，例如从 0.82 降到 0.7。
增加 top_k，例如从 4096 增加到 8000。
检查两张图是否确实有重叠区域。
```

如果可视化图片太大：

```text
使用第 7 节生成 preview jpg。
减少 drawMatches 中绘制的匹配数量，例如只画前 200 条。
```
