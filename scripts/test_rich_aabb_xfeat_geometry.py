#!/usr/bin/env python3
"""Run XFeat on a RICH AABB-style sample with generated-depth geometry checks.

Important: in this workspace RICH depth/*.npy files are Depth Anything outputs,
not metric RICH ground-truth depth. Use test_rich_aabb_xfeat_mesh_geometry.py
for metric cross-camera static-scene validation against the official scan mesh.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.xfeat import XFeat


DATA_ROOT = Path("/workspace/data/RICH/RICH_4Human3R/Training")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--source_sequence", default="BBQ_001_guitar")
    parser.add_argument("--cam_a", type=int, default=0)
    parser.add_argument("--cam_b", type=int, default=1)
    parser.add_argument("--start_frame", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=4096)
    parser.add_argument("--min_cossim", type=float, default=0.82)
    parser.add_argument("--max_dim", type=int, default=1200)
    parser.add_argument("--depth_scale", type=float, default=0.001, help="Scale uint depth to camera-pose units.")
    parser.add_argument("--reproj_thresh", type=float, default=24.0)
    parser.add_argument("--depth_rel_thresh", type=float, default=0.25)
    parser.add_argument("--ransac_thresh", type=float, default=4.0)
    parser.add_argument("--max_draw", type=int, default=500)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def seq_name(source_sequence, cam):
    return f"{source_sequence}_cam_{cam:02d}"


def frame_path(data_root, seq, frame, subdir, suffix):
    return Path(data_root) / seq / subdir / f"{frame:08d}{suffix}"


def load_rgb(data_root, seq, frame):
    path = frame_path(data_root, seq, frame, "rgb", ".png")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img, path


def load_depth(data_root, seq, frame, depth_scale):
    path = frame_path(data_root, seq, frame, "depth", ".npy")
    depth = np.load(path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    if depth_scale is not None:
        depth *= float(depth_scale)
    return depth


def load_camera(data_root, seq, frame):
    path = frame_path(data_root, seq, frame, "cam", ".npz")
    cam = np.load(path)
    return cam["pose"].astype(np.float32), cam["intrinsics"].astype(np.float32)


def load_mask(data_root, seq, frame, shape):
    path = frame_path(data_root, seq, frame, "mask", ".png")
    if not path.exists():
        return np.zeros(shape[:2], dtype=bool)
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return np.zeros(shape[:2], dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def resize_for_matching(img, max_dim):
    h, w = img.shape[:2]
    if max_dim is None or max_dim <= 0 or max(h, w) <= max_dim:
        return img, 1.0, 1.0
    scale = float(max_dim) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    sx = float(w) / float(new_w)
    sy = float(h) / float(new_h)
    return resized, sx, sy


def to_original_coords(kpts, sx, sy):
    if len(kpts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = np.asarray(kpts, dtype=np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def sample_depth(depth, x, y, radius=4):
    h, w = depth.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    x0 = max(0, xi - radius)
    x1 = min(w, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius + 1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def project_cur_to_ref(x_cur, y_cur, depth_cur, c2w_cur, k_cur, c2w_ref, k_ref):
    pix_cur = np.array([x_cur, y_cur, 1.0], dtype=np.float32)
    xyz_cur = np.linalg.inv(k_cur) @ pix_cur * depth_cur
    xyz_cur_h = np.concatenate([xyz_cur, np.ones(1, dtype=np.float32)])
    xyz_world = c2w_cur @ xyz_cur_h
    xyz_ref = np.linalg.inv(c2w_ref) @ xyz_world
    z_ref = float(xyz_ref[2])
    if z_ref <= 1e-6:
        return None
    pix_ref = k_ref @ xyz_ref[:3]
    x_ref = float(pix_ref[0] / pix_ref[2])
    y_ref = float(pix_ref[1] / pix_ref[2])
    return x_ref, y_ref, z_ref


def compute_ransac_inliers(mkpts0, mkpts1, thresh):
    if len(mkpts0) < 4:
        return np.zeros((len(mkpts0),), dtype=bool)
    try:
        _, inliers = cv2.findHomography(
            mkpts0,
            mkpts1,
            cv2.USAC_MAGSAC,
            thresh,
            maxIters=5000,
            confidence=0.999,
        )
    except Exception:
        _, inliers = cv2.findHomography(
            mkpts0,
            mkpts1,
            cv2.RANSAC,
            thresh,
            maxIters=5000,
            confidence=0.999,
        )
    if inliers is None:
        return np.zeros((len(mkpts0),), dtype=bool)
    return inliers.reshape(-1).astype(bool)


def evaluate_gt_geometry(mkpts_ref_orig, mkpts_cur_orig, data, args):
    depth_ref = data["depth_ref"]
    depth_cur = data["depth_cur"]
    c2w_ref = data["c2w_ref"]
    c2w_cur = data["c2w_cur"]
    k_ref = data["k_ref"]
    k_cur = data["k_cur"]
    mask_ref = data["mask_ref"]
    mask_cur = data["mask_cur"]
    h_ref, w_ref = depth_ref.shape[:2]
    h_cur, w_cur = depth_cur.shape[:2]

    items = []
    for idx, (p_ref, p_cur) in enumerate(zip(mkpts_ref_orig, mkpts_cur_orig)):
        x_ref, y_ref = map(float, p_ref)
        x_cur, y_cur = map(float, p_cur)

        ref_inside = 0 <= x_ref < w_ref and 0 <= y_ref < h_ref
        cur_inside = 0 <= x_cur < w_cur and 0 <= y_cur < h_cur
        on_human = False
        if ref_inside:
            on_human = on_human or bool(mask_ref[int(round(y_ref)), int(round(x_ref))])
        if cur_inside:
            on_human = on_human or bool(mask_cur[int(round(y_cur)), int(round(x_cur))])

        depth_c = sample_depth(depth_cur, x_cur, y_cur)
        depth_r = sample_depth(depth_ref, x_ref, y_ref)
        reproj_error = None
        depth_rel_error = None
        projected_in_image = False
        gt_inlier = False

        if ref_inside and cur_inside and not on_human and depth_c is not None:
            projected = project_cur_to_ref(x_cur, y_cur, depth_c, c2w_cur, k_cur, c2w_ref, k_ref)
            if projected is not None:
                px, py, z_ref = projected
                projected_in_image = 0 <= px < w_ref and 0 <= py < h_ref
                reproj_error = float(math.hypot(px - x_ref, py - y_ref))
                if depth_r is not None:
                    depth_rel_error = float(abs(z_ref - depth_r) / max(abs(depth_r), 1e-6))
                depth_ok = depth_rel_error is None or depth_rel_error < args.depth_rel_thresh
                gt_inlier = bool(projected_in_image and reproj_error < args.reproj_thresh and depth_ok)

        items.append(
            {
                "idx": int(idx),
                "ref_xy_original": [x_ref, y_ref],
                "cur_xy_original": [x_cur, y_cur],
                "on_human": bool(on_human),
                "depth_cur": depth_c,
                "depth_ref": depth_r,
                "projected_in_image": bool(projected_in_image),
                "reproj_error_px": reproj_error,
                "depth_rel_error": depth_rel_error,
                "gt_inlier": bool(gt_inlier),
            }
        )
    return items


def draw_matches(img0, img1, mkpts0, mkpts1, indices, out_path, color, title):
    kp0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts0]
    kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in mkpts1]
    matches = [cv2.DMatch(int(i), int(i), 0.0) for i in indices]
    vis = cv2.drawMatches(
        img0,
        kp0,
        img1,
        kp1,
        matches,
        None,
        matchColor=color,
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1] - 1, 1800), 54), (0, 0, 0), -1)
    cv2.putText(vis, title, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def make_preview(src_path, max_w=1400):
    src = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if src is None:
        return None
    scale = min(1.0, float(max_w) / float(src.shape[1]))
    preview = cv2.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    out_path = src_path.with_name(src_path.stem + "_preview.jpg")
    cv2.imwrite(str(out_path), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return out_path


def run_pair(xfeat, pair, out_dir, args):
    img_ref, img_ref_path = load_rgb(args.data_root, pair["ref_seq"], pair["ref_frame"])
    img_cur, img_cur_path = load_rgb(args.data_root, pair["cur_seq"], pair["cur_frame"])

    img_ref_match, sx_ref, sy_ref = resize_for_matching(img_ref, args.max_dim)
    img_cur_match, sx_cur, sy_cur = resize_for_matching(img_cur, args.max_dim)

    mkpts_ref, mkpts_cur = xfeat.match_xfeat(
        img_ref_match,
        img_cur_match,
        top_k=args.top_k,
        min_cossim=args.min_cossim,
    )
    mkpts_ref = np.asarray(mkpts_ref, dtype=np.float32)
    mkpts_cur = np.asarray(mkpts_cur, dtype=np.float32)
    mkpts_ref_orig = to_original_coords(mkpts_ref, sx_ref, sy_ref)
    mkpts_cur_orig = to_original_coords(mkpts_cur, sx_cur, sy_cur)

    data = {
        "depth_ref": load_depth(args.data_root, pair["ref_seq"], pair["ref_frame"], args.depth_scale),
        "depth_cur": load_depth(args.data_root, pair["cur_seq"], pair["cur_frame"], args.depth_scale),
        "mask_ref": load_mask(args.data_root, pair["ref_seq"], pair["ref_frame"], img_ref.shape),
        "mask_cur": load_mask(args.data_root, pair["cur_seq"], pair["cur_frame"], img_cur.shape),
    }
    data["c2w_ref"], data["k_ref"] = load_camera(args.data_root, pair["ref_seq"], pair["ref_frame"])
    data["c2w_cur"], data["k_cur"] = load_camera(args.data_root, pair["cur_seq"], pair["cur_frame"])

    ransac_mask = compute_ransac_inliers(mkpts_ref, mkpts_cur, args.ransac_thresh)
    eval_items = evaluate_gt_geometry(mkpts_ref_orig, mkpts_cur_orig, data, args)
    gt_mask = np.array([item["gt_inlier"] for item in eval_items], dtype=bool)
    human_mask = np.array([item["on_human"] for item in eval_items], dtype=bool)
    reproj = [item["reproj_error_px"] for item in eval_items if item["reproj_error_px"] is not None]
    depth_rel = [item["depth_rel_error"] for item in eval_items if item["depth_rel_error"] is not None]

    pair_dir = out_dir / pair["name"]
    pair_dir.mkdir(parents=True, exist_ok=True)
    raw_indices = np.arange(len(mkpts_ref))[: args.max_draw]
    ransac_indices = np.flatnonzero(ransac_mask)[: args.max_draw]
    gt_indices = np.flatnonzero(gt_mask)[: args.max_draw]
    if len(gt_indices) == 0:
        gt_indices = raw_indices

    draw_specs = [
        (
            "xfeat_raw_matches.png",
            raw_indices,
            (0, 255, 255),
            f"{pair['name']} raw: {len(mkpts_ref)}",
        ),
        (
            "xfeat_ransac_matches.png",
            ransac_indices if len(ransac_indices) else raw_indices,
            (0, 220, 0),
            f"{pair['name']} RANSAC: {int(ransac_mask.sum())}/{len(mkpts_ref)}",
        ),
        (
            "xfeat_gt_geometry_matches.png",
            gt_indices,
            (0, 220, 0) if gt_mask.any() else (0, 0, 255),
            f"{pair['name']} GT geom: {int(gt_mask.sum())}/{len(mkpts_ref)}",
        ),
    ]
    previews = []
    for filename, indices, color, title in draw_specs:
        out_path = pair_dir / filename
        draw_matches(img_ref_match, img_cur_match, mkpts_ref, mkpts_cur, indices, out_path, color, title)
        preview = make_preview(out_path)
        if preview is not None:
            previews.append(str(preview))

    summary = {
        "name": pair["name"],
        "ref": {
            "seq": pair["ref_seq"],
            "frame": int(pair["ref_frame"]),
            "image": str(img_ref_path),
            "shape": list(img_ref.shape),
            "match_shape": list(img_ref_match.shape),
        },
        "cur": {
            "seq": pair["cur_seq"],
            "frame": int(pair["cur_frame"]),
            "image": str(img_cur_path),
            "shape": list(img_cur.shape),
            "match_shape": list(img_cur_match.shape),
        },
        "top_k": int(args.top_k),
        "min_cossim": float(args.min_cossim),
        "max_dim": int(args.max_dim),
        "depth_scale": float(args.depth_scale),
        "thresholds": {
            "reproj_thresh": float(args.reproj_thresh),
            "depth_rel_thresh": float(args.depth_rel_thresh),
            "ransac_thresh": float(args.ransac_thresh),
        },
        "geometry_source_warning": "RICH depth/*.npy is generated Depth Anything depth in this workspace, not metric GT depth.",
        "raw_matches": int(len(mkpts_ref)),
        "homography_ransac_inliers": int(ransac_mask.sum()),
        "homography_ransac_inlier_ratio": float(ransac_mask.sum() / max(len(mkpts_ref), 1)),
        "gt_geometry_inliers": int(gt_mask.sum()),
        "gt_geometry_inlier_ratio": float(gt_mask.sum() / max(len(mkpts_ref), 1)),
        "gt_inliers_inside_ransac": int((gt_mask & ransac_mask).sum()),
        "matches_on_human": int(human_mask.sum()),
        "matches_on_human_ratio": float(human_mask.sum() / max(len(mkpts_ref), 1)),
        "reproj_error_mean_px": float(np.mean(reproj)) if reproj else None,
        "reproj_error_median_px": float(np.median(reproj)) if reproj else None,
        "depth_rel_error_mean": float(np.mean(depth_rel)) if depth_rel else None,
        "depth_rel_error_median": float(np.median(depth_rel)) if depth_rel else None,
        "previews": previews,
    }

    with open(pair_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "matches": eval_items}, f, indent=2)
    return summary


def write_contact_sheet(out_dir, pair_summaries):
    images = []
    for summary in pair_summaries:
        for name in ["xfeat_raw_matches_preview.jpg", "xfeat_ransac_matches_preview.jpg", "xfeat_gt_geometry_matches_preview.jpg"]:
            path = out_dir / summary["name"] / name
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                images.append((summary["name"], name.replace("_preview.jpg", ""), img))
    if not images:
        return None

    cell_w = max(img.shape[1] for _, _, img in images)
    cell_h = max(img.shape[0] for _, _, img in images) + 42
    sheet = np.zeros((cell_h * len(images), cell_w, 3), dtype=np.uint8)
    for row, (pair_name, title, img) in enumerate(images):
        y0 = row * cell_h
        sheet[y0 : y0 + 42, :] = 20
        cv2.putText(
            sheet,
            f"{pair_name} / {title}",
            (12, y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_img = y0 + 42
        sheet[y_img : y_img + img.shape[0], : img.shape[1]] = img
    out_path = out_dir / "rich_xfeat_contact_sheet.jpg"
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return str(out_path)


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = (
            f"outputs/rich_xfeat_aabb_{args.source_sequence}"
            f"_cam{args.cam_a:02d}_cam{args.cam_b:02d}_f{args.start_frame:08d}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_a = seq_name(args.source_sequence, args.cam_a)
    seq_b = seq_name(args.source_sequence, args.cam_b)
    t = args.start_frame
    pairs = [
        {"name": "a_contiguous_t_t1", "ref_seq": seq_a, "ref_frame": t, "cur_seq": seq_a, "cur_frame": t + 1},
        {"name": "aabb_shot_t1_t2", "ref_seq": seq_a, "ref_frame": t + 1, "cur_seq": seq_b, "cur_frame": t + 2},
        {"name": "b_contiguous_t2_t3", "ref_seq": seq_b, "ref_frame": t + 2, "cur_seq": seq_b, "cur_frame": t + 3},
    ]

    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    print("RICH AABB sample:")
    for pair in pairs:
        print(f"  {pair['name']}: {pair['ref_seq']}@{pair['ref_frame']} -> {pair['cur_seq']}@{pair['cur_frame']}")

    xfeat = XFeat(top_k=args.top_k)
    pair_summaries = []
    for pair in pairs:
        print("\nRunning pair:", pair["name"])
        summary = run_pair(xfeat, pair, out_dir, args)
        pair_summaries.append(summary)
        print(json.dumps(summary, indent=2))

    contact_sheet = write_contact_sheet(out_dir, pair_summaries)
    final_summary = {
        "aabb_views": [
            {"view": 0, "seq": seq_a, "frame": t, "shot_label": 0},
            {"view": 1, "seq": seq_a, "frame": t + 1, "shot_label": 0},
            {"view": 2, "seq": seq_b, "frame": t + 2, "shot_label": 1},
            {"view": 3, "seq": seq_b, "frame": t + 3, "shot_label": 0},
        ],
        "pair_summaries": pair_summaries,
        "contact_sheet": contact_sheet,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("\nsaved:", out_dir / "summary.json")
    if contact_sheet:
        print("saved:", contact_sheet)


if __name__ == "__main__":
    main()
