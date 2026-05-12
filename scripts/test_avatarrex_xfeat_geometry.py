#!/usr/bin/env python3
"""Run XFeat on the AvatarReX AABB pair and validate matches with GT geometry."""

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


DATA_ROOT = Path("/workspace/data/Avatarrex/avatarrex_zzr_output/Training")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img0", default=str(REPO_ROOT / "data/aabb_ref_22010708_00000304.png"))
    parser.add_argument("--img1", default=str(REPO_ROOT / "data/aabb_cur_22010710_00000305.png"))
    parser.add_argument("--ref_seq", default="22010708")
    parser.add_argument("--ref_frame", type=int, default=304)
    parser.add_argument("--cur_seq", default="22010710")
    parser.add_argument("--cur_frame", type=int, default=305)
    parser.add_argument("--top_k", type=int, default=4096)
    parser.add_argument("--min_cossim", type=float, default=0.82)
    parser.add_argument("--reproj_thresh", type=float, default=32.0)
    parser.add_argument("--depth_rel_thresh", type=float, default=0.20)
    parser.add_argument("--ransac_thresh", type=float, default=4.0)
    parser.add_argument("--max_draw", type=int, default=500)
    parser.add_argument("--out_dir", default="outputs/avatarrex_xfeat_a5b5")
    return parser.parse_args()


def frame_path(seq, frame, subdir, suffix):
    return DATA_ROOT / seq / subdir / f"{frame:08d}{suffix}"


def load_depth(seq, frame):
    path = frame_path(seq, frame, "depth", ".npy")
    depth = np.load(path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_camera(seq, frame):
    path = frame_path(seq, frame, "cam", ".npz")
    cam = np.load(path)
    return cam["pose"].astype(np.float32), cam["intrinsics"].astype(np.float32)


def load_mask(seq, frame, shape):
    path = frame_path(seq, frame, "mask", ".png")
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


def sample_depth(depth, x, y, radius=2):
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


def evaluate_gt_geometry(mkpts_ref, mkpts_cur, data, args):
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
    for idx, (p_ref, p_cur) in enumerate(zip(mkpts_ref, mkpts_cur)):
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

        items.append({
            "idx": idx,
            "ref_xy": [x_ref, y_ref],
            "cur_xy": [x_cur, y_cur],
            "on_human": bool(on_human),
            "depth_cur": depth_c,
            "depth_ref": depth_r,
            "projected_in_image": bool(projected_in_image),
            "reproj_error_px": reproj_error,
            "depth_rel_error": depth_rel_error,
            "gt_inlier": bool(gt_inlier),
        })

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
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1] - 1, 1300), 48), (0, 0, 0), -1)
    cv2.putText(vis, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img0 = cv2.imread(args.img0, cv2.IMREAD_COLOR)
    img1 = cv2.imread(args.img1, cv2.IMREAD_COLOR)
    if img0 is None:
        raise RuntimeError(f"Failed to read {args.img0}")
    if img1 is None:
        raise RuntimeError(f"Failed to read {args.img1}")

    print("image0:", args.img0, img0.shape)
    print("image1:", args.img1, img1.shape)
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

    xfeat = XFeat(top_k=args.top_k)
    mkpts0, mkpts1 = xfeat.match_xfeat(
        img0,
        img1,
        top_k=args.top_k,
        min_cossim=args.min_cossim,
    )
    mkpts0 = np.asarray(mkpts0, dtype=np.float32)
    mkpts1 = np.asarray(mkpts1, dtype=np.float32)
    print("raw_matches:", len(mkpts0))

    data = {
        "depth_ref": load_depth(args.ref_seq, args.ref_frame),
        "depth_cur": load_depth(args.cur_seq, args.cur_frame),
        "mask_ref": load_mask(args.ref_seq, args.ref_frame, img0.shape),
        "mask_cur": load_mask(args.cur_seq, args.cur_frame, img1.shape),
    }
    data["c2w_ref"], data["k_ref"] = load_camera(args.ref_seq, args.ref_frame)
    data["c2w_cur"], data["k_cur"] = load_camera(args.cur_seq, args.cur_frame)

    ransac_mask = compute_ransac_inliers(mkpts0, mkpts1, args.ransac_thresh)
    eval_items = evaluate_gt_geometry(mkpts0, mkpts1, data, args)
    gt_mask = np.array([item["gt_inlier"] for item in eval_items], dtype=bool)
    human_mask = np.array([item["on_human"] for item in eval_items], dtype=bool)
    reproj = [item["reproj_error_px"] for item in eval_items if item["reproj_error_px"] is not None]
    depth_rel = [item["depth_rel_error"] for item in eval_items if item["depth_rel_error"] is not None]

    raw_indices = np.arange(len(mkpts0))[:args.max_draw]
    ransac_indices = np.flatnonzero(ransac_mask)[:args.max_draw]
    gt_indices = np.flatnonzero(gt_mask)[:args.max_draw]
    if len(gt_indices) == 0:
        gt_indices = raw_indices

    draw_matches(
        img0,
        img1,
        mkpts0,
        mkpts1,
        raw_indices,
        out_dir / "xfeat_raw_matches.png",
        (0, 255, 255),
        f"XFeat raw matches: {len(mkpts0)}",
    )
    draw_matches(
        img0,
        img1,
        mkpts0,
        mkpts1,
        ransac_indices if len(ransac_indices) else raw_indices,
        out_dir / "xfeat_ransac_matches.png",
        (0, 220, 0),
        f"XFeat homography RANSAC inliers: {int(ransac_mask.sum())}/{len(mkpts0)}",
    )
    draw_matches(
        img0,
        img1,
        mkpts0,
        mkpts1,
        gt_indices,
        out_dir / "xfeat_gt_geometry_matches.png",
        (0, 220, 0) if gt_mask.any() else (0, 0, 255),
        f"GT geometry inliers: {int(gt_mask.sum())}/{len(mkpts0)}",
    )

    for name in ["xfeat_raw_matches.png", "xfeat_ransac_matches.png", "xfeat_gt_geometry_matches.png"]:
        src = cv2.imread(str(out_dir / name), cv2.IMREAD_COLOR)
        if src is None:
            continue
        max_w = 1200
        scale = min(1.0, max_w / src.shape[1])
        preview = cv2.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / name.replace(".png", "_preview.jpg")), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    summary = {
        "img0": args.img0,
        "img1": args.img1,
        "image0_shape": list(img0.shape),
        "image1_shape": list(img1.shape),
        "top_k": args.top_k,
        "min_cossim": args.min_cossim,
        "thresholds": {
            "reproj_thresh": args.reproj_thresh,
            "depth_rel_thresh": args.depth_rel_thresh,
            "ransac_thresh": args.ransac_thresh,
        },
        "raw_matches": int(len(mkpts0)),
        "homography_ransac_inliers": int(ransac_mask.sum()),
        "homography_ransac_inlier_ratio": float(ransac_mask.sum() / max(len(mkpts0), 1)),
        "gt_geometry_inliers": int(gt_mask.sum()),
        "gt_geometry_inlier_ratio": float(gt_mask.sum() / max(len(mkpts0), 1)),
        "matches_on_human": int(human_mask.sum()),
        "reproj_error_mean_px": float(np.mean(reproj)) if reproj else None,
        "reproj_error_median_px": float(np.median(reproj)) if reproj else None,
        "depth_rel_error_mean": float(np.mean(depth_rel)) if depth_rel else None,
        "depth_rel_error_median": float(np.median(depth_rel)) if depth_rel else None,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "matches": eval_items}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("saved:", out_dir / "summary.json")
    print("saved:", out_dir / "xfeat_raw_matches.png")
    print("saved:", out_dir / "xfeat_ransac_matches.png")
    print("saved:", out_dir / "xfeat_gt_geometry_matches.png")


if __name__ == "__main__":
    main()
