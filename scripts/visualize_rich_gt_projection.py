#!/usr/bin/env python3
"""Visualize generated-depth projection correspondences between two RICH views.

Important: in this workspace RICH depth/*.npy files are Depth Anything outputs,
not metric RICH ground-truth depth. For real cross-camera static-scene geometry,
use visualize_rich_mesh_correspondences.py instead.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import test_rich_aabb_xfeat_geometry as rich


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    parser.add_argument("--source_sequence", default="BBQ_001_juggle")
    parser.add_argument("--cam_a", type=int, default=3)
    parser.add_argument("--cam_b", type=int, default=4)
    parser.add_argument("--frame_a", type=int, default=101)
    parser.add_argument("--frame_b", type=int, default=102)
    parser.add_argument("--num_points", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth_scale", type=float, default=0.001)
    parser.add_argument("--max_dim", type=int, default=1400)
    parser.add_argument("--min_border", type=int, default=80)
    parser.add_argument("--pose_mode", choices=["c2w", "w2c", "both"], default="both")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def seq_name(source_sequence, cam):
    return f"{source_sequence}_cam_{cam:02d}"


def convert_pose(pose, mode):
    if mode == "c2w":
        return pose
    if mode == "w2c":
        return np.linalg.inv(pose)
    raise ValueError(f"Bad pose mode: {mode}")


def project_ref_to_cur(x_ref, y_ref, depth_ref, c2w_ref, k_ref, c2w_cur, k_cur):
    pix_ref = np.array([x_ref, y_ref, 1.0], dtype=np.float32)
    xyz_ref = np.linalg.inv(k_ref) @ pix_ref * depth_ref
    xyz_ref_h = np.concatenate([xyz_ref, np.ones(1, dtype=np.float32)])
    xyz_world = c2w_ref @ xyz_ref_h
    xyz_cur = np.linalg.inv(c2w_cur) @ xyz_world
    z_cur = float(xyz_cur[2])
    if z_cur <= 1e-6:
        return None
    pix_cur = k_cur @ xyz_cur[:3]
    x_cur = float(pix_cur[0] / pix_cur[2])
    y_cur = float(pix_cur[1] / pix_cur[2])
    return x_cur, y_cur, z_cur


def valid_pixel(mask, depth, x, y, min_border):
    h, w = depth.shape[:2]
    if x < min_border or x >= w - min_border or y < min_border or y >= h - min_border:
        return False
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if depth[yi, xi] <= 0 or not np.isfinite(depth[yi, xi]):
        return False
    if mask is not None and mask[yi, xi]:
        return False
    return True


def pick_gt_points(
    depth_ref,
    mask_ref,
    depth_cur,
    mask_cur,
    c2w_ref,
    k_ref,
    c2w_cur,
    k_cur,
    num_points,
    seed,
    min_border,
):
    rng = np.random.default_rng(seed)
    h, w = depth_ref.shape[:2]
    grid_y = np.linspace(min_border, h - min_border - 1, 18)
    grid_x = np.linspace(min_border, w - min_border - 1, 24)
    candidates = []
    for y in grid_y:
        for x in grid_x:
            xj = float(x + rng.uniform(-40, 40))
            yj = float(y + rng.uniform(-40, 40))
            if not valid_pixel(mask_ref, depth_ref, xj, yj, min_border):
                continue
            xi = int(round(xj))
            yi = int(round(yj))
            d_ref = float(depth_ref[yi, xi])
            projected = project_ref_to_cur(xj, yj, d_ref, c2w_ref, k_ref, c2w_cur, k_cur)
            if projected is None:
                continue
            x_cur, y_cur, z_cur = projected
            if not valid_pixel(mask_cur, depth_cur, x_cur, y_cur, min_border):
                continue
            d_cur = rich.sample_depth(depth_cur, x_cur, y_cur, radius=4)
            if d_cur is None:
                continue
            depth_rel_error = abs(z_cur - d_cur) / max(abs(d_cur), 1e-6)
            # Keep only projections that land on a surface with compatible depth.
            if depth_rel_error > 0.25:
                continue
            candidates.append(
                {
                    "ref_xy": [float(xj), float(yj)],
                    "cur_xy": [float(x_cur), float(y_cur)],
                    "depth_ref": d_ref,
                    "projected_depth_cur": float(z_cur),
                    "sampled_depth_cur": float(d_cur),
                    "depth_rel_error": float(depth_rel_error),
                }
            )

    if not candidates:
        return []

    # Pick spatially spread points by sorting candidates into coarse image order.
    rng.shuffle(candidates)
    selected = []
    min_dist = min(h, w) / max(num_points, 1) * 0.45
    for item in candidates:
        xy = np.array(item["ref_xy"], dtype=np.float32)
        if all(np.linalg.norm(xy - np.array(prev["ref_xy"], dtype=np.float32)) > min_dist for prev in selected):
            selected.append(item)
        if len(selected) >= num_points:
            break
    if len(selected) < num_points:
        for item in candidates:
            if item not in selected:
                selected.append(item)
            if len(selected) >= num_points:
                break
    return selected[:num_points]


def resize_for_vis(img, max_dim):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img.copy(), 1.0, 1.0
    scale = float(max_dim) / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, float(new_w) / float(w), float(new_h) / float(h)


def draw_projection(img_ref, img_cur, points, title, out_path, max_dim):
    ref_vis, sx_ref, sy_ref = resize_for_vis(img_ref, max_dim)
    cur_vis, sx_cur, sy_cur = resize_for_vis(img_cur, max_dim)
    h = max(ref_vis.shape[0], cur_vis.shape[0])
    w = ref_vis.shape[1] + cur_vis.shape[1]
    canvas = np.zeros((h + 54, w, 3), dtype=np.uint8)
    canvas[:54, :] = 20
    canvas[54 : 54 + ref_vis.shape[0], : ref_vis.shape[1]] = ref_vis
    canvas[54 : 54 + cur_vis.shape[0], ref_vis.shape[1] : ref_vis.shape[1] + cur_vis.shape[1]] = cur_vis
    cv2.putText(canvas, title, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (128, 255, 0),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 128),
    ]
    x_offset = ref_vis.shape[1]
    for idx, item in enumerate(points):
        color = colors[idx % len(colors)]
        x_ref, y_ref = item["ref_xy"]
        x_cur, y_cur = item["cur_xy"]
        p_ref = (int(round(x_ref * sx_ref)), int(round(54 + y_ref * sy_ref)))
        p_cur = (int(round(x_offset + x_cur * sx_cur)), int(round(54 + y_cur * sy_cur)))
        cv2.line(canvas, p_ref, p_cur, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, p_ref, 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p_cur, 7, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (p_ref[0] + 8, p_ref[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, str(idx), (p_cur[0] + 8, p_cur[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def crop_with_border(img, x, y, radius):
    h, w = img.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    x0 = max(0, xi - radius)
    x1 = min(w, xi + radius)
    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius)
    crop = img[y0:y1, x0:x1]
    out = np.zeros((radius * 2, radius * 2, 3), dtype=img.dtype)
    oy = max(0, radius - yi)
    ox = max(0, radius - xi)
    out[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    return out


def draw_crop_pairs(img_ref, img_cur, points, title, out_path, crop_radius=96):
    if not points:
        canvas = np.zeros((80, 800, 3), dtype=np.uint8)
        cv2.putText(canvas, "No valid projected points", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_path), canvas)
        return

    crop_size = crop_radius * 2
    row_h = crop_size + 42
    col_w = crop_size * 2 + 36
    canvas = np.zeros((54 + row_h * len(points), col_w, 3), dtype=np.uint8)
    canvas[:54, :] = 20
    cv2.putText(canvas, title, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    for idx, item in enumerate(points):
        y0 = 54 + idx * row_h
        x_ref, y_ref = item["ref_xy"]
        x_cur, y_cur = item["cur_xy"]
        ref_crop = crop_with_border(img_ref, x_ref, y_ref, crop_radius)
        cur_crop = crop_with_border(img_cur, x_cur, y_cur, crop_radius)
        canvas[y0 : y0 + crop_size, :crop_size] = ref_crop
        canvas[y0 : y0 + crop_size, crop_size + 36 : crop_size * 2 + 36] = cur_crop
        cv2.circle(canvas, (crop_radius, y0 + crop_radius), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (crop_size + 36 + crop_radius, y0 + crop_radius), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "A", (8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "B", (crop_size + 44, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        label = f"#{idx} depth_rel={item['depth_rel_error']:.3f}"
        cv2.putText(canvas, label, (8, y0 + crop_size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def run_mode(args, mode, out_dir):
    seq_a = seq_name(args.source_sequence, args.cam_a)
    seq_b = seq_name(args.source_sequence, args.cam_b)
    img_a, path_a = rich.load_rgb(args.data_root, seq_a, args.frame_a)
    img_b, path_b = rich.load_rgb(args.data_root, seq_b, args.frame_b)
    depth_a = rich.load_depth(args.data_root, seq_a, args.frame_a, args.depth_scale)
    depth_b = rich.load_depth(args.data_root, seq_b, args.frame_b, args.depth_scale)
    mask_a = rich.load_mask(args.data_root, seq_a, args.frame_a, img_a.shape)
    mask_b = rich.load_mask(args.data_root, seq_b, args.frame_b, img_b.shape)
    pose_a, k_a = rich.load_camera(args.data_root, seq_a, args.frame_a)
    pose_b, k_b = rich.load_camera(args.data_root, seq_b, args.frame_b)
    c2w_a = convert_pose(pose_a, mode)
    c2w_b = convert_pose(pose_b, mode)

    ab_points = pick_gt_points(
        depth_a,
        mask_a,
        depth_b,
        mask_b,
        c2w_a,
        k_a,
        c2w_b,
        k_b,
        args.num_points,
        args.seed,
        args.min_border,
    )
    ba_points = pick_gt_points(
        depth_b,
        mask_b,
        depth_a,
        mask_a,
        c2w_b,
        k_b,
        c2w_a,
        k_a,
        args.num_points,
        args.seed + 1,
        args.min_border,
    )

    ab_path = out_dir / f"gt_projection_{mode}_A_to_B.png"
    ba_path = out_dir / f"gt_projection_{mode}_B_to_A.png"
    ab_crop_path = out_dir / f"gt_projection_{mode}_A_to_B_crops.png"
    ba_crop_path = out_dir / f"gt_projection_{mode}_B_to_A_crops.png"
    draw_projection(
        img_a,
        img_b,
        ab_points,
        f"Generated-depth projection {mode}: {seq_a}@{args.frame_a} -> {seq_b}@{args.frame_b}",
        ab_path,
        args.max_dim,
    )
    draw_crop_pairs(
        img_a,
        img_b,
        ab_points,
        f"Generated-depth crop pairs {mode}: {seq_a}@{args.frame_a} -> {seq_b}@{args.frame_b}",
        ab_crop_path,
    )
    draw_projection(
        img_b,
        img_a,
        ba_points,
        f"Generated-depth projection {mode}: {seq_b}@{args.frame_b} -> {seq_a}@{args.frame_a}",
        ba_path,
        args.max_dim,
    )
    draw_crop_pairs(
        img_b,
        img_a,
        ba_points,
        f"Generated-depth crop pairs {mode}: {seq_b}@{args.frame_b} -> {seq_a}@{args.frame_a}",
        ba_crop_path,
    )
    return {
        "pose_mode": mode,
        "A": {"seq": seq_a, "frame": args.frame_a, "image": str(path_a)},
        "B": {"seq": seq_b, "frame": args.frame_b, "image": str(path_b)},
        "A_to_B_points": ab_points,
        "B_to_A_points": ba_points,
        "A_to_B_image": str(ab_path),
        "B_to_A_image": str(ba_path),
        "A_to_B_crops": str(ab_crop_path),
        "B_to_A_crops": str(ba_crop_path),
    }


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = (
            f"outputs/rich_gt_projection_{args.source_sequence}"
            f"_cam{args.cam_a:02d}_cam{args.cam_b:02d}"
            f"_f{args.frame_a:08d}_{args.frame_b:08d}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["c2w", "w2c"] if args.pose_mode == "both" else [args.pose_mode]
    summaries = [run_mode(args, mode, out_dir) for mode in modes]

    summary = {
        "args": vars(args),
        "summaries": summaries,
        "note": "Points are sampled from background pixels with generated Depth Anything depth, not metric RICH GT depth. This is not valid for cross-camera GT validation.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
