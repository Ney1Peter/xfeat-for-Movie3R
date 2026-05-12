#!/usr/bin/env python3
"""Visualize cross-camera RICH correspondences from the official static scan mesh.

This is the metric-geometry diagnostic to use for RICH static background.
It samples the same scan mesh vertex, projects it into two calibrated cameras,
and keeps only vertices that are z-buffer visible in both images.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from visualize_rich_mesh_projection import load_calibration, load_ply_vertices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich_root", default="/workspace/data/RICH")
    parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    parser.add_argument("--source_sequence", default="BBQ_001_juggle")
    parser.add_argument("--cam_a", type=int, default=3)
    parser.add_argument("--cam_b", type=int, default=4)
    parser.add_argument("--frame_a", type=int, default=101)
    parser.add_argument("--frame_b", type=int, default=102)
    parser.add_argument("--num_points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_dim", type=int, default=1400)
    parser.add_argument("--z_tol", type=float, default=0.03)
    parser.add_argument("--min_border", type=int, default=120)
    parser.add_argument("--crop_radius", type=int, default=96)
    parser.add_argument("--pose_mode", choices=["xml_w2c", "xml_as_c2w_inverted"], default="xml_w2c")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def seq_name(source_sequence, cam):
    return f"{source_sequence}_cam_{cam:02d}"


def load_rgb(data_root, seq, frame):
    path = Path(data_root) / seq / "rgb" / f"{frame:08d}.png"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img, path


def load_mask(data_root, seq, frame, shape):
    path = Path(data_root) / seq / "mask" / f"{frame:08d}.png"
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


def camera_transform(rich_root, cam, mode):
    calib_path = Path(rich_root) / "scan_calibration" / "BBQ" / "calibration" / f"{cam:03d}.xml"
    w2c, c2w, intrinsics = load_calibration(calib_path)
    if mode == "xml_w2c":
        return w2c, intrinsics, calib_path
    return c2w, intrinsics, calib_path


def projected_vertices(xyz, transform, intrinsics, image_shape, max_dim):
    h, w = image_shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    hs = max(1, int(round(h * scale)))
    ws = max(1, int(round(w * scale)))

    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float32)], axis=1)
    cam = (transform @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    pix = (intrinsics @ cam.T).T
    denom = pix[:, 2]
    finite = np.isfinite(denom) & (np.abs(denom) > 1e-12)
    x = np.full(len(xyz), np.nan, dtype=np.float32)
    y = np.full(len(xyz), np.nan, dtype=np.float32)
    x[finite] = pix[finite, 0] / denom[finite]
    y[finite] = pix[finite, 1] / denom[finite]

    xs = np.zeros(len(xyz), dtype=np.int32)
    ys = np.zeros(len(xyz), dtype=np.int32)
    xs[finite] = np.round(x[finite] * scale).astype(np.int32)
    ys[finite] = np.round(y[finite] * scale).astype(np.int32)
    inside = finite & (z > 1e-4) & (xs >= 0) & (xs < ws) & (ys >= 0) & (ys < hs)
    flat = np.full(len(xyz), -1, dtype=np.int64)
    flat[inside] = ys[inside].astype(np.int64) * ws + xs[inside].astype(np.int64)

    zbuf = np.full(hs * ws, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, flat[inside], z[inside].astype(np.float32))
    visible = np.zeros(len(xyz), dtype=bool)
    visible[inside] = z[inside] <= zbuf[flat[inside]] + 1e-6

    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "z": z.astype(np.float32),
        "inside": inside,
        "visible": visible,
        "flat": flat,
        "zbuf": zbuf,
        "scale": scale,
        "scaled_shape": (hs, ws),
    }


def mask_lookup(mask, x, y):
    h, w = mask.shape[:2]
    finite = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 1e8) & (np.abs(y) < 1e8)
    xi = np.zeros(len(x), dtype=np.int32)
    yi = np.zeros(len(y), dtype=np.int32)
    xi[finite] = np.round(x[finite]).astype(np.int32)
    yi[finite] = np.round(y[finite]).astype(np.int32)
    ok = finite & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.ones(len(x), dtype=bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


def choose_spread(indices, pa, num_points, seed, image_shape):
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, dtype=np.int64)
    rng.shuffle(indices)
    selected = []
    h, w = image_shape[:2]
    min_dist = min(h, w) / max(num_points, 1) * 0.42
    for idx in indices:
        xy = np.array([pa["x"][idx], pa["y"][idx]], dtype=np.float32)
        if all(np.linalg.norm(xy - np.array([pa["x"][prev], pa["y"][prev]], dtype=np.float32)) > min_dist for prev in selected):
            selected.append(int(idx))
        if len(selected) >= num_points:
            return selected
    return [int(i) for i in indices[:num_points]]


def resize_for_vis(img, max_dim):
    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale == 1.0:
        return img.copy(), scale
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA), scale


def draw_matches(img_a, img_b, points, out_path, title, max_dim):
    vis_a, sa = resize_for_vis(img_a, max_dim)
    vis_b, sb = resize_for_vis(img_b, max_dim)
    h = max(vis_a.shape[0], vis_b.shape[0])
    canvas = np.zeros((h + 54, vis_a.shape[1] + vis_b.shape[1], 3), dtype=np.uint8)
    canvas[:54] = 20
    canvas[54 : 54 + vis_a.shape[0], : vis_a.shape[1]] = vis_a
    canvas[54 : 54 + vis_b.shape[0], vis_a.shape[1] : vis_a.shape[1] + vis_b.shape[1]] = vis_b
    cv2.putText(canvas, title, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    colors = [(255, 0, 0), (0, 255, 0), (0, 180, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
    xoff = vis_a.shape[1]
    for i, item in enumerate(points):
        color = colors[i % len(colors)]
        xa, ya = item["a_xy"]
        xb, yb = item["b_xy"]
        pa = (int(round(xa * sa)), int(round(54 + ya * sa)))
        pb = (int(round(xoff + xb * sb)), int(round(54 + yb * sb)))
        cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pa, 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pb, 5, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (pa[0] + 6, pa[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, str(i), (pb[0] + 6, pb[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
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


def draw_crops(img_a, img_b, points, out_path, title, radius):
    if not points:
        canvas = np.zeros((80, 800, 3), dtype=np.uint8)
        cv2.putText(canvas, "No visible shared mesh points", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_path), canvas)
        return
    crop_size = radius * 2
    row_h = crop_size + 42
    canvas = np.zeros((54 + row_h * len(points), crop_size * 2 + 42, 3), dtype=np.uint8)
    canvas[:54] = 20
    cv2.putText(canvas, title, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    for i, item in enumerate(points):
        y0 = 54 + i * row_h
        xa, ya = item["a_xy"]
        xb, yb = item["b_xy"]
        canvas[y0 : y0 + crop_size, :crop_size] = crop_with_border(img_a, xa, ya, radius)
        canvas[y0 : y0 + crop_size, crop_size + 42 : crop_size * 2 + 42] = crop_with_border(img_b, xb, yb, radius)
        cv2.circle(canvas, (radius, y0 + radius), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (crop_size + 42 + radius, y0 + radius), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"#{i} z=({item['a_z']:.2f},{item['b_z']:.2f})", (8, y0 + crop_size + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = (
            f"outputs/rich_mesh_correspondences_{args.source_sequence}"
            f"_cam{args.cam_a:02d}_cam{args.cam_b:02d}"
            f"_f{args.frame_a:08d}_{args.frame_b:08d}_{args.pose_mode}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_a = seq_name(args.source_sequence, args.cam_a)
    seq_b = seq_name(args.source_sequence, args.cam_b)
    img_a, img_path_a = load_rgb(args.data_root, seq_a, args.frame_a)
    img_b, img_path_b = load_rgb(args.data_root, seq_b, args.frame_b)
    mask_a = load_mask(args.data_root, seq_a, args.frame_a, img_a.shape)
    mask_b = load_mask(args.data_root, seq_b, args.frame_b, img_b.shape)

    transform_a, k_a, calib_a = camera_transform(args.rich_root, args.cam_a, args.pose_mode)
    transform_b, k_b, calib_b = camera_transform(args.rich_root, args.cam_b, args.pose_mode)
    mesh_path = Path(args.rich_root) / "scan_calibration" / "BBQ" / "scan_camcoord.ply"
    xyz, rgb = load_ply_vertices(mesh_path)

    pa = projected_vertices(xyz, transform_a, k_a, img_a.shape, args.max_dim)
    pb = projected_vertices(xyz, transform_b, k_b, img_b.shape, args.max_dim)
    shared = pa["inside"] & pb["inside"] & pa["visible"] & pb["visible"]
    shared &= pa["z"] <= pa["zbuf"][pa["flat"]].clip(max=np.inf) + args.z_tol
    shared &= pb["z"] <= pb["zbuf"][pb["flat"]].clip(max=np.inf) + args.z_tol

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    shared &= (pa["x"] >= args.min_border) & (pa["x"] < w_a - args.min_border)
    shared &= (pa["y"] >= args.min_border) & (pa["y"] < h_a - args.min_border)
    shared &= (pb["x"] >= args.min_border) & (pb["x"] < w_b - args.min_border)
    shared &= (pb["y"] >= args.min_border) & (pb["y"] < h_b - args.min_border)
    shared &= ~mask_lookup(mask_a, pa["x"], pa["y"])
    shared &= ~mask_lookup(mask_b, pb["x"], pb["y"])

    candidate_indices = np.flatnonzero(shared)
    selected_indices = choose_spread(candidate_indices, pa, args.num_points, args.seed, img_a.shape)
    points = []
    for idx in selected_indices:
        points.append(
            {
                "vertex_index": int(idx),
                "mesh_xyz": [float(v) for v in xyz[idx]],
                "mesh_rgb": [int(v) for v in rgb[idx]],
                "a_xy": [float(pa["x"][idx]), float(pa["y"][idx])],
                "b_xy": [float(pb["x"][idx]), float(pb["y"][idx])],
                "a_z": float(pa["z"][idx]),
                "b_z": float(pb["z"][idx]),
            }
        )

    match_path = out_dir / "mesh_visible_correspondences.png"
    crop_path = out_dir / "mesh_visible_correspondence_crops.png"
    title = f"RICH scan mesh correspondences: {seq_a}@{args.frame_a} -> {seq_b}@{args.frame_b} ({args.pose_mode})"
    draw_matches(img_a, img_b, points, match_path, title, args.max_dim)
    draw_crops(img_a, img_b, points, crop_path, title, args.crop_radius)

    summary = {
        "args": vars(args),
        "image_a": str(img_path_a),
        "image_b": str(img_path_b),
        "calib_a": str(calib_a),
        "calib_b": str(calib_b),
        "mesh_path": str(mesh_path),
        "mesh_vertices": int(len(xyz)),
        "candidates_visible_in_both": int(len(candidate_indices)),
        "selected_points": int(len(points)),
        "match_image": str(match_path),
        "crop_image": str(crop_path),
        "points": points,
        "note": "These correspondences are generated from the same static scan mesh vertices and do not use generated depth/*.npy files.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
