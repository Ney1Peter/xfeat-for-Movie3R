#!/usr/bin/env python3
"""Project the official RICH static scan mesh into RGB cameras.

This diagnostic intentionally does not use the generated depth/*.npy files.
Those depths are Depth Anything predictions in this workspace, not RICH metric
ground-truth depth. The scan mesh plus XML calibration is the correct source
for checking cross-camera static-scene geometry.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich_root", default="/workspace/data/RICH")
    parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    parser.add_argument("--source_sequence", default="BBQ_001_juggle")
    parser.add_argument("--cam", type=int, default=3)
    parser.add_argument("--frame", type=int, default=101)
    parser.add_argument("--max_dim", type=int, default=1400)
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--out_dir", default="outputs/rich_mesh_projection_check")
    return parser.parse_args()


def read_opencv_matrix(root, name):
    node = root.find(name)
    if node is None:
        raise ValueError(f"missing matrix node: {name}")
    rows = int(node.find("rows").text)
    cols = int(node.find("cols").text)
    values = [float(x) for x in node.find("data").text.split()]
    return np.asarray(values, dtype=np.float32).reshape(rows, cols)


def load_calibration(calib_path):
    root = ET.parse(calib_path).getroot()
    w2c_3x4 = read_opencv_matrix(root, "CameraMatrix")
    intrinsics = read_opencv_matrix(root, "Intrinsics")
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = w2c_3x4
    c2w = np.linalg.inv(w2c)
    return w2c, c2w, intrinsics


def load_ply_vertices(ply_path):
    with open(ply_path, "rb") as f:
        vertex_count = None
        header_len = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY ended before end_header")
            header_len += len(line)
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded.startswith("element vertex"):
                vertex_count = int(decoded.split()[-1])
            if decoded == "end_header":
                break
        if vertex_count is None:
            raise ValueError("PLY header has no vertex count")
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
                ("a", "u1"),
            ]
        )
        f.seek(header_len)
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    rgb = np.stack([vertices["b"], vertices["g"], vertices["r"]], axis=1).astype(np.uint8)
    return xyz, rgb


def resize_image(img, max_dim):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img.copy(), 1.0
    scale = float(max_dim) / float(max(h, w))
    resized = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def project_points(xyz, rgb, transform, intrinsics, image_shape, scale):
    h, w = image_shape[:2]
    hs = int(round(h * scale))
    ws = int(round(w * scale))

    xyz_h = np.concatenate([xyz, np.ones((len(xyz), 1), dtype=np.float32)], axis=1)
    cam = (transform @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    valid = z > 1e-4
    cam = cam[valid]
    z = z[valid]
    rgb = rgb[valid]

    pix = (intrinsics @ cam.T).T
    x = pix[:, 0] / pix[:, 2]
    y = pix[:, 1] / pix[:, 2]
    xs = np.round(x * scale).astype(np.int32)
    ys = np.round(y * scale).astype(np.int32)
    inside = (xs >= 0) & (xs < ws) & (ys >= 0) & (ys < hs)
    xs = xs[inside]
    ys = ys[inside]
    z = z[inside]
    rgb = rgb[inside]

    zbuf = np.full(hs * ws, np.inf, dtype=np.float32)
    flat = ys * ws + xs
    np.minimum.at(zbuf, flat, z.astype(np.float32))
    front = z <= zbuf[flat] + 1e-6

    color = np.zeros((hs * ws, 3), dtype=np.uint8)
    color[flat[front]] = rgb[front]
    valid_pix = np.isfinite(zbuf)
    mesh_img = color.reshape(hs, ws, 3)
    mask = valid_pix.reshape(hs, ws)
    return mesh_img, mask, int(len(xs)), int(mask.sum())


def overlay_mesh(img, mesh_img, mask, radius):
    if radius > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        mesh_img = cv2.dilate(mesh_img, kernel)
    out = img.copy()
    out[mask] = (0.45 * out[mask] + 0.55 * mesh_img[mask]).astype(np.uint8)
    return out


def write_comparison(img, mesh_img, overlay, out_path):
    h = max(img.shape[0], mesh_img.shape[0], overlay.shape[0])
    w = img.shape[1] + mesh_img.shape[1] + overlay.shape[1]
    canvas = np.zeros((h + 48, w, 3), dtype=np.uint8)
    canvas[:48] = 20
    x = 0
    for title, panel in [("rgb", img), ("projected mesh", mesh_img), ("overlay", overlay)]:
        canvas[48 : 48 + panel.shape[0], x : x + panel.shape[1]] = panel
        cv2.putText(canvas, title, (x + 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        x += panel.shape[1]
    cv2.imwrite(str(out_path), canvas)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = f"{args.source_sequence}_cam_{args.cam:02d}"
    image_path = Path(args.data_root) / seq / "rgb" / f"{args.frame:08d}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    image_small, scale = resize_image(image, args.max_dim)

    calib_path = Path(args.rich_root) / "scan_calibration" / "BBQ" / "calibration" / f"{args.cam:03d}.xml"
    scan_mesh_path = Path(args.rich_root) / "scan_calibration" / "BBQ" / "scan_camcoord.ply"
    w2c, c2w, intrinsics = load_calibration(calib_path)
    xyz, rgb = load_ply_vertices(scan_mesh_path)

    summaries = []
    for mode, transform in [("xml_w2c", w2c), ("xml_as_c2w_inverted", c2w)]:
        mesh_img, mask, projected_points, covered_pixels = project_points(xyz, rgb, transform, intrinsics, image.shape, scale)
        overlay = overlay_mesh(image_small, mesh_img, mask, args.point_radius)
        out_path = out_dir / f"mesh_projection_{seq}_f{args.frame:08d}_{mode}.jpg"
        mesh_only_path = out_dir / f"mesh_only_{seq}_f{args.frame:08d}_{mode}.jpg"
        overlay_path = out_dir / f"mesh_overlay_{seq}_f{args.frame:08d}_{mode}.jpg"
        cv2.imwrite(str(mesh_only_path), mesh_img)
        cv2.imwrite(str(overlay_path), overlay)
        write_comparison(image_small, mesh_img, overlay, out_path)
        summaries.append(
            {
                "mode": mode,
                "projected_points": projected_points,
                "covered_pixels": covered_pixels,
                "covered_ratio": float(covered_pixels / max(mask.size, 1)),
                "image": str(out_path),
                "mesh_only": str(mesh_only_path),
                "overlay": str(overlay_path),
            }
        )

    summary = {
        "args": vars(args),
        "image_path": str(image_path),
        "calibration_path": str(calib_path),
        "mesh_path": str(scan_mesh_path),
        "image_shape": list(image.shape),
        "vis_scale": float(scale),
        "summaries": summaries,
        "note": "xml_w2c should align with the RGB background if RICH XML CameraMatrix maps scan/world coordinates to camera coordinates.",
    }
    summary_path = out_dir / f"mesh_projection_{seq}_f{args.frame:08d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
