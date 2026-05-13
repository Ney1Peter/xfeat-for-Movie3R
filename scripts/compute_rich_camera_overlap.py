#!/usr/bin/env python3
"""Compute RICH camera-pair overlap from the official static scan mesh.

The overlap is measured on mesh vertices that are visible in each camera. By
default, vertices landing on the per-frame human mask are removed so the metric
is closer to the static-background anchor budget used by Movie3R diagnostics.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from visualize_rich_mesh_correspondences import camera_transform, mask_lookup, projected_vertices
from visualize_rich_mesh_projection import load_ply_vertices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich_root", default="/workspace/data/RICH")
    parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    parser.add_argument("--source_sequence", default="BBQ_001_juggle")
    parser.add_argument("--cams", default=None, help="Comma-separated camera ids. Defaults to discovered sequence cams.")
    parser.add_argument("--frame", type=int, default=None, help="Use the same frame for all cams. Defaults to each cam's first RGB frame.")
    parser.add_argument("--max_dim", type=int, default=1400)
    parser.add_argument("--z_tol", type=float, default=0.03)
    parser.add_argument("--sort_by", choices=["overlap_min", "jaccard", "shared_visible"], default="overlap_min")
    parser.add_argument("--include_human_mask", action="store_true", help="Do not remove vertices that project onto the human mask.")
    parser.add_argument("--pose_mode", choices=["xml_w2c", "xml_as_c2w_inverted"], default="xml_w2c")
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def seq_name(source_sequence, cam):
    return f"{source_sequence}_cam_{cam:02d}"


def discover_cams(data_root, source_sequence):
    root = Path(data_root)
    cams = []
    for seq_dir in sorted(root.glob(f"{source_sequence}_cam_*")):
        if not seq_dir.is_dir():
            continue
        try:
            cams.append(int(seq_dir.name.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return cams


def parse_cams(args):
    if args.cams:
        return [int(item) for item in args.cams.split(",") if item.strip()]
    cams = discover_cams(args.data_root, args.source_sequence)
    if not cams:
        raise RuntimeError(f"no camera sequences found for {args.source_sequence} under {args.data_root}")
    return cams


def first_frame(data_root, seq):
    rgb_dir = Path(data_root) / seq / "rgb"
    frames = sorted(rgb_dir.glob("*.png"))
    if not frames:
        raise RuntimeError(f"no RGB frames found: {rgb_dir}")
    return int(frames[0].stem)


def load_image(data_root, seq, frame):
    path = Path(data_root) / seq / "rgb" / f"{frame:08d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image, path


def load_mask(data_root, seq, frame, shape):
    path = Path(data_root) / seq / "mask" / f"{frame:08d}.png"
    if not path.exists():
        return np.zeros(shape[:2], dtype=bool), None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return np.zeros(shape[:2], dtype=bool), None
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127, path


def visible_vertices_for_camera(xyz, args, cam):
    seq = seq_name(args.source_sequence, cam)
    frame = args.frame if args.frame is not None else first_frame(args.data_root, seq)
    image, image_path = load_image(args.data_root, seq, frame)
    transform, intrinsics, calib_path = camera_transform(args.rich_root, cam, args.pose_mode)
    proj = projected_vertices(xyz, transform, intrinsics, image.shape, args.max_dim)
    visible = proj["inside"] & proj["visible"]
    visible &= proj["z"] <= proj["zbuf"][proj["flat"]] + args.z_tol

    masked_vertices = 0
    mask_path = None
    if not args.include_human_mask:
        mask, mask_path = load_mask(args.data_root, seq, frame, image.shape)
        on_human = mask_lookup(mask, proj["x"], proj["y"])
        masked_vertices = int(np.count_nonzero(visible & on_human))
        visible &= ~on_human

    return {
        "cam": int(cam),
        "seq": seq,
        "frame": int(frame),
        "image": str(image_path),
        "calib_path": str(calib_path),
        "mask_path": str(mask_path) if mask_path is not None else None,
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "scaled_shape": [int(proj["scaled_shape"][0]), int(proj["scaled_shape"][1])],
        "visible": visible,
        "visible_count": int(np.count_nonzero(visible)),
        "masked_visible_vertices": masked_vertices,
    }


def pair_metrics(cam_a, cam_b, visible_a, visible_b):
    count_a = int(np.count_nonzero(visible_a))
    count_b = int(np.count_nonzero(visible_b))
    shared = int(np.count_nonzero(visible_a & visible_b))
    union = int(np.count_nonzero(visible_a | visible_b))
    min_count = max(min(count_a, count_b), 1)
    max_count = max(max(count_a, count_b), 1)
    return {
        "cam_a": int(cam_a["cam"]),
        "cam_b": int(cam_b["cam"]),
        "seq_a": cam_a["seq"],
        "seq_b": cam_b["seq"],
        "frame_a": int(cam_a["frame"]),
        "frame_b": int(cam_b["frame"]),
        "visible_a": count_a,
        "visible_b": count_b,
        "shared_visible": shared,
        "union_visible": union,
        "overlap_a": float(shared / max(count_a, 1)),
        "overlap_b": float(shared / max(count_b, 1)),
        "overlap_min": float(shared / min_count),
        "overlap_max": float(shared / max_count),
        "jaccard": float(shared / max(union, 1)),
    }


def sort_pairs(pairs, sort_by):
    return sorted(pairs, key=lambda item: (item[sort_by], item["shared_visible"]), reverse=True)


def write_csv(path, pairs):
    fields = [
        "rank",
        "cam_a",
        "cam_b",
        "frame_a",
        "frame_b",
        "visible_a",
        "visible_b",
        "shared_visible",
        "overlap_a",
        "overlap_b",
        "overlap_min",
        "overlap_max",
        "jaccard",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, item in enumerate(pairs, 1):
            row = {field: item.get(field) for field in fields if field != "rank"}
            row["rank"] = rank
            writer.writerow(row)


def color_for_value(value):
    value = float(np.clip(value, 0.0, 1.0))
    color = cv2.applyColorMap(np.array([[int(round(value * 255))]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0]
    return tuple(int(c) for c in color)


def write_heatmap(path, cams, matrix, title):
    cell = 78
    margin_left = 94
    margin_top = 82
    h = margin_top + cell * len(cams) + 28
    w = margin_left + cell * len(cams) + 28
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    for i, cam in enumerate(cams):
        x = margin_left + i * cell
        y = margin_top + i * cell
        cv2.putText(canvas, f"cam{cam:02d}", (x + 6, margin_top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"cam{cam:02d}", (12, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    for r in range(len(cams)):
        for c in range(len(cams)):
            value = matrix[r, c]
            x0 = margin_left + c * cell
            y0 = margin_top + r * cell
            color = color_for_value(value)
            cv2.rectangle(canvas, (x0, y0), (x0 + cell - 2, y0 + cell - 2), color, -1)
            text = f"{value:.2f}"
            cv2.putText(canvas, text, (x0 + 12, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def build_matrix(cams, pairs, key):
    cam_to_i = {cam: i for i, cam in enumerate(cams)}
    matrix = np.eye(len(cams), dtype=np.float32)
    for item in pairs:
        i = cam_to_i[item["cam_a"]]
        j = cam_to_i[item["cam_b"]]
        matrix[i, j] = matrix[j, i] = float(item[key])
    return matrix


def main():
    args = parse_args()
    if args.out_dir is None:
        frame_part = f"f{args.frame:08d}" if args.frame is not None else "first_frame"
        mask_part = "with_human" if args.include_human_mask else "static_bg"
        args.out_dir = f"outputs/rich_camera_overlap_{args.source_sequence}_{frame_part}_{mask_part}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cams = parse_cams(args)
    mesh_path = Path(args.rich_root) / "scan_calibration" / "BBQ" / "scan_camcoord.ply"
    xyz, _ = load_ply_vertices(mesh_path)
    print(f"mesh vertices: {len(xyz)} {mesh_path}")

    camera_items = []
    for cam in cams:
        item = visible_vertices_for_camera(xyz, args, cam)
        camera_items.append(item)
        print(
            f"cam{cam:02d} frame={item['frame']} visible={item['visible_count']} "
            f"masked={item['masked_visible_vertices']}"
        )

    pairs = []
    for i, cam_a in enumerate(camera_items):
        for cam_b in camera_items[i + 1 :]:
            pairs.append(pair_metrics(cam_a, cam_b, cam_a["visible"], cam_b["visible"]))
    ranked_pairs = sort_pairs(pairs, args.sort_by)

    camera_summary = []
    for item in camera_items:
        summary = dict(item)
        summary.pop("visible")
        camera_summary.append(summary)

    report = {
        "args": vars(args),
        "mesh_path": str(mesh_path),
        "geometry_source": "RICH scan_camcoord.ply + XML calibration, not generated depth/*.npy",
        "overlap_definition": {
            "shared_visible": "count of static mesh vertices visible in both cameras after optional human-mask removal",
            "overlap_min": "shared_visible / min(visible_a, visible_b)",
            "jaccard": "shared_visible / union_visible",
            "default_sort": args.sort_by,
        },
        "cameras": camera_summary,
        "pair_overlaps": ranked_pairs,
    }

    summary_path = out_dir / "overlap_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_dir / "overlap_pairs_sorted.csv", ranked_pairs)
    write_csv(out_dir / "training_pairs_overlap_sorted.csv", ranked_pairs)

    sorted_cams = [item["cam"] for item in camera_items]
    for key in ["overlap_min", "jaccard"]:
        matrix = build_matrix(sorted_cams, pairs, key)
        np.savetxt(out_dir / f"{key}_matrix.csv", matrix, delimiter=",", fmt="%.6f")
        write_heatmap(out_dir / f"{key}_matrix.jpg", sorted_cams, matrix, f"{args.source_sequence} {key}")

    print(json.dumps({"out_dir": str(out_dir), "top_pairs": ranked_pairs[:10]}, indent=2))


if __name__ == "__main__":
    main()
