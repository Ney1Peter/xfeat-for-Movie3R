#!/usr/bin/env python3
"""Run XFeat on RICH AABB samples and validate matches against the scan mesh."""

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
from test_rich_aabb_xfeat_geometry import (
    compute_ransac_inliers,
    draw_matches,
    load_mask,
    load_rgb,
    make_preview,
    resize_for_matching,
    to_original_coords,
)
from visualize_rich_mesh_correspondences import camera_transform, mask_lookup, projected_vertices
from visualize_rich_mesh_projection import load_ply_vertices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich_root", default="/workspace/data/RICH")
    parser.add_argument("--data_root", default="/workspace/data/RICH/RICH_4Human3R/Training")
    parser.add_argument("--source_sequence", default="BBQ_001_juggle")
    parser.add_argument("--cam_a", type=int, default=3)
    parser.add_argument("--cam_b", type=int, default=4)
    parser.add_argument("--start_frame", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=8192)
    parser.add_argument("--min_cossim", type=float, default=0.9)
    parser.add_argument("--match_mode", choices=["sparse", "semidense"], default="sparse")
    parser.add_argument("--max_dim", type=int, default=1200)
    parser.add_argument("--mesh_max_dim", type=int, default=1400)
    parser.add_argument("--mesh_lookup_radius", type=int, default=4)
    parser.add_argument("--mesh_z_tol", type=float, default=0.03)
    parser.add_argument("--reproj_thresh", type=float, default=24.0)
    parser.add_argument("--ransac_thresh", type=float, default=4.0)
    parser.add_argument("--fundamental_thresh", type=float, default=2.0)
    parser.add_argument("--max_draw", type=int, default=500)
    parser.add_argument("--out_dir", default=None)
    return parser.parse_args()


def seq_name(source_sequence, cam):
    return f"{source_sequence}_cam_{cam:02d}"


def cam_from_seq(seq):
    return int(seq.rsplit("_", 1)[-1])


def build_visible_vertex_map(xyz, rich_root, seq, image_shape, mesh_max_dim, mesh_z_tol):
    cam = cam_from_seq(seq)
    transform, intrinsics, calib_path = camera_transform(rich_root, cam, "xml_w2c")
    proj = projected_vertices(xyz, transform, intrinsics, image_shape, mesh_max_dim)
    visible = proj["inside"] & proj["visible"]
    visible &= proj["z"] <= proj["zbuf"][proj["flat"]] + mesh_z_tol

    hs, ws = proj["scaled_shape"]
    idx_map = np.full(hs * ws, -1, dtype=np.int32)
    z_map = np.full(hs * ws, np.inf, dtype=np.float32)
    indices = np.flatnonzero(visible)
    flat = proj["flat"][indices]
    z = proj["z"][indices]
    order = np.argsort(z)[::-1]
    idx_map[flat[order]] = indices[order].astype(np.int32)
    z_map[flat[order]] = z[order].astype(np.float32)
    return {
        "seq": seq,
        "calib_path": str(calib_path),
        "proj": proj,
        "idx_map": idx_map.reshape(hs, ws),
        "z_map": z_map.reshape(hs, ws),
        "visible": visible,
    }


def compute_visible_overlap(ref_map, cur_map, mask_ref, mask_cur):
    visible_ref = ref_map["visible"].copy()
    visible_cur = cur_map["visible"].copy()
    ref_human = mask_lookup(mask_ref, ref_map["proj"]["x"], ref_map["proj"]["y"])
    cur_human = mask_lookup(mask_cur, cur_map["proj"]["x"], cur_map["proj"]["y"])
    visible_ref &= ~ref_human
    visible_cur &= ~cur_human

    visible_ref_count = int(np.count_nonzero(visible_ref))
    visible_cur_count = int(np.count_nonzero(visible_cur))
    shared_visible = int(np.count_nonzero(visible_ref & visible_cur))
    union_visible = int(np.count_nonzero(visible_ref | visible_cur))
    return {
        "visible_ref": visible_ref_count,
        "visible_cur": visible_cur_count,
        "shared_visible": shared_visible,
        "union_visible": union_visible,
        "overlap_ref": float(shared_visible / max(visible_ref_count, 1)),
        "overlap_cur": float(shared_visible / max(visible_cur_count, 1)),
        "overlap_min": float(shared_visible / max(min(visible_ref_count, visible_cur_count), 1)),
        "overlap_max": float(shared_visible / max(max(visible_ref_count, visible_cur_count), 1)),
        "jaccard": float(shared_visible / max(union_visible, 1)),
    }


def lookup_vertex(mesh_map, x, y, radius):
    proj = mesh_map["proj"]
    scale = proj["scale"]
    xs = float(x) * scale
    ys = float(y) * scale
    xi = int(round(xs))
    yi = int(round(ys))
    idx_map = mesh_map["idx_map"]
    h, w = idx_map.shape[:2]
    x0 = max(0, xi - radius)
    x1 = min(w, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(h, yi + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = idx_map[y0:y1, x0:x1]
    valid = patch >= 0
    if not np.any(valid):
        return None
    yy, xx = np.nonzero(valid)
    cand = patch[yy, xx].astype(np.int64)
    px = x0 + xx
    py = y0 + yy
    d2 = (px.astype(np.float32) - xs) ** 2 + (py.astype(np.float32) - ys) ** 2
    best = int(np.argmin(d2))
    return int(cand[best])


def on_mask(mask, x, y):
    h, w = mask.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return True
    return bool(mask[yi, xi])


def reproj_error_for_vertex(idx, target_xy, target_map):
    proj = target_map["proj"]
    if idx is None or not target_map["visible"][idx]:
        return None
    tx, ty = target_xy
    px = float(proj["x"][idx])
    py = float(proj["y"][idx])
    if not np.isfinite(px) or not np.isfinite(py):
        return None
    return float(math.hypot(px - float(tx), py - float(ty)))


def evaluate_mesh_geometry(mkpts_ref_orig, mkpts_cur_orig, ref_map, cur_map, mask_ref, mask_cur, args):
    items = []
    for idx, (p_ref, p_cur) in enumerate(zip(mkpts_ref_orig, mkpts_cur_orig)):
        x_ref, y_ref = map(float, p_ref)
        x_cur, y_cur = map(float, p_cur)
        human = on_mask(mask_ref, x_ref, y_ref) or on_mask(mask_cur, x_cur, y_cur)
        cur_vertex = None if human else lookup_vertex(cur_map, x_cur, y_cur, args.mesh_lookup_radius)
        ref_vertex = None if human else lookup_vertex(ref_map, x_ref, y_ref, args.mesh_lookup_radius)
        err_cur_to_ref = None if human else reproj_error_for_vertex(cur_vertex, (x_ref, y_ref), ref_map)
        err_ref_to_cur = None if human else reproj_error_for_vertex(ref_vertex, (x_cur, y_cur), cur_map)
        errs = [e for e in [err_cur_to_ref, err_ref_to_cur] if e is not None]
        best_err = min(errs) if errs else None
        mesh_inlier = bool(best_err is not None and best_err < args.reproj_thresh)
        items.append(
            {
                "idx": int(idx),
                "ref_xy_original": [x_ref, y_ref],
                "cur_xy_original": [x_cur, y_cur],
                "on_human": bool(human),
                "ref_vertex": ref_vertex,
                "cur_vertex": cur_vertex,
                "reproj_error_cur_to_ref_px": err_cur_to_ref,
                "reproj_error_ref_to_cur_px": err_ref_to_cur,
                "best_mesh_reproj_error_px": best_err,
                "mesh_inlier": mesh_inlier,
            }
        )
    return items


def compute_fundamental_inliers(mkpts0, mkpts1, thresh):
    if len(mkpts0) < 8:
        return np.zeros((len(mkpts0),), dtype=bool)
    try:
        _, inliers = cv2.findFundamentalMat(
            mkpts0,
            mkpts1,
            cv2.USAC_MAGSAC,
            thresh,
            0.999,
            10000,
        )
    except Exception:
        _, inliers = cv2.findFundamentalMat(
            mkpts0,
            mkpts1,
            cv2.FM_RANSAC,
            thresh,
            0.999,
        )
    if inliers is None:
        return np.zeros((len(mkpts0),), dtype=bool)
    return inliers.reshape(-1).astype(bool)


def run_pair(xfeat, xyz, map_cache, pair, out_dir, args):
    img_ref, img_ref_path = load_rgb(args.data_root, pair["ref_seq"], pair["ref_frame"])
    img_cur, img_cur_path = load_rgb(args.data_root, pair["cur_seq"], pair["cur_frame"])
    img_ref_match, sx_ref, sy_ref = resize_for_matching(img_ref, args.max_dim)
    img_cur_match, sx_cur, sy_cur = resize_for_matching(img_cur, args.max_dim)

    # **========== 原始代码：只使用 sparse XFeat matching ==========**
    # mkpts_ref, mkpts_cur = xfeat.match_xfeat(
    #     img_ref_match,
    #     img_cur_match,
    #     top_k=args.top_k,
    #     min_cossim=args.min_cossim,
    # )
    # **========== 新代码：支持 sparse / semi-dense XFeat matching ==========**
    if args.match_mode == "semidense":
        mkpts_ref, mkpts_cur = xfeat.match_xfeat_star(
            img_ref_match,
            img_cur_match,
            top_k=args.top_k,
        )
    else:
        mkpts_ref, mkpts_cur = xfeat.match_xfeat(
            img_ref_match,
            img_cur_match,
            top_k=args.top_k,
            min_cossim=args.min_cossim,
        )
    # **========== 结束 ==========**
    mkpts_ref = np.asarray(mkpts_ref, dtype=np.float32)
    mkpts_cur = np.asarray(mkpts_cur, dtype=np.float32)
    mkpts_ref_orig = to_original_coords(mkpts_ref, sx_ref, sy_ref)
    mkpts_cur_orig = to_original_coords(mkpts_cur, sx_cur, sy_cur)

    for seq, img in [(pair["ref_seq"], img_ref), (pair["cur_seq"], img_cur)]:
        if seq not in map_cache:
            map_cache[seq] = build_visible_vertex_map(xyz, args.rich_root, seq, img.shape, args.mesh_max_dim, args.mesh_z_tol)

    mask_ref = load_mask(args.data_root, pair["ref_seq"], pair["ref_frame"], img_ref.shape)
    mask_cur = load_mask(args.data_root, pair["cur_seq"], pair["cur_frame"], img_cur.shape)
    visible_overlap = compute_visible_overlap(map_cache[pair["ref_seq"]], map_cache[pair["cur_seq"]], mask_ref, mask_cur)
    ransac_mask = compute_ransac_inliers(mkpts_ref, mkpts_cur, args.ransac_thresh)
    fundamental_mask = compute_fundamental_inliers(mkpts_ref, mkpts_cur, args.fundamental_thresh)
    eval_items = evaluate_mesh_geometry(
        mkpts_ref_orig,
        mkpts_cur_orig,
        map_cache[pair["ref_seq"]],
        map_cache[pair["cur_seq"]],
        mask_ref,
        mask_cur,
        args,
    )
    mesh_mask = np.array([item["mesh_inlier"] for item in eval_items], dtype=bool)
    human_mask = np.array([item["on_human"] for item in eval_items], dtype=bool)
    reproj = [item["best_mesh_reproj_error_px"] for item in eval_items if item["best_mesh_reproj_error_px"] is not None]

    pair_dir = out_dir / pair["name"]
    pair_dir.mkdir(parents=True, exist_ok=True)
    raw_indices = np.arange(len(mkpts_ref))[: args.max_draw]
    ransac_indices = np.flatnonzero(ransac_mask)[: args.max_draw]
    mesh_indices = np.flatnonzero(mesh_mask)[: args.max_draw]
    if len(mesh_indices) == 0:
        mesh_indices = raw_indices

    draw_specs = [
        ("xfeat_raw_matches.png", raw_indices, (0, 255, 255), f"{pair['name']} raw: {len(mkpts_ref)}"),
        (
            "xfeat_ransac_matches.png",
            ransac_indices if len(ransac_indices) else raw_indices,
            (0, 220, 0),
            f"{pair['name']} RANSAC: {int(ransac_mask.sum())}/{len(mkpts_ref)}",
        ),
        (
            "xfeat_mesh_geometry_matches.png",
            mesh_indices,
            (0, 220, 0) if mesh_mask.any() else (0, 0, 255),
            f"{pair['name']} mesh geom: {int(mesh_mask.sum())}/{len(mkpts_ref)}",
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
        "ref": {"seq": pair["ref_seq"], "frame": int(pair["ref_frame"]), "image": str(img_ref_path)},
        "cur": {"seq": pair["cur_seq"], "frame": int(pair["cur_frame"]), "image": str(img_cur_path)},
        "top_k": int(args.top_k),
        "min_cossim": float(args.min_cossim),
        "match_mode": args.match_mode,
        "match_max_dim": int(args.max_dim),
        "mesh_max_dim": int(args.mesh_max_dim),
        "mesh_lookup_radius": int(args.mesh_lookup_radius),
        "mesh_z_tol": float(args.mesh_z_tol),
        "raw_matches": int(len(mkpts_ref)),
        "homography_ransac_inliers": int(ransac_mask.sum()),
        "fundamental_ransac_inliers": int(fundamental_mask.sum()),
        "mesh_geometry_inliers": int(mesh_mask.sum()),
        "overlap_min": visible_overlap["overlap_min"],
        "mesh_visible_overlap": visible_overlap,
        "mesh_geometry_inlier_ratio": float(mesh_mask.sum() / max(len(mkpts_ref), 1)),
        "mesh_inliers_inside_ransac": int((mesh_mask & ransac_mask).sum()),
        "mesh_inliers_inside_fundamental": int((mesh_mask & fundamental_mask).sum()),
        "matches_on_human": int(human_mask.sum()),
        "mesh_reproj_error_mean_px": float(np.mean(reproj)) if reproj else None,
        "mesh_reproj_error_median_px": float(np.median(reproj)) if reproj else None,
        "previews": previews,
    }
    with open(pair_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "matches": eval_items}, f, indent=2)
    return summary


def write_contact_sheet(out_dir, summaries):
    images = []
    for summary in summaries:
        for name in ["xfeat_raw_matches_preview.jpg", "xfeat_ransac_matches_preview.jpg", "xfeat_mesh_geometry_matches_preview.jpg"]:
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
        sheet[y0 : y0 + 42] = 20
        cv2.putText(sheet, f"{pair_name} / {title}", (12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        sheet[y0 + 42 : y0 + 42 + img.shape[0], : img.shape[1]] = img
    out_path = out_dir / "rich_xfeat_mesh_contact_sheet.jpg"
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return str(out_path)


def main():
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = (
            f"outputs/rich_xfeat_mesh_aabb_{args.source_sequence}"
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
    mesh_path = Path(args.rich_root) / "scan_calibration" / "BBQ" / "scan_camcoord.ply"
    xyz, _ = load_ply_vertices(mesh_path)
    print("mesh vertices:", len(xyz), mesh_path)
    xfeat = XFeat(top_k=args.top_k)
    map_cache = {}
    summaries = []
    for pair in pairs:
        print("\nRunning pair:", pair["name"])
        summaries.append(run_pair(xfeat, xyz, map_cache, pair, out_dir, args))

    contact_sheet = write_contact_sheet(out_dir, summaries)
    report = {
        "args": vars(args),
        "mesh_path": str(mesh_path),
        "geometry_source": "RICH scan_camcoord.ply + XML calibration, not generated depth/*.npy",
        "pairs": summaries,
        "contact_sheet": contact_sheet,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
