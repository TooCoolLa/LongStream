import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import c2w_in_view_space, selected_frame_indices, world_to_view


def _natural_sort_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def detect_sequences(output_root: str) -> List[str]:
    if not os.path.isdir(output_root):
        return []
    seqs = []
    for entry in os.listdir(output_root):
        seq_dir = os.path.join(output_root, entry)
        if os.path.isdir(seq_dir) and os.path.isdir(os.path.join(seq_dir, "poses")):
            seqs.append(entry)
    return sorted(seqs, key=_natural_sort_key)


def _parse_abs_pose_txt(path: str) -> Tuple[np.ndarray, List[int]]:
    frames = []
    mats = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            frame_id = int(vals[0])
            r = np.array([float(v) for v in vals[1:10]]).reshape(3, 3)
            t = np.array([float(v) for v in vals[10:13]])
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = r
            w2c[:3, 3] = t
            mats.append(w2c)
            frames.append(frame_id)
    return np.stack(mats, axis=0), frames


def _parse_intri_txt(path: str) -> Tuple[np.ndarray, List[int]]:
    frames = []
    mats = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            frame_id = int(vals[0])
            fx, fy, cx, cy = (
                float(vals[1]),
                float(vals[2]),
                float(vals[3]),
                float(vals[4]),
            )
            k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            mats.append(k)
            frames.append(frame_id)
    return np.stack(mats, axis=0), frames


def _load_images_from_dir(rgb_dir: str) -> Optional[np.ndarray]:
    patterns = ["frame_*.png", "frame_*.jpg", "frame_*.jpeg"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(rgb_dir, pat)))
    if not paths:
        return None
    paths = sorted(paths, key=_natural_sort_key)
    try:
        from PIL import Image

        imgs = []
        for p in paths:
            img = np.array(Image.open(p))
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            imgs.append(img)
        return np.stack(imgs, axis=0)
    except ImportError:
        return None


def _load_depth_frames(depth_dir: str, num_frames: int) -> Optional[np.ndarray]:
    paths = sorted(
        glob.glob(os.path.join(depth_dir, "frame_*.npy")), key=_natural_sort_key
    )
    if not paths:
        return None
    depths = []
    for p in paths:
        depths.append(np.load(p))
    if not depths:
        return None
    return np.stack(depths, axis=0)


def _load_sky_masks_from_dir(
    sky_dir: str, num_frames: int, min_frame_id: int = 0
) -> Optional[np.ndarray]:
    patterns = ["images__frame_*.jpg", "frame_*.png", "frame_*.jpg", "*.jpg", "*.png"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(sky_dir, pat)))
    if not paths:
        return None
    paths = sorted(set(paths), key=_natural_sort_key)
    try:
        from PIL import Image

        masks = []
        for p in paths:
            mask = np.array(Image.open(p))
            if mask.ndim == 3:
                mask = mask[..., 0]
            masks.append(mask)
        if not masks:
            return None

        base_name = os.path.basename(paths[0])
        digits = re.findall(r"(\d+)", base_name)
        if digits:
            first_file_id = int(digits[-1])
            offset = first_file_id - min_frame_id
        else:
            offset = 0

        if offset > 0:
            masks = [np.zeros_like(masks[0])] * offset + masks
        elif offset < 0:
            masks = masks[-offset:]

        if len(masks) > num_frames:
            masks = masks[:num_frames]
        elif len(masks) < num_frames:
            pad = [np.zeros_like(masks[-1])] * (num_frames - len(masks))
            masks = masks + pad

        return np.stack(masks, axis=0)
    except ImportError:
        return None


def _read_ply_binary(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with open(path, "rb") as f:
        header_end = 0
        for _ in range(1000):
            line = f.readline()
            if line.decode("ascii").strip() == "end_header":
                header_end = f.tell()
                break

    with open(path, "rb") as f:
        f.seek(header_end)
        num_verts = 0
        f.seek(0)
        for _ in range(100):
            hl = f.readline().decode("ascii").strip()
            if hl.startswith("element vertex"):
                num_verts = int(hl.split()[-1])
            if hl == "end_header":
                break

        has_color = False
        f.seek(0)
        for _ in range(100):
            hl = f.readline().decode("ascii").strip()
            if "red" in hl or "green" in hl or "blue" in hl:
                has_color = True
            if hl == "end_header":
                break

    if has_color:
        vertex_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
    else:
        vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

    with open(path, "rb") as f:
        f.seek(header_end)
        data = np.fromfile(f, dtype=vertex_dtype, count=num_verts)

    points = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(
        np.float32
    )
    if has_color:
        colors = np.stack(
            [data["red"], data["green"], data["blue"]], axis=-1
        ).astype(np.uint8)
    else:
        colors = None

    return points, colors


def _maybe_downsample(points, colors, max_points, seed=0):
    if max_points is None or points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[keep], colors[keep] if colors is not None else None


def _find_point_clouds(seq_dir: str) -> Dict[str, str]:
    sources = {}
    points_root = os.path.join(seq_dir, "points")
    if not os.path.isdir(points_root):
        return sources

    for branch_name, subdirs in [
        ("point_head", ["point_head"]),
        ("depth_projection", ["dpt_unproj"]),
    ]:
        for subdir in subdirs:
            full_ply = os.path.join(points_root, subdir, f"{subdir}_full.ply")
            if os.path.isfile(full_ply):
                sources[f"{branch_name}_full"] = full_ply
            frame_dir = os.path.join(points_root, subdir)
            if os.path.isdir(frame_dir):
                frame_plies = sorted(
                    glob.glob(os.path.join(frame_dir, "frame_*.ply")),
                    key=_natural_sort_key,
                )
                if frame_plies:
                    sources[f"{branch_name}_frames"] = frame_dir
    return sources


def load_runpy_session(seq_dir: str) -> Dict:
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    pose_txt = os.path.join(seq_dir, "poses", "abs_pose.txt")
    intri_txt = os.path.join(seq_dir, "poses", "intri.txt")

    if not os.path.isfile(pose_txt):
        raise FileNotFoundError(f"abs_pose.txt not found: {pose_txt}")
    if not os.path.isfile(intri_txt):
        raise FileNotFoundError(f"intri.txt not found: {intri_txt}")

    w2c, pose_frames = _parse_abs_pose_txt(pose_txt)
    intri, intri_frames = _parse_intri_txt(intri_txt)

    num_frames = len(pose_frames)
    if num_frames == 0:
        raise ValueError("No frames found in pose file")

    rgb_dir = os.path.join(seq_dir, "images", "rgb")
    images = None
    if os.path.isdir(rgb_dir):
        images = _load_images_from_dir(rgb_dir)

    depth_dir = os.path.join(seq_dir, "depth", "dpt")
    depth = None
    if os.path.isdir(depth_dir):
        depth = _load_depth_frames(depth_dir, num_frames)

    sky_dir = os.path.join(seq_dir, "sky_masks")
    sky_masks = None
    if os.path.isdir(sky_dir):
        sky_masks = _load_sky_masks_from_dir(sky_dir, num_frames, min(pose_frames))

    has_sky = sky_masks is not None
    sky_removed_ratio = None
    if has_sky:
        sky_removed_ratio = float(1.0 - (sky_masks > 0).mean())

    point_sources = _find_point_clouds(seq_dir)

    meta = {
        "session_dir": seq_dir,
        "source": "runpy",
        "num_frames": num_frames,
        "height": int(images.shape[1]) if images is not None else 518,
        "width": int(images.shape[2]) if images is not None else 518,
        "has_sky_masks": has_sky,
        "sky_removed_ratio": sky_removed_ratio,
        "point_sources": list(point_sources.keys()),
    }

    return {
        "seq_dir": seq_dir,
        "w2c": w2c,
        "intri": intri,
        "images": images,
        "depth": depth,
        "sky_masks": sky_masks,
        "point_sources": point_sources,
        "metadata": meta,
        "frame_ids": pose_frames,
    }


def _origin_shift(w2c_all) -> np.ndarray:
    first = c2w_in_view_space(w2c_all[0])
    return first[:3, 3].copy()


def _sample_colors_from_images(
    images: np.ndarray,
    points: np.ndarray,
    w2c: np.ndarray,
    intri: np.ndarray,
) -> np.ndarray:
    if images.ndim == 4:
        h, w = images.shape[1:3]
    else:
        h, w = images.shape[:2]
        images = images[np.newaxis, ...]

    fx = intri[0, 0]
    fy = intri[1, 1]
    cx = intri[0, 2]
    cy = intri[1, 2]

    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = (R @ points.T + t[:, None]).T

    valid_z = pts_cam[:, 2] > 1e-6
    u = np.zeros(len(pts_cam), dtype=int)
    v = np.zeros(len(pts_cam), dtype=int)
    if valid_z.any():
        z_safe = np.clip(pts_cam[valid_z, 2], 1e-6, None)
        u[valid_z] = (pts_cam[valid_z, 0] * fx / z_safe + cx).astype(int)
        v[valid_z] = (pts_cam[valid_z, 1] * fy / z_safe + cy).astype(int)

    valid = (
        valid_z
        & (u >= 0)
        & (u < w)
        & (v >= 0)
        & (v < h)
    )
    colors = np.full((points.shape[0], 3), 128, dtype=np.uint8)
    colors[valid] = images[0, v[valid], u[valid]]
    return colors


def collect_points_from_runpy(
    session: Dict,
    branch: str,
    display_mode: str,
    frame_index: int,
    mask_sky: bool,
    max_points: Optional[int],
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .common import branch_key

    branch = branch_key(branch)
    meta = session["metadata"]
    frame_ids = selected_frame_indices(
        meta["num_frames"], frame_index, display_mode
    )
    if not frame_ids:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.zeros(3, dtype=np.float64),
        )

    w2c = session["w2c"]
    images = session["images"]
    sky = session["sky_masks"] if mask_sky else None
    point_sources = session["point_sources"]
    depth = session["depth"]
    intri = session["intri"]

    origin_shift = _origin_shift(w2c)

    source_key = f"{branch}_full"
    use_full_ply = source_key in point_sources

    if use_full_ply:
        ply_path = point_sources[source_key]
        points, colors = _read_ply_binary(ply_path)
        if colors is None:
            if images is not None:
                colors = _sample_colors_from_images(
                    images, points, w2c[0], intri[0]
                )
            else:
                colors = np.full((points.shape[0], 3), 128, dtype=np.uint8)

        points_view = (
            world_to_view(points.astype(np.float64)) - origin_shift[None]
        )
        points_view, colors = _maybe_downsample(
            points_view.astype(np.float32), colors, max_points, seed
        )
        return points_view, colors, origin_shift

    frame_source_key = f"{branch}_frames"
    if frame_source_key in point_sources:
        frame_dir = point_sources[frame_source_key]
        rng = np.random.default_rng(seed)
        points_list = []
        colors_list = []

        for idx in frame_ids:
            frame_id = (
                session["frame_ids"][idx]
                if idx < len(session["frame_ids"])
                else idx
            )
            ply_path = os.path.join(frame_dir, f"frame_{frame_id:06d}.ply")
            if not os.path.isfile(ply_path):
                continue

            pts, cols = _read_ply_binary(ply_path)
            if cols is None and images is not None:
                cols = _sample_colors_from_images(
                    images[idx : idx + 1],
                    pts,
                    w2c[idx : idx + 1],
                    intri[idx : idx + 1],
                )
            if cols is None:
                cols = np.full((pts.shape[0], 3), 128, dtype=np.uint8)

            pts_view = (
                world_to_view(pts.astype(np.float64)) - origin_shift[None]
            )
            points_list.append(pts_view.astype(np.float32))
            colors_list.append(cols)

        if points_list:
            all_pts = np.concatenate(points_list, axis=0)
            all_cols = np.concatenate(colors_list, axis=0)
            all_pts, all_cols = _maybe_downsample(
                all_pts, all_cols, max_points, seed
            )
            return all_pts, all_cols, origin_shift

    if (
        branch == "depth_projection"
        and depth is not None
        and images is not None
    ):
        rng = np.random.default_rng(seed)
        points_list = []
        colors_list = []
        per_frame_budget = None
        if max_points is not None and max_points > 0:
            per_frame_budget = max(max_points // max(len(frame_ids), 1), 1)

        for idx in frame_ids:
            depth_i = depth[idx]
            valid = (np.isfinite(depth_i) & (depth_i > 0)).reshape(-1)
            if sky is not None and idx < len(sky):
                valid &= sky[idx].reshape(-1) > 0
            flat = np.flatnonzero(valid)
            if flat.size == 0:
                continue

            from .geometry import _depth_points_from_flat, _sample_flat_indices

            flat = _sample_flat_indices(flat, per_frame_budget, rng)
            pts_world = _depth_points_from_flat(
                depth_i, intri[idx], w2c[idx], flat
            )
            pts_view = (
                world_to_view(pts_world.astype(np.float64)) - origin_shift[None]
            )

            points_list.append(pts_view.astype(np.float32))
            colors_list.append(
                images[idx].reshape(-1, 3)[flat].astype(np.uint8)
            )

        if points_list:
            all_pts = np.concatenate(points_list, axis=0)
            all_cols = np.concatenate(colors_list, axis=0)
            all_pts, all_cols = _maybe_downsample(
                all_pts, all_cols, max_points, seed
            )
            return all_pts, all_cols, origin_shift

    return (
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.uint8),
        origin_shift,
    )


def camera_geometry_from_runpy(
    session: Dict,
    display_mode: str,
    frame_index: int,
    camera_scale_ratio: float,
    points_hint=None,
) -> Tuple[np.ndarray, list, np.ndarray]:
    meta = session["metadata"]
    frame_ids = selected_frame_indices(
        meta["num_frames"], frame_index, display_mode
    )
    w2c = session["w2c"]
    intri = session["intri"]
    origin_shift = _origin_shift(w2c)

    center_points = np.array(
        [
            c2w_in_view_space(w2c[idx], origin_shift)[:3, 3]
            for idx in frame_ids
        ],
        dtype=np.float64,
    )
    center_extent = 1.0
    if len(center_points) > 1:
        center_extent = float(
            np.linalg.norm(
                center_points.max(axis=0) - center_points.min(axis=0)
            )
        )

    point_extent = 0.0
    if points_hint is not None and len(points_hint) > 0:
        lo = np.percentile(points_hint, 5, axis=0)
        hi = np.percentile(points_hint, 95, axis=0)
        point_extent = float(np.linalg.norm(hi - lo))

    extent = max(center_extent, point_extent, 1.0)
    depth_scale = extent * float(camera_scale_ratio)

    h = meta["height"]
    w = meta["width"]

    from .geometry import _frustum_corners_camera

    centers = []
    frustums = []
    for idx in frame_ids:
        c2w_view = c2w_in_view_space(w2c[idx], origin_shift)
        center = c2w_view[:3, 3]
        corners_cam = _frustum_corners_camera(
            intri[idx], (h, w), depth_scale
        )
        corners_world = (c2w_view[:3, :3] @ corners_cam.T).T + center[None]
        centers.append(center)
        frustums.append((center, corners_world))
    return np.asarray(centers, dtype=np.float64), frustums, origin_shift


def load_frame_previews_from_runpy(session: Dict, frame_index: int):
    meta = session["metadata"]
    frame_index = int(np.clip(frame_index, 0, meta["num_frames"] - 1))
    images = session["images"]
    depth = session["depth"]

    if images is None:
        rgb = np.zeros(
            (meta["height"], meta["width"], 3), dtype=np.uint8
        )
    else:
        rgb = np.array(images[frame_index])

    if depth is None:
        depth_color = np.zeros(
            (meta["height"], meta["width"], 3), dtype=np.uint8
        )
    else:
        from longstream.utils.depth import colorize_depth

        depth_color = colorize_depth(np.array(depth[frame_index]), cmap="plasma")

    label = f"Frame {frame_index + 1}/{meta['num_frames']}"
    return rgb, depth_color, label
