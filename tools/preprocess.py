import argparse
import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from perceptalign.config import DFS_N_FFT, DFS_WIN_LENGTH, FINAL_IMAGE_SIZE, NUM_KEYPOINTS


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


SCENE3_USER_TO_LAYOUT = {
    "user1": "A",
    "user2": "A",
    "user3": "B",
    "user4": "B",
    "user5": "C",
    "user6": "C",
}


def _infer_scene_name(scene_root: str, scene_name: Optional[str]) -> str:
    if scene_name:
        return scene_name
    base = os.path.basename(os.path.normpath(scene_root))
    return base if base else "SceneX"


def _infer_layout(scene: str, user: str) -> str:
    if scene.lower() == "scene3":
        return SCENE3_USER_TO_LAYOUT.get(user, "unknown")
    return "single"


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_val, max_val = torch.min(tensor), torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(tensor)


def complex_division(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    num_complex = torch.complex(num, torch.zeros_like(num)) if not num.is_complex() else num
    den_complex = torch.complex(den, torch.zeros_like(den)) if not den.is_complex() else den
    den_abs = torch.abs(den_complex)
    den_safe = den_complex + eps * (den_abs < eps).float()
    return num_complex / den_safe


def compute_feature(csi_data: torch.Tensor, func, target_size: Tuple[int, int]) -> torch.Tensor:
    feature = func(csi_data)
    feature = feature.unsqueeze(0).unsqueeze(0)
    feature_resized = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
    return normalize_tensor(feature_resized.squeeze(0))


def _load_keypoints3d(keypoints_dir: str, num_keypoints: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    keypoint_files = sorted(glob.glob(os.path.join(keypoints_dir, "*.json")))
    if not keypoint_files:
        return None
    all_xyz = []
    all_conf = []
    for kp_file in keypoint_files:
        with open(kp_file, "r") as f:
            kp_data = json.load(f)
        if kp_data and kp_data[0].get("keypoints3d"):
            kp = torch.tensor(kp_data[0]["keypoints3d"], dtype=torch.float32)[:num_keypoints, :]
            xyz = kp[:, :3]
            conf = kp[:, 3] if kp.shape[1] >= 4 else torch.ones((kp.shape[0],), dtype=torch.float32)
            all_xyz.append(xyz)
            all_conf.append(conf)
    if not all_xyz:
        return None
    return torch.stack(all_xyz, dim=0), torch.stack(all_conf, dim=0)  # ([T,K,3], [T,K])


def _load_geometry_config(scene_root: str, geometry_json: Optional[str]) -> Optional[dict]:
    cand = geometry_json
    if cand is None:
        auto = os.path.join(scene_root, "geometry_config.json")
        cand = auto if os.path.exists(auto) else None
    if cand is None:
        return None
    with open(cand, "r") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else None


def _parse_geometry(
    geo: dict,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    tx = geo.get("tx") or geo.get("tx_coords") or geo.get("tx_coordinates")
    rx = geo.get("rx") or geo.get("rx_coords") or geo.get("rx_coordinates")
    mat = geo.get("scene_matrix") or geo.get("benchmark_matrix") or geo.get("world_to_benchmark")
    rx_mask = geo.get("rx_mask") or geo.get("receiver_mask") or geo.get("rxmask")

    tx_t = torch.tensor(tx, dtype=torch.float32) if tx is not None else None
    rx_t = torch.tensor(rx, dtype=torch.float32) if rx is not None else None
    mat_t = torch.tensor(mat, dtype=torch.float32) if mat is not None else None
    rx_mask_t = torch.tensor(rx_mask, dtype=torch.bool) if rx_mask is not None else None

    if tx_t is not None and tx_t.numel() == 3:
        tx_t = tx_t.view(3)
    else:
        tx_t = None
    if rx_t is not None and rx_t.ndim == 2 and rx_t.shape[-1] == 3:
        rx_t = rx_t.to(torch.float32)
    else:
        rx_t = None
    if mat_t is not None and mat_t.shape == (4, 4):
        mat_t = mat_t.to(torch.float32)
    else:
        mat_t = None
    if rx_mask_t is not None and rx_mask_t.ndim == 1:
        rx_mask_t = rx_mask_t.view(-1).to(torch.bool)
    else:
        rx_mask_t = None

    return tx_t, rx_t, mat_t, rx_mask_t


def _select_geometry_for_layout(geo: dict, layout: str) -> dict:
    """
    Support Scene3-style multi-layout geometry configs.

    Expected JSON patterns:
      - Single layout (Scene1/2/4/5): {"tx": [...], "rx": [[...],[...],[...]], "scene_matrix": [[...]]}
      - Multi layout (Scene3): {"scene_matrix": [[...]], "layouts": {"A": {...}, "B": {...}, "C": {...}}}
    """
    layouts = geo.get("layouts")
    if not isinstance(layouts, dict):
        return geo

    cand = None
    for k in (layout, str(layout), f"layout{layout}", f"Layout{layout}"):
        if k in layouts:
            cand = layouts.get(k)
            break
    if not isinstance(cand, dict):
        return geo

    merged = dict(geo)
    merged.update(cand)
    return merged


def _load_csi_mat_three_receivers(csi_dir: str) -> Optional[List[torch.Tensor]]:
    csi_mat_files = glob.glob(os.path.join(csi_dir, "*.mat"))
    if len(csi_mat_files) < 3:
        return None

    csi_paths: Dict[str, str] = {}
    for f_path in csi_mat_files:
        name = os.path.basename(f_path)
        if "-r1.mat" in name:
            csi_paths["r1"] = f_path
        elif "-r2.mat" in name:
            csi_paths["r2"] = f_path
        elif "-r3.mat" in name:
            csi_paths["r3"] = f_path
    if len(csi_paths) < 3:
        return None

    csi_rxs = []
    for rx_id in ["r1", "r2", "r3"]:
        mat_path = csi_paths[rx_id]
        with h5py.File(mat_path, "r") as f:
            csi_h5_raw = None
            if "csi" not in f:
                raise ValueError(f"No 'csi' key found in {mat_path}.")
            if isinstance(f["csi"], h5py.Dataset):
                csi_h5_raw = f["csi"][()]
            elif isinstance(f["csi"], h5py.Group) and "csi" in f["csi"]:
                csi_h5_raw = f["csi"]["csi"][()]
            else:
                raise ValueError(f"Unsupported HDF5 'csi' field structure in {mat_path}.")

            if csi_h5_raw.dtype.names and "real" in csi_h5_raw.dtype.names and "imag" in csi_h5_raw.dtype.names:
                csi_complex = csi_h5_raw["real"].astype(np.float32) + 1j * csi_h5_raw["imag"].astype(np.float32)
            else:
                csi_complex = csi_h5_raw.astype(np.complex64)

            # Expected raw: [ant, subcarrier, time] or similar; existing pipeline uses permute(2,1,0).
            csi_rxs.append(torch.from_numpy(csi_complex).permute(2, 1, 0))

    return csi_rxs  # list of 3 tensors, each [T, subcarrier, ant]


def process_action_instance(
    action_path: str,
    *,
    num_keypoints: int,
    dfs_n_fft: int,
    dfs_win_length: int,
    final_image_size: Tuple[int, int],
) -> Optional[Dict]:
    csi_dir = os.path.join(action_path, "csi_mat")
    keypoints_dir = os.path.join(action_path, "default", "smplx", "keypoints3d")
    if not os.path.exists(csi_dir) or not os.path.exists(keypoints_dir):
        return None

    kp_loaded = _load_keypoints3d(keypoints_dir, num_keypoints=num_keypoints)
    if kp_loaded is None:
        return None
    keypoints_xyz, keypoints_conf = kp_loaded
    n_kp_frames = int(keypoints_xyz.shape[0])

    csi_rxs = _load_csi_mat_three_receivers(csi_dir)
    if csi_rxs is None:
        return None

    min_t = min(int(csi.shape[0]) for csi in csi_rxs)
    t_segment = min_t // n_kp_frames
    if t_segment <= 0:
        return None

    win_length = min(t_segment, int(dfs_win_length))
    hop_length = max(1, win_length // 4)

    all_frames_data = []
    for i in range(n_kp_frames):
        receivers_data = []
        for rx_idx in range(3):
            csi_t = csi_rxs[rx_idx][i * t_segment : (i + 1) * t_segment, :, :]
            # Antenna quotient (antenna0 / antenna1)
            csi_quotient = complex_division(csi_t[:, :, 0], csi_t[:, :, 1]).T

            phase = compute_feature(torch.angle(csi_quotient), lambda x: x, final_image_size)
            amp = compute_feature(torch.abs(csi_quotient), lambda x: x, final_image_size)

            csi_for_stft = csi_quotient
            if int(csi_for_stft.shape[1]) < int(dfs_n_fft):
                pad_amount = int(dfs_n_fft) - int(csi_for_stft.shape[1])
                csi_for_stft = F.pad(csi_for_stft, (0, pad_amount), "constant", 0)

            stft_res = torch.stft(
                csi_for_stft,
                n_fft=int(dfs_n_fft),
                hop_length=int(hop_length),
                win_length=int(win_length),
                return_complex=True,
                center=True,
            )
            dfs = compute_feature(torch.abs(stft_res).mean(dim=0), lambda x: x, final_image_size)

            receivers_data.append(torch.cat((phase, amp, dfs), dim=0))  # [3,H,W]
        all_frames_data.append(torch.stack(receivers_data, dim=0))  # [Nr,3,H,W]

    final_csi = torch.stack(all_frames_data, dim=0)  # [T,Nr,3,H,W]

    return {
        "csi_data": final_csi,
        "keypoints": keypoints_xyz,
        "keypoints_conf": keypoints_conf,
    }


def _load_existing_manifest(manifest_path: str) -> List[dict]:
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r") as f:
        # Detect JSON array vs JSONL.
        first = ""
        while True:
            ch = f.read(1)
            if ch == "":
                return []
            if ch.isspace():
                continue
            first = ch
            break
        f.seek(0)

        if first == "[":
            obj = json.load(f)
            return obj if isinstance(obj, list) else []

        items = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items


def _write_manifest_jsonl(manifest_path: str, items: List[dict]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _write_manifest_json(manifest_path: str, items: List[dict]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _derive_manifest_paths(manifest_path: str) -> Tuple[str, str]:
    """Return (jsonl_path, json_path) based on a user-provided path (or prefix)."""
    if manifest_path.endswith(".jsonl"):
        return manifest_path, manifest_path[:-1]
    if manifest_path.endswith(".json"):
        return manifest_path + "l", manifest_path
    return manifest_path + ".jsonl", manifest_path + ".json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess one SceneX folder into action-level .pt + manifest.")
    parser.add_argument("--scene_root", type=str, required=True, help="Path to SceneX directory (contains user*/action*/ *-*-*).")
    parser.add_argument("--out_root", type=str, required=True, help="Output root directory for preprocessed data.")
    parser.add_argument("--scene_name", type=str, default=None, help="Override scene name (default: basename(scene_root)).")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Manifest path (.json or .jsonl) OR prefix (default: <out_root>/manifest.json). "
        "Both manifest.json and manifest.jsonl will be written for convenience.",
    )
    parser.add_argument("--overwrite_scene", action="store_true", help="Remove existing entries/files of this scene in manifest/out dir.")
    parser.add_argument("--max_instances", type=int, default=0, help="Debug: process only first N instances (0=all).")
    parser.add_argument(
        "--geometry_json",
        type=str,
        default=None,
        help="Optional geometry_config.json. Can be single-layout (tx/rx/scene_matrix) or Scene3 multi-layout via `layouts:{A,B,C}`.",
    )
    parser.add_argument("--apply_scene_transform", action="store_true", help="Apply scene_matrix to keypoints and subtract tx (requires geometry).")
    parser.add_argument("--num_keypoints", type=int, default=NUM_KEYPOINTS)
    parser.add_argument("--dfs_n_fft", type=int, default=DFS_N_FFT)
    parser.add_argument("--dfs_win_length", type=int, default=DFS_WIN_LENGTH)
    args = parser.parse_args()

    scene_root = os.path.abspath(args.scene_root)
    out_root = os.path.abspath(args.out_root)
    scene_name = _infer_scene_name(scene_root, args.scene_name)
    manifest_path = os.path.abspath(args.manifest) if args.manifest else os.path.join(out_root, "manifest.json")

    if not os.path.exists(scene_root):
        raise FileNotFoundError(f"scene_root not found: {scene_root}")

    out_scene_dir = os.path.join(out_root, scene_name)
    geo_obj = _load_geometry_config(scene_root, args.geometry_json)
    geo_cache: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = {}
    missing_transform_warned = set()

    def get_geometry(layout: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if geo_obj is None:
            return None, None, None, None
        if layout in geo_cache:
            return geo_cache[layout]
        sel = _select_geometry_for_layout(geo_obj, layout)
        tx, rx, mat, rx_mask = _parse_geometry(sel)
        geo_cache[layout] = (tx, rx, mat, rx_mask)
        return geo_cache[layout]

    if geo_obj is not None:
        layouts = geo_obj.get("layouts")
        if isinstance(layouts, dict) and layouts:
            logging.info(f"[geometry] loaded: multi-layout keys={sorted(list(layouts.keys()))}")
        else:
            tx0, rx0, mat0, _rxmask0 = _parse_geometry(geo_obj)
            logging.info(
                f"[geometry] loaded: tx={tuple(tx0.tolist()) if tx0 is not None else None} "
                f"rx={'ok' if rx0 is not None else None} matrix={'ok' if mat0 is not None else None}"
            )
    elif args.apply_scene_transform:
        logging.warning("[geometry] --apply_scene_transform set but no geometry_config.json found; will skip transform.")

    out_pt_dir = os.path.join(out_scene_dir, "pt")
    os.makedirs(out_pt_dir, exist_ok=True)

    all_instances = sorted(glob.glob(os.path.join(scene_root, "user*", "action*", "*-*-*")))
    if args.max_instances and int(args.max_instances) > 0:
        all_instances = all_instances[: int(args.max_instances)]
    logging.info(f"[scan] scene={scene_name} instances={len(all_instances)}")

    if args.overwrite_scene:
        # Clean old pt files for this scene
        if os.path.exists(out_pt_dir):
            for f in glob.glob(os.path.join(out_pt_dir, "*.pt")):
                try:
                    os.remove(f)
                except OSError:
                    pass

    new_manifest_items: List[dict] = []
    saved = 0
    skipped = 0

    pbar = tqdm(all_instances, desc=f"Preprocess {scene_name}")
    for action_path in pbar:
        rel = os.path.relpath(action_path, scene_root)
        parts = rel.split(os.sep)
        if len(parts) < 3:
            skipped += 1
            continue
        user, action, instance = parts[0], parts[1], parts[2]
        layout = _infer_layout(scene_name, user)

        try:
            processed = process_action_instance(
                action_path,
                num_keypoints=int(args.num_keypoints),
                dfs_n_fft=int(args.dfs_n_fft),
                dfs_win_length=int(args.dfs_win_length),
                final_image_size=FINAL_IMAGE_SIZE,
            )
        except Exception as e:
            processed = None
            logging.warning(f"[skip] {action_path} error: {e}")

        if processed is None:
            skipped += 1
            continue

        pt_name = f"{user}_{action}_{instance}.pt"
        pt_path = os.path.join(out_pt_dir, pt_name)
        payload = dict(processed)
        payload.update(
            {
                "scene": scene_name,
                "user": user,
                "action": action,
                "instance": instance,
                "layout": layout,
            }
        )

        tx_coords, rx_coords, scene_matrix, rx_mask = get_geometry(layout)
        if args.apply_scene_transform and not ((tx_coords is not None) and (scene_matrix is not None)):
            if layout not in missing_transform_warned:
                missing_transform_warned.add(layout)
                logging.warning(
                    f"[geometry] missing tx/scene_matrix for layout={layout}; will skip --apply_scene_transform for these samples."
                )

        if args.apply_scene_transform and (tx_coords is not None) and (scene_matrix is not None):
            xyz = payload["keypoints"]  # [T,K,3]
            ones = torch.ones((xyz.shape[0], xyz.shape[1], 1), dtype=xyz.dtype)
            kp_h = torch.cat([xyz, ones], dim=-1)  # [T,K,4]
            kp_trans_h = kp_h @ scene_matrix.T
            w = kp_trans_h[..., 3:4].clamp(min=1e-9)
            xyz2 = kp_trans_h[..., :3] / w
            payload["keypoints"] = xyz2 - tx_coords.view(1, 1, 3)

        if tx_coords is not None:
            payload["tx_coords"] = tx_coords.clone()
        if rx_coords is not None:
            payload["rx_coords"] = rx_coords.clone()
        if rx_mask is not None:
            payload["rx_mask"] = rx_mask.clone()
        if scene_matrix is not None:
            payload["scene_matrix"] = scene_matrix.clone()

        torch.save(payload, pt_path)

        new_manifest_items.append(
            {
                "scene": scene_name,
                "user": user,
                "action": action,
                "instance": instance,
                "layout": layout,
                "pt_relpath": os.path.relpath(pt_path, out_root),
            }
        )
        saved += 1

    old_items = _load_existing_manifest(manifest_path)
    if args.overwrite_scene:
        old_items = [x for x in old_items if x.get("scene") != scene_name]
    merged = old_items + new_manifest_items
    manifest_jsonl_path, manifest_json_path = _derive_manifest_paths(manifest_path)
    _write_manifest_jsonl(manifest_jsonl_path, merged)
    _write_manifest_json(manifest_json_path, merged)

    scene_manifest_path = os.path.join(out_scene_dir, "manifest.json")
    scene_jsonl_path, scene_json_path = _derive_manifest_paths(scene_manifest_path)
    _write_manifest_jsonl(scene_jsonl_path, new_manifest_items)
    _write_manifest_json(scene_json_path, new_manifest_items)

    logging.info(
        f"[done] scene={scene_name} saved={saved} skipped={skipped} "
        f"manifest={manifest_json_path} (also {manifest_jsonl_path}) scene_manifest={scene_json_path} (also {scene_jsonl_path})"
    )


if __name__ == "__main__":
    main()


