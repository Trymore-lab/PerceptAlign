import argparse
import contextlib
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import yaml

from perceptalign.models.perceptalign import posenet


def _resolve_path(path_str: str, base_dir: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(base_dir, path_str))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_manifest(manifest_path: str) -> List[dict]:
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
            if not isinstance(obj, list):
                raise ValueError(f"manifest json must be a list, got: {type(obj)}")
            return obj

        items: List[dict] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items


def _split_train_val(items: List[dict], *, val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if not items:
        return [], []
    r = random.Random(seed)
    idx = list(range(len(items)))
    r.shuffle(idx)
    n_val = max(1, int(round(len(items) * float(val_ratio)))) if val_ratio > 0 else 0
    val_idx = set(idx[:n_val])
    train, val = [], []
    for i, it in enumerate(items):
        (val if i in val_idx else train).append(it)
    return train, val


def _select_by_protocol(items: List[dict], cfg: dict) -> Tuple[List[dict], List[dict]]:
    proto = cfg["protocol"]
    ptype = proto["type"]

    if ptype == "cross_layout":
        scene = proto["scene"]
        train_layouts = set(proto["train_layouts"])
        test_layouts = set(proto.get("test_layouts") or proto.get("test_layout") or [])
        if not test_layouts:
            raise ValueError("cross_layout requires protocol.test_layouts (or test_layout).")
        train_pool = [x for x in items if x.get("scene") == scene and x.get("layout") in train_layouts]
        test_set = [x for x in items if x.get("scene") == scene and x.get("layout") in test_layouts]
        return train_pool, test_set

    if ptype == "cross_scene":
        train_scenes = set(proto["train_scenes"])
        test_scene = proto["test_scene"]
        test_layout = proto.get("test_layout")
        train_pool = [x for x in items if x.get("scene") in train_scenes]
        if test_layout:
            test_set = [x for x in items if x.get("scene") == test_scene and x.get("layout") == test_layout]
        else:
            test_set = [x for x in items if x.get("scene") == test_scene]
        return train_pool, test_set

    if ptype == "cross_subject":
        scene = proto["scene"]
        test_user = proto.get("test_user")
        scene_items = [x for x in items if x.get("scene") == scene]
        users = sorted({x.get("user") for x in scene_items if x.get("user")})
        if not users:
            raise ValueError(f"No users found in manifest for scene={scene}")
        if test_user in (None, "", "random"):
            r = random.Random(int(cfg["experiment"]["seed"]))
            test_user = r.choice(users)
            print(f"[protocol] cross_subject random test_user={test_user}")
        train_user_n = proto.get("train_users") or proto.get("num_train_users")
        train_user_n = int(train_user_n) if train_user_n not in (None, "") else None
        candidate_users = [u for u in users if u != test_user]
        if train_user_n is not None and train_user_n > 0:
            r = random.Random(int(cfg["experiment"]["seed"]) + 997)
            if train_user_n > len(candidate_users):
                train_user_n = len(candidate_users)
            train_users = sorted(r.sample(candidate_users, k=train_user_n))
        else:
            train_users = sorted(candidate_users)
        print(f"[protocol] cross_subject train_users={train_users} (n={len(train_users)})")
        train_pool = [x for x in scene_items if x.get("user") in set(train_users)]
        test_set = [x for x in scene_items if x.get("user") == test_user]
        return train_pool, test_set

    raise ValueError(f"Unknown protocol.type: {ptype}")


class ManifestPTDataset(Dataset):
    def __init__(self, pt_paths: List[str], max_seq_len: int):
        self.pt_paths = list(pt_paths)
        self.max_seq_len = int(max_seq_len)

    def __len__(self) -> int:
        return len(self.pt_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.pt_paths[idx]
        data = torch.load(path, map_location="cpu")
        csi = data["csi_data"]
        kp = data["keypoints"]
        kp_conf = data.get("keypoints_conf")
        tx = data.get("tx_coords")
        rx = data.get("rx_coords")
        rx_mask = data.get("rx_mask")

        if csi.shape[0] > self.max_seq_len:
            csi = csi[: self.max_seq_len]
            kp = kp[: self.max_seq_len]
            if kp_conf is not None:
                kp_conf = kp_conf[: self.max_seq_len]

        return {
            "csi_data": csi,
            "keypoints": kp,
            "keypoints_conf": kp_conf,
            "tx_coords": tx,
            "rx_coords": rx,
            "rx_mask": rx_mask,
        }


def collate_sequences(batch: List[dict]) -> dict:
    csi_list = [x["csi_data"] for x in batch]
    kp_list = [x["keypoints"] for x in batch]
    conf_list = [x.get("keypoints_conf") for x in batch]
    rxmask_list = [x.get("rx_mask") for x in batch]

    seq_lengths = torch.tensor([int(x.shape[0]) for x in csi_list], dtype=torch.long)
    csi_padded = pad_sequence(csi_list, batch_first=True, padding_value=0.0)
    kp_padded = pad_sequence(kp_list, batch_first=True, padding_value=0.0)
    mask = torch.arange(int(seq_lengths.max()))[None, :] < seq_lengths[:, None]

    has_conf = all(c is not None for c in conf_list)
    conf_padded = pad_sequence(conf_list, batch_first=True, padding_value=0.0) if has_conf else None

    tx_list = [x.get("tx_coords") for x in batch]
    rx_list = [x.get("rx_coords") for x in batch]
    if all(isinstance(t, torch.Tensor) and t.numel() == 3 for t in tx_list):
        tx = torch.stack([t.to(torch.float32) for t in tx_list], dim=0)
    else:
        tx = None
    if all(isinstance(r, torch.Tensor) and r.ndim == 2 and r.shape[-1] == 3 for r in rx_list):
        rx = torch.stack([r.to(torch.float32) for r in rx_list], dim=0)
    else:
        rx = None

    if all(isinstance(m, torch.Tensor) and m.ndim == 1 for m in rxmask_list):
        rx_mask = torch.stack([m.to(torch.bool) for m in rxmask_list], dim=0)
    else:
        rx_mask = None

    return {
        "csi_data": csi_padded,
        "keypoints": kp_padded,
        "keypoints_conf": conf_padded,
        "mask": mask,
        "tx_coords": tx,
        "rx_coords": rx,
        "rx_mask": rx_mask,
    }


@dataclass
class Metrics:
    mpjpe_m: float
    pck_20: float
    pck_50: float


def _compute_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> Metrics:
    err = torch.norm(pred - gt, dim=-1)  # [B,T,K]
    denom = valid.sum().clamp(min=1)
    mpjpe = (err * valid).sum() / denom
    pck20 = (((err < 0.02) & valid).sum() / denom).to(torch.float32)
    pck50 = (((err < 0.05) & valid).sum() / denom).to(torch.float32)
    return Metrics(mpjpe_m=float(mpjpe.item()), pck_20=float(pck20.item()), pck_50=float(pck50.item()))


def eval_loop(model: torch.nn.Module, loader: DataLoader, device: torch.device, *, amp: bool) -> Metrics:
    model.eval()
    mpjpe_sum = 0.0
    p20_sum = 0.0
    p50_sum = 0.0
    batches = 0

    autocast_ctx = torch.cuda.amp.autocast(enabled=True) if (amp and device.type == "cuda") else contextlib.nullcontext()
    with torch.no_grad():
        for batch in loader:
            csi = batch["csi_data"].to(device, non_blocking=True)
            gt = batch["keypoints"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            conf = batch.get("keypoints_conf")
            tx = batch.get("tx_coords")
            rx = batch.get("rx_coords")
            rx_mask = batch.get("rx_mask")

            kp_valid = torch.ones(gt.shape[:-1], device=device, dtype=torch.bool)
            if conf is not None:
                kp_valid = conf.to(device, non_blocking=True) > 0.0

            rel_rx = None
            if tx is not None and rx is not None:
                tx = tx.to(device, non_blocking=True)
                rx = rx.to(device, non_blocking=True)
                rel_rx = rx - tx.unsqueeze(1)
            if rx_mask is not None:
                rx_mask = rx_mask.to(device, non_blocking=True)

            with autocast_ctx:
                if rel_rx is not None:
                    pred = model(csi, mask=mask, rel_rx_coords=rel_rx, rx_mask=rx_mask)
                else:
                    pred = model(csi, mask=mask, rx_mask=rx_mask)

            valid = mask.unsqueeze(-1) & kp_valid
            m = _compute_metrics(pred, gt, valid)
            mpjpe_sum += m.mpjpe_m
            p20_sum += m.pck_20
            p50_sum += m.pck_50
            batches += 1

    if batches == 0:
        return Metrics(mpjpe_m=float("inf"), pck_20=0.0, pck_50=0.0)
    return Metrics(mpjpe_m=mpjpe_sum / batches, pck_20=p20_sum / batches, pck_50=p50_sum / batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PerceptAlign with YAML configs + checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg_path = os.path.abspath(args.config)
    cfg_dir = os.path.dirname(cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    _seed_all(int(cfg["experiment"]["seed"]))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    pre_root = _resolve_path(cfg["paths"]["preprocessed_root"], cfg_dir)
    manifest_path = _resolve_path(cfg["paths"]["manifest"], cfg_dir)
    all_items = _load_manifest(manifest_path)
    train_pool, test_items = _select_by_protocol(all_items, cfg)
    train_items, val_items = _split_train_val(train_pool, val_ratio=float(cfg["train"]["val_ratio"]), seed=int(cfg["experiment"]["seed"]))

    if args.split == "train":
        items = train_items
    elif args.split == "val":
        items = val_items
    else:
        items = test_items

    pt_paths = []
    for it in items:
        rel = it.get("pt_relpath")
        if not rel:
            continue
        p = os.path.join(pre_root, rel)
        if os.path.exists(p):
            pt_paths.append(p)

    if not pt_paths:
        raise RuntimeError(f"No samples found for split={args.split}. Check manifest/protocol filters.")

    ds = ManifestPTDataset(pt_paths, max_seq_len=int(cfg["model"]["max_seq_len"]))
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_sequences,
    )

    rel_rx_default = torch.zeros((3, 3), dtype=torch.float32)
    model = posenet(
        num_keypoints=int(cfg["model"]["num_keypoints"]),
        rel_rx_coords=rel_rx_default,
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        pos_enc_depth=int(cfg["model"]["pos_enc_depth"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)

    m = eval_loop(model, dl, device, amp=bool(args.amp))
    print(f"[{args.split}] MPJPE={m.mpjpe_m*1000:.2f} mm | PCK@20={m.pck_20*100:.2f}% | PCK@50={m.pck_50*100:.2f}%")


if __name__ == "__main__":
    main()



