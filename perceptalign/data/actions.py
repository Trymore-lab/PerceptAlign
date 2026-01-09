import glob
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from perceptalign.config import RX_COORDINATES, TX_COORDINATES


class VideoFrameDataset(Dataset):
    def __init__(self, data_dir: str, max_seq_len: int, *, is_train: bool = False):
        self.data_dir = data_dir
        self.max_seq_len = int(max_seq_len)
        self.is_train = bool(is_train)
        self.file_paths = []

        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
            paths_list = [self.file_paths]
            dist.broadcast_object_list(paths_list, src=0)
            self.file_paths = paths_list[0]
        else:
            self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

        if not self.file_paths:
            raise RuntimeError(f"No .pt files found in: {data_dir}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[index]
        data = torch.load(file_path, map_location="cpu")

        csi_data = data["csi_data"]
        keypoints = data["keypoints"]

        tx_coords = data.get("tx_coords", TX_COORDINATES)
        rx_coords = data.get("rx_coords", RX_COORDINATES)
        rx_mask = data.get("rx_mask", torch.ones((csi_data.shape[1],), dtype=torch.bool))

        if not isinstance(tx_coords, torch.Tensor):
            tx_coords = torch.tensor(tx_coords, dtype=torch.float32)
        if not isinstance(rx_coords, torch.Tensor):
            rx_coords = torch.tensor(rx_coords, dtype=torch.float32)
        if not isinstance(rx_mask, torch.Tensor):
            rx_mask = torch.tensor(rx_mask, dtype=torch.bool)

        tx_coords = tx_coords.to(torch.float32)
        rx_coords = rx_coords.to(torch.float32)
        rx_mask = rx_mask.to(torch.bool)

        if csi_data.shape[0] > self.max_seq_len:
            csi_data = csi_data[: self.max_seq_len]
            keypoints = keypoints[: self.max_seq_len]

        return csi_data, keypoints, tx_coords, rx_coords, rx_mask


def collate_fn_sequences(batch):
    csi_list, kp_list, tx_list, rx_list, rxmask_list = zip(*batch)
    seq_lengths = torch.tensor([len(seq) for seq in csi_list], dtype=torch.long)
    csi_padded = pad_sequence(csi_list, batch_first=True, padding_value=0.0)
    kp_padded = pad_sequence(kp_list, batch_first=True, padding_value=0.0)
    mask = torch.arange(int(seq_lengths.max()))[None, :] < seq_lengths[:, None]
    tx_coords = torch.stack(tx_list, dim=0)
    rx_coords = torch.stack(rx_list, dim=0)
    rx_mask = torch.stack(rxmask_list, dim=0)
    return {
        "csi_data": csi_padded,
        "keypoints": kp_padded,
        "mask": mask,
        "tx_coords": tx_coords,
        "rx_coords": rx_coords,
        "rx_mask": rx_mask,
    }


