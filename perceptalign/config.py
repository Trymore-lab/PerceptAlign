import os
import torch


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and str(v).strip() != "" else default


DATA_ROOT = _env("PERCEPTALIGN_DATA_ROOT", "data/raw")
PREPROCESSED_ACTIONS_ROOT = _env("PERCEPTALIGN_PREPROCESSED_ACTIONS_ROOT", "data/preprocessed_actions")
WEIGHTS_DIR = _env("PERCEPTALIGN_WEIGHTS_DIR", "weights")

NUM_KEYPOINTS = int(_env("PERCEPTALIGN_NUM_KEYPOINTS", "25"))

TX_COORDINATES = torch.tensor([0.62851124, -1.08806139, 0.01993734], dtype=torch.float32)
RX_COORDINATES = torch.tensor([
    [0.41202504, 2.97651455, 0.02596234],   # Rx1
    [1.30517124, 1.76991134, 0.000208976278],# Rx2
    [-3.40882228, -1.20002765, -0.07493947], # Rx3
], dtype=torch.float32)

SCENE_2_BENCHMARK_MATRIX = torch.tensor([
    [1.941177424045289479e-01, 1.167825188066586029e-02, 9.809087218068667235e-01, 8.347051851260479660e-01],
    [3.995411874468585145e-03, 9.999114308150333086e-01, -1.269516480343271803e-02, 1.093013572122540911e+00],
    [-9.809701008530902211e-01, 6.383491085972775592e-03, 1.940538901280592365e-01, 4.013500324675971509e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
], dtype=torch.float32)

DFS_N_FFT = int(_env("PERCEPTALIGN_DFS_N_FFT", "64"))
DFS_WIN_LENGTH = int(_env("PERCEPTALIGN_DFS_WIN_LENGTH", "64"))
FINAL_IMAGE_SIZE = (224, 224)

MODEL_CONFIG = {
    "num_keypoints": NUM_KEYPOINTS,
    "num_layers": int(_env("PERCEPTALIGN_NUM_LAYERS", "4")),
    "num_heads": int(_env("PERCEPTALIGN_NUM_HEADS", "8")),
    "pos_enc_depth": int(_env("PERCEPTALIGN_POS_ENC_DEPTH", "10")),
    "max_seq_len": int(_env("PERCEPTALIGN_MAX_SEQ_LEN", "120")),
}

MODEL_SAVE_FILENAME = _env(
    "PERCEPTALIGN_MODEL_SAVE_FILENAME",
    f"perceptalign_best_mpjpe_kp{NUM_KEYPOINTS}.pth",
)
MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, MODEL_SAVE_FILENAME)

TRAIN_CONFIG = {
    "batch_size": int(_env("PERCEPTALIGN_BATCH_SIZE", "1")),
    "gradient_accumulation_steps": int(_env("PERCEPTALIGN_GRAD_ACCUM", "16")),
    "num_epochs": int(_env("PERCEPTALIGN_NUM_EPOCHS", "200")),
    "initial_learning_rate": float(_env("PERCEPTALIGN_LR", "1e-4")),
    "patience": int(_env("PERCEPTALIGN_LR_PATIENCE", "10")),
    "factor": float(_env("PERCEPTALIGN_LR_FACTOR", "0.5")),
    "min_lr": float(_env("PERCEPTALIGN_MIN_LR", "1e-7")),
    "train_seq_len": int(_env("PERCEPTALIGN_TRAIN_SEQ_LEN", "96")),
    "receiver_dropout_prob": float(_env("PERCEPTALIGN_RX_DROPOUT", "0.10")),
    "canonicalize_layout": _env("PERCEPTALIGN_CANONICALIZE_LAYOUT", "0") == "1",
    "yaw_aug_deg": float(_env("PERCEPTALIGN_YAW_AUG_DEG", "180.0")),
    "yaw_aug_warmup_epochs": int(_env("PERCEPTALIGN_YAW_AUG_WARMUP_EPOCHS", "10")),
    "ignore_invalid_keypoints": _env("PERCEPTALIGN_IGNORE_INVALID_KP", "1") == "1",
    "invalid_kp_eps_m": float(_env("PERCEPTALIGN_INVALID_KP_EPS_M", "1e-4")),
}


