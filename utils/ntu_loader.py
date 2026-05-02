# utils/ntu_loader.py — NTU RGB+D Dataset Loader for Ghost-Vision
"""
Loads the OpenMMLab ntu60_3d.pkl file and maps NTU's 60 action classes
down to Ghost-Vision's 5 classes:
    0 = Normal
    1 = Fall
    2 = Pre-Fall Risk
    3 = Motionless
    4 = Sit Down
"""
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch

# ── NTU action IDs that map to each Ghost-Vision class ──────
# NTU labels are 0-indexed (so A043 Fall = index 42)
NTU_TO_GHOSTVISION = {
    # Fall (label 1)
    42: 1,   # A043 — fall down
    # Pre-Fall Risk (label 2)
    39: 2,   # A040 — staggering
    # Motionless (label 3)
    40: 3,   # A041 — headache (person hunched, near-motionless)
    41: 3,   # A042 — chest pain (person hunched, near-motionless)
    # Sit Down (label 4)
    26: 4,   # A027 — sit down
    # Normal — common daily activities used as negative class
    0:  0,   # A001 — drink water
    1:  0,   # A002 — eat meal
    2:  0,   # A003 — brushing teeth
    7:  0,   # A008 — stand up
    8:  0,   # A009 — clapping
    9:  0,   # A010 — reading
    10: 0,   # A011 — writing
    11: 0,   # A012 — tear up paper
    22: 0,   # A023 — hand waving
    23: 0,   # A024 — kicking something
    27: 0,   # A028 — stand up (second variant)
}

# NTU skeleton has 25 joints — adjacency for Ghost-Vision graph builder
NTU_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),          # spine
    (20, 4), (4, 5), (5, 6), (6, 7),            # left arm
    (20, 8), (8, 9), (9, 10), (10, 11),         # right arm
    (0, 16), (16, 17), (17, 18), (18, 19),      # left leg
    (0, 12), (12, 13), (13, 14), (14, 15),      # right leg
    (2, 24), (24, 22), (22, 23),                # left hand
    (2, 21),                                     # right hand thumb
]

NUM_NTU_JOINTS = 25


class NTUDataset(Dataset):
    """
    PyTorch Dataset for NTU RGB+D skeleton data.
    Handles loading, filtering, normalization, and augmentation.
    """
    def __init__(self, pkl_path, split="xsub_train",
                 sequence_length=30, augment=False):
        self.sequence_length = sequence_length
        self.augment = augment

        print(f"📂 Loading NTU RGB+D from {pkl_path} ...")
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        split_ids = set(raw["split"][split])
        annotations = raw["annotations"]

        self.samples = []   # list of (keypoint_array, ghost_vision_label)
        skipped = 0

        for ann in annotations:
            if ann["frame_dir"] not in split_ids:
                continue
            ntu_label = ann["label"]
            if ntu_label not in NTU_TO_GHOSTVISION:
                skipped += 1
                continue

            gv_label = NTU_TO_GHOSTVISION[ntu_label]
            # keypoint shape: (persons, frames, joints, coords)
            # take first person only, drop confidence if present
            kp = ann["keypoint"][0]   # (frames, 25, 3)
            self.samples.append((kp, gv_label))

        # Count per class
        from collections import Counter
        label_counts = Counter(s[1] for s in self.samples)
        print(f"✅ Loaded {len(self.samples)} sequences from '{split}'")
        print(f"   Skipped {skipped} sequences (NTU classes not in mapping)")
        print(f"   Class distribution:")
        class_names = ["Normal", "Fall", "Pre-Fall Risk", "Motionless", "Sit Down"]
        for i, name in enumerate(class_names):
            print(f"     {i} {name}: {label_counts.get(i, 0)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kp, label = self.samples[idx]
        kp = self._pad_or_trim(kp)           # (seq_len, 25, 3)
        kp = self._normalize(kp)
        if self.augment:
            kp = self._augment(kp)
        # (seq_len, 25, 3) → (3, seq_len, 25) for ST-GCN
        tensor = torch.FloatTensor(kp.transpose(2, 0, 1))
        return tensor, label

    def _pad_or_trim(self, kp):
        T = kp.shape[0]
        if T == self.sequence_length:
            return kp
        elif T < self.sequence_length:
            pad = np.tile(kp[-1:], (self.sequence_length - T, 1, 1))
            return np.concatenate([kp, pad], axis=0)
        else:
            idx = np.linspace(0, T - 1, self.sequence_length, dtype=int)
            return kp[idx]

    def _normalize(self, kp):
        """Center on hip joint (joint 0 in NTU) and scale by torso height"""
        kp = kp.copy().astype(np.float32)
        center = kp[:, 0:1, :]              # hip center
        kp = kp - center
        torso = np.mean(np.linalg.norm(kp[:, 2, :] - kp[:, 0, :], axis=1))
        if torso > 1e-6:
            kp = kp / torso
        return kp

    def _augment(self, kp):
        kp = kp.copy()
        # Random rotation around Y axis (±15 degrees)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            kp = kp @ R.T
        # Random horizontal flip
        if np.random.random() > 0.5:
            kp[:, :, 0] *= -1
        # Random Gaussian noise
        if np.random.random() > 0.5:
            kp += np.random.normal(0, 0.01, kp.shape)
        # Random temporal crop and resize
        if np.random.random() > 0.5:
            T = kp.shape[0]
            start = np.random.randint(0, T // 4)
            end = np.random.randint(3 * T // 4, T)
            cropped = kp[start:end]
            idx = np.linspace(0, len(cropped) - 1, T, dtype=int)
            kp = cropped[idx]
        return kp


def build_ntu_adjacency():
    """Build normalized adjacency matrix for 25-joint NTU skeleton"""
    import torch
    A = np.zeros((NUM_NTU_JOINTS, NUM_NTU_JOINTS))
    for i, j in NTU_CONNECTIONS:
        A[i][j] = 1
        A[j][i] = 1
    A = A + np.eye(NUM_NTU_JOINTS)
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.FloatTensor(A_norm)


if __name__ == "__main__":
    ds = NTUDataset("ntu60_3d.pkl", split="xsub_train", augment=True)
    x, y = ds[0]
    print(f"\nSample tensor shape : {x.shape}")   # (3, 30, 25)
    print(f"Sample label        : {y}")

    A = build_ntu_adjacency()
    print(f"Adjacency matrix    : {A.shape}")      # (25, 25)