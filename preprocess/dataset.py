"""第 6 步：数据集划分 + 数据增强。

- AudioDataset：读取 features/*.npy（单通道 Mel），按需做增强（SpecAugment + 谱图噪声）
- build_split：按类别分层划分 训练/验证/测试（默认 8:1:1）

增强只对“训练”打开，验证/测试关闭。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import config
from utils.utils import get_classes


# ----------------------------------------------------------------------
# SpecAugment
# ----------------------------------------------------------------------
def spec_augment(mel: np.ndarray,
                 time_mask_num: int = 2,
                 freq_mask_num: int = 2,
                 time_mask_ratio: float = 0.1,
                 freq_mask_ratio: float = 0.15,
                 noise_std: float = 0.01,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """在 Mel 谱图上随机遮掩时间/频率块，并叠加少量高斯噪声。

    mel: 形状 (n_mels, n_frames)，值域约 [0,1]。返回同形状数组。
    对应规划“SpecAugment：随机遮掩时间块/频率块”。
    """
    if rng is None:
        rng = np.random.default_rng()
    mel = mel.copy()
    n_mels, n_frames = mel.shape

    # 频率遮掩（横向条带）
    for _ in range(freq_mask_num):
        f = int(rng.integers(0, max(1, int(n_mels * freq_mask_ratio)) + 1))
        if f > 0 and n_mels - f > 0:
            f0 = int(rng.integers(0, n_mels - f))
            mel[f0:f0 + f, :] = mel.mean()

    # 时间遮掩（纵向条带）
    for _ in range(time_mask_num):
        t = int(rng.integers(0, max(1, int(n_frames * time_mask_ratio)) + 1))
        if t > 0 and n_frames - t > 0:
            t0 = int(rng.integers(0, n_frames - t))
            mel[:, t0:t0 + t] = mel.mean()

    # 高斯噪声（小数据很关键，对应规划“加高斯噪声”）
    if noise_std > 0:
        mel = mel + rng.normal(0, noise_std, mel.shape).astype(np.float32)

    return np.clip(mel, 0.0, 1.0)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class AudioDataset(Dataset):
    """单通道 Mel 谱数据集。

    - 复制成 3 通道，按 ImageNet 均值方差标准化 → 直接喂 ResNet。
    - augment=True 时在 __getitem__ 里做 SpecAugment（训练用）。
    """

    def __init__(self,
                 items: List[Tuple[Path, int]],
                 augment: bool = False,
                 seed: int = config.SEED):
        self.items = items
        self.augment = augment
        self._mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
        # 每个样本一个独立 RNG，增强可复现且 worker 安全
        self._base_rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: Path) -> np.ndarray:
        # allow_pickle=False 缓存的纯数组
        mel = np.load(path)
        # 保证形状一致：宽不足补零，超长截断
        n_mels, n_frames = config.N_MELS, config.N_FRAMES
        if mel.shape[0] != n_mels:
            mel = mel[:n_mels] if mel.shape[0] > n_mels else np.pad(
                mel, ((0, n_mels - mel.shape[0]), (0, 0)))
        if mel.shape[1] != n_frames:
            mel = mel[:, :n_frames] if mel.shape[1] > n_frames else np.pad(
                mel, ((0, 0), (0, n_frames - mel.shape[1])))
        return mel.astype(np.float32)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        mel = self._load(path)
        if self.augment:
            mel = spec_augment(mel, rng=np.random.default_rng(self._base_rng.integers(1 << 31)))

        # 3 通道复制 + ImageNet 标准化
        x = torch.from_numpy(mel).unsqueeze(0).repeat(3, 1, 1)   # (3, n_mels, n_frames)
        x = (x - self._mean) / self._std
        return x, int(label)


# ----------------------------------------------------------------------
# 划分
# ----------------------------------------------------------------------
def build_split(data_dir: Path = config.FEATURE_DIR,
                ratio: Tuple[float, float, float] = config.SPLIT_RATIO,
                seed: int = config.SEED,
                data_filter: str | None = None):
    """分层划分 训练/验证/测试。

    扫描 features/<源>/<类别>/*.npy，按类别各自 shuffle 后按比例切分，
    再合并，避免小类别全部落到同一集合。
    返回 (train_items, val_items, test_items, classes)。

    data_filter: None/'all'=全部(默认) / 'public'=仅公开 / 'self'=仅自录
    """
    data_dir = Path(data_dir)
    # 按数据源筛选
    if data_filter in (None, "all"):
        source_names = ["dataset_public", "dataset_self"]
    elif data_filter == "public":
        source_names = ["dataset_public"]
    elif data_filter == "self":
        source_names = ["dataset_self"]
    else:
        source_names = ["dataset_public", "dataset_self"]

    # 先用全部类别做候选（保证顺序稳定），再按实际有样本的过滤
    candidate_classes = get_classes()

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    classes = []   # 只保留当前 data_filter 下确有样本的类别
    # 先收集每个类别的文件，统一编号后再划分
    per_class_files = {}

    for cls in candidate_classes:
        # 合并符合筛选条件的数据源（features/<源>/<类别>/*.npy）下的该类样本
        files = []
        if data_dir.exists():
            for source_dir in sorted(data_dir.iterdir()):
                if not source_dir.is_dir() or source_dir.name.startswith("."):
                    continue
                # 按数据源筛选：只纳入要求的源目录
                if source_dir.name not in source_names:
                    continue
                cdir = source_dir / cls
                if cdir.exists():
                    files.extend(sorted(cdir.glob("*.npy")))
        if not files:
            continue
        classes.append(cls)
        rng.shuffle(files)
        per_class_files[cls] = files

    # 统一编号（0..N-1），保证 train/eval/predict 类别索引一致
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for cls in classes:
        files = [(f, class_to_idx[cls]) for f in per_class_files[cls]]
        n = len(files)
        # 保证小数据也能切出 val/test：n>=3 时各至少 1 个给 val/test
        if n >= 3:
            n_val = max(1, int(n * ratio[1]))
            n_test = max(1, int(n * ratio[2]))
            n_train = max(1, n - n_val - n_test)
            # 防止 train 被挤没
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test
            train += files[:n_train]
            val += files[n_train:n_train + n_val]
            test += files[n_train + n_val:n_train + n_val + n_test]
        else:
            # 极少样本全部进训练
            train += files

    print(f"[*] 划分结果：train={len(train)} val={len(val)} test={len(test)} "
          f"类别={len(classes)}")
    return train, val, test, classes


def make_loaders(seed: int = config.SEED, data_filter: str | None = None):
    """构造 train/val/test 三个 DataLoader。data_filter 见 build_split。"""
    train_items, val_items, test_items, classes = build_split(seed=seed,
                                                              data_filter=data_filter)
    train_ds = AudioDataset(train_items, augment=True, seed=seed)
    val_ds = AudioDataset(val_items, augment=False, seed=seed)
    test_ds = AudioDataset(test_items, augment=False, seed=seed)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, generator=g, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS)
    return train_loader, val_loader, test_loader, classes


if __name__ == "__main__":
    # 快速自检：看看划分数量
    tr, va, te, cls = build_split()
    print("类别：", cls)
