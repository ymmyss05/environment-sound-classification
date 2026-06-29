"""第 5 步：特征提取 —— wav → Mel 频谱图，缓存为 .npy。

流程：wav → 重采样/单声道/定长 → melspectrogram → power_to_db → 0~1 归一化。
输出形状 (N_MELS, N_FRAMES) 的 float32。训练时再复制成 3 通道。

用法：
    python extract.py            # 处理 dataset/ 下所有 wav
    python extract.py --force    # 强制重新生成（忽略已有缓存）
"""
from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm

from utils import config
from utils.utils import audio_files_in, get_classes


def wav_to_mel(path: str | Path,
               sr: int = config.SR,
               n_mels: int = config.N_MELS,
               n_fft: int = config.N_FFT,
               hop_length: int = config.HOP_LENGTH,
               fixed_sec: float = config.FIXED_SEC) -> np.ndarray:
    """一段音频 → Mel(dB) 频谱图，已归一化到 [0, 1]。

    返回形状 (n_mels, n_frames)。
    """
    # 读取并预处理（含重采样、单声道、定长补零/截断）
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    y = y.astype(np.float32)

    target_len = int(sr * fixed_sec)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]

    # Mel 频谱 → 转分贝
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=n_fft, n_mels=n_mels, power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)            # 大致 [-80, 0]

    # 归一化到 [0, 1]：减最小、除以范围，避免除零
    mel_min, mel_max = mel_db.min(), mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-6)
    return mel_norm.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="提取 Mel 频谱图并缓存为 .npy")
    parser.add_argument("--force", action="store_true", help="强制重新生成缓存")
    args = parser.parse_args()

    pairs = audio_files_in()
    if not pairs:
        print(f"[!] 没有找到任何 wav。请先准备数据：")
        print(f"    公开数据：python prepare_data.py --esc50 <目录>")
        print(f"    自录数据：python record.py <类别> --count 20")
        print(f"    数据目录：{config.DATA_PUBLIC_DIR} 和 {config.DATA_SELF_DIR}")
        classes = get_classes()
        print(f"    当前默认类别：{classes}")
        return

    classes = get_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[*] 发现 {len(pairs)} 个 wav，{len(classes)} 个类别：{classes}")

    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    skipped, done = 0, 0
    for wav, cls in tqdm(pairs, desc="提取 Mel"):
        # 缓存路径：features/<源目录名>/<类别>/<文件名>.npy
        # 不同数据源（public/self）分目录存放，避免同名冲突
        source_name = wav.parent.parent.name   # dataset_public 或 dataset_self
        rel = Path(source_name) / cls / wav.stem
        out = config.FEATURE_DIR / f"{rel}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not args.force:
            skipped += 1
            continue
        mel = wav_to_mel(wav)
        np.save(out, mel, allow_pickle=False)   # 只存 mel，标签从路径推断
        done += 1

    print(f"[√] 完成：新生成 {done} 个，跳过已有 {skipped} 个。")
    print(f"    缓存目录：{config.FEATURE_DIR}")
    print(f"    示例文件形状：单通道 Mel (n_mels={config.N_MELS}, "
          f"n_frames={config.N_FRAMES})")


if __name__ == "__main__":
    main()
