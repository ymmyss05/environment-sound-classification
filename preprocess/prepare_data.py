"""把公开数据集（ESC-50 / UrbanSound8K）整理成项目标准格式。

输出：dataset_public/<中文类别>/NNNN.wav
  - 统一 16kHz 单声道
  - 不足 2 秒补零、超长按 1 秒步长滑窗切多段（顺便扩数据）

用法：
    # 1) 先把下载好的数据集放到对应位置（见下方说明），然后运行：
    python prepare_data.py --esc50 "ESC-50"
    python prepare_data.py --urban "UrbanSound8K"
    python prepare_data.py --esc50 "ESC-50" --urban "UrbanSound8K"   # 两个一起
    python prepare_data.py --clean                                   # 清空 dataset_public/

数据集下载位置说明：
    ESC-50:  https://github.com/karolpiczak/ESC-50
        下载后目录形如：ESC-50/audio/*.wav  ESC-50/meta/esc50.csv
    UrbanSound8K:  https://www.kaggle.com/datasets/chrisfilo/urbansound8k
        下载后目录形如：UrbanSound8K/audio/fold*/*.wav  UrbanSound8K/metadata/UrbanSound8K.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from utils import config
from utils.utils import get_classes


# ----------------------------------------------------------------------
# 音频标准化：重采样 16k、单声道、定长 2s，超长滑窗切多段
# ----------------------------------------------------------------------
def standardize_audio(src: str | Path,
                      dst_dir: Path,
                      idx: int,
                      sr: int = config.SR,
                      fixed_sec: float = config.FIXED_SEC,
                      hop_sec: float = 1.0) -> int:
    """读取 src，标准化后存到 dst_dir/<idx:04d>.wav。超长按 hop_sec 步长切多段。

    返回：实际生成的段数（0 表示该文件不可用）。
    """
    target = int(sr * fixed_sec)
    hop = int(sr * hop_sec)
    try:
        y, _ = librosa.load(str(src), sr=sr, mono=True)
    except Exception as e:
        print(f"  [跳过] 读取失败 {src}: {e}")
        return 0
    y = y.astype(np.float32)

    segments = []
    if len(y) < target:
        segments.append(np.pad(y, (0, target - len(y))))
    else:
        # 滑窗切多段，最后不足 target 用末尾补齐
        starts = list(range(0, max(1, len(y) - target + 1), hop))
        if not starts:
            starts = [0]
        for s in starts:
            seg = y[s:s + target]
            if len(seg) < target:
                seg = np.pad(seg, (0, target - len(seg)))
            segments.append(seg)

    for i, seg in enumerate(segments):
        out = dst_dir / f"{idx:04d}{'' if i == 0 else f'_{i}'}.wav"
        sf.write(str(out), seg, sr, subtype="PCM_16")
    return len(segments)


# ----------------------------------------------------------------------
# ESC-50
# ----------------------------------------------------------------------
def process_esc50(root: str | Path):
    """处理 ESC-50。依赖 meta/esc50.csv 里的 filename,category 两列。"""
    root = Path(root)
    csv_path = root / "meta" / "esc50.csv"
    audio_dir = root / "audio"
    if not csv_path.exists():
        print(f"[!] 找不到 {csv_path}，请确认 ESC-50 目录结构正确")
        print("    期望结构：ESC-50/meta/esc50.csv  ESC-50/audio/*.wav")
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[*] ESC-50：共 {len(rows)} 条记录")

    # 统计每类产出数量
    counter = {}
    out_root = config.DATA_PUBLIC_DIR
    for row in tqdm(rows, desc="ESC-50"):
        category = row.get("category", "").strip()
        label = config.ESC50_TO_LABEL.get(category)
        if not label:
            continue   # 不在映射表里的类别跳过
        src = audio_dir / row["filename"]
        if not src.exists():
            continue
        dst_dir = out_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        counter.setdefault(label, 0)
        idx = counter[label] + 1
        n = standardize_audio(src, dst_dir, idx)
        counter[label] += n

    _print_summary("ESC-50", counter)


# ----------------------------------------------------------------------
# UrbanSound8K
# ----------------------------------------------------------------------
def process_urban(root: str | Path):
    """处理 UrbanSound8K。依赖 metadata/UrbanSound8K.csv 里的 slice_file_name,class。"""
    root = Path(root)
    csv_path = root / "metadata" / "UrbanSound8K.csv"
    audio_root = root / "audio"
    if not csv_path.exists():
        print(f"[!] 找不到 {csv_path}，请确认 UrbanSound8K 目录结构正确")
        print("    期望结构：UrbanSound8K/metadata/UrbanSound8K.csv  UrbanSound8K/audio/fold*/*.wav")
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[*] UrbanSound8K：共 {len(rows)} 条记录")

    counter = {}
    out_root = config.DATA_PUBLIC_DIR
    for row in tqdm(rows, desc="UrbanSound8K"):
        cls = row.get("class", "").strip()
        label = config.URBAN_TO_LABEL.get(cls)
        if not label:
            continue
        fold = row.get("fold", "fold1")
        src = audio_root / f"fold{fold}" / row["slice_file_name"]
        if not src.exists():
            continue
        dst_dir = out_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        counter.setdefault(label, 0)
        idx = counter[label] + 1
        n = standardize_audio(src, dst_dir, idx)
        counter[label] += n

    _print_summary("UrbanSound8K", counter)


def _print_summary(name: str, counter: dict):
    print(f"\n[√] {name} 整理完成：")
    if not counter:
        print("    （没有匹配到任何类别，请检查 config.py 里的映射表）")
        return
    total = 0
    for label in sorted(counter):
        print(f"    {label:>8}: {counter[label]} 段")
        total += counter[label]
    print(f"    {'合计':>8}: {total} 段")


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="整理公开数据集为标准格式")
    parser.add_argument("--esc50", type=str, default=None,
                        help="ESC-50 根目录（含 audio/ 和 meta/esc50.csv）")
    parser.add_argument("--urban", type=str, default=None,
                        help="UrbanSound8K 根目录（含 audio/fold*/ 和 metadata/）")
    parser.add_argument("--clean", action="store_true",
                        help="清空 dataset_public/（重新整理前用）")
    parser.add_argument("--print-map", action="store_true",
                        help="打印当前类别映射表后退出")
    args = parser.parse_args()

    if args.print_map:
        print("ESC-50 → 中文类别映射：")
        for k, v in config.ESC50_TO_LABEL.items():
            print(f"  {k:>20} → {v}")
        print("\nUrbanSound8K → 中文类别映射：")
        for k, v in config.URBAN_TO_LABEL.items():
            print(f"  {k:>20} → {v}")
        return

    if args.clean:
        if config.DATA_PUBLIC_DIR.exists():
            shutil.rmtree(config.DATA_PUBLIC_DIR)
        config.DATA_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[√] 已清空 {config.DATA_PUBLIC_DIR}")
        return

    if not args.esc50 and not args.urban:
        parser.print_help()
        print("\n[!] 请至少指定 --esc50 或 --urban 之一")
        print("    示例：python prepare_data.py --esc50 \"ESC-50\"")
        return

    config.DATA_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    if args.esc50:
        process_esc50(args.esc50)
    if args.urban:
        process_urban(args.urban)

    print(f"\n[√] 数据已写入 {config.DATA_PUBLIC_DIR}")
    print("    下一步：python extract.py   # 提取 Mel 频谱")


if __name__ == "__main__":
    main()
