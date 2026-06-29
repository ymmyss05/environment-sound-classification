from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from utils import config
from utils.utils import get_classes

# 支持的音频扩展名
AUDIO_EXTS = {".m4a", ".mp3", ".aac", ".wma", ".amr", ".ogg", ".wav", ".3gp"}


def get_ffmpeg() -> str | None:
    """获取 ffmpeg 可执行文件路径。"""
    # 优先用 imageio_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    # 其次用系统的
    import shutil
    return shutil.which("ffmpeg")


def audio_to_wav(src: str | Path, dst: str | Path, ffmpeg: str | None) -> bool:
    """用 ffmpeg 把任意音频转成 wav（16k 单声道）。返回是否成功。"""
    src, dst = str(src), str(dst)
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-i", src, "-ar", str(config.SR),
               "-ac", "1", "-vn", dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0
    # 没有 ffmpeg，尝试用 librosa（仅对 wav 有效）
    try:
        y, _ = librosa.load(src, sr=config.SR, mono=True)
        sf.write(dst, y, config.SR)
        return True
    except Exception:
        return False


def standardize_length(wav_path: str | Path, out_dir: Path, idx: int,
                       sr: int = config.SR, fixed_sec: float = config.FIXED_SEC,
                       hop_sec: float = 1.0) -> int:
    """把已转好的 wav 统一时长为 2 秒：短的补零，长的滑窗切多段。

    返回生成的段数。同时做音量峰值归一化。
    """
    target = int(sr * fixed_sec)
    hop = int(sr * hop_sec)
    try:
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    except Exception as e:
        print(f"  [跳过] 读取失败 {wav_path}: {e}")
        return 0
    y = y.astype(np.float32)

    # 音量峰值归一化（防止不同录音音量差异大）
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak * 0.9

    segments = []
    if len(y) < target:
        segments.append(np.pad(y, (0, target - len(y))))
    else:
        starts = list(range(0, max(1, len(y) - target + 1), hop))
        if not starts:
            starts = [0]
        for s in starts:
            seg = y[s:s + target]
            if len(seg) < target:
                seg = np.pad(seg, (0, target - len(seg)))
            segments.append(seg)

    for i, seg in enumerate(segments):
        name = f"{idx:04d}{'' if i == 0 else f'_{i}'}.wav"
        sf.write(str(out_dir / name), seg, sr, subtype="PCM_16")
    return len(segments)


def import_one(src: str | Path, label: str, ffmpeg: str | None,
               out_root: Path = config.DATA_SELF_DIR) -> int:
    """导入单个音频文件到 dataset_self/<label>/ 下。返回生成段数。"""
    src = Path(src)
    if not src.exists():
        print(f"[!] 文件不存在: {src}")
        return 0

    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(out_dir.glob("*.wav"))) + 1

    # 临时 wav
    tmp_wav = out_dir / f"_tmp_{idx:04d}.wav"
    print(f"  转换 {src.name} ({src.suffix}) → wav ...")
    ok = audio_to_wav(src, tmp_wav, ffmpeg)
    if not ok:
        print(f"  [失败] 无法转换 {src}（需要 ffmpeg 才能读 {src.suffix}）")
        return 0

    # 标准化时长
    n = standardize_length(tmp_wav, out_dir, idx)
    # 删除临时文件
    try:
        tmp_wav.unlink()
    except Exception:
        pass
    print(f"  ✓ {src.name} → {label}/{idx:04d}*.wav，生成 {n} 段")
    return n


def main():
    parser = argparse.ArgumentParser(description="导入手机录音到项目标准格式")
    parser.add_argument("input", type=str,
                        help="音频文件路径，或文件夹(配合--batch)，或通配符")
    parser.add_argument("--label", type=str, default=None,
                        help="类别名（如 雨声）。批量模式且不指定时用子文件夹名")
    parser.add_argument("--batch", action="store_true",
                        help="批量导入文件夹下所有音频")
    parser.add_argument("--out", choices=["self", "public"], default="self",
                        help="存到自录目录(默认)或公开目录")
    args = parser.parse_args()

    ffmpeg = get_ffmpeg()
    if ffmpeg:
        print(f"[*] ffmpeg: {ffmpeg}")
    else:
        print("[!] 未找到 ffmpeg，只能导入 wav 格式")
        print("    安装：pip install imageio-ffmpeg")

    out_root = config.DATA_SELF_DIR if args.out == "self" else config.DATA_PUBLIC_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    # 收集要导入的文件
    if args.batch:
        in_path = Path(args.input)
        if in_path.is_dir():
            # 文件夹模式：如果没指定 label，按子文件夹名作为类别
            if args.label:
                files = []
                for ext in AUDIO_EXTS:
                    files += list(in_path.rglob(f"*{ext}"))
                pairs = [(f, args.label) for f in files]
            else:
                pairs = []
                for sub in sorted(in_path.iterdir()):
                    if sub.is_dir():
                        for ext in AUDIO_EXTS:
                            for f in sub.glob(f"*{ext}"):
                                pairs.append((f, sub.name))
        else:
            # 通配符模式
            matched = glob.glob(args.input)
            if not args.label:
                print("[!] 通配符模式必须指定 --label")
                return
            pairs = [(Path(f), args.label) for f in matched]
    else:
        # 单文件
        if not args.label:
            print("[!] 单文件导入必须指定 --label")
            return
        pairs = [(Path(args.input), args.label)]

    if not pairs:
        print("[!] 没有找到可导入的音频文件")
        print(f"    支持的格式: {', '.join(AUDIO_EXTS)}")
        return

    print(f"[*] 共 {len(pairs)} 个文件待导入 → {out_root.name}/")
    # 按类别统计
    from collections import Counter
    cnt = Counter(l for _, l in pairs)
    for label, n in cnt.items():
        print(f"    {label}: {n} 个源文件")

    total = 0
    for src, label in tqdm(pairs, desc="导入"):
        total += import_one(src, label, ffmpeg, out_root)

    print(f"\n[√] 导入完成：共生成 {total} 段标准 wav → {out_root}")
    print("    下一步：python extract.py")


if __name__ == "__main__":
    main()
