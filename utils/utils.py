from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import librosa

from . import config

def setup_matplotlib_font():
    """让 matplotlib 能正常显示中文（类别名/标题含中文）。

    按平台挑常见中文字体；找不到就用 minus 符号兜底，避免报错。
    在每个绘图入口调用一次即可。
    """
    import matplotlib
    from matplotlib import font_manager
    # Windows / macOS / Linux 常见中文字体名
    candidates = ["Microsoft YaHei", "SimHei", "SimSun",
                  "PingFang SC", "Heiti SC", "Noto Sans CJK SC",
                  "WenQuanYi Zen Hei", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen]
        matplotlib.rcParams["axes.unicode_minus"] = False
    # 找不到不报错：最多图里中文显示成方框，不影响流程


# ----------------------------------------------------------------------
# 可复现性
# ----------------------------------------------------------------------
def set_seed(seed: int = config.SEED) -> None:
    """固定 random / numpy / torch 的随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------
# 类别表读写
# ----------------------------------------------------------------------
def get_classes(data_dir=None) -> List[str]:
    """扫描数据目录下的一级子目录作为类别列表（按字母序稳定排序）。

    默认合并扫描所有数据源目录（public + self）。若 data_dir 指定则只扫该目录。
    若没有任何数据，则回退到 config.DEFAULT_CLASSES。
    """
    dirs = [Path(data_dir)] if data_dir else [Path(d) for d in config.DATA_DIRS]
    names = set()
    for d in dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_dir() and not p.name.startswith(".") and len(list(p.glob("*.wav"))) > 0:
                names.add(p.name)
    if names:
        return sorted(names)
    return list(config.DEFAULT_CLASSES)


def save_classes(classes: List[str]) -> None:
    """保存类别顺序（训练时调用，预测时必须用同一顺序）。"""
    config.CLASSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CLASSES_FILE, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)


def load_classes() -> List[str]:
    """读取类别顺序；文件不存在则扫描 dataset/。"""
    if config.CLASSES_FILE.exists():
        with open(config.CLASSES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return get_classes()


# ----------------------------------------------------------------------
# 音频读取 / 预处理（对应规划第 4 步）
# ----------------------------------------------------------------------
def load_audio(path: str | Path,
               sr: int = config.SR,
               fixed_sec: float = config.FIXED_SEC) -> np.ndarray:
    """读取音频并做统一预处理：重采样 → 单声道 → 固定时长。

    短的右侧补零，长的截断；返回 1D float32 数组。
    （滑窗切多段的逻辑在 extract.py 里实现，这里只做“单段对齐”。）
    """
    target_len = int(sr * fixed_sec)
    # librosa.load 自带 resample 和 mono，会自动转单声道
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    y = y.astype(np.float32)

    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    return y


def audio_files_in(data_dir=None):
    """遍历 <数据目录>/<类别>/*.wav，返回 [(文件路径, 类别名), ...]。

    默认合并扫描 config.DATA_DIRS（public + self）。可指定单个 data_dir。
    """
    dirs = [Path(data_dir)] if data_dir else [Path(d) for d in config.DATA_DIRS]
    pairs = []
    for d in dirs:
        if not d.exists():
            continue
        for class_dir in sorted(d.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue
            for wav in sorted(class_dir.glob("*.wav")):
                pairs.append((wav, class_dir.name))
    return pairs
