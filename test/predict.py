from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils import config
from preprocess.extract import wav_to_mel
from model.resnet18 import build_model, device_select
from test.evaluate import load_checkpoint


def preprocess(path: str | Path) -> torch.Tensor:
    """wav → 3 通道标准化 Mel 张量 (1, 3, n_mels, n_frames)。"""
    mel = wav_to_mel(path)   # (n_mels, n_frames) 已归一化
    # 保证宽度
    if mel.shape[1] < config.N_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, config.N_FRAMES - mel.shape[1])))
    else:
        mel = mel[:, :config.N_FRAMES]
    x = torch.from_numpy(mel).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).float()
    mean = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def predict(path: str | Path, model, classes, device, topk: int = 1):
    x = preprocess(path).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    k = min(topk, len(classes))
    topv, topi = torch.topk(probs, k)
    return [(classes[int(i)], float(v)) for v, i in zip(topv, topi)], probs.cpu().numpy()


def save_spectrogram(path: str | Path, out: Path, title: str = ""):
    """保存 Mel 频谱图（论文“Mel 频谱图示例”用）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.utils import setup_matplotlib_font
    setup_matplotlib_font()

    mel = wav_to_mel(path)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("时间帧"); ax.set_ylabel("Mel 频带")
    ax.set_title(title or Path(path).name)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="对单段音频做预测并可视化")
    parser.add_argument("audio", type=str, help="wav 文件路径")
    parser.add_argument("--model", type=str, default="best.pth")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--save-spec", action="store_true", help="同时保存 Mel 频谱图")
    args = parser.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"[!] 文件不存在：{audio}")
        return
    device = device_select()
    model_path = config.MODEL_DIR / args.model
    if not model_path.exists():
        print(f"[!] 找不到权重 {model_path}，请先训练。")
        return
    ckpt, classes = load_checkpoint(model_path)
    model = build_model(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    results, _ = predict(audio, model, classes, device, topk=args.topk)
    print(f"\n音频：{audio.name}")
    print("Top-K 预测：")
    for name, p in results:
        print(f"  {name:>8}  {p*100:6.2f}%")

    if args.save_spec:
        out = config.OUTPUT_DIR / f"spec_{audio.stem}.png"
        save_spectrogram(audio, out, title=f"{audio.name} → {results[0][0]}")
        print(f"[√] 频谱图已保存：{out}")


if __name__ == "__main__":
    main()
