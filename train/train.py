"""第 8 步：训练 —— 早停 + 保存最佳验证模型 + loss/acc 曲线。

对应规划：
    优化器 AdamW(lr=1e-3) + CosineAnnealingLR + CrossEntropyLoss
    早停 patience=10，batch_size=32，epochs=50，保存最佳 val_acc 模型。

用法：
    python train.py                 # 从零训练（ImageNet 预训练 backbone）
    python train.py --epochs 30
    python train.py --freeze 3      # 先冻结 backbone 训 3 轮再解冻
"""
from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import config
from utils.utils import set_seed, save_classes
from preprocess.dataset import make_loaders
from model.resnet18 import build_model, device_select, count_parameters


# ----------------------------------------------------------------------
# 训练 / 验证一轮
# ----------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="训练音频分类 ResNet18")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument("--freeze", type=int, default=0,
                        help="先冻结 backbone 训几轮再解冻（0=不冻结）")
    parser.add_argument("--out", type=str, default="best.pth", help="权重保存文件名")
    parser.add_argument("--resume", type=str, default=None,
                        help="加载已有权重再微调（两阶段微调用），如 best.pth")
    parser.add_argument("--data", type=str, default=None,
                        choices=["all", "public", "self"],
                        help="使用哪些数据：all=全部(默认) public=仅公开 self=仅自录")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = device_select()
    print(f"[*] 设备：{device}")

    # 数据（可按需筛选数据源）
    train_loader, val_loader, _, classes = make_loaders(seed=config.SEED,
                                                        data_filter=args.data)
    num_classes = len(classes)
    print(f"[*] 数据源：{args.data or 'all'}  类别数：{num_classes}  类别：{classes}")
    save_classes(classes)   # 保存类别顺序，供 evaluate/predict 使用

    # 模型
    freeze_now = args.freeze > 0
    model = build_model(num_classes=num_classes, pretrained=True,
                        freeze_backbone=freeze_now).to(device)

    # 两阶段微调：加载已有权重（类别数必须一致，否则报错提示）
    start_epoch_msg = ""
    if args.resume:
        resume_path = config.MODEL_DIR / args.resume
        if not resume_path.exists():
            print(f"[!] 找不到 {resume_path}，无法 resume，改为从头训练")
        else:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            old_classes = ckpt.get("classes", [])
            if len(old_classes) != num_classes:
                print(f"[!] 类别数不匹配（原 {len(old_classes)} ≠ 现 {num_classes}）")
                print(f"    原类别：{old_classes}")
                print(f"    现类别：{classes}")
                print("    将只加载 backbone 权重，fc 层随机初始化")
                # 只加载除 fc 外的权重（backbone）
                backbone = {k: v for k, v in ckpt["model"].items()
                            if not k.startswith("fc.")}
                missing = model.load_state_dict(backbone, strict=False)
                print(f"    缺失键：{len(missing.missing_keys)} 个（fc 层）")
            else:
                # 类别顺序可能不同，需要校验
                if old_classes != classes:
                    print(f"[!] 类别顺序不一致，将按当前顺序重新对齐")
                model.load_state_dict(ckpt["model"])
            start_epoch_msg = f"（从 {args.resume} 加载权重继续微调）"
            print(f"[*] 已加载权重 {resume_path} {start_epoch_msg}")

    print(f"[*] 可训练参数：{count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, bad, history = 0.0, 0, {"train_loss": [], "train_acc": [],
                                      "val_loss": [], "val_acc": []}
    save_path = config.MODEL_DIR / args.out
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    val_empty = len(val_loader.dataset) == 0   # 数据极少时可能为空

    for epoch in range(1, args.epochs + 1):
        # 解冻阶段切换
        if freeze_now and epoch > args.freeze:
            print(f"[*] 第 {epoch} 轮：解冻 backbone，全网络微调")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=args.lr * 0.1,
                              weight_decay=config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.freeze)
            freeze_now = False

        t0 = time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        if val_empty:
            # 数据极少、无验证集：回退用 train 指标
            va_loss, va_acc = tr_loss, tr_acc
        else:
            va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        if not freeze_now:   # 解冻后用新调度器；冻结阶段不动 lr
            scheduler.step()
        dt = time() - t0

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        print(f"[{epoch:02d}/{args.epochs}] {dt:5.1f}s  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}")

        monitor = va_acc
        if monitor > best_acc or (val_empty and epoch == 1):
            best_acc, bad = max(best_acc, monitor), 0
            torch.save({"model": model.state_dict(),
                        "classes": classes,
                        "config": vars(args)}, save_path)
            print(f"    ↑ 新最佳，已保存到 {save_path}")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[!] {args.patience} 轮无提升，早停。")
                break

    # 保存训练曲线
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_history(history)
    print(f"[√] 训练结束。最佳 val_acc={best_acc:.4f}")
    print(f"    权重：{save_path}  曲线：{config.OUTPUT_DIR/'history.json'}")


def plot_history(history: dict):
    """绘制训练 loss/acc 曲线（对应规划“训练 loss/acc 曲线”）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.utils import setup_matplotlib_font
    setup_matplotlib_font()

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "training_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
