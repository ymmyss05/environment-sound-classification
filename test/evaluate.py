"""第 9 步：评估 + 可视化。

输出（对应规划第 9 步评估表）：
- 指标表：准确率/精确率/召回率/F1（per-class + macro/weighted）→ metrics.json
- classification_report 文本
- 混淆矩阵 confusion_matrix.png
- ROC 曲线 + AUC roc_curve.png（one-vs-rest）
- 每类准确率柱状图 per_class_acc.png

用法：
    python evaluate.py                       # 用 models/best.pth 在测试集上评估
    python evaluate.py --model best.pth
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

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_curve, auc)
from sklearn.preprocessing import label_binarize

from utils import config
from utils.utils import set_seed, setup_matplotlib_font
from preprocess.dataset import build_split, AudioDataset
from model.resnet18 import build_model, device_select

# 绘图前统一设置中文字体（本文件多处绘图用中文标签）
setup_matplotlib_font()


def load_checkpoint(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes")
    if classes is None:
        from utils.utils import load_classes
        classes = load_classes()
    return ckpt, classes


@torch.no_grad()
def collect_predictions(model, items, device, batch_size=config.BATCH_SIZE):
    """逐批推理，收集 logits/标签。"""
    ds = AudioDataset(items, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def main():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--model", type=str, default="best.pth")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--data", type=str, default=None,
                        choices=["all", "public", "self"],
                        help="使用哪些数据构建测试集：all=全部(默认) public=仅公开 self=仅自录")
    args = parser.parse_args()

    set_seed(config.SEED)
    device = device_select()
    model_path = config.MODEL_DIR / args.model
    if not model_path.exists():
        print(f"[!] 找不到权重 {model_path}，请先运行 train.py")
        return

    ckpt, classes = load_checkpoint(model_path)
    num_classes = len(classes)
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[*] 已加载 {model_path}，类别 {num_classes} 个")

    # 用同样的 seed 重建划分，取测试/验证集（按 --data 筛选数据源）
    train_it, val_it, test_it, split_classes = build_split(seed=config.SEED,
                                                            data_filter=args.data)
    # 如果 split 中的类别与模型类别不一致，只保留模型认识的类别
    if split_classes != classes:
        print(f"[!] 数据集类别({len(split_classes)})与模型类别({len(classes)})不一致")
        print(f"    数据集类别：{split_classes}")
        print(f"    模型类别：{classes}")
        # 只保留模型能识别的类别的样本
        class_to_idx = {c: i for i, c in enumerate(classes)}
        test_it = [(p, class_to_idx[c]) for p, c_idx in test_it
                   if (c := split_classes[c_idx]) in class_to_idx]
        val_it = [(p, class_to_idx[c]) for p, c_idx in val_it
                  if (c := split_classes[c_idx]) in class_to_idx]
        print(f"    过滤后 {args.split} 集样本数：{len(test_it)}")
    print(f"[*] {args.split} 集样本数：{len(test_it)}")
    items = test_it if args.split == "test" else val_it
    if not items:
        print(f"[!] {args.split} 集为空，无法评估。")
        return
    print(f"[*] {args.split} 集样本数：{len(items)}")

    logits, labels = collect_predictions(model, items, device)
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(1)

    # ---------- 指标 ----------
    acc = float((preds == labels).mean())
    p, r, f1, s = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)

    print(f"\n=== {args.split} 集结果 ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro    P/R/F1 : {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
    print(f"Weighted P/R/F1 : {weighted_p:.4f} / {weighted_r:.4f} / {weighted_f1:.4f}")
    print("\n" + classification_report(labels, preds,
          target_names=classes, zero_division=0, digits=4))

    metrics = {
        "split": args.split, "accuracy": acc,
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f1},
        "per_class": {classes[i]: {"precision": p[i], "recall": r[i],
                                   "f1": f1[i], "support": int(s[i])}
                      for i in range(num_classes)},
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---------- 混淆矩阵 ----------
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    np.save(config.OUTPUT_DIR / f"confusion_matrix_{args.split}.npy", cm)
    plot_confusion(cm, classes, args.split)

    # ---------- ROC + AUC ----------
    plot_roc(labels, probs, classes, args.split)

    # ---------- 每类准确率柱状图 ----------
    plot_per_class_acc(cm, classes, args.split)

    print(f"\n[√] 结果已保存到 {config.OUTPUT_DIR}")


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, classes, split: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.7),
                                    max(5, len(classes) * 0.6)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title(f"混淆矩阵 ({split})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, f"{v}\n{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / f"confusion_matrix_{split}.png", dpi=150)
    plt.close(fig)


def plot_roc(labels, probs, classes, split: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(classes)
    y_bin = label_binarize(labels, classes=list(range(n)))   # (N, n) 或 (N,1)
    if y_bin.shape[1] == 1:   # 二分类补一列
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = []
    for i in range(n):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            a = auc(fpr, tpr)
        except ValueError:
            continue
        aucs.append(a)
        ax.plot(fpr, tpr, label=f"{classes[i]} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC 曲线 ({split})  mean AUC={np.mean(aucs):.3f}")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / f"roc_curve_{split}.png", dpi=150)
    plt.close(fig)


def plot_per_class_acc(cm: np.ndarray, classes, split: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recall = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.7), 4))
    ax.bar(range(len(classes)), recall, color="#4C72B0")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Recall")
    ax.set_title(f"每类召回率 ({split})"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / f"per_class_acc_{split}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
