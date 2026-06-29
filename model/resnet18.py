from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torchvision.models as tv_models

from utils import config


def build_model(num_classes: int,
                pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    """构建 ResNet18 分类器。

    Args:
        num_classes: 类别数。
        pretrained: 是否加载 ImageNet 预训练权重。
        freeze_backbone: True 则冻结除 fc 外的所有层（只训练分类头）。
    """
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    # 改最后一层为对应类别数
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def device_select() -> torch.device:
    """选择设备：有 CUDA 用 CUDA，否则 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量（论文“实验设置”可引用）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dev = device_select()
    net = build_model(num_classes=10, pretrained=False).to(dev)
    print(f"设备：{dev}  可训练参数：{count_parameters(net):,}")
    # 前向自检：输入 (B, 3, N_MELS, N_FRAMES)
    x = torch.randn(2, 3, config.N_MELS, config.N_FRAMES).to(dev)
    out = net(x)
    print("前向输出形状：", tuple(out.shape))   # (2, 10)
