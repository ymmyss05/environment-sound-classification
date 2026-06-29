# 基于ResNet18与Mel频谱图的环境声音分类

利用 ImageNet 预训练的 ResNet18 对环境声音进行分类。将音频转换为 Mel 频谱图作为二维特征，通过迁移学习实现 10 类环境声音分类。

## 项目结构

```
人类/
├── model/              # 模型定义
│   ├── __init__.py
│   └── resnet18.py     # ResNet18 + ImageNet 预训练
├── preprocess/         # 数据预处理
│   ├── __init__.py
│   ├── prepare_data.py # 公开数据集整理 (ESC-50/UrbanSound8K)
│   ├── extract.py      # wav → Mel 频谱图 → .npy
│   ├── dataset.py      # 数据集划分 + SpecAugment 增强
│   ├── record.py       # 麦克风录音采集
│   └── import_phone.py # 手机录音导入 (m4a→wav)
├── train/              # 训练
│   ├── __init__.py
│   └── train.py        # AdamW + CosineLR + 早停
├── test/               # 评估与预测
│   ├── __init__.py
│   ├── evaluate.py     # 指标 + 混淆矩阵 + ROC/AUC
│   └── predict.py      # 单段音频预测
├── utils/              # 工具
│   ├── __init__.py
│   ├── config.py       # 集中配置
│   └── utils.py        # 随机种子/类别读写/音频读取
├── pyproject.toml
├── run_train.sh
└── .gitignore
```

## 快速开始

```bash
# 1. 安装依赖
pip install -e .

# 2. 准备数据 (可选，如已有数据可跳过)
python preprocess/prepare_data.py --esc50 "ESC-50"

# 3. 提取 Mel 频谱图特征
python preprocess/extract.py

# 4. 训练 (200轮)
bash run_train.sh
# 或直接运行:
python train/train.py --epochs 200 --data all --out best.pth

# 5. 评估
python test/evaluate.py --model best.pth --data all

# 6. 预测单段音频
python test/predict.py path/to/audio.wav --model best.pth
```

## 类别

雨声、键盘敲击、拍手、说话、水流、咳嗽、汽车、鸟鸣、警笛、关门

## 数据来源

- 公开数据: ESC-50 (3680 samples, 9 classes)
- 自录数据: 449 samples, 6 classes (含说话类)
- 合计: 4129 samples, 10 classes
