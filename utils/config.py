
import os
from pathlib import Path


# ============================================================
# ffmpeg 设置
# ============================================================
def _setup_ffmpeg():
    """把 imageio-ffmpeg 的 ffmpeg 加入 PATH，让 librosa/audioread 能读 m4a/mp3。

    系统没装独立 ffmpeg 时，用 pip 装的 imageio-ffmpeg 提供 ffmpeg 二进制。
    在 import config 时自动调用一次。
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        d = os.path.dirname(exe)
        if d and d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass  # 没装也能跑，只是不能直接读 m4a


_setup_ffmpeg()

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"        # 原始音频（兼容旧版）：按类别建子文件夹
# 多数据源目录：公开数据 + 自录数据分开存放，extract 时自动合并
DATA_PUBLIC_DIR = BASE_DIR / "dataset_public"   # 公开数据集整理后（ESC-50/UrbanSound8K 等）
DATA_SELF_DIR = BASE_DIR / "dataset_self"       # 自录数据
DATA_DIRS = [DATA_PUBLIC_DIR, DATA_SELF_DIR]    # 所有原始数据目录（自动跳过不存在的）
FEATURE_DIR = BASE_DIR / "features"    # extract.py 缓存的 Mel 频谱图 (.npy)
MODEL_DIR = BASE_DIR / "models"        # 训练保存的权重 (.pth)
OUTPUT_DIR = BASE_DIR / "output"       # 评估图、指标、曲线
CLASSES_FILE = MODEL_DIR / "classes.json"   # 类别顺序（训练时保存，评估/预测时读取）

# ============================================================
# 音频 / 特征参数（对应规划第 4、5 步）
# ============================================================
SR = 16000            # 统一采样率
FIXED_SEC = 2         # 每段固定时长（秒）
FIXED_LEN = SR * FIXED_SEC          # 固定采样点数 = 32000
N_MELS = 128          # Mel 频带数
N_FFT = 1024          # FFT 窗口
HOP_LENGTH = 512      # 步长
WIN_LENGTH = N_FFT    # 窗长
# 一段音频对应的 Mel 时间帧数：1 + FIXED_LEN // HOP_LENGTH
N_FRAMES = 1 + FIXED_LEN // HOP_LENGTH   # = 63

# ============================================================
# 训练超参数（对应规划第 8 步）
# ============================================================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10
NUM_WORKERS = 0       # Windows 下多进程 dataloader 容易出问题，默认 0

# 数据集划分比例（训练 : 验证 : 测试）
SPLIT_RATIO = (0.8, 0.1, 0.1)

# ImageNet 标准化参数：3 通道 Mel 谱当作 RGB 图片喂 ResNet 时使用
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 默认类别（若 dataset/ 下没有对应文件夹，则用此列表）
DEFAULT_CLASSES = ["雨声", "键盘敲击", "拍手", "说话", "水流",
                   "咳嗽", "汽车", "鸟鸣", "警笛", "关门"]

# ESC-50 类别 → 中文类别 的映射（prepare_data.py 使用）
# ESC-50 官方类别名见 https://github.com/karolpiczak/ESC-50/blob/master/meta/esc50.csv
# 注意：ESC-50 实际类别名以 esc50.csv 为准（如 toilet_flush 而非 flush_toilet）
ESC50_TO_LABEL = {
    "rain": "雨声", "sea_waves": "雨声", "wind": "雨声", "thunderstorm": "雨声",
    "keyboard_typing": "键盘敲击",
    "clapping": "拍手",
    # ESC-50 没有纯"说话/人声对话"类，该类由自录数据补齐
    "water_drops": "水流", "pouring_water": "水流", "toilet_flush": "水流",
    "coughing": "咳嗽", "sneezing": "咳嗽",
    # 鸟鸣：ESC-50 实际有的鸟/虫鸣类
    "rooster": "鸟鸣", "hen": "鸟鸣", "chirping_birds": "鸟鸣", "crow": "鸟鸣",
    "crickets": "鸟鸣", "frog": "鸟鸣",
    "siren": "警笛",
    "door_wood_knock": "关门", "door_wood_creaks": "关门", "can_opening": "关门",
    # 汽车类：ESC-50 实际有 car_horn / engine
    "car_horn": "汽车", "engine": "汽车",
}

# UrbanSound8K 类别 → 中文类别 映射（prepare_data.py 使用）
URBAN_TO_LABEL = {
    "car_horn": "汽车", "engine_idling": "汽车",
    "siren": "警笛",
    "dog_bark": "狗叫",   # 多余类别可保留或丢弃
    "street_music": "说话",  # 可选映射
}


def ensure_dirs():
    """创建运行所需的所有目录。"""
    for d in [DATA_PUBLIC_DIR, DATA_SELF_DIR, FEATURE_DIR, MODEL_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
