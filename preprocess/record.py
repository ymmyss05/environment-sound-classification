"""录音采集辅助（对应规划 record.py + 第 3 步采集规范）。

按规范自动产出：单声道、16kHz、固定时长(默认2秒)的 wav，
按类别自动归入 dataset/<类别>/NNN.wav，序号自动递增。

用法：
    python record.py 雨声              # 录一段 2 秒到 dataset/雨声/001.wav
    python record.py 拍手 --seconds 3 --count 20
    python record.py 雨声 --device 2   # 指定输入设备编号
"""
from __future__ import annotations

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import sys
import wave
from pathlib import Path

from utils import config


def list_devices(sd):
    """打印可用输入设备，帮助用户选 --device。"""
    print("可用输入设备：")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            print(f"  [{i}] {d['name']}  (in={d['max_input_channels']}, "
                  f"{d.get('default_samplerate', '?')}Hz)")


def next_index(class_dir: Path) -> int:
    """dataset/<类别>/ 下已有多少 wav，返回下一个序号。"""
    exist = list(class_dir.glob("*.wav"))
    return len(exist) + 1


def record_one(sd, seconds: float, sr: int, device):
    """录制一段，返回 float32 单声道数组。"""
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    return audio.reshape(-1)


def save_wav(path: Path, audio, sr: int):
    """保存为 16-bit PCM wav。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (audio * 32767).clip(-32768, 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data)


def main():
    parser = argparse.ArgumentParser(description="录音采集辅助")
    parser.add_argument("label", type=str, help="类别名（如 雨声）")
    parser.add_argument("--seconds", type=float, default=config.FIXED_SEC, help="每段时长(秒)")
    parser.add_argument("--count", type=int, default=1, help="连续录制段数")
    parser.add_argument("--device", type=int, default=None, help="输入设备编号")
    parser.add_argument("--sr", type=int, default=config.SR, help="采样率")
    parser.add_argument("--list-devices", action="store_true", help="列出设备后退出")
    parser.add_argument("--out", type=str, default="self",
                        choices=["self", "public"],
                        help="存到自录目录(self, 默认) 或 公开目录(public)")
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except Exception as e:   # 未安装或无音频后端
        print("[!] 需要 sounddevice：pip install sounddevice")
        print(f"    ({e})")
        sys.exit(1)

    if args.list_devices:
        list_devices(sd); return

    base_dir = config.DATA_SELF_DIR if args.out == "self" else config.DATA_PUBLIC_DIR
    class_dir = base_dir / args.label
    print(f"[*] 目标目录：{class_dir}  采样率 {args.sr}Hz  单声道  "
          f"每段 {args.seconds}s × {args.count} 段")

    for _ in range(args.count):
        idx = next_index(class_dir)
        out = class_dir / f"{idx:03d}.wav"
        print(f"  准备录制 {out.name}，3 秒后开始 …")
        sd.sleep(3000)
        print("  ● 录制中 …")
        audio = record_one(sd, args.seconds, args.sr, args.device)
        save_wav(out, audio, args.sr)
        print(f"  ■ 已保存 {out}")

    print(f"[√] 完成。可用 python extract.py 提取特征。")


if __name__ == "__main__":
    main()
