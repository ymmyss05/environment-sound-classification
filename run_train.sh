#!/bin/bash
# Run training: 200 epochs with all data (public + self)
# Usage: bash run_train.sh [epochs] [data_filter]

EPOCHS=${1:-200}
DATA=${2:-all}

cd "$(dirname "$0")"
python train/train.py --epochs $EPOCHS --data $DATA --out best.pth
