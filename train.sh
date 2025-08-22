#!/usr/bin/env bash
set -euo pipefail

# === Cấu hình CUDA và PyTorch ===
export CUDA_VISIBLE_DEVICES=4  # Sử dụng GPU 0 (thay đổi nếu cần)

# === Cấu hình training ===
DATA_DIR="/work/cuc.buithi/brats_challenge/BraTS2021"
JSON_LIST="/work/cuc.buithi/brats_challenge/BraTS2021/brats21_folds.json"
RUN_NAME="swin_unetr_brats"  # Tên experiment
FOLD=0
LR=8e-4        # Giảm learning rate để ổn định hơn
EPOCHS=800     # Giảm epochs để test nhanh hơn
BATCH=1        # Training batch size
SW_BATCH=8    # Giảm sliding window batch size để tránh OOM
ROI=64         # Giảm ROI size để tránh OOM
WORKERS=2      # Số worker cho DataLoader
FLIP_P=0.5     # Xác suất flip augmentation
DISTRIBUTED=0  # 1 để bật distributed training

# === Kiểm tra và chuẩn bị dữ liệu ===
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory $DATA_DIR not found!"
    exit 1
fi

if [ ! -f "$JSON_LIST" ]; then
    echo "ERROR: JSON file $JSON_LIST not found!"
    exit 1
fi

# === Chuẩn bị thư mục ===
RUNS_DIR="runs"
EXP_DIR="${RUNS_DIR}/${RUN_NAME}"
mkdir -p "$EXP_DIR"

# Copy script để reproduce
cp "$0" "${EXP_DIR}/train.sh"

# === Cấu hình distributed ===
DIST_FLAG=$([ "$DISTRIBUTED" -eq 1 ] && echo "--distributed" || echo "")
if [ "$DISTRIBUTED" -eq 1 ]; then
    # Thêm các biến môi trường cho distributed
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500
fi

# === Training command ===
echo "Starting training with config:"
echo "- Run name: $RUN_NAME"
echo "- Batch size: $BATCH"
echo "- Learning rate: $LR"
echo "- ROI size: ${ROI}x${ROI}x${ROI}"
echo "- Workers: $WORKERS"
echo "- Device: $CUDA_VISIBLE_DEVICES"

# === Kiểm tra main.py tồn tại ===
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found in current directory!"
    exit 1
fi

# === Chạy training ===
python -u main.py \
    --logdir "$RUN_NAME" \
    --fold "$FOLD" \
    --data_dir "$DATA_DIR" \
    --json_list "$JSON_LIST" \
    --max_epochs "$EPOCHS" \
    --batch_size "$BATCH" \
    --sw_batch_size "$SW_BATCH" \
    --optim_name adamw \
    --optim_lr "$LR" \
    --reg_weight 1e-5 \
    --lrschedule warmup_cosine \
    --warmup_epochs 20 \
    --feature_size 48 \
    --in_channels 4 \
    --out_channels 3 \
    --workers "$WORKERS" \
    --roi_x "$ROI" \
    --roi_y "$ROI" \
    --roi_z "$ROI" \
    --infer_overlap 0.7 \
    --RandFlipd_prob "$FLIP_P" \
    --RandRotate90d_prob 0.2 \
    --save_checkpoint \
    --val_every 10 \
    $DIST_FLAG \
    2>&1 | tee "${EXP_DIR}/train.log"

echo "Training finished! Check logs at: ${EXP_DIR}/train.log"
