#!/bin/bash
#SBATCH --job-name=brats
#SBATCH --partition=dgx-small
#SBATCH --account=ddt_acc23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==== 1. Load environment ====

source ~/.bashrc
conda activate swin               # đổi tên env nếu khác

# ==== 2. In thông tin job ====
echo "Start job at $(date)"
echo "Node: $SLURMD_NODENAME | GPUs: $CUDA_VISIBLE_DEVICES"

# ==== 3. Chạy training ====
CUDA_VISIBLE_DEVICES=1 python main.py \
  --json_list=/work/cuc.buithi/brats_challenge/BraTS2021/brats21_folds.json \
  --data_dir=/work/cuc.buithi/brats_challenge/BraTS2021 \
  --logdir=./runs/swinunetr_brats21 \
  --fold=0 \
  --optim_lr=0.0008 \
  --max_epochs=300 \
  --batch_size=1 \
  --sw_batch_size=4 \
  --roi_x=128 --roi_y=128 --roi_z=128 \
  --in_channels=4 --spatial_dims=3 \
  --infer_overlap=0.7 \
  --RandFlipd_prob=0.5 \
  --cache_dataset \
  --save_checkpoint \
  --lrschedule=warmup_cosine \
  --val_every=10 \
  --use_checkpoint

echo "End job at $(date)"
