#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Chọn GPU 2

import torch
import torch.nn as nn
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from utils.data_utils import get_loader
import argparse

def test_overfit():
    """Test overfit trên 1-5 samples để kiểm tra model có học được không"""
    
    # Setup args giống main.py
    class Args:
        def __init__(self):
            self.data_dir = "/work/cuc.buithi/brats_challenge/BraTS2021"
            self.json_list = "/work/cuc.buithi/brats_challenge/BraTS2021/brats21_folds.json"
            self.fold = 0
            self.batch_size = 1
            self.workers = 2
            self.roi_x = self.roi_y = self.roi_z = 128  # Giảm để tránh OOM
            self.a_min = -175.0
            self.a_max = 250.0
            self.b_min = 0.0
            self.b_max = 1.0
            self.space_x = self.space_y = 1.5
            self.space_z = 2.0
            self.RandFlipd_prob = 0.5
            self.RandRotate90d_prob = 0.2
            self.RandScaleIntensityd_prob = 0.1
            self.RandShiftIntensityd_prob = 0.1
            self.cache_dataset = False
            self.test_mode = False
            self.distributed = False
            self.rank = 0
    
    args = Args()
    
    # Load data
    print("Loading data...")
    loader = get_loader(args)
    train_loader = loader[0]
    
    # Lấy 1 sample để overfit
    data_iter = iter(train_loader)
    batch_data = next(data_iter)
    
    # Handle different data formats
    if isinstance(batch_data, dict):
        sample_data = batch_data["image"]
        sample_target = batch_data["label"]
    else:
        sample_data, sample_target = batch_data
    
    print(f"Data shape: {sample_data.shape}")
    print(f"Target shape: {sample_target.shape}")
    print(f"Target unique values: {torch.unique(sample_target)}")
    
    # Model
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=False,
    ).cuda()
    
    # Loss & Optimizer
    dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Metrics
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, logit_thresh=0.5)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=True)
    
    # Move to GPU
    sample_data = sample_data.cuda()
    sample_target = sample_target.cuda()
    
    print("\nStarting overfit test...")
    print("=" * 50)
    
    for epoch in range(100):
        model.train()
        
        # Forward
        logits = model(sample_data)
        loss = dice_loss(logits, sample_target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluation mỗi 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Predictions
                pred_sigmoid = post_sigmoid(logits)
                pred_discrete = post_pred(pred_sigmoid)
                
                # Dice metric
                dice_metric.reset()
                dice_metric(y_pred=[pred_discrete], y=[sample_target])
                dice_scores = dice_metric.aggregate()
                
                # Handle different number of classes
                if len(dice_scores) >= 3:
                    print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | "
                          f"Dice TC: {dice_scores[0].item():.4f} | "
                          f"Dice WT: {dice_scores[1].item():.4f} | "
                          f"Dice ET: {dice_scores[2].item():.4f}")
                else:
                    dice_str = " | ".join([f"Dice {i}: {score.item():.4f}" for i, score in enumerate(dice_scores)])
                    print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | {dice_str}")
                print(f"         | Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}] | "
                      f"Sigmoid range: [{pred_sigmoid.min().item():.3f}, {pred_sigmoid.max().item():.3f}]")
                
                # Kiểm tra có predictions > 0.5 không
                positive_preds = (pred_sigmoid > 0.5).sum().item()
                total_voxels = pred_sigmoid.numel()
                print(f"         | Positive predictions: {positive_preds}/{total_voxels} "
                      f"({100*positive_preds/total_voxels:.2f}%)")
                print("-" * 50)
    
    print("\nOverfit test completed!")
    print("Nếu Dice không tăng sau 100 epochs → có lỗi trong code")
    print("Nếu Dice tăng → model OK, cần train lâu hơn")

if __name__ == "__main__":
    test_overfit()