python test.py \
  --json_list=/work/cuc.buithi/brats_challenge/BraTS2021/brats21_folds.json \
  --data_dir=/work/cuc.buithi/brats_challenge/BraTS2021 \
  --feature_size=48 \
  --infer_overlap=0.7 \
  --pretrained_model_name model.pt \
  --pretrained_dir /work/cuc.buithi/brats_challenge/swinunetr_code/research-contributions/SwinUNETR/BRATS21/runs/runs/swinunetr_brats21 \
  --fold 1 