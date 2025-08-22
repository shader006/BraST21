import os
import json
import random
from glob import glob

def generate_dataset_json(data_dir, output_json_path, val_ratio=0.2, seed=42, fold=0):
    """
    Tạo file dataset.json tương thích với hàm `datafold_read(...)` của MONAI Swin UNETR.

    Args:
        data_dir (str): Thư mục chứa các case BraTS2021_xxxxx.
        output_json_path (str): Đường dẫn file JSON đầu ra.
        val_ratio (float): Tỷ lệ validation.
        seed (int): Seed ngẫu nhiên.
        fold (int): Số fold để gán cho validation.
    """
    random.seed(seed)
    all_cases = sorted(glob(os.path.join(data_dir, "BraTS2021_*")))
    print(f"Found {len(all_cases)} cases.")

    random.shuffle(all_cases)
    split_index = int(len(all_cases) * (1 - val_ratio))
    train_cases = all_cases[:split_index]
    val_cases = all_cases[split_index:]

    def build_entry(case_path, is_val=False):
        case_id = os.path.basename(case_path)
        entry = {
            "image": [
                os.path.join(case_id, f"{case_id}_t1.nii.gz"),
                os.path.join(case_id, f"{case_id}_t1ce.nii.gz"),
                os.path.join(case_id, f"{case_id}_t2.nii.gz"),
                os.path.join(case_id, f"{case_id}_flair.nii.gz")
            ],
            "label": os.path.join(case_id, f"{case_id}_seg.nii.gz")
        }
        if is_val:
            entry["fold"] = fold
        return entry

    dataset = {
        "training": [build_entry(p, is_val=False) for p in train_cases] +
                    [build_entry(p, is_val=True) for p in val_cases]
    }

    with open(output_json_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Saved dataset JSON to: {output_json_path}")
    print(f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)}")


# Ví dụ sử dụng:
if __name__ == "__main__":
    data_dir = r"/work/cuc.buithi/brats_challenge/BraTS2021"  # <-- chỉnh sửa đường dẫn này
    output_json = "dataset.json"
    generate_dataset_json(data_dir, output_json)
