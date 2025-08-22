#!/usr/bin/env python3
# make_brats21_folds_nocli.py
# Tạo brats21_folds.json cho BraTS21 theo format MONAI/Swin UNETR
# KHÔNG dùng argparse. TỰ NHẬN BIẾT có/không thư mục TrainingData/.
# Thêm chế độ chia FOLD "balanced" để phân bố đều nhất có thể.

from pathlib import Path
import json, re, random
from collections import Counter

# ==== CẤU HÌNH (SỬA TẠI ĐÂY) ================================================
DATA_DIR   = Path("/work/cuc.buithi/brats_challenge/BraTS2021")  # gốc dữ liệu (các case nằm trực tiếp ở đây, hoặc trong TrainingData/)
OUT_JSON   = DATA_DIR / "brats21_folds.json"                     # file json đầu ra
N_FOLDS    = 5                                                   # số fold
ASSIGN     = "balanced"                                          # "balanced" | "roundrobin" | "random"
SEED       = 42                                                  # seed cho random/balanced
# ============================================================================

REQ_MODAL_SUFFIXES = ["flair", "t1", "t1ce", "t2"]
CASE_DIR_PREFIX = "BraTS2021_"
CASE_NAME_RE = re.compile(r"BraTS2021_(\d+)$")

def _case_dirs(root: Path):
    """Trả về danh sách thư mục case (BraTS2021_XXXXX) dưới root."""
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(CASE_DIR_PREFIX)])

def find_cases(base_dir: Path):
    """Nếu tồn tại TrainingData/ thì dùng; nếu không, quét trực tiếp base_dir."""
    td = base_dir / "TrainingData"
    if td.is_dir():
        return _case_dirs(td), td
    return _case_dirs(base_dir), base_dir

def build_entry(case_dir: Path, base_dir: Path):
    case_name = case_dir.name
    imgs = []
    for suf in REQ_MODAL_SUFFIXES:
        f = case_dir / f"{case_name}_{suf}.nii.gz"
        if not f.is_file():
            raise FileNotFoundError(f"Thiếu modality {suf}: {f}")
        imgs.append(f)
    seg = case_dir / f"{case_name}_seg.nii.gz"
    if not seg.is_file():
        raise FileNotFoundError(f"Thiếu label: {seg}")
    # đường dẫn tương đối so với base_dir (DATA_DIR)
    rel_imgs = [str(p.relative_to(base_dir).as_posix()) for p in imgs]
    rel_seg  = str(seg.relative_to(base_dir).as_posix())
    return {"image": rel_imgs, "label": rel_seg}

def extract_numeric_id(case_name: str) -> int:
    m = CASE_NAME_RE.match(case_name)
    return int(m.group(1)) if m else 0

def assign_folds(cases, n_folds: int, method: str, seed: int):
    """Trả về dict: {case_name: fold_id} theo phương pháp chỉ định."""
    if method not in {"balanced", "roundrobin", "random"}:
        raise ValueError(f"assign method không hợp lệ: {method}")

    # ROUNDROBIN: theo ID % n_folds (ổn định nếu ID liên tục)
    if method == "roundrobin":
        folds = {}
        for c in cases:
            cid = extract_numeric_id(c.name)
            folds[c.name] = cid % n_folds
        return folds

    # RANDOM: ngẫu nhiên (xấp xỉ đều), reproducible theo seed
    if method == "random":
        rng = random.Random(seed)
        folds = {}
        for c in cases:
            folds[c.name] = rng.randrange(n_folds)
        return folds

    # BALANCED: xáo trộn rồi cắt đúng kích thước mục tiêu cho từng fold
    # => phân bố đều nhất có thể, ví dụ 1251 -> [251, 250, 250, 250, 250]
    rng = random.Random(seed)
    idx = list(range(len(cases)))
    rng.shuffle(idx)

    N = len(cases)
    base = N // n_folds
    extra = N % n_folds
    sizes = [base + (1 if i < extra else 0) for i in range(n_folds)]

    folds = {}
    pos = 0
    for f, sz in enumerate(sizes):
        for k in idx[pos:pos+sz]:
            folds[cases[k].name] = f
        pos += sz
    return folds

def main():
    base_dir = DATA_DIR.resolve()
    cases, scan_root = find_cases(base_dir)
    if not cases:
        raise SystemExit(f"Không tìm thấy case nào trong {scan_root}")

    fold_map = assign_folds(cases, N_FOLDS, ASSIGN, SEED)

    training_list, skipped = [], 0
    for c in cases:
        try:
            entry = build_entry(c, base_dir)
            entry["fold"] = fold_map[c.name]
            training_list.append(entry)
        except FileNotFoundError as e:
            skipped += 1
            print(f"[WARN] Bỏ qua {c.name}: {e}")

    data = {"training": training_list}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # thống kê số lượng theo fold
    cnt = Counter(d["fold"] for d in training_list)
    per_fold = ", ".join(f"fold {k}: {v}" for k, v in sorted(cnt.items()))
    print(f"Đã viết {OUT_JSON} với {len(training_list)} case. Bỏ qua {skipped} case thiếu file.")
    print("Phân bố theo fold ->", per_fold)
    ok = all(len(d["image"]) == 4 and "label" in d and "fold" in d for d in training_list)
    print("Định dạng hợp lệ:", ok)

if __name__ == "__main__":
    main()
