"""
Dataset Setup — Run ONCE in Colab before training.
Downloads and organizes:
  1. BUSI          → /content/datasets/ultrasound
  2. BreakHis      → /content/datasets/histopathology
  3. Chest X-ray   → /content/datasets/chest_xray

Requires: kaggle.json uploaded to /content/
"""

import os
import shutil
import random
from pathlib import Path

DATA_ROOT = '/content/datasets'


def setup_kaggle():
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    src = '/content/kaggle.json'
    dst = os.path.expanduser('~/.kaggle/kaggle.json')
    if os.path.exists(src):
        shutil.copy(src, dst)
        os.chmod(dst, 0o600)
        print("✅ Kaggle API configured!")
    else:
        raise FileNotFoundError("Upload kaggle.json to /content/ first!\nGet it from: kaggle.com → Account → API → Create New Token")


def download_datasets():
    os.makedirs(DATA_ROOT, exist_ok=True)
    print("\n[1/3] Downloading BUSI Ultrasound Dataset...")
    os.system(f'kaggle datasets download -d aryashah2003/breast-ultrasound-images-dataset -p {DATA_ROOT}/raw_us --unzip -q')
    print("[2/3] Downloading BreakHis Histopathology Dataset...")
    os.system(f'kaggle datasets download -d forderation/breakhis -p {DATA_ROOT}/raw_histo --unzip -q')
    print("[3/3] Downloading Chest X-ray Dataset...")
    os.system(f'kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p {DATA_ROOT}/raw_xray --unzip -q')
    print("✅ All downloads complete!")


def organize_split(src_benign, src_malignant, dest_dir,
                   val_ratio=0.15, test_ratio=0.10, seed=42):
    random.seed(seed)
    for label, src in [('benign', src_benign), ('malignant', src_malignant)]:
        src_path = Path(src)
        if not src_path.exists():
            print(f"  ⚠️  Not found: {src} — skipping")
            continue
        files = [f for f in src_path.rglob('*')
                 if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        random.shuffle(files)
        n       = len(files)
        n_val   = int(n * val_ratio)
        n_test  = int(n * test_ratio)
        n_train = n - n_val - n_test
        splits  = {
            'train': files[:n_train],
            'val':   files[n_train:n_train + n_val],
            'test':  files[n_train + n_val:]
        }
        for split, imgs in splits.items():
            out = Path(dest_dir) / split / label
            out.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, out / img.name)
        print(f"  {label}: {n_train} train | {n_val} val | {n_test} test")


def organize_all():
    print("\n[Organizing] Ultrasound (BUSI)...")
    organize_split(
        src_benign    = f'{DATA_ROOT}/raw_us/Dataset_BUSI_with_GT/benign',
        src_malignant = f'{DATA_ROOT}/raw_us/Dataset_BUSI_with_GT/malignant',
        dest_dir      = f'{DATA_ROOT}/ultrasound'
    )
    print("\n[Organizing] Histopathology (BreakHis 400X)...")
    organize_split(
        src_benign    = f'{DATA_ROOT}/raw_histo/BreaKHis_v1/histology_slides/breast/benign/SOB',
        src_malignant = f'{DATA_ROOT}/raw_histo/BreaKHis_v1/histology_slides/breast/malignant/SOB',
        dest_dir      = f'{DATA_ROOT}/histopathology'
    )
    print("\n[Organizing] Chest X-ray...")
    organize_split(
        src_benign    = f'{DATA_ROOT}/raw_xray/chest_xray/train/NORMAL',
        src_malignant = f'{DATA_ROOT}/raw_xray/chest_xray/train/PNEUMONIA',
        dest_dir      = f'{DATA_ROOT}/chest_xray'
    )
    print("\n✅ All datasets organized!")
    # Summary
    for mod in ['ultrasound', 'histopathology', 'chest_xray']:
        for split in ['train', 'val', 'test']:
            p = Path(f'{DATA_ROOT}/{mod}/{split}')
            if p.exists():
                count = sum(1 for _ in p.rglob('*')
                            if _.suffix.lower() in ['.jpg', '.jpeg', '.png'])
                print(f"  {mod}/{split}: {count} images")


if __name__ == '__main__':
    setup_kaggle()
    download_datasets()
    organize_all()
