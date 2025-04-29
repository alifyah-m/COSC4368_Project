import os, shutil

TRAIN_DIR = "data/raw/train"
NORMAL_DIR = os.path.join(TRAIN_DIR, "NORMAL")
PNEU_DIR   = os.path.join(TRAIN_DIR, "PNEUMONIA")
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(PNEU_DIR,   exist_ok=True)

for fname in os.listdir(TRAIN_DIR):
    src = os.path.join(TRAIN_DIR, fname)
    if os.path.isdir(src):
        continue
    # classify by filename substring
    if "NORMAL" in fname.upper():
        dst = os.path.join(NORMAL_DIR, fname)
    else:
        dst = os.path.join(PNEU_DIR, fname)
    shutil.move(src, dst)

print(f"✅ Moved {len(os.listdir(NORMAL_DIR))} files to NORMAL/")
print(f"✅ Moved {len(os.listdir(PNEU_DIR))} files to PNEUMONIA/")
