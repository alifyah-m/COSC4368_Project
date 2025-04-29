import os
import pandas as pd

TRAIN_DIR = "data/raw/train"
records = []

# For each class subfolder (NORMAL, PNEUMONIA)
for cls in ["NORMAL", "PNEUMONIA"]:
    cls_path = os.path.join(TRAIN_DIR, cls)
    for fname in os.listdir(cls_path):
        # build relative image path and label
        records.append({
            "X_ray_image_name": f"{cls}/{fname}",
            "Label": 0 if cls == "NORMAL" else 1
        })

# Create DataFrame and write CSV
df = pd.DataFrame(records)
df.to_csv("data/raw/train_labels.csv", index=False)
print(f"✅ Created train_labels.csv with {len(df)} entries.")
