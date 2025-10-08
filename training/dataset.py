import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class SkinDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        """
        Args:
            csv_path (str): Path to metadata_final.csv
            img_dir (str): Directory containing ISIC_*.jpg files
            transform (callable, optional): Optional transform to apply to each image
        """
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Clean up possible missing or invalid rows
        self.data = self.data.dropna(subset=["image", "label_id"])
        self.data["image"] = self.data["image"].astype(str)
        print(f"✅ Loaded {len(self.data)} entries from {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = f"{row['image']}.jpg" if not row['image'].endswith('.jpg') else row['image']
        img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"❌ Could not open image: {img_path}") from e

        label = int(row["label_id"])
        age = torch.tensor(row.get("age", 0) if not pd.isna(row.get("age", 0)) else 0, dtype=torch.float32)
        sex = 1.0 if str(row.get("sex", "")).lower() == "male" else 0.0  # simple numeric encoding

        if self.transform:
            image = self.transform(image)

        return image, label, {"age": age, "sex": sex, "localization": row.get("localization", "unknown")}
