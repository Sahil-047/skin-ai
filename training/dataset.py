import os
from typing import Tuple
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SkinDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, non_skin_dir=None, dermnet_dir=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.non_skin_dir = non_skin_dir
        self.dermnet_dir = dermnet_dir
        
        # Clean up possible missing or invalid rows
        self.data = self.data.dropna(subset=["image", "label_id"])
        self.data["image"] = self.data["image"].astype(str)
        print(f"Loaded {len(self.data)} entries from {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = f"{row['image']}.jpg" if not row['image'].endswith('.jpg') else row['image']
        
        # Choose directory based on condition
        dx = str(row.get('dx', '')).lower()
        
        if dx == 'non_skin' and self.non_skin_dir:
            img_path = os.path.join(self.non_skin_dir, img_name)
        elif dx in ['acne', 'athletes_foot', 'cellulitis', 'cold_sores', 'eczema', 'fungal_infection', 'heat_rash'] and self.dermnet_dir:
            # Map athletes_foot (in metadata) to athlete's_foot (folder name)
            folder_name = "athlete's_foot" if dx == "athletes_foot" else dx
            img_path = os.path.join(self.dermnet_dir, folder_name, img_name)
        else:
            img_path = os.path.join(self.img_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Could not open image: {img_path}") from e

        label = int(row["label_id"])

        if self.transform:
            image = self.transform(image)
        
        # Convert metadata to simple types for batching
        age = float(row.get("age", 0)) if pd.notna(row.get("age")) else 0.0
        sex = str(row.get("sex", "unknown"))
        localization = str(row.get("localization", "unknown"))

        return image, label, {"age": age, "sex": sex, "localization": localization}
