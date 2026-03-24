# ============================================================
# dataset.py — CT / ADDA Segmentation Dataset with Albumentations
# ============================================================
import pandas as pd, numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SegDataset(Dataset):
    def __init__(self, csv_path, root_dir, augment=True, img_size=384):
        self.data = pd.read_csv(csv_path)
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.img_size = img_size

        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def fix_path(self, p):
        """Fixes relative or absolute paths to work across OS/colab."""
        p = str(p).strip().replace("\\", "/")
        # If already absolute or contains root, leave it
        if str(self.root_dir) in p:
            return p
        # Otherwise, prepend root_dir
        return str((self.root_dir / p).resolve())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.fix_path(row["ct_path"])
        mask_path = self.fix_path(row["label_path"])

        # Load as grayscale (CT single channel)
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 0.5).astype(np.float32)

        # Convert grayscale to 3-channel
        img = np.stack([img, img, img], axis=-1)

        transformed = self.transform(image=img, mask=mask)
        img_t, mask_t = transformed["image"], transformed["mask"].unsqueeze(0)
        return img_t, mask_t

