import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ChexpertDataset(Dataset):
    def __init__(self, img_path=None, csv_file=None, transform=None, inference=False):
        if inference:
            self.data_frame = pd.DataFrame([img_path], columns=['path'])
        else:
            self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.inference = inference
        self.label_map = {
            "No Finding": 0,
            "Cardiomegaly": 1,
            "Edema": 2,
            "Pneumothorax": 3,
            "Pleural Effusion": 4
        }

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.inference:
            return image, 50, img_path
        else:
            labels = self.data_frame.iloc[idx, 5:10].values
            labels = np.array(labels, dtype=np.float32)
            label = np.where(labels == 1)[0][0]
            label = torch.tensor(label, dtype=torch.long)
            return image, label, img_path