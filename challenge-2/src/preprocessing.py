"""

Author: Vaibhav Sharma, Shreya Khantal, Prasanna Saxena
Team Name: Team Cygnus
Team Members: Vaibhav Sharma, Shreya Khantal, Prasanna Saxena
Public Board Rank: 20

"""

# preprocessing.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Replace these values with your dataset-specific ones if known
DEFAULT_MEAN = [0.4886, 0.3984, 0.3095]
DEFAULT_STD = [0.1494, 0.1462, 0.1414]


class SoilImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_test=False):
        self.dataframe = dataframe
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['file_path']
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, self.dataframe.iloc[idx]['image_id']
        else:
            label = self.dataframe.iloc[idx]['label']
            return img, label


def get_transforms(img_size=224, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transforms, val_transforms
