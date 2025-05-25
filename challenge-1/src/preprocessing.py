"""

Author: Vaibhav Sharma, Shreya Khantal, Prasanna Saxena
Team Name: Team Cygnus
Team Members: Vaibhav Sharma, Shreya Khantal, Prasanna Saxena
Leaderboard Rank: 91

"""

# Here you add all the preprocessing related details for the task completed from Kaggle.
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Constants
IMAGE_SIZE = 224

# Dataset Class
class SoilDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image
        else:
            label = self.df.iloc[idx]['soil_label']
            return image, label

# Compute Dataset Mean and Std
def compute_dataset_stats(df, img_dir):
    pixel_sum = np.zeros(3)
    pixel_sum_sq = np.zeros(3)
    n_pixels = 0

    for idx in range(len(df)):
        img_name = df.iloc[idx]['image_id']
        img_path = os.path.join(img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
            pixel_sum += image.sum(axis=(0, 1))
            pixel_sum_sq += (image ** 2).sum(axis=(0, 1))
            n_pixels += image.shape[0] * image.shape[1]
        except:
            continue

    mean = pixel_sum / n_pixels
    std = np.sqrt(pixel_sum_sq / n_pixels - mean ** 2)
    return mean.tolist(), std.tolist()

# Data Transforms
def get_transforms(mean, std):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform
