"""

Author: Vaibhav Sharma, Shreya Khantal, Prasanna Saxena
Team Name: Team Cygnus
Team Members: Vaibhav Sharma, Shreya Khantal, Prasanna Saxena
Public Board Rank: 20

"""
# postprocessing.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


def predict_test_set(model, test_df, transform, device='cpu', batch_size=32):
    from preprocessing import SoilImageDataset

    model.eval()
    test_dataset = SoilImageDataset(test_df, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            for image_id, pred, prob in zip(image_ids, preds.cpu(), probs[:, 1].cpu()):
                results.append({
                    'image_id': image_id,
                    'label': int(pred),
                    'soil_probability': float(prob)
                })

    submission_df = pd.DataFrame(results)
    return submission_df


def save_submission(submission_df, filename='submission.csv'):
    submission_df[['image_id', 'label']].to_csv(filename, index=False)
    print(f"Saved to {filename}")


def visualize_predictions(submission_df, test_df, n=5):
    high_conf_soil = submission_df[submission_df['label'] == 1].sort_values('soil_probability', ascending=False).head(n)
    low_conf_soil = submission_df[submission_df['label'] == 1].sort_values('soil_probability').head(n)
    high_conf_nonsoil = submission_df[submission_df['label'] == 0].sort_values('soil_probability').head(n)

    plt.figure(figsize=(20, 8))

    for i, (_, row) in enumerate(high_conf_soil.iterrows()):
        img_path = test_df[test_df['image_id'] == row['image_id']]['file_path'].values[0]
        plt.subplot(3, n, i + 1)
        plt.imshow(Image.open(img_path))
        plt.title(f"Soil: {row['soil_probability']:.2f}")
        plt.axis('off')

    for i, (_, row) in enumerate(low_conf_soil.iterrows()):
        img_path = test_df[test_df['image_id'] == row['image_id']]['file_path'].values[0]
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(Image.open(img_path))
        plt.title(f"Soil: {row['soil_probability']:.2f}")
        plt.axis('off')

    for i, (_, row) in enumerate(high_conf_nonsoil.iterrows()):
        img_path = test_df[test_df['image_id'] == row['image_id']]['file_path'].values[0]
        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(Image.open(img_path))
        plt.title(f"Non-Soil: {1 - row['soil_probability']:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
