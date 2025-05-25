"""

Author: Vaibhav Sharma, Shreya Khantal, Prasanna Saxena
Team Name: Team Cygnus
Team Members: Vaibhav Sharma, Shreya Khantal, Prasanna Saxena
Leaderboard Rank: 91

"""
# Here you add all the post-processing related details for the task completed from Kaggle.

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Prediction function
def predict(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)

# Evaluation function
def evaluate(model, dataloader, device, true_labels, label_names):
    preds = predict(model, dataloader, device)
    report = classification_report(true_labels, preds, target_names=label_names)
    cm = confusion_matrix(true_labels, preds)
    f1_macro = f1_score(true_labels, preds, average='macro')
    f1_min = np.min(f1_score(true_labels, preds, average=None))
    return {
        "report": report,
        "confusion_matrix": cm,
        "f1_macro": f1_macro,
        "f1_min": f1_min
    }

# Create submission
def create_submission(image_ids, predictions, label_mapping, output_path="submission.csv"):
    labels = [label_mapping[pred] for pred in predictions]
    df = pd.DataFrame({
        "image_id": image_ids,
        "soil_type": labels
    })
    df.to_csv(output_path, index=False)
    return df
