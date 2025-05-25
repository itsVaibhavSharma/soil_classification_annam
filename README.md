# Soil Classification - Deep Learning Project

This repository contains the implementation for two soil-related image classification challenges using deep learning (ResNet50). The project includes both multi-class soil type classification and binary soil vs non-soil classification.

---

## Project Structure

```
├── challenge-1/                  # Multi-class soil type classification
│   ├── data/                     # Dataset and helper script
│   │   └── download.sh           # Script to download dataset
│   │
│   ├── docs/cards/              # Documentation and metrics
│   │   ├── architecture.png     # Model architecture diagram
│   │   └── ml-metrics.json      # Final evaluation metrics (F1 scores)
│   │
│   ├── notebooks/               # Main Jupyter notebooks
│   │   ├── training.ipynb       # Model training notebook
│   │   └── inference.ipynb      # Inference and submission notebook
│   │
│   ├── src/                     # Python modules for modular code
│   │   ├── preprocessing.py     # Dataset class and transforms
│   │   └── postprocessing.py    # Prediction, evaluation, submission
│   │
│   └── requirements.txt         # Python dependencies
│
├── challenge-2/                  # Binary soil vs non-soil classification
│   ├── data/                     # Dataset and helper script
│   │   └── download.sh           # Script to download dataset
│   │
│   ├── docs/cards/              # Documentation and metrics
│   │   ├── architecture.png     # Model architecture diagram
│   │   └── ml-metrics.json      # Final evaluation metrics (F1 scores)
│   │
│   ├── notebooks/               # Main Jupyter notebooks
│   │   ├── training.ipynb       # Model training notebook
│   │   └── inference.ipynb      # Inference and submission notebook
│   │
│   ├── src/                     # Python modules for modular code
│   │   ├── preprocessing.py     # Dataset class and transforms
│   │   └── postprocessing.py    # Prediction, evaluation, submission
│   │
│   └── requirements.txt         # Python dependencies
│
└── README.md                     # This file
```

---

## Challenge Overview

### Challenge 1: Soil Type Classification
Multi-class classification of soil images into four categories:
- Alluvial soil
- Black Soil
- Clay soil
- Red soil

### Challenge 2: Soil vs Non-Soil Classification
Binary classification to determine whether an image contains soil or not:
- Soil
- Non-Soil

---

## Notebooks

Each challenge contains two main notebooks:

### `training.ipynb`
- Loads and preprocesses the dataset
- Computes dataset statistics and class weights
- Applies augmentations and transformations
- Trains a **ResNet50** model using PyTorch
- Tracks and logs metrics (accuracy, F1 macro/min)
- Saves the best model to `best_resnet50.pth`

### `inference.ipynb`
- Loads the trained model
- Applies test-time preprocessing
- Generates predictions on test images
- Saves predictions to `submission.csv` in competition format

---

## Key Modules

Both challenges share similar module structure:

### `src/preprocessing.py`
- `SoilDataset`: PyTorch-compatible dataset loader
- `compute_dataset_stats`: Calculates mean and std for normalization
- `get_transforms`: Defines augmentation and normalization pipelines

### `src/postprocessing.py`
- `predict`: Runs inference on the test/validation set
- `evaluate`: Computes F1 scores, confusion matrix, classification report
- `create_submission`: Generates submission CSV from predictions

---

## Model Info

- Model: `ResNet50`, `ResNet18` (pretrained backbone)
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss` with class weights
- Augmentations: Flip, Rotation, Affine transform
- Metrics:
  - Challenge 1: **Minimum F1-score across all classes**
  - Challenge 2: **F1-score for binary classification**

---

## Final Performance

### Challenge 1 Results
From `challenge-1/docs/cards/ml-metrics.json`:

| Soil Type       | F1 Score |
|------------------|----------|
| Alluvial soil    | 0.98     |
| Black Soil       | 0.98     |
| Clay soil        | 0.98     |
| Red soil         | 0.99     |

### Challenge 2 Results
From `challenge-2/docs/cards/ml-metrics.json`:

| Class        | F1 Score |
|--------------|----------|
| Soil         | 0.99     |
| Non-Soil     | 0.99     |

---

## Dataset Format

Both challenges follow the same dataset structure:

```
data/
├── train/
├── test/
├── train_labels.csv
└── test_ids.csv
```

Each row in `train_labels.csv` must contain:
- `image_id`
- `soil_type` (Challenge 1) or `class` (Challenge 2)

Each row in `test_ids.csv` must contain:
- `image_id`

---

## Installation

Navigate to the specific challenge directory and install dependencies:

```bash
cd challenge-1  # or challenge-2
pip install -r requirements.txt
```

If you're using GPU with CUDA:

```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Submission Format

Output CSV (`submission.csv`) format:

**Challenge 1:**
```csv
image_id,soil_type
img_123.jpg,Red soil
img_456.jpg,Alluvial soil
...
```

**Challenge 2:**
```csv
image_id,class
img_123.jpg,Soil
img_456.jpg,Non-Soil
...
```

---

## How to Run Training

### Option 1: Run via Jupyter Notebook Interface (Recommended)

1. **Activate your environment** (if using `venv` or `conda`):
   ```bash
   conda activate your_env_name
   # or
   source venv/bin/activate
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Navigate to the specific challenge**:
   ```
   challenge-1/notebooks/training.ipynb
   # or
   challenge-2/notebooks/training.ipynb
   ```

4. **Open the notebook** and run each cell sequentially:
   - It will train the ResNet50 model
   - Logs validation accuracy, F1-scores (macro, min)
   - Saves the best model as `best_resnet50.pth` in the current directory

### Option 2: Run via Command Line (Non-interactive)

You can run the notebook headlessly and export output:

```bash
cd challenge-1/notebooks  # or challenge-2/notebooks
jupyter nbconvert --to notebook --execute training.ipynb --output training_output.ipynb
```

---

## Output Files

- `best_resnet50.pth`: Saved PyTorch model (after training)
- `training_output.ipynb`: (Optional) Notebook with outputs (if using CLI)
- `submission.csv`: Competition submission file

---

## Authors

**Team: SoilClassifiers**

- Vaibhav Sharma
- Shreya Khantal
- Prasanna Saxena

---

## References

- `architecture.png`: High-level model flow diagram (in each challenge's docs/cards/)
- `ml-metrics.json`: Final F1 scores per class (in each challenge's docs/cards/)

---

## License

This project is for academic and research purposes only.