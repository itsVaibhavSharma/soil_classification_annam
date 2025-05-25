```markdown
# 🧪 Soil Type Classification - Deep Learning Project

This repository contains the implementation for soil type classification using deep learning (ResNet50). The model classifies soil images into four categories: Alluvial soil, Black Soil, Clay soil, and Red soil.

---

## 📁 Project Structure

```

challenge-1/
├── data/                         # Dataset and helper script
│   └── download.sh               # Script to download dataset
│
├── docs/cards/                  # Documentation and metrics
│   ├── architecture.png         # Model architecture diagram
│   └── ml-metrics.json          # Final evaluation metrics (F1 scores)
│
├── notebooks/                   # Main Jupyter notebooks
│   ├── training.ipynb           # Model training notebook
│   └── inference.ipynb          # Inference and submission notebook
│
├── src/                         # Python modules for modular code
│   ├── preprocessing.py         # Dataset class and transforms
│   └── postprocessing.py        # Prediction, evaluation, submission
│
├── requirements.txt             # Python dependencies 
```

---

## 📚 Notebooks

### 🔧 `training.ipynb`
- Located at `challenge-1/notebooks/training.ipynb`
- Loads and preprocesses the dataset
- Computes dataset statistics and class weights
- Applies augmentations and transformations
- Trains a **ResNet50** model using PyTorch
- Tracks and logs metrics (accuracy, F1 macro/min)
- Saves the best model to `best_resnet50.pth`

### 🔍 `inference.ipynb`
- Loads the trained model
- Applies test-time preprocessing
- Generates predictions on test images
- Saves predictions to `submission.csv` in competition format

---

## 🛠️ Key Modules

### `src/preprocessing.py`
- `SoilDataset`: PyTorch-compatible dataset loader
- `compute_dataset_stats`: Calculates mean and std for normalization
- `get_transforms`: Defines augmentation and normalization pipelines

### `src/postprocessing.py`
- `predict`: Runs inference on the test/validation set
- `evaluate`: Computes F1 scores, confusion matrix, classification report
- `create_submission`: Generates submission CSV from predictions

---

## 🧠 Model Info

- Model: `ResNet50` (pretrained backbone)
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss` with class weights
- Augmentations: Flip, Rotation, Affine transform
- Metric: **Minimum F1-score across all classes**

---

## 📊 Final Performance

From `ml-metrics.json` (see `docs/cards/`):

| Soil Type       | F1 Score |
|------------------|----------|
| Alluvial soil    | 0.98     |
| Black Soil       | 0.98     |
| Clay soil        | 0.98     |
| Red soil         | 0.99     |

---

## 🧪 Dataset Format

The dataset should be placed as:

```

data/
├── train/
├── test/
├── train\_labels.csv
└── test\_ids.csv

````

Each row in `train_labels.csv` must contain:
- `image_id`
- `soil_type`

Each row in `test_ids.csv` must contain:
- `image_id`

---

## 🔧 Installation

Install dependencies:

```bash
pip install -r requirements.txt
````

If you're using GPU with CUDA:

```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📤 Submission Format

Output CSV (`submission.csv`) must have:

```
image_id,soil_type
img_123.jpg,Red soil
img_456.jpg,Alluvial soil
...
```

---

## 👨‍💻 Authors

Team: **SoilClassifiers**

* Vaibhav Sharma
* Shreya Khantal
* Prasanna Saxena

---

## 🖼️ References

* `architecture.png`: High-level model flow diagram
* `ml-metrics.json`: Final F1 scores per class

---

## 📄 License

This project is for academic and research purposes only.

