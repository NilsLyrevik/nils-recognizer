# Nils Recognizer

A simple image recognition project using PyTorch and ResNet18.  
The repository supports two modes:

1. **Generic ImageNet classification** – run predictions with a pretrained ResNet18 on ImageNet.
2. **Custom classifier** – fine-tuned ResNet18 to distinguish between images of Nils and not Nils.

---

## Project Structure

.
├── data/ # Training and validation datasets
│ ├── train/
│ │ ├── nils/ # Images of Nils
│ │ └── not_nils/ # Images of others
│ └── val/ # Validation split
├── models/ # Saved models
├── src/ # Source code
│ ├── train.py # Training script
│ └── predict.py # Prediction script
└── requirements.txt # Python dependencies


---

## Setup

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt

2. Ensure you have PyTorch installed. For GPU support, install with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Or for CPU-only:
pip install torch torchvision torchaudio

## Training

Organize your dataset as follows:
data/
├── train/
│   ├── nils/
│   └── not_nils/
└── val/
    ├── nils/
    └── not_nils/

Then run:
python src/train.py

## Prediction
Run predictions with either the pretrained ImageNet model or the fine-tuned Nils model.

Using ImageNet:
python src/predict.py imagenet path/to/image.jpg

Using Nils recognizer:
python src/predict.py nils path/to/image.jpg

