# src/predict.py
import sys
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

# Load pretrained model & weights
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Preprocessing pipeline (from weights transforms)
preprocess = weights.transforms()

# Labels from weights metadata
LABELS = weights.meta["categories"]

def predict(image_path: str) -> str:
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)

    return LABELS[predicted.item()]

# Allow running from CLI
if __name__ == "__main__":
    image_path = sys.argv[1]
    print(predict(image_path))
# src/predict.py
import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained model once
model = models.resnet18(pretrained=True)
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load labels
with open("imagenet_classes.txt") as f:
    LABELS = [line.strip() for line in f.readlines()]

def predict(image_path: str) -> str:
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)

    return LABELS[predicted.item()]
# src/predict.py
import sys
import torch
from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image

# Load pretrained model & weights
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Preprocessing pipeline (from weights transforms)
preprocess = weights.transforms()

# Labels from weights metadata
LABELS = weights.meta["categories"]

def predict(image_path: str) -> str:
    image = Image.open(image_path)
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)

    return LABELS[predicted.item()]

# Allow running from CLI
if __name__ == "__main__":
    image_path = sys.argv[1]
    print(predict(image_path))
