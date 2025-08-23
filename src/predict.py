# src/predict.py
import sys
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

# Default (ImageNet) weights
imagenet_weights = ResNet18_Weights.DEFAULT
imagenet_model = models.resnet18(weights=imagenet_weights)
imagenet_model.eval()
imagenet_preprocess = imagenet_weights.transforms()
imagenet_labels = imagenet_weights.meta["categories"]

def predict_imagenet(image_path: str) -> str:
    image = Image.open(image_path)
    input_tensor = imagenet_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = imagenet_model(input_tensor)
        _, predicted = outputs.max(1)
    return imagenet_labels[predicted.item()]

# Custom (Nils) model
def load_nils_model(model_path="models/nils_resnet18.pth", class_names=["nils", "not_nils"]):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, class_names

custom_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_nils(image_path: str, model, class_names) -> str:
    image = Image.open(image_path)
    input_tensor = custom_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
    return class_names[predicted.item()]

# CLI usage
if __name__ == "__main__":
    mode = sys.argv[1]   # "imagenet" or "nils"
    image_path = sys.argv[2]

    if mode == "imagenet":
        print(predict_imagenet(image_path))
    elif mode == "nils":
        model, classes = load_nils_model()
        print(predict_nils(image_path, model, classes))
