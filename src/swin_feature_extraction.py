import torch
from torchvision import transforms
from timm import create_model
from PIL import Image
import os

# Load trained Swin model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin_model = create_model("swin_base_patch4_window7_224", num_classes=2, pretrained=False)
swin_model.load_state_dict(torch.load("models/swin_model_custom.pth"))
swin_model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(face_folder):
    feature_list = []

    for face in sorted(os.listdir(face_folder)):
        face_path = os.path.join(face_folder, face)
        img = Image.open(face_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = swin_model.forward_features(img)

        feature_list.append(features.squeeze().flatten())

    return torch.stack(feature_list)

# Example Usage
# features = extract_features("dataset/test_faces/")
