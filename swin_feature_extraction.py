import torch
from torchvision import transforms
from timm import create_model
from PIL import Image
import os

# Load trained Swin model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin_model = create_model("swin_base_patch4_window7_224", num_classes=2, pretrained=False)
swin_model.load_state_dict(torch.load("models/swin_model_best.pth"))
swin_model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_and_save_features(face_root_folder, feature_output_folder):
    """
    Extract features for real and fake faces and store them in a single file per category.

    Parameters:
        face_root_folder (str): Path to extracted faces (should contain 'real/' and 'fake/')
        feature_output_folder (str): Path where extracted features will be saved
    """
    categories = ["real", "fake"]
    os.makedirs(feature_output_folder, exist_ok=True)

    for category in categories:
        face_folder = os.path.join(face_root_folder, category)
        all_features = []

        for face in sorted(os.listdir(face_folder)):
            face_path = os.path.join(face_folder, face)
            img = Image.open(face_path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = swin_model.forward_features(img)

            feature_tensor = features.squeeze().flatten()
            all_features.append(feature_tensor)

        # Save all features of this category in a single file
        torch.save(torch.stack(all_features), os.path.join(feature_output_folder, f"{category}.pt"))
        print(f"✅ Features for {category} saved in {feature_output_folder}/{category}.pt")

# Example Usage
# extract_and_save_features("data/extracted_faces/", "dataset/extracted_features/")
