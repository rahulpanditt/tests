import os
import torch
from lstm_model import CustomLSTM
from swin_feature_extraction import extract_features

# Load trained LSTM model
lstm_model = CustomLSTM()
lstm_model.load_state_dict(torch.load("models/lstm_model_custom.pth"))
lstm_model.eval()

def detect_deepfake(face_folder):
    features = extract_features(face_folder).unsqueeze(0)
    
    with torch.no_grad():
        prediction = lstm_model(features)
        predicted_label = torch.argmax(prediction, dim=1).item()

    return "Deepfake" if predicted_label == 1 else "Real"

# Aggregating face-level detections for a video
def detect_video(video_faces_folder):
    deepfake_count = 0
    total_faces = 0

    for face in os.listdir(video_faces_folder):
        face_path = os.path.join(video_faces_folder, face)
        label = detect_deepfake(face_path)

        if label == "Deepfake":
            deepfake_count += 1
        total_faces += 1

    detection_ratio = deepfake_count / total_faces if total_faces > 0 else 0
    return "Deepfake" if detection_ratio > 0.5 else "Real"

# Example Usage
# print(detect_video("data/test_faces/"))
