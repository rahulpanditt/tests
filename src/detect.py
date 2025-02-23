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

# Example Usage
# print(detect_deepfake("dataset/test_faces/"))
