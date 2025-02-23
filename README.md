# **Deepfake Detection Model - Training & Inference Guide**

This repository contains the implementation of a **Deepfake Detection Model** using **Swin Transformer** for feature extraction and **LSTM** for classification.

## **Prerequisites**

Ensure you have Python **3.8+** installed. Then, install the required dependencies:

```bash
pip install torch torchvision timm facenet-pytorch opencv-python pillow numpy scikit-learn
```

### **Check GPU Availability (Optional but Recommended)**

```python
import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
```

---

## **Project Structure**

```
Deepfake-Detection/
├── data/
│   ├── videos/                  # Raw video files (real & fake videos)
│   │   ├── real/
│   │   │   ├── real_video_1.mp4
│   │   │   ├── real_video_2.mp4
│   │   ├── fake/
│   │   │   ├── fake_video_1.mp4
│   │   │   ├── fake_video_2.mp4
│   ├── extracted_frames/        # Frames extracted from videos
│   │   ├── real/
│   │   ├── fake/
│   ├── extracted_faces/         # Faces extracted from frames
│   │   ├── real/
│   │   ├── fake/
│   ├── train_faces/             # Training dataset (real/fake faces)
│   │   ├── real/
│   │   ├── fake/
│   ├── test_faces/              # Testing dataset (real/fake faces)
│   │   ├── real/
│   │   ├── fake/
├── dataset/
│   ├── extracted_features/      # Features extracted using Swin Transformer
│   │   ├── real.pt              # All real face features stored in a single file
│   │   ├── fake.pt              # All fake face features stored in a single file
│   ├── test_features/           # Features for evaluation
│   │   ├── real.pt
│   │   ├── fake.pt
├── models/
│   ├── swin_model_custom.pth    # Trained Swin Transformer model
│   ├── lstm_model_custom.pth    # Trained LSTM model
├── extract_frames.py            # Extract frames from video
├── extract_faces.py             # Detect and extract faces
├── swin_feature_extraction.py   # Feature extraction using Swin Transformer
├── train_swim.py                # Train the Swin Transformer model
├── train_lstm.py                # Train the LSTM model
├── detect.py                    # Deepfake detection script
├── evaluate.py                  # Model evaluation script
├── README.md                    # Project documentation
```

---

## **Step 6: Model Evaluation**

To assess the performance of the trained deepfake detection model, you need to evaluate it on a labeled test dataset.

### **1. Prepare a Test Dataset**
- Ensure the test dataset contains both **real** and **deepfake** faces.
- The extracted features for the test dataset should be in:
  ```
  dataset/test_features/
  ├── real.pt     # Pre-extracted real face features
  ├── fake.pt     # Pre-extracted deepfake face features
  ```

### **2. Run Model Evaluation**
Run the evaluation script:
```bash
python evaluate.py
```
This will compute the model's **accuracy, precision, recall, and F1-score** and display the results.

---

### **License**
This project is open-source and free to use.

