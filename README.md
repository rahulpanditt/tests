# **Deepfake Detection Model - Training & Inference Guide**

This repository contains the implementation of a **Deepfake Detection Model** using **Swin Transformer** for feature extraction and **LSTM** for classification.

## **Prerequisites**

Ensure you have Python **3.8+** installed. Then, install the required dependencies:

```bash
pip install torch torchvision timm facenet-pytorch opencv-python pillow numpy
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
│   │   ├── real/
│   │   ├── fake/
├── models/
│   ├── swin_model_custom.pth    # Trained Swin Transformer model
│   ├── lstm_model_custom.pth    # Trained LSTM model
├── extract_frames.py            # Extract frames from video
├── extract_faces.py             # Detect and extract faces
├── swin_feature_extraction.py   # Feature extraction using Swin Transformer
├── train_swim.py                # Train the Swin Transformer model
├── train_lstm.py                # Train the LSTM model
├── detect.py                    # Deepfake detection script
├── README.md                    # Project documentation
```

---

## **Step 1: Prepare the Dataset**

Before training, you need to extract **frames** and **faces** from videos.

### **1. Organize Raw Videos with Labels**

- Store raw videos inside `data/videos/`.
- Create two separate subfolders:
  ```
  data/videos/
  ├── real/
  │   ├── real_video_1.mp4
  │   ├── real_video_2.mp4
  │   └── ...
  ├── fake/
  │   ├── fake_video_1.mp4
  │   ├── fake_video_2.mp4
  │   └── ...
  ```
- Ensure the **real** videos contain genuine content, while **fake** videos are deepfake-generated.

### **2. Extract Frames from Videos**

- Extract frames using:

```bash
python extract_frames.py --input data/videos/real/video.mp4 --output data/extracted_frames/real/
python extract_frames.py --input data/videos/fake/video.mp4 --output data/extracted_frames/fake/
```

- Extracted frames will be stored in corresponding real and fake folders.

### **3. Extract Faces from Frames**

- Detect and crop faces from frames:

```bash
python extract_faces.py --input data/extracted_frames/real/ --output data/extracted_faces/real/
python extract_faces.py --input data/extracted_frames/fake/ --output data/extracted_faces/fake/
```

- Cropped faces will be saved in respective real and fake folders.

---

## **Step 2: Train the Swin Transformer Model**

The **Swin Transformer** extracts features from the detected faces.

### **1. Ensure Dataset is Ready**

Organize `data/train_faces/` as follows:

```
data/train_faces/
├── real/
│   ├── face_1.jpg
│   ├── face_2.jpg
│   └── ...
├── fake/
│   ├── face_1.jpg
│   ├── face_2.jpg
│   └── ...
```

### **2. Train the Swin Model**

Run the training script:

```bash
python train_swim.py
```

This will train the Swin Transformer and save the model weights to:

```
models/swin_model_custom.pth
```

---

## **Step 3: Extract Features Using Swin Transformer**

Extract features from the trained Swin model:

```bash
python swin_feature_extraction.py --input data/extracted_faces/ --output dataset/extracted_features/
```

- This will generate feature vectors and save them in `dataset/extracted_features/`.

---

## **Step 4: Train the LSTM Model**

The **LSTM model** takes extracted features and classifies faces as **real** or **deepfake**.

### **1. Ensure Extracted Features Exist**

`dataset/extracted_features/` should contain feature files.

### **2. Train the LSTM Model**

Run the training script:

```bash
python train_lstm.py
```

This will train the LSTM model and save it as:

```
models/lstm_model_custom.pth
```

---

### **License**
This project is open-source and free to use.
