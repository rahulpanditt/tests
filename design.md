                     ┌─────────────────────────────┐
                     │        Input Video          │
                     └───────────┬─────────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Extract Frames       │  ← extract_frames.py
                     │   (OpenCV)             │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Extract Faces        │  ← extract_faces.py
                     │   (MTCNN)              │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Fine-Tune Swin       │  ← train_swin.py
                     │   (Custom Dataset)     │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Extract Features     │  ← extract_features.py
                     │   (Fine-Tuned Swin)    │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Train LSTM Model    │  ← train_lstm.py
                     │   (Temporal Analysis) │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Evaluate Model       │  ← evaluate.py
                     │   (Confusion Matrix,   │
                     │   ROC Curve, Accuracy)│
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │   Predict Deepfake     │  ← detect.py
                     │   (Swin + LSTM)        │
                     └───────────┬───────────┘
                                 │
                     ┌───────────▼───────────┐
                     │    Output: Real /     │
                     │    Deepfake           │
                     └───────────────────────┘
