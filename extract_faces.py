from facenet_pytorch import MTCNN
from PIL import Image
import os

# Initialize MTCNN with improved settings
mtcnn = MTCNN(keep_all=True, min_face_size=20, thresholds=[0.6, 0.7, 0.7])

def extract_faces(frame_folder, output_folder):
    """
    Detects and extracts faces from frames using MTCNN.
    
    Parameters:
        frame_folder (str): Folder containing extracted frames.
        output_folder (str): Folder where cropped faces will be saved.
    """
    label = "real" if "real" in frame_folder.lower() else "fake"
    labeled_output_folder = os.path.join(output_folder, label)
    os.makedirs(labeled_output_folder, exist_ok=True)
    
    for frame in sorted(os.listdir(frame_folder)):
        frame_path = os.path.join(frame_folder, frame)
        
        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"Error reading {frame_path}: {e}")
            continue
        
        faces, _ = mtcnn.detect(img)
        
        print(f"Processing {frame} - Detected faces: {len(faces) if faces is not None else 0}")

        if faces is not None:
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face)
                face_crop = img.crop((x1, y1, x2, y2)).resize((224, 224))
                face_crop.save(os.path.join(labeled_output_folder, f"{frame[:-4]}_face_{i}.jpg"))

    print(f"✅ Faces extracted and saved in {labeled_output_folder}")

# Example Usage
# extract_faces("data/extracted_frames/real", "data/extracted_faces/")
