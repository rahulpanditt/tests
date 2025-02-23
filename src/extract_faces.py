from facenet_pytorch import MTCNN
from PIL import Image
import os

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

def extract_faces(frame_folder, output_folder):
    """
    Detects and extracts faces from frames using MTCNN.
    
    Parameters:
        frame_folder (str): Folder containing extracted frames.
        output_folder (str): Folder where cropped faces will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for frame in sorted(os.listdir(frame_folder)):
        frame_path = os.path.join(frame_folder, frame)
        img = Image.open(frame_path)
        
        faces, _ = mtcnn.detect(img)
        
        if faces is not None:
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = map(int, face)
                face_crop = img.crop((x1, y1, x2, y2)).resize((224, 224))
                face_crop.save(os.path.join(output_folder, f"face_{i}.jpg"))
    
    print(f"✅ Faces extracted and saved in {output_folder}")

# Example Usage
# extract_faces("data/extracted_frames/", "data/extracted_faces/")
