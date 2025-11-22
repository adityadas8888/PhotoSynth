import torch
import cv2
import os
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

class Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_model = None
        self.grounding_dino_model = None
        self.face_app = None
        self._load_models()

    def _load_models(self):
        """
        Loads SAM 3, Grounding DINO, and InsightFace.
        """
        print("🔍 Loading Detection Models...")
        
        # --- InsightFace (Face Detection & Embedding) ---
        # providers=['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace Loaded")

        # --- Grounding DINO & SAM 3 (Placeholders for now) ---
        print("✅ SAM 3 + Grounding DINO (Placeholder) Loaded")

    def run_detection(self, image_path):
        """
        Runs full detection pipeline:
        1. Face Detection & Embedding (InsightFace)
        2. Object Detection (Grounding DINO) - Placeholder
        3. Segmentation (SAM 3) - Placeholder
        """
        print(f"🔍 Running detection on {image_path}...")
        
        # 1. Face Analysis
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error reading image: {image_path}")
            return {}

        faces = self.face_app.get(img)
        face_data = []
        
        faces_dir = os.path.expanduser("~/personal/PhotoSynth/faces_crop")
        os.makedirs(faces_dir, exist_ok=True)

        for i, face in enumerate(faces):
            # Save Crop
            bbox = face.bbox.astype(int)
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Generate unique face ID (hash of embedding or file+index)
            # For now, simple filename
            face_filename = f"{os.path.basename(image_path)}_{i}.jpg"
            crop_path = os.path.join(faces_dir, face_filename)
            
            if crop.size > 0:
                cv2.imwrite(crop_path, crop)
            
            face_data.append({
                "bbox": bbox.tolist(),
                "embedding": face.embedding.tolist(), # 512-d vector
                "crop_path": crop_path,
                "det_score": float(face.det_score)
            })

        print(f"   Found {len(faces)} faces.")
        
        return {
            "status": "SUCCESS", 
            "faces": face_data,
            "objects": [] # Placeholder for DINO results
        }
